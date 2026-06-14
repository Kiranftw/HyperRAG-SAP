from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import requests
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import json
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from llm.models import OllamaLLM, NvidiaLLM, SAPLLM
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from Tools import AgentTools, SearchInternet, SaveDocumentRequest, LOGGER
from prompts.planning import PLANNING_SYSTEM_PROMPT, VALIDATION_PROMPT, REACT_EXECUTION_PROMPT
from manifest_pipeline import *
import subprocess
from langchain.agents import create_agent
TOOLS = AgentTools()
MODELS = {
    "ollama": OllamaLLM,
    "nvidia": NvidiaLLM,
    "sap": SAPLLM
}

@dataclass
class RunState:
    # User Context
    run_id: str
    user_goal: str
    session_id: str
    # Planning
    plan: List[dict] = field(default_factory=list)
    current_task: Optional[str] = None
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    #ReAct Loop
    thoughts: List[str] = field(default_factory=list)
    actions: List[dict] = field(default_factory=list)
    observations: List[dict] = field(default_factory=list)
    reflections: List[str] = field(default_factory=list)
    #tools info and registry
    tool_history: List[dict] = field(default_factory=list)
    active_tool: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)
    #memory
    working_memory: Dict[str, Any] = field(default_factory=dict)
    retrieved_memories: List[dict] = field(default_factory=list)
    retrieved_documents: List[dict] = field(default_factory=list)
    #validation and error handling
    validation_errors: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    status: str = "running"
    total_tool_calls: int = 0
    total_tokens: int = 0

class BaseAgent(OllamaLLM):
    def __init__(self, model_name="gpt-oss:120b-cloud", default_config: Optional[Any] = None):
        super().__init__(model_name=model_name, default_config=default_config)

class ManagerAgent(BaseAgent):
    def build_plan_prompt(self, state: RunState, tool_manifest: str = "") -> str:
        tools_context = f"\nAvailable Tools:\n{tool_manifest}\n" if tool_manifest else ""
        return f"""{PLANNING_SYSTEM_PROMPT}
            User Goal:
            {state.user_goal}
            Session ID:
            {state.session_id}
            Run ID:
            {state.run_id}
            Existing Context:
            - Current status: {state.status}
            - Retry count: {state.retry_count}
            - Retrieved documents: {len(state.retrieved_documents)}
            - Retrieved memories: {len(state.retrieved_memories)}{tools_context}
            Task:
            Create a structured execution plan for this goal.
            Requirements:
            - Break the goal into small executable tasks
            - Include dependencies where needed
            - Include verification criteria for each task
            - Choose a worker_type for each task
            - Return ONLY valid JSON
            - Do not include markdown or explanations
            Schema:
            {{
            "goal_analysis": {{
                "primary_goal": "string",
                "task_type": "string",
                "complexity": "simple|moderate|complex",
                "requires_retrieval": true,
                "requires_document_processing": true,
                "requires_file_output": false,
                "requires_code_execution": false,
                "requires_web_search": false
            }},
            "tasks": [
                {{
                "task_id": "task_1",
                "objective": "string",
                "status": "pending",
                "dependencies": [],
                "priority": 1,
                "verification": "string",
                "worker_type": "generic",
                "acceptance_criteria": ["string"]
                }}
            ],
            "execution_strategy": {{
                "loop_style": "sequential|conditional|iterative",
                "retry_policy": "string",
                "fallback_policy": "string"
            }}
            }}"""

    def normalize_tasks(self, tasks: List[dict]) -> List[dict]:
        normalized = []
        for idx, task in enumerate(tasks):
            normalized.append({
                "task_id": task.get("task_id", f"task_{idx}"),
                "objective": task.get("objective", "").strip(),
                "status": task.get("status", "pending"),
                "dependencies": task.get("dependencies", []),
                "priority": task.get("priority", idx + 1),
                "verification": task.get("verification", "Task completed successfully"),
                "worker_type": task.get("worker_type", "generic"),
                "acceptance_criteria": task.get("acceptance_criteria", []),
                "result": None,
                "error": None
            })
        return normalized

    def plan(self, state: RunState, tool_registry: Optional[Any] = None) -> RunState:
        # Build the tool manifest from the registry so the LLM knows what's available
        inbuilt_tools_manifest: List[dict] = []
        if tool_registry:
            inbuilt_tools_manifest = tool_registry.build_tool_manifest()
            LOGGER.info(f"Loaded len({len(inbuilt_tools_manifest)}), tools manifest: {inbuilt_tools_manifest}")

        prompt = self.build_plan_prompt(state, inbuilt_tools_manifest)
        response = self.generate(prompt=prompt)
        LOGGER.info(f"Planning response: {response}")
        try:
            parsed = json.loads(response)
            state.plan = self.normalize_tasks(parsed.get("tasks", []))
            state.working_memory["goal_analysis"] = parsed.get("goal_analysis", {})
            state.working_memory["execution_strategy"] = parsed.get("execution_strategy", {})
        except Exception as e:
            LOGGER.error(f"Planning failed: {e}")
            state.validation_errors.append(f"Planning failed: {str(e)}")
            state.plan = [{
                "task_id": "task_0",
                "objective": state.user_goal,
                "status": "pending",
                "dependencies": [],
                "priority": 1,
                "verification": "Task completed successfully",
                "worker_type": "generic",
                "acceptance_criteria": [],
                "result": None,
                "error": str(e)
            }]
        return state

    def get_next_task(self, state: RunState) -> Optional[dict]:
        completed = {t["task_id"] for t in state.plan if t["status"] == "done"}
        for task in sorted(state.plan, key=lambda x: x.get("priority", 999)):
            if task["status"] != "pending":
                continue
            if all(dep in completed for dep in task.get("dependencies", [])):
                return task
        return None

    def assign_task(self, state: RunState, worker_pool: Any) -> RunState:
        task = self.get_next_task(state)
        if not task:
            return state
        worker = worker_pool.get_worker(task.get("worker_type", "generic"))
        if worker is None:
            task["status"] = "failed"
            task["error"] = f"No worker found for role: {task.get('worker_type', 'generic')}"
            state.failed_tasks.append(task["task_id"])
            return state
        task["status"] = "assigned"
        state.current_task = task["task_id"]
        state.active_tool = None
        worker.submit(task)
        if task["status"] == "done":
            self.update_result(state, task["task_id"], result=task.get("result"))
            state.completed_tasks.append(task["task_id"])
        else:
            self.update_result(state, task["task_id"], error=task.get("error", "Execution failed"))
            state.failed_tasks.append(task["task_id"])
        return state

    def update_result(self, state: RunState, task_id: str, result: Any = None, error: Optional[str] = None):
        for task in state.plan:
            if task["task_id"] != task_id:
                continue
            if error:
                task["status"] = "failed"
                task["error"] = error
                state.validation_errors.append(f"{task_id}: {error}")
            else:
                task["status"] = "done"
                task["result"] = result
                state.results[task_id] = result
            state.current_task = None
            return state
        return state

    def is_complete(self, state: RunState) -> bool:
        return len(state.plan) > 0 and all(t["status"] == "done" for t in state.plan)

    def run(self, state: RunState, worker_pool: Any, tool_registry: Optional[Any] = None) -> RunState:
        if not state.plan:
            state = self.plan(state, tool_registry)
        if self.is_complete(state):
            state.status = "complete"
            return state
        state = self.assign_task(state, worker_pool)
        if self.is_complete(state):
            state.status = "complete"
        elif any(t["status"] == "failed" for t in state.plan):
            state.status = "running"
        return state

class ToolRegistry:
    def __init__(self, agent_tools: Optional[AgentTools] = None):
        self._agent_tools = agent_tools or TOOLS
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._discover()
    def _discover(self) -> None:
        #discover only public methods defined directly on AgentTools (not inherited from AgenticRAG):
        import inspect
        own_methods = set(AgentTools.__dict__.keys())
        for name, method in inspect.getmembers(self._agent_tools, predicate=inspect.ismethod):
            if name.startswith("_"):
                continue
            if name not in own_methods:
                continue
            self.tools[name] = {
                "name": name,
                "description": inspect.getdoc(method) or "No description",
                "callable": method,
            }
    def get(self, name: str):
        return self.tools.get(name)
    def list_tools(self) -> List[str]:
        return list(self.tools.keys())
    def build_tool_manifest(self) -> str:
        manifest = []
        for name, tool in self.tools.items():
            manifest.append({
                "name": tool["name"],
                "description": tool["description"],
            })
        return json.dumps(manifest, indent=2)  

def main():
    question = input("Enter your question: ").strip()
    document_context = input("Paste document text or leave blank: ").strip()
    manager = ManagerAgent()
    registry = ToolRegistry()
    state = RunState(
        run_id="run_001",
        user_goal=question,
        session_id="session_001"
    )
    if document_context:
        state.working_memory["document_context"] = document_context
    state = manager.plan(state, registry)
    print(json.dumps(state.working_memory.get("goal_analysis", {}), indent=2))
    print(json.dumps(state.working_memory.get("execution_strategy", {}), indent=2))
    print(json.dumps(state.plan, indent=2))
    return state

if __name__ == "__main__":
    state = main()
    print(state)