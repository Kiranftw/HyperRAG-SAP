from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import requests
import os
import sys
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

# Add tools directory to path so we can import tool modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tools_dir = os.path.join(parent_dir, "tools")
if tools_dir not in sys.path:
    sys.path.append(tools_dir)
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
    plan: Dict[str, Any] = field(default_factory=lambda: {"nodes": [], "edges": []})
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    # Memory
    working_memory: Dict[str, Any] = field(default_factory=dict)
    retrieved_memories: List[dict] = field(default_factory=list)
    retrieved_documents: List[dict] = field(default_factory=list)
    # Validation and error handling
    validation_errors: List[str] = field(default_factory=list)
    retry_count: int = 0
    status: str = "running"

class BaseAgent(OllamaLLM):
    def __init__(self, model_name="gpt-oss:120b-cloud", default_config: Optional[Any] = None):
        super().__init__(model_name=model_name, default_config=default_config)

class ReActAgent(BaseAgent):
    def __init__(self, model_name="gpt-oss:120b-cloud", default_config: Optional[Any] = None) -> None:
        #this is main part 
        super().__init__(model_name=model_name, default_config=default_config)

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
        create a DAG plan as per the example!
        """

    def normalize_dag(self, dag: dict) -> Dict[str, Any]:
        nodes = []
        for idx, node in enumerate(dag.get("nodes", [])):
            nodes.append({
                "task_id": node.get("task_id", f"task_{idx}"),
                "objective": node.get("objective", "").strip(),
                "status": node.get("status", "pending"),
                "priority": node.get("priority", idx + 1),
                "verification": node.get("verification", "Task completed successfully"),
                "worker_type": node.get("worker_type", "generic"),
                "acceptance_criteria": node.get("acceptance_criteria", []),
                "result": None,
                "error": None
            })
        edges = []
        valid_ids = {n["task_id"] for n in nodes}
        for edge in dag.get("edges", []):
            src, dst = edge.get("from", ""), edge.get("to", "")
            if src in valid_ids and dst in valid_ids:
                edges.append({
                    "from": src,
                    "to": dst,
                    "type": edge.get("type", "control")
                })
        return {"nodes": nodes, "edges": edges}

    def plan(self, state: RunState, tool_registry: Optional[Any] = None, mcp_tools: Optional[List[dict]] = None) -> RunState:
        # Build the tool manifest from the registry and merge with MCP tools
        combined_tools: List[dict] = []
        if tool_registry:
            try:
                inbuilt_manifest_str = tool_registry.build_tool_manifest()
                if inbuilt_manifest_str:
                    combined_tools.extend(json.loads(inbuilt_manifest_str))
            except Exception as e:
                LOGGER.error(f"Failed to parse inbuilt tools manifest: {e}")
        if mcp_tools:
            combined_tools.extend(mcp_tools)

        tool_manifest_str = json.dumps(combined_tools, indent=2) if combined_tools else ""
        prompt = self.build_plan_prompt(state, tool_manifest_str)
        response = self.generate(prompt=prompt)
        #storing the response in FS for debugging DAG
        filepath = os.path.join("/home/kiranftw/HyperRAG-SAP/agents/generated", f"plan_{state.run_id}.json")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            parsed_response = json.loads(response)
            with open(filepath, "w") as f:
                LOGGER.info(f"saving response in {filepath}")
                json.dump(parsed_response, f, indent=2)
        except json.JSONDecodeError as e:
            LOGGER.error(f"Failed to parse response: {e}")
            # write raw response for debugging purposes
            try:
                with open(filepath, "w") as f:
                    f.write(response)
            except Exception:
                pass
            raise ValueError(f"Invalid JSON response from planning agent: {e}")
        try:
            parsed = json.loads(response)
            state.plan = self.normalize_dag(parsed.get("dag", {}))
            state.working_memory["goal_analysis"] = parsed.get("goal_analysis", {})
            state.working_memory["execution_strategy"] = parsed.get("execution_strategy", {})
        except Exception as e:
            LOGGER.error(f"Planning failed: {e}")
            state.validation_errors.append(f"Planning failed: {str(e)}")
            state.plan = {
                "nodes": [{
                    "task_id": "task_0",
                    "objective": state.user_goal,
                    "status": "pending",
                    "priority": 1,
                    "verification": "Task completed successfully",
                    "worker_type": "generic",
                    "acceptance_criteria": [],
                    "result": None,
                    "error": str(e)
                }],
                "edges": []
            }
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
    tool_manifest_files = [
        "/home/kiranftw/HyperRAG-SAP/tools/p2p_mcp_server.py",
        "/home/kiranftw/HyperRAG-SAP/tools/mcp_full_server.py"
    ]
    manifestpath = "/home/kiranftw/HyperRAG-SAP/agents/manifest.json"
    data = generate_manifest_from_files(tool_manifest_files, manifestpath)
    tool_info = [
        {
            "name": tool.get("name"),
            "description": tool.get("description")
        }
        for tool in data.get("tools", [])
    ]
    agent = ReActAgent()
    registry = ToolRegistry()
    state = RunState(
        run_id="run_001",
        user_goal=question,
        session_id="session_001"
    )
    if document_context:
        state.working_memory["document_context"] = document_context

    # Phase 1: Plan
    state = agent.plan(state, registry, tool_info)
    print("\n═══ DAG PLAN ═══")
    print(json.dumps(state.plan, indent=2))

    # Phase 2: Execute
    print("\n═══ EXECUTING PLAN ═══")
    state = execute_plan(state, agent, registry)

    # Phase 3: Results
    print("\n═══ RESULTS ═══")
    print(f"Status: {state.status}")
    print(f"Completed: {state.completed_tasks}")
    print(f"Failed:    {state.failed_tasks}")
    for node in state.plan.get("nodes", []):
        print(f"  [{node['status']:>8}] {node['task_id']}: {node.get('result') or node.get('error', '')[:100]}")
    return state

if __name__ == "__main__":
    state = main()
    print(state)