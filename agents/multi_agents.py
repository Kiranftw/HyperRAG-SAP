from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import requests
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import json
from langchain_ollama import ChatOllama
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from Tools import AgentTools, SearchInternet, SaveDocumentRequest, LOGGER
from prompts.planning import PLANNING_SYSTEM_PROMPT, NEXT_STEP_PROMPT

@dataclass
class RunState:
    user_goal: str
    subtasks: List[dict] = field(default_factory=list)
    results: Dict[str, str] = field(default_factory=dict)
    status: str = "running"

@dataclass
class GenerationConfig:
    temperature: float = 0.2
    max_tokens: int = 6000
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    stream: bool = False
    thinking_mode: bool = False
    extra_body: Optional[Dict[str, Any]] = None

class TokenTrackerCallback(BaseCallbackHandler):
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        response_filepath = "/home/kiranftw/HyperRAG-SAP/agents/generated/responses.json"
        os.makedirs(os.path.dirname(response_filepath), exist_ok=True)
        with open(response_filepath, "w") as f:
            json.dump(response, f, indent=2)
            LOGGER.info(f"Response saved to {response_filepath}")
        for gen_list in response.generations:
            for gen in gen_list:
                if hasattr(gen, "message"):
                    usage = getattr(gen.message, "usage_metadata", None)
                    if usage:
                        self.total_prompt_tokens += usage.get("input_tokens", 0)
                        self.total_completion_tokens += usage.get("output_tokens", 0)
                        self.total_tokens += usage.get("total_tokens", 0)
                        continue

class OllamaLLM:
    def __init__(
        self,
        model_name="gpt-oss:120b-cloud",
        default_config: Optional[GenerationConfig] = None
    ):
        self.model_name = model_name
        self.default_config = default_config or GenerationConfig()
        self.token_tracker = TokenTrackerCallback()
        self.llm = ChatOllama(
            model=model_name,
            temperature=self.default_config.temperature,
            num_ctx=10000,
            callbacks=[self.token_tracker],
            extra_body={
                "thinking_mode": self.default_config.thinking_mode,
            },
        )
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ):
        if not prompt: return None
        response = self.llm.invoke(prompt)
        return response.content

class BaseAgent:
    def __init__(self, llm):
        self.llm = llm

class PlannerAgent(BaseAgent):
    def build_prompt(self, state):
        return f"""
        {PLANNING_SYSTEM_PROMPT}
        You are creating an execution plan for an autonomous AI system.
        User Goal:
        {state.user_goal}
        Requirements:
        - Break the task into executable subtasks
        - Keep steps meaningful and high-level
        - Avoid tiny micro-steps
        - Preserve execution order
        - Include verification-oriented tasks where useful
        - Tasks should be independently executable
        Return ONLY valid JSON.
        Schema:
        {{
            "tasks": [
                {{
                    "task_id": "task_1",
                    "objective": "Describe the task objective",
                    "status": "pending",
                    "dependencies": [],
                    "priority": 1,
                    "verification": "How success is validated"
                }}
            ]
        }}
        """
    def normalize_tasks(self, tasks):
        normalized = []
        for idx, task in enumerate(tasks):
            normalized.append({
                "task_id": task.get("task_id", f"task_{idx}"),
                "objective": task.get("objective", "").strip(),
                "status": task.get("status", "pending"),
                "dependencies": task.get("dependencies", []),
                "priority": task.get("priority", idx + 1),
                "verification": task.get(
                    "verification",
                    "Task completed successfully"
                ),
                "result": None,
                "error": None
            })
        return normalized

    def run(self, state):
        prompt = self.build_prompt(state)
        response = self.llm.generate(prompt)
        try:
            parsed = json.loads(response)
            tasks = parsed.get("tasks", [])
            state.subtasks = self.normalize_tasks(tasks)
        except Exception as e:
            LOGGER.error(f"Planner parsing failed: {e}")
            state.subtasks = [{
                "task_id": "task_0",
                "objective": state.user_goal,
                "status": "pending",
                "dependencies": [],
                "priority": 1,
                "verification": "Task completed successfully",
                "result": None,
                "error": str(e)
            }]
        return state

class WorkerAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm)
        self.agent_tools = AgentTools()
        # 1. Automatically discover and load all tools from AgentTools!
        # LangChain reads the docstrings and type hints we just added to build the tools.
        self.tools = []
        for name in dir(self.agent_tools):
            # Ignore internal python methods (like __init__) and helper methods
            if not name.startswith("_") and name != "get_langchain_tools":
                func = getattr(self.agent_tools, name)
                if callable(func):
                    try:
                        self.tools.append(StructuredTool.from_function(func))
                    except ValueError:
                        pass # Skip methods that can't be converted to tools
        # 2. Create an agent that automatically handles tool execution

        # Note: We pass self.llm.llm to pass the actual ChatOllama instance
        self.react_agent = create_react_agent(self.llm.llm, tools=self.tools)

    def run(self, task, state):
        prompt = f"""
        Execute this task using the tools available to you if necessary.

        Task:
        {task['objective']}
        """
        # 3. Run the tool-calling loop
        response = self.react_agent.invoke({"messages": [HumanMessage(content=prompt)]})
        # The final answer after tools have been used is the last message
        result = response["messages"][-1].content 
        state.results[task["task_id"]] = result
        return result

if __name__ == "__main__":
    planner = PlannerAgent(OllamaLLM())
    state = RunState(user_goal="Extract the data from the document")
    planner.run(state)
    print(state.subtasks)