from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import requests
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchai

@dataclass
class RunState:
    user_goal: str
    subtasks: List[dict] = field(default_factory=list)
    results: Dict[str, str] = field(default_factory=dict)
    status: str = "running"
@dataclass
class RunState:
    user_goal: str
    subtasks: List[dict] = field(default_factory=list)
    
@dataclass
class GenerationConfig:
    temperature: float = 0.2
    max_tokens: int = 2048
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    stream: bool = False
    thinking_mode: bool = False
    extra_body: Optional[Dict[str, Any]] = None

class NvidiaLLM:
    def __init__(
        self,
        model_name="bytedance/seed-oss-36b-instruct",
        default_config: Optional[GenerationConfig] = None
    ):
        self.model_name = model_name
        self.api_key = os.getenv("SEED_OSS_MODEL")
        self.default_config = default_config or GenerationConfig()
        self.llm = ChatNVIDIA(
            model=model_name,
            api_key=self.api_key
        )
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ):
        cfg = config or self.default_config
        invoke_params = {
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "top_p": cfg.top_p,
        }
        # Optional params
        if cfg.stop:
            invoke_params["stop"] = cfg.stop
        if cfg.extra_body:
            invoke_params.update(cfg.extra_body)
        # Thinking mode example
        if cfg.thinking_mode:
            invoke_params["extra_body"] = {
                "thinking": True
            }
        response = self.llm.invoke(
            prompt,
            **invoke_params
        )
        return response.content

class BaseAgent:
    def __init__(self, llm):
        self.llm = llm

class PlannerAgent(BaseAgent):
    def run(self, state):
        prompt = f"""
        Break this goal into executable subtasks.
        Goal:
        {state.user_goal}
        Return bullet points.
        """
        response = self.llm.generate(prompt)
        tasks = []
        for idx, line in enumerate(response.split("\n")):
            if line.strip():
                tasks.append({
                    "task_id": f"task_{idx}",
                    "objective": line.strip()
                })
        state.subtasks = tasks
        return state

class WorkerAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm)
        self.agent_tools = AgentTools()
        
        # 1. Convert your AgentTools methods into standard LangChain tools
        self.tools = [
            StructuredTool.from_function(
                func=self.agent_tools.search_internet,
                name="search_internet",
                description="Search the internet for SAP documentation or general information.",
                args_schema=SearchInternet, 
            ),
            StructuredTool.from_function(
                func=self.agent_tools.save_documents,
                name="save_documents",
                description="Save extracted data, lists, or text to a local file.",
                args_schema=SaveDocumentRequest,
            ),
            StructuredTool.from_function(
                func=self.agent_tools.process_urls,
                name="process_urls",
                description="Scrape and extract text content from a list of web URLs.",
            )
        ]
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
