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
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from Tools import AgentTools, SearchInternet, SaveDocumentRequest, LOGGER
from prompts.planning import PLANNING_SYSTEM_PROMPT, NEXT_STEP_PROMPT
from manifest_pipeline import *
from langchain.agents import create_agent
TOOLS = AgentTools()

@dataclass
class RunState:
    user_goal:str
    run_id:str="default"
    subtasks:List[dict]=field(default_factory=list)
    results:Dict[str,Any]=field(default_factory=dict)
    status:str="running"
    current_task:Optional[str]=None

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
        self.chat_model = ChatOllama(
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
        response = self.chat_model.invoke(prompt)
        return response.content

class BaseAgent(OllamaLLM):
    def __init__(self, model_name="gpt-oss:120b-cloud", default_config: Optional[GenerationConfig] = None):
        super().__init__(model_name=model_name, default_config=default_config)
        # (test_query, response, print, return were removed to prevent __init__ from returning a value)
 

class ManagerAgent(BaseAgent):
    def build_plan_prompt(self, state:RunState)->str:
        return f"""{PLANNING_SYSTEM_PROMPT}
            You are creating an execution plan for an autonomous AI system.
            User Goal:
            {state.user_goal}

            Requirements:
            - Break the task into executable subtasks
            - Keep steps meaningful and high-level
            - Preserve execution order
            - Include verification-oriented tasks where useful
            - Each task should be independently executable
            - Return ONLY valid JSON

            Schema:
            {{
            "tasks":[
                {{
                "task_id":"task_1",
                "objective":"Describe the task objective",
                "status":"pending",
                "dependencies":[],
                "priority":1,
                "verification":"How success is validated",
                "worker_type":"generic",
                "acceptance_criteria":["..."]
                }}
            ]
            }}
        """
    def normalize_tasks(self, tasks:List[dict])->List[dict]:
        normalized=[]
        for idx,task in enumerate(tasks):
            normalized.append({
                "task_id":task.get("task_id",f"task_{idx}"),
                "objective":task.get("objective","").strip(),
                "status":task.get("status","pending"),
                "dependencies":task.get("dependencies",[]),
                "priority":task.get("priority",idx+1),
                "verification":task.get("verification","Task completed successfully"),
                "worker_type":task.get("worker_type","generic"),
                "result":None,
                "error":None
            })
        return normalized

    def plan(self, state:RunState)->RunState:
        prompt=self.build_plan_prompt(state)
        response=self.generate(prompt)
        try:
            parsed=json.loads(response)
            state.subtasks=self.normalize_tasks(parsed.get("tasks",[]))
        except Exception as e:
            LOGGER.error(f"Planning failed: {e}")
            state.subtasks=[{
                "task_id":"task_0",
                "objective":state.user_goal,
                "status":"pending",
                "dependencies":[],
                "priority":1,
                "verification":"Task completed successfully",
                "result":None,
                "error":str(e)
            }]
        return state

    def get_next_task(self,state:RunState)->Optional[dict]:
        completed={t["task_id"] for t in state.subtasks if t["status"]=="done"}
        for task in sorted(state.subtasks,key=lambda x:x.get("priority",999)):
            if task["status"]!="pending":
                continue
            if all(dep in completed for dep in task.get("dependencies",[])):
                return task
        return None

    def assign_task(self, state:RunState, worker_pool:Any)->RunState:
        task=self.get_next_task(state)
        if not task:
            return state
        worker=worker_pool.get_worker(task.get("worker_type", "generic"))
        if worker is None:
            return state
        task["status"]="assigned"
        state.current_task=task["task_id"]
        worker.submit(task)
        if task["status"] == "done":
            self.update_result(state, task["task_id"], result=task["result"])
        else:
            self.update_result(state, task["task_id"], error=task.get("error", "Execution failed"))
        return state

    def update_result(self,state:RunState,task_id:str,result:Any=None,error:Optional[str]=None):
        for task in state.subtasks:
            if task["task_id"]!=task_id:
                continue
            if error:
                task["status"]="failed"
                task["error"]=error
            else:
                task["status"]="done"
                task["result"]=result
                state.results[task_id]=result
            state.current_task=None
            return state
        return state

    def is_complete(self,state:RunState)->bool:
        return len(state.subtasks)>0 and all(t["status"]=="done" for t in state.subtasks)

    def run(self,state:RunState,worker_pool:Any)->RunState:
        if not state.subtasks:
            state=self.plan(state)
        if self.is_complete(state):
            state.status="complete"
            return state
        state=self.assign_task(state,worker_pool)
        if self.is_complete(state):
            state.status="complete"
        return state

class WorkerAgent(BaseAgent):
    def __init__(
        self,
        model_name="gpt-oss:120b-cloud",
        tools=None,
        role="generic"
    ):
        super().__init__(model_name=model_name)
        self.role = role
        self.agent_tools = tools or AgentTools()
        self.tools = []
        self.capabilities = []
        for name in dir(self.agent_tools):
            if name.startswith("_") or name == "hybrid_search":
                continue
            func = getattr(self.agent_tools, name)
            if callable(func):
                try:
                    tool = StructuredTool.from_function(func)
                    self.tools.append(tool)
                    self.capabilities.append(tool.name)
                except Exception:
                    pass
        #https://reference.langchain.com/python/langchain/agents/factory/create_agent#related-docs use this documnentation foir further reference
        self.react_agent = create_agent(
            model=self.chat_model,
            tools=self.tools,
            name=self.role,
            debug=True
        )

    def submit(self, task: dict) -> dict:
        objective = task.get("objective", "")
        LOGGER.info(f"Worker {self.role} executing task: {objective}")
        try:
            response = self.react_agent.invoke({"messages": [HumanMessage(content=objective)]})
            messages = response.get("messages", [])
            result = messages[-1].content if messages else "No output"
            task["result"] = result
            task["status"] = "done"
        except Exception as e:
            LOGGER.error(f"Worker {self.role} failed to execute task: {e}")
            task["status"] = "failed"
            task["error"] = str(e)
        return task

RAG_TOOL_NAMES = ("sap_knowledge_search", "query_decomposition", "search_internet")

class RetrivalAugumentedGenerationAgent(BaseAgent):
    """Dedicated worker for retrieval: local KB (hybrid_search via sap_knowledge_search), query decomposition, and web search."""
    def __init__(self, model_name="gpt-oss:120b-cloud", tools=None, role="rag"):
        super().__init__(model_name=model_name)
        self.role = role
        self.agent_tools = tools or TOOLS
        self.tools = []
        for name in RAG_TOOL_NAMES:
            func = getattr(self.agent_tools, name, None)
            if not callable(func):
                continue
            try:
                self.tools.append(StructuredTool.from_function(func))
            except Exception:
                LOGGER.warning(f"RAG agent: could not register tool {name}")
        self.react_agent = create_agent(
            model=self.chat_model,
            tools=self.tools,
            name="Retrieval Augmented Generation Agent",
            debug=True,
        )

    def submit(self, task: dict) -> dict:
        objective = task.get("objective", "")
        LOGGER.info(f"RAG worker {self.role} executing task: {objective}")
        try:
            response = self.react_agent.invoke({"messages": [HumanMessage(content=objective)]})
            messages = response.get("messages", [])
            result = messages[-1].content if messages else "No output"
            task["result"] = result
            task["status"] = "done"
        except Exception as e:
            LOGGER.error(f"RAG worker {self.role} failed to execute task: {e}")
            task["status"] = "failed"
            task["error"] = str(e)
        return task

        
class WorkerPool:
    def __init__(self, model_name="gpt-oss:120b-cloud"):
        self.model_name = model_name
        self.workers = {}

    def get_worker(self, role: str = "generic"):
        if role not in self.workers:
            if role in ("rag", "retrieval", "retrival_augmented_generation"):
                self.workers[role] = RetrivalAugumentedGenerationAgent(
                    model_name=self.model_name, role=role
                )
            else:
                self.workers[role] = WorkerAgent(model_name=self.model_name, role=role)
        return self.workers[role]

if __name__ == "__main__":
    planner = ManagerAgent()
    state = RunState(
        run_id="test", 
        user_goal="Search the web for the current population of Tokyo, and then save a document called tokyo.txt with the population."
    )
    pool = WorkerPool()
    # Run loop
    for _ in range(10):  # limit execution iterations
        if state.status == "complete":
            break
        state = planner.run(state, pool)
    print("Final Subtasks:")
    print(json.dumps(state.subtasks, indent=2))