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
TOOLS = AgentTools()
MODELS = {
    "ollama": OllamaLLM,
    "nvidia": NvidiaLLM,
    "sap": SAPLLM
}
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

def test():
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

if __name__ == "__main__":
    state = test()
    print(state)