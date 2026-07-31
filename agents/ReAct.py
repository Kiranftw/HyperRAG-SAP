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
    path = r"\\wsl.localhost\Ubuntu-22.04\home\kiranftw\sap-cloud-alm-ai\mcp\cloud_alm_mcp.py"
    path = normalize_file_path(path)
    manifestpath = "/home/kiranftw/HyperRAG-SAP/agents/manifest.json"
    data = generate_manifest_from_files([path], manifestpath)
    return data
    paths_raw = input(
        "Enter MCP file path(s) (comma-separated; Windows/WSL UNC/Linux paths supported): "
    ).strip()
    tool_manifest_files = [
        "\\wsl.localhost\Ubuntu-22.04\home\kiranftw\sap-cloud-alm-ai\mcp\cloud_alm_mcp.py"
    ]
    manifestpath = "/home/kiranftw/HyperRAG-SAP/agents/manifest.json"
    data = generate_manifest_from_files(tool_manifest_files, manifestpath)
    return data
    

if __name__ == "__main__":
    state = test()
    print(state)