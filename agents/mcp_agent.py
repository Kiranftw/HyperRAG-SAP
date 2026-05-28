from __future__ import annotations
import json
import logging
import importlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from manifest_pipeline import generate_manifest_from_files
from multi_agents import BaseAgent
import sys
from pathlib import Path
tools_dir = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(tools_dir))

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
MANIFEST_OUTPUT_PATH = "/home/kiranftw/HyperRAG-SAP/agents/datasets/manifest_output.json"
TOKEN_RE = re.compile(r"[A-Za-z0-9_\-|]+")

def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t.lower() for t in TOKEN_RE.findall(text)]

@dataclass
class ToolManifest:
    name: str
    description: str = ""
    module: str = ""
    function_name: str = ""
    parameters: list[dict[str, Any]] = field(default_factory=list)
    source_file: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    @property
    def param_names(self) -> list[str]:
        return [p.get("name", "") for p in self.parameters if p.get("name")]
    @property
    def search_text(self) -> str:
        parts = [
            self.name,
            self.description,
            self.module,
            self.function_name,
            " ".join(self.param_names),
            " ".join(f"{p.get('name','')} {p.get('description','')}" for p in self.parameters),
        ]
        return " ".join(parts)

class ManifestLoader:
    @staticmethod
    def generate_manifest(mcp_files: list[str | Path], output_path: str | Path = MANIFEST_OUTPUT_PATH) -> dict[str, Any]:
        py_files = [str(path) for path in mcp_files if str(path).endswith(".py")]
        if not py_files:
            LOGGER.warning("No valid python files found for manifest generation.")
            return {}
        output_path = str(Path(output_path))
        LOGGER.info("Generating manifest for %d python files.", len(py_files))
        manifest_dump: dict[str, Any] = generate_manifest_from_files(
            file_paths=py_files,
            output_path=output_path,
        )
        LOGGER.info("Manifest generated successfully at: %s", output_path)
        return manifest_dump
    @staticmethod
    def load_manifest(path: str | Path) -> dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    @staticmethod
    def load_tools(path: str | Path) -> list[ToolManifest]:
        manifest = ManifestLoader.load_manifest(path)
        tools: list[ToolManifest] = []
        for tool in manifest.get("tools", []):
            tools.append(
                ToolManifest(
                    name=tool.get("name", ""),
                    description=tool.get("description", ""),
                    module=tool.get("module", ""),
                    function_name=tool.get("function_name", ""),
                    parameters=tool.get("parameters", []),
                    source_file=tool.get("source_file", ""),
                    raw=tool,
                )
            )
        LOGGER.info("Loaded %d tools from manifest.", len(tools))
        return tools

class MCPToolRegistry:
    def __init__(self):
        self.tools: dict[str, ToolManifest] = {}

    def register_tool(self, tool: ToolManifest) -> None:
        self.tools[tool.name.lower()] = tool

    def register_many(self, tools: list[ToolManifest]) -> None:
        for tool in tools:
            self.register_tool(tool)

    def get_tool(self, tool_name: str) -> Optional[ToolManifest]:
        return self.tools.get(tool_name.lower())

    def list_tools(self) -> list[ToolManifest]:
        return list(self.tools.values())

    def search(self, query: str, top_k: int = 5) -> list[ToolManifest]:
        q_tokens = set(tokenize(query))
        scored: list[tuple[float, ToolManifest]] = []
        for tool in self.tools.values():
            score = 0.0
            name = tool.name.lower()
            desc = tool.description.lower()
            tool_tokens = set(tokenize(tool.search_text))
            param_tokens = set(tokenize(" ".join(tool.param_names)))
            if query.lower() == name:
                score += 100.0
            if name.startswith(query.lower()):
                score += 50.0
            if query.lower() in name:
                score += 25.0
            overlap = len(q_tokens & tool_tokens)
            score += overlap * 4.0
            score += len(q_tokens & param_tokens) * 2.0
            if any(tok in desc for tok in q_tokens):
                score += 3.0
            scored.append((score, tool))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [tool for score, tool in scored[:top_k] if score > 0]

class ToolResolver:
    def __init__(self):
        self.cache: dict[str, Callable[..., Any]] = {}

    def resolve(self, tool: ToolManifest) -> Callable[..., Any]:
        cache_key = f"{tool.module}:{tool.function_name}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        if not tool.module:
            raise ValueError(f"{tool.name} missing module")
        if not tool.function_name:
            raise ValueError(f"{tool.name} missing function_name")
        module = importlib.import_module(tool.module)
        func = getattr(module, tool.function_name, None)
        if func is None or not callable(func):
            raise AttributeError(f"Could not resolve callable for {tool.name}")
        self.cache[cache_key] = func
        return func

class MCPRuntime:
    def __init__(self, registry: MCPToolRegistry):
        self.registry = registry
        self.resolver = ToolResolver()
    def execute(self, tool_name: str, payload: Any) -> Any:
        tool = self.registry.get_tool(tool_name)
        if not tool:
            raise KeyError(f"Tool not found: {tool_name}")
        func = self.resolver.resolve(tool)
        return func(payload)

class MCPDiscoveryService:
    def __init__(self, registry: MCPToolRegistry):
        self.registry = registry
    def discover(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        tools = self.registry.search(query, top_k=top_k)
        return [
            {
                "name": t.name,
                "description": t.description,
                "module": t.module,
                "function_name": t.function_name,
                "parameters": t.parameters,
                "source_file": t.source_file,
            }
            for t in tools
        ]
class ToolPolicyManager:
    READ_ONLY_PREFIXES = ("list", "get", "lookup", "search", "show")
    DANGEROUS_PREFIXES = ("delete", "remove", "drop", "unassign")
    @classmethod
    def requires_confirmation(cls, tool_name: str) -> bool:
        name = tool_name.lower()
        return name.startswith(cls.DANGEROUS_PREFIXES)

class MCPAgent(BaseAgent):
    def __init__(self, manifest_path: str, model_name: str = "gpt-oss:120b-cloud", role: str = "assistant"):
        super().__init__()
        self.registry = MCPToolRegistry()
        self.discovery = MCPDiscoveryService(self.registry)
        self.runtime = MCPRuntime(self.registry)
        self.policy_manager = ToolPolicyManager()
        self.load_manifest(manifest_path)
        self.model_name = model_name
        self.role = role

    def load_manifest(self, path: str) -> None:
        tools = ManifestLoader.load_tools(path)
        self.registry.register_many(tools)
    def discover_tools(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self.discovery.discover(query, top_k=top_k)
    def execute_tool(self, tool_name: str, payload: Any) -> Any:
        if self.policy_manager.requires_confirmation(tool_name):
            raise PermissionError(f"Execution of tool '{tool_name}' requires confirmation due to potential risks.")
        return self.runtime.execute(tool_name, payload)
    def list_available_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "module": t.module,
                "function_name": t.function_name,
                "parameters": t.parameters,
                "source_file": t.source_file,
            }
            for t in self.registry.list_tools()
        ]
    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "role": self.role,
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "module": t.module,
                    "function_name": t.function_name,
                    "parameters": t.parameters,
                    "source_file": t.source_file,
                }
                for t in self.registry.list_tools()
            ],
        }
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


#just an sample test func to check manifest working or not!
def build_default_manifest() -> dict[str, Any]:
    return ManifestLoader.generate_manifest(
        mcp_files=[
            "/home/kiranftw/HyperRAG-SAP/tools/mcp_full_server.py",
            "/home/kiranftw/HyperRAG-SAP/tools/p2p_mcp_server.py",
        ],
        output_path=MANIFEST_OUTPUT_PATH,
    )

# if __name__ == "__main__":
#     if not Path(MANIFEST_OUTPUT_PATH).exists():
#         build_default_manifest()

#     agent = MCPAgent(manifest_path=MANIFEST_OUTPUT_PATH)
#     query = "list distribution channels"
#     candidates = agent.discover_tools(query, top_k=5)
#     print("Candidate tools:")
#     #all tools that we have discovered from mco files
#     print(json.dumps(candidates, indent=2, ensure_ascii=False))
#     import asyncio
#     if candidates:
#         chosen_tool = "list_available_company_codes"
#         payload = json.dumps([{
#             "country_code": "IN"
#         }])
#         func = agent.runtime.resolver.resolve(agent.registry.get_tool(chosen_tool))
#         result = asyncio.run(func(payload))
#         print(result)

if __name__ == "__main__":

    # STEP 1:
    # Generate manifest if it does not exist

    if not Path(MANIFEST_OUTPUT_PATH).exists():

        LOGGER.info("Manifest does not exist. Generating manifest...")

        ManifestLoader.generate_manifest(
            mcp_files=[
                "/home/kiranftw/HyperRAG-SAP/tools/mcp_full_server.py",
                "/home/kiranftw/HyperRAG-SAP/tools/p2p_mcp_server.py",
            ],
            output_path=MANIFEST_OUTPUT_PATH,
        )

    # STEP 2:
    # Load tools from manifest

    LOGGER.info("Loading tools from manifest...")

    tools = ManifestLoader.load_tools(
        MANIFEST_OUTPUT_PATH
    )

    LOGGER.info("Loaded %d tools", len(tools))

    # STEP 3:
    # Create registry

    registry = MCPToolRegistry()

    # STEP 4:
    # Register all tools

    registry.register_many(tools)

    LOGGER.info(
        "Registered %d tools in registry",
        len(registry.list_tools())
    )

    # STEP 5:
    # Manual interactive testing

    while True:

        print("\n")
        print("=" * 80)
        print("MCP TOOL SEARCH TEST")
        print("=" * 80)

        query = input(
            "\nEnter search query (or 'exit'): "
        ).strip()

        if query.lower() == "exit":
            break

        results = registry.search(
            query=query,
            top_k=5,
        )

        print("\n")
        print(f"Top matches for: {query}")
        print("-" * 80)

        if not results:
            print("No matching tools found.")
            continue

        for idx, tool in enumerate(results, start=1):

            print(f"\n[{idx}] {tool.name}")

            print(f"Description:")
            print(f"  {tool.description}")

            print(f"Module:")
            print(f"  {tool.module}")

            print(f"Function:")
            print(f"  {tool.function_name}")

            print(f"Parameters:")
            print(
                json.dumps(
                    tool.parameters,
                    indent=2,
                    ensure_ascii=False,
                )
            )

            print(f"Source File:")
            print(f"  {tool.source_file}")

            print("-" * 80)