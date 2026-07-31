from __future__ import annotations
import importlib
import json
import logging
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from langchain_huggingface import HuggingFaceEmbeddings

from manifest_pipeline import generate_manifest_from_files, normalize_file_path

tools_dir = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(tools_dir))

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
MANIFEST_OUTPUT_PATH = os.path.join(tools_dir, "manifest.json")
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
        py_files = [
            normalize_file_path(str(path))
            for path in mcp_files
            if normalize_file_path(str(path)).endswith(".py")
        ]
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
            LOGGER.info("Loading tool: %s", tool.get("name", ""))
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

class BaseAgent:
    pass


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

def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)

@dataclass
class ToolRecord:
    tool_id: int
    name: str
    description: str
    module: str
    function_name: str
    parameters_json: str
    source_file: str
    manifest_path: str
    created_at: str
    updated_at: str

class DynamicToolDiscovery:
    def __init__(
        self,
        db_path: str = "tool_database_test.db",
        manifest_path: str = "/home/kiranftw/HyperRAG-SAP/agents/manifest.json",
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.db_path = normalize_file_path(db_path)
        self.manifest_path = normalize_file_path(manifest_path)
        manifest_data = ManifestLoader.load_manifest(self.manifest_path)
        if not manifest_data:
            raise ValueError("No manifest data found")
        self.tools = manifest_data.get("tools", [])

        self.tool_embeddings_model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tool_embeddings_model = HuggingFaceEmbeddings(
            model_name=self.tool_embeddings_model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.view_weights: dict[str, float] = {
            "name_desc": 0.45,
            "signature": 0.35,
            "rich": 0.20,
        }
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()
    
    def _embed_text(self, text: str) -> list[float]:
        return self.tool_embeddings_model.embed_query(text)
    
    def _create_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tools (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                module TEXT,
                function_name TEXT,
                parameters_json TEXT,
                source_file TEXT,
                manifest_path TEXT,
                raw_tool_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(name, module, function_name, source_file)
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_id INTEGER NOT NULL,
                view_name TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(tool_id) REFERENCES tools(id) ON DELETE CASCADE,
                UNIQUE(tool_id, view_name)
            )
            """
        )
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_tools_name ON tools(name)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_tools_function ON tools(function_name)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_tool_id ON tool_embeddings(tool_id)")
        self.connection.commit()

    def _load_manifest_tools(self, manifest_data: Any) -> list[dict[str, Any]]:
        if isinstance(manifest_data, dict):
            if "tools" in manifest_data and isinstance(manifest_data["tools"], list):
                return manifest_data["tools"]
            if "data" in manifest_data and isinstance(manifest_data["data"], list):
                return manifest_data["data"]
            if "items" in manifest_data and isinstance(manifest_data["items"], list):
                return manifest_data["items"]
        if isinstance(manifest_data, list):
            return manifest_data
        raise ValueError("Unsupported manifest format. Expected dict with tools/items/data or a list.")

    def _tool_views(self, tool: dict[str, Any]) -> dict[str, str]:
        name = str(tool.get("name", "") or "")
        description = str(tool.get("description", "") or "")
        module = str(tool.get("module", "") or "")
        function_name = str(tool.get("function_name", "") or "")
        parameters = tool.get("parameters", [])
        param_text = " ".join(
            f"{p.get('name', '')} {p.get('type', '')} {p.get('description', '')}"
            for p in parameters
            if isinstance(p, dict)
        )
        return {
            "name_desc": f"{name} {description}".strip(),
            "signature": f"{module} {function_name} {param_text}".strip(),
            "rich": f"{name} {description} {module} {function_name} {param_text} {tool.get('source_file', '')}".strip(),
        }

    def _upsert_tool(self, tool: dict[str, Any], manifest_path: str) -> int:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        name = str(tool.get("name", "") or "")
        description = str(tool.get("description", "") or "")
        module = str(tool.get("module", "") or "")
        function_name = str(tool.get("function_name", "") or "")
        parameters = tool.get("parameters", tool.get("parameters_json", []))
        source_file = str(tool.get("source_file", "") or "")
        raw_tool_json = _safe_json(tool)
        self.connection.execute(
            """
            INSERT INTO tools (name, description, module, function_name, parameters_json, source_file, manifest_path, raw_tool_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name, module, function_name, source_file)
            DO UPDATE SET
                description=excluded.description,
                parameters_json=excluded.parameters_json,
                manifest_path=excluded.manifest_path,
                raw_tool_json=excluded.raw_tool_json,
                updated_at=excluded.updated_at
            """,
            (name, description, module, function_name, _safe_json(parameters), source_file, manifest_path, raw_tool_json, now, now),
        )
        row = self.connection.execute(
            """
            SELECT id FROM tools
            WHERE name=? AND module=? AND function_name=? AND source_file=?
            """,
            (name, module, function_name, source_file),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Failed to upsert tool: {name}")
        return int(row["id"])
    
    def _upsert_embeddings(self, tool_id: int, views: dict[str, str]) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for view_name, text in views.items():
            embedding = self._embed_text(text)
            self.connection.execute(
                """
                INSERT INTO tool_embeddings (tool_id, view_name, embedding_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(tool_id, view_name)
                DO UPDATE SET embedding_json=excluded.embedding_json, created_at=excluded.created_at
                """,
                (tool_id, view_name, _safe_json(embedding), now),
            )
    
    def ingest_manifest(self, manifest_data: Any, manifest_path: Optional[str] = None) -> dict[str, Any]:
        manifest_path = normalize_file_path(manifest_path or self.manifest_path)
        tools = self._load_manifest_tools(manifest_data)
        inserted = 0
        updated = 0
        for tool in tools:
            tool_id = self._upsert_tool(tool, manifest_path)
            self._upsert_embeddings(tool_id, self._tool_views(tool))
            inserted += 1
            updated += 1
        self.connection.commit()
        return {
            "status": "ok",
            "manifest_path": manifest_path,
            "tools_processed": len(tools),
            "tools_upserted": inserted,
            "embeddings_upserted": updated * len(self.view_weights),
        }
    
    def discover(self, path: str) -> dict[str, Any]:
        path = normalize_file_path(path)
        data = generate_manifest_from_files([path], self.manifest_path)
        if isinstance(data, str) and os.path.exists(data):
            with open(data, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        elif isinstance(data, dict):
            manifest_data = data
        else:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        return self.ingest_manifest(manifest_data, self.manifest_path)
    
    def search_tools(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        query_embedding = np.array(self.tool_embeddings_model.embed_query(query), dtype=np.float32)
        rows = self.connection.execute(
            """
            SELECT
                t.id AS tool_id,
                t.name,
                t.description,
                t.module,
                t.function_name,
                t.parameters_json,
                t.source_file,
                t.manifest_path,
                e.view_name,
                e.embedding_json
            FROM tools t
            JOIN tool_embeddings e ON t.id = e.tool_id
            """
        ).fetchall()
        scored: dict[int, dict[str, Any]] = {}
        for row in rows:
            tool_id = int(row["tool_id"])
            view_name = str(row["view_name"])
            stored_embedding = np.array(json.loads(row["embedding_json"]), dtype=np.float32)
            score = _cosine_similarity(query_embedding, stored_embedding) * self.view_weights.get(view_name, 0.1)
            if tool_id not in scored:
                scored[tool_id] = {
                    "tool_id": tool_id,
                    "name": row["name"],
                    "description": row["description"],
                    "module": row["module"],
                    "function_name": row["function_name"],
                    "parameters_json": row["parameters_json"],
                    "source_file": row["source_file"],
                    "manifest_path": row["manifest_path"],
                    "score": 0.0,
                    "matched_views": [],
                }
            scored[tool_id]["score"] += score
            scored[tool_id]["matched_views"].append({"view_name": view_name, "view_score": round(float(score), 6)})
        ranked = sorted(scored.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]
    
    def close(self) -> None:
        self.connection.close()


def run_min_test(
    query: str = "manage projects in SAP Cloud ALM",
    manifest_path: str = "/home/kiranftw/HyperRAG-SAP/agents/manifest.json",
    db_path: str = "/tmp/tool_database_test.db",
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Minimal end-to-end test: load manifest -> ingest -> semantic search."""
    manifest_path = normalize_file_path(manifest_path)
    discovery = DynamicToolDiscovery(db_path=db_path, manifest_path=manifest_path)
    try:
        manifest_data = ManifestLoader.load_manifest(manifest_path)
        ingest_result = discovery.ingest_manifest(manifest_data, manifest_path)
        LOGGER.info("Ingest result: %s", ingest_result)
        results = discovery.search_tools(query, top_k=top_k)
        for rank, tool in enumerate(results, start=1):
            LOGGER.info("#%d %s (score=%.4f)", rank, tool["name"], tool["score"])
        return results
    finally:
        discovery.close()

if __name__ == "__main__":
    results = run_min_test()
    print(results)
