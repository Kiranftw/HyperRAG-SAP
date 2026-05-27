from __future__ import annotations
from Tools import LOGGER
import ast
import json
import os
from pathlib import Path
from typing import Any, Optional, Union
from pydantic import BaseModel, Field

class ParameterSchema(BaseModel):
    name: str
    type: str = "Any"
    required: bool = True
    default: Optional[Any] = None
    description: Optional[str] = None

class ToolManifest(BaseModel):
    name: str
    description: str
    module: str
    function_name: str
    parameters: list[ParameterSchema] = Field(default_factory=list)
    source_file: str

class ManifestOutput(BaseModel):
    total_files: int
    total_tools: int
    tools: list[ToolManifest]
    skipped_files: list[str] = Field(default_factory=list)
    errors: list[dict] = Field(default_fakctory=list)

def get_annotation(annotation_node) -> str:
    if annotation_node is None:
        return "Any"
    try:
        return ast.unparse(annotation_node)
    except Exception:
        return "Any"

def extract_fastmcp_description(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Optional[str]:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Call):
            for keyword in decorator.keywords:
                if keyword.arg == "description":
                    if isinstance(keyword.value, ast.Constant):
                        return str(keyword.value.value)
    return None

def extract_docstring(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
    # 1. Check for description in @mcp.tool(description="...")
    desc = extract_fastmcp_description(node)
    if desc:
        return desc.strip()
    # 2. Fallback to function body docstring
    doc = ast.get_docstring(node)
    if not doc:
        return "No description available"
    return doc.strip()

def is_fastmcp_tool(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
    for decorator in node.decorator_list:
        # @mcp.tool()
        if isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Attribute):
                if decorator.func.attr == "tool":
                    return True
        # @tool
        elif isinstance(decorator, ast.Name):
            if decorator.id == "tool":
                return True
    return False

def extract_fastmcp_name(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Optional[str]:
    """
    Extract tool name from:
    @mcp.tool(name="xyz")
    """
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Call):
            for keyword in decorator.keywords:
                if keyword.arg == "name":
                    if isinstance(keyword.value, ast.Constant):
                        return str(keyword.value.value)
    return None

def extract_parameters(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> list[ParameterSchema]:
    parameters = []
    total_args = len(node.args.args)
    total_defaults = len(node.args.defaults)
    default_offset = total_args - total_defaults
    for idx, arg in enumerate(node.args.args):
        if arg.arg == "self":
            continue
        param_type = get_annotation(arg.annotation)
        required = True
        default_value = None
        if idx >= default_offset:
            required = False
            default_node = node.args.defaults[idx - default_offset]
            try:
                default_value = ast.literal_eval(default_node)
            except Exception:
                default_value = str(default_node)
        parameters.append(
            ParameterSchema(
                name=arg.arg,
                type=param_type,
                required=required,
                default=default_value,
            )
        )
    return parameters

def generate_manifest_from_files(
        file_paths: list[str],
        output_path: str,
    ) -> dict:
    py_files = [f for f in file_paths if f.endswith(".py")]
    manifest_tools = []
    skipped_files = []
    errors = []
    for filepath in py_files:
        if not os.path.exists(filepath):
            skipped_files.append(filepath)
            errors.append(
                {
                    "file": filepath,
                    "error": "File does not exist",
                }
            )
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                source_code = f.read()
            tree = ast.parse(source_code)
            LOGGER.info(f"PARSING FILE: {filepath}")
            module_name = Path(filepath).stem
            LOGGER.info(f"MODULE NAME: {module_name}")
            for node in ast.walk(tree):
                # only functions
                if not isinstance(
                    node,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                    ),
                ):
                    continue
                # skip private/internal funcs
                if node.name.startswith("_"):
                    continue
                # only FastMCP tools
                if not is_fastmcp_tool(node):
                    continue
                description = extract_docstring(node)
                parameters = extract_parameters(node)
                tool_name = (
                    extract_fastmcp_name(node)
                    or node.name
                )
                tool_manifest = ToolManifest(
                    name=tool_name,
                    description=description,
                    module=module_name,
                    function_name=node.name,
                    parameters=parameters,
                    source_file=filepath,
                )
                manifest_tools.append(tool_manifest)
        except Exception as e:
            errors.append(
                {
                    "file": filepath,
                    "error": str(e),
                }
            )
    # remove duplicates
    unique_tools = {}
    for tool in manifest_tools:
        if tool.name not in unique_tools:
            unique_tools[tool.name] = tool
    final_tools = list(unique_tools.values())
    manifest = ManifestOutput(
        total_files=len(py_files),
        total_tools=len(final_tools),
        tools=final_tools,
        skipped_files=skipped_files,
        errors=errors,
    )
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            manifest.model_dump(),
            f,
            indent=2,
            ensure_ascii=False,
        )
    return manifest.model_dump()