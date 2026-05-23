from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from Tools import ToolManager
from RAG.agentic_rag import AgenticRAG

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

class TaskType(str, Enum):
    MANAGER = "manager"
    SUB_MANAGER = "sub_manager"
    WORKER = "worker"

    RESEARCH = "research"
    PLANNING = "planning"
    VALIDATION = "validation"

    COMPANY = "company"
    COMPANY_CODE = "company_code"
    SALES_ORG = "sales_org"
    PLANT = "plant"
    STORAGE_LOCATION = "storage_location"
    SHIPPING_POINT = "shipping_point"
    WAREHOUSE = "warehouse"

    CONFIGURATION = "configuration"
    TOOL_EXECUTION = "tool_execution"

    SYNTHESIS = "synthesis"

class EntityType(str, Enum):
    COMPANY = "company"
    COMPANY_CODE = "company_code"
    SALES_ORG = "sales_org"
    PLANT = "plant"
    STORAGE_LOCATION = "storage_location"
    SHIPPING_POINT = "shipping_point"
    WAREHOUSE = "warehouse"

class ToolSpec(BaseModel):
    tool_name: str
    description: str
    server_name: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    tags: List[str] = Field(default_factory=list)

class MCPManifest(BaseModel):
    server_name: str
    server_type: str
    tools: List[ToolSpec] = Field(default_factory=list)
    version: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SubTask(BaseModel):
    task_id: str
    task_type: TaskType
    objective: str
    assigned_agent: Optional[str] = None
    depends_on: List[str] = Field(default_factory=list)
    priority: int = 5
    metadata: Dict[str, Any] = Field(default_factory=dict)
    input_payload: Dict[str, Any] = Field(default_factory=dict)
    expected_output: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING

class AgentResult(BaseModel):
    task_id: str
    agent_name: str
    status: TaskStatus
    output: Any
    confidence: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    execution_time: Optional[float] = None
    token_usage: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


@dataclass
class RunState:    # CORE
    run_id: str
    user_goal: str
    original_input: Any = None
    parsed_input: List[ExcelRow] = field(default_factory=list)
    # TASKS
    subtasks: List[SubTask] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    active_tasks: List[str] = field(default_factory=list)
    # RESULTS
    results: Dict[str, AgentResult] = field(default_factory=dict)
    # ORG STRUCTURE
    org_blueprint: Dict[str, EntityNode] = field(default_factory=dict)
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)
    entity_registry: Dict[str, EntityNode] = field(default_factory=dict)
    validated_entities: List[str] = field(default_factory=list)
    # TOOLING
    available_tools: Dict[str, ToolSpec] = field(default_factory=dict)
    tool_cache: Dict[str, Any] = field(default_factory=dict)
    tool_failures: List[str] = field(default_factory=list)
    active_mcp_servers: List[str] = field(default_factory=list)
    # MEMORY
    notes: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    retrieved_documents: List[RetrievedDocument] = field(default_factory=list)
    # EXECUTION
    execution_plan: Optional[ExecutionPlan] = None
    execution_context: ExecutionContext = field(
        default_factory=ExecutionContext
    )
    execution_status: str = "running"
    retry_count: int = 0
    # VALIDATION
    validation_errors: List[ValidationErrorModel] = field(
        default_factory=list
    )
    warnings: List[str] = field(default_factory=list)
    # METRICS
    total_tokens_used: int = 0
    total_execution_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)

class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    async def run(
        self,
        task: SubTask,
        state: RunState
    ) -> AgentResult:
        raise NotImplementedError

    def create_result(
        self,
        task: SubTask,
        output: Any,
        confidence: float = 0.8,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentResult:

        return AgentResult(
            task_id=task.task_id,
            agent_name=self.name,
            status=TaskStatus.COMPLETED,
            output=output,
            confidence=confidence,
            metadata=metadata or {}
        )