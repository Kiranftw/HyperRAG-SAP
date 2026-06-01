import sys
from pathlib import Path
# Add the agents directory to sys.path to allow absolute imports of other agent modules
agents_dir = Path(__file__).resolve().parents[1]
if str(agents_dir) not in sys.path:
    sys.path.insert(0, str(agents_dir))
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import requests
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import json
from langchain_ollama import ChatOllama
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler
from gen_ai_hub.proxy import get_proxy_client
from gen_ai_hub.orchestration.service import OrchestrationService
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.template import Template, TemplateValue
from gen_ai_hub.orchestration.models.message import SystemMessage as OrchestrationSystemMessage, UserMessage as OrchestrationUserMessage
from langchain_core.outputs import LLMResult
from Tools import AgentTools, SearchInternet, SaveDocumentRequest, LOGGER
from prompts.planning import PLANNING_SYSTEM_PROMPT, NEXT_STEP_PROMPT
from manifest_pipeline import *
import subprocess
from langchain.agents import create_agent
TOOLS = AgentTools()

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
        try:
            try:
                data = response.model_dump()
            except AttributeError:
                data = response.dict()
            with open(response_filepath, 'w') as f:
                json.dump(data, f, indent=4)
            LOGGER.info(f"Response saved to {response_filepath}")
        except Exception as e:
            LOGGER.error(f"Failed to serialize/save response to {response_filepath}: {e}")
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

class NvidiaLLM():
    def __init__(
        self,
        model_name ="z-ai/glm-5.1",
        default_config: Optional[GenerationConfig] = None
    ):
        self.model_name = model_name
        self.default_config = default_config or GenerationConfig()
        self.token_tracker = TokenTrackerCallback()
        self.chat_model = ChatNVIDIA(
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

class SAPLLM:
    def __init__(
        self,
        model_name="gpt-4o",
        default_config: Optional[GenerationConfig] = None
    ):
        self.model_name = model_name
        self.default_config = default_config or GenerationConfig()
        self.token_tracker = TokenTrackerCallback()
        self.proxy_client = get_proxy_client("gen-ai-hub")
        self.orchestration_service = OrchestrationService(proxy_client=self.proxy_client)

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ):
        if not prompt:
            LOGGER.info(f"No prompt found!")
            return None
        config = config or self.default_config
        llm_model = LLM(
            name=self.model_name,
            parameters={
                "temperature": config.temperature,
            }
        )
        orchestration_config = OrchestrationConfig(
            llm=llm_model,
            template=Template(
                messages=[
                    OrchestrationUserMessage(content="{{?prompt_text}}")
                ]
            )
        )
        response = self.orchestration_service.run(
            config=orchestration_config,
            template_values=[
                TemplateValue(name="prompt_text", value=prompt)
            ]
        )
        try:
            usage = response.orchestration_result.usage
            if usage:
                self.token_tracker.total_prompt_tokens += usage.prompt_tokens
                self.token_tracker.total_completion_tokens += usage.completion_tokens
                self.token_tracker.total_tokens += usage.total_tokens
                LOGGER.info(f"SAPLLM usage tracked: input_tokens={usage.prompt_tokens}, output_tokens={usage.completion_tokens}")
                return response.orchestration_result.choices[0].message.content
        except Exception as e:
            LOGGER.error(f"Failed to track tokens for SAPLLM: {e}")
        return response.orchestration_result.choices[0].message.content

if __name__ == "__main__":
    llm = SAPLLM()
    data = llm.generate("Hello, how are you?")
    print(data)