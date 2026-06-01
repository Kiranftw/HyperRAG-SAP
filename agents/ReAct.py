"""
ReAct Agent Implementation
==========================
This module implements a state-of-the-art ReAct (Reasoning and Acting) Agent.
The agent alternates between thinking (reasoning) and taking actions (tools execution)
to solve complex enterprise orchestration tasks.

Workflow:
Goal -> Thought -> Action -> Execute -> Observe -> Reflect -> Goal Achieved? -> Finish
"""

import sys
import os
import re
import json
import inspect
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel

# Ensure proper path configuration for direct running
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Tools import AgentTools, LOGGER
from llm.models import OllamaLLM, SAPLLM, NvidiaLLM

class ReActAgent:
    def __init__(
        self,
        llm: Optional[Any] = None,
        tools: Optional[AgentTools] = None,
        max_steps: int = 10,
    ):
        """
        Initializes the ReAct Agent.
        
        Args:
            llm: An instance of OllamaLLM, SAPLLM, or NvidiaLLM. Defaults to NvidiaLLM.
            tools: An instance of AgentTools. Defaults to a new AgentTools.
            max_steps: Maximum number of execution steps before stopping.
        """
        self.llm = llm or NvidiaLLM()
        self.tools = tools or AgentTools()
        self.max_steps = max_steps
        self.history: List[Dict[str, Any]] = []
        self.available_tools_metadata = self._discover_tools()

    def _discover_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Dynamically discovers execution methods from the tools instance,
        filtering out private and helper functions.
        """
        tools_metadata = {}
        for name in dir(self.tools):
            # Filter out private methods, base class overrides, and specialized helpers
            if name.startswith("_") or name in [
                "hybrid_search", "validate_extension", "validate_file_size",
                "ROOT", "workspace_dir", "cohere_reranker", "sap_knowledge_search",
                "embeddings", "es_client", "faiss_index", "pdf_loader"
            ]:
                continue
            
            func = getattr(self.tools, name)
            if callable(func):
                # Retrieve docstring and signature
                doc = func.__doc__ or "No description available."
                sig = inspect.signature(func)
                params_info = {}
                
                # Check if first parameter is a Pydantic BaseModel
                params = list(sig.parameters.values())
                if len(params) > 0:
                    first_param = params[0]
                    anno = first_param.annotation
                    if isinstance(anno, type) and issubclass(anno, BaseModel):
                        # Extract fields from the BaseModel
                        for field_name, field_obj in anno.model_fields.items():
                            params_info[field_name] = {
                                "type": str(field_obj.annotation),
                                "description": field_obj.description or ""
                            }
                    else:
                        # Standard function parameters
                        for p in params:
                            if p.name != "self":
                                params_info[p.name] = {
                                    "type": str(p.annotation),
                                    "default": str(p.default) if p.default != inspect.Parameter.empty else "Required"
                                }
                
                tools_metadata[name] = {
                    "description": doc.strip(),
                    "parameters": params_info,
                    "func": func
                }
        return tools_metadata

    def _build_tools_description(self) -> str:
        """Formats the available tools into a clean readable string for the LLM prompt."""
        descriptions = []
        for name, meta in self.available_tools_metadata.items():
            param_str = json.dumps(meta["parameters"], indent=2)
            descriptions.append(
                f"- **{name}**:\n"
                f"  Description: {meta['description']}\n"
                f"  Arguments Schema:\n{param_str}\n"
            )
        return "\n".join(descriptions)

    def _extract_json(self, llm_output: str) -> Dict[str, Any]:
        """
        Extracts and parses a JSON block from the LLM's response.
        Enforces robust fallback strategies to handle parsing errors.
        """
        # Try direct JSON parsing
        try:
            return json.loads(llm_output.strip())
        except Exception:
            pass

        # Try extracting markdown json codeblocks
        json_pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
        match = json_pattern.search(llm_output)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except Exception:
                pass

        # Generic markdown codeblock extraction
        block_pattern = re.compile(r"```\s*(.*?)\s*```", re.DOTALL)
        match = block_pattern.search(llm_output)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except Exception:
                pass

        # Extract everything inside curly braces
        braces_pattern = re.compile(r"(\{.*\})", re.DOTALL)
        match = braces_pattern.search(llm_output)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except Exception:
                pass

        raise ValueError(f"Failed to extract structured JSON from response: {llm_output}")

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """
        Executes a tool dynamically with the parsed arguments.
        
        Args:
            tool_name: The name of the tool to execute.
            tool_args: Dictionary of arguments.
        """
        if tool_name not in self.available_tools_metadata:
            return f"Error: Tool '{tool_name}' is not recognized. Please choose from: {list(self.available_tools_metadata.keys())}"
        
        meta = self.available_tools_metadata[tool_name]
        func = meta["func"]
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        
        LOGGER.info(f"Executing tool {tool_name} with arguments: {tool_args}")
        try:
            # TODO(security): Validate file operations and commands for directory traversal and command injection
            if len(params) > 0:
                first_param = params[0]
                anno = first_param.annotation
                if isinstance(anno, type) and issubclass(anno, BaseModel):
                    # Populate and validate Pydantic request object
                    validated_request = anno(**tool_args)
                    result = func(validated_request)
                else:
                    # Pass normal args
                    result = func(**tool_args)
            else:
                result = func()
            return str(result)
        except Exception as e:
            LOGGER.error(f"Execution error on tool '{tool_name}': {e}")
            return f"Execution Error: {str(e)}"

    def run(self, goal: str) -> str:
        """
        Executes the ReAct loop to solve the user's goal.
        
        Args:
            goal: The task or query defined by the user.
        """
        LOGGER.info(f"ReAct Agent initiated with goal: {goal}")
        self.history = []
        tools_desc = self._build_tools_description()
        
        system_prompt = f"""You are an autonomous ReAct (Reasoning & Acting) Agent.
Your objective is to solve the user's goal through structured steps:
1. Thought: Reason about the current progress and what to do next.
2. Action: Choose a single tool from the available tools list, or 'finish' to complete.
3. Action Input: The specific parameters/payload for the selected action.
4. Observation: The system will run the tool and show you the execution result.
5. Reflection: Self-critique step (*Did the action help progress? What should I do next?*).

Available Tools:
{tools_desc}

Format:
For every step, you MUST respond with a single, valid JSON block following this schema:
{{
  "thought": "What I know, what I need to do next, and why.",
  "action": "tool_name",
  "action_input": {{
    "parameter_name": "value"
  }}
}}

If you have completed the goal, gathered all required information, or successfully performed the task, you MUST use the 'finish' action:
{{
  "thought": "I have successfully completed the task.",
  "action": "finish",
  "action_input": {{
    "final_answer": "Your detailed final answer summarizing findings and actions taken."
  }}
}}

Important Guidelines:
- Return ONLY the raw JSON block. Do not include extra conversational text outside the JSON.
- Maintain consistent progress. If a tool fails, reflect on the error and try a different parameter or tool.
"""

        for step in range(1, self.max_steps + 1):
            LOGGER.info(f"--- ReAct Step {step} of {self.max_steps} ---")
            
            # Format history
            history_str = ""
            for idx, h in enumerate(self.history):
                history_str += (
                    f"\n[Step {idx+1}]\n"
                    f"Thought: {h.get('thought')}\n"
                    f"Action: {h.get('action')}\n"
                    f"Action Input: {json.dumps(h.get('action_input'))}\n"
                    f"Observation: {h.get('observation')}\n"
                    f"Reflection: {h.get('reflection')}\n"
                )
            
            prompt = f"{system_prompt}\n\nUser Goal: {goal}\n\nExecution History:\n{history_str}\nLet's generate the next step now."
            
            # Query the LLM
            response = self.llm.generate(prompt)
            if not response:
                return "Failed to get response from the LLM."
            
            # Parse structured response
            try:
                parsed = self._extract_json(response)
            except Exception as e:
                LOGGER.warning(f"Failed to parse JSON response. LLM Output:\n{response}")
                # Provide error as observation to recover
                observation = f"Error parsing your JSON output: {e}. Please return ONLY valid JSON."
                self.history.append({
                    "thought": "Failed to output valid JSON format.",
                    "action": "invalid_format",
                    "action_input": {},
                    "observation": observation,
                    "reflection": "I must correct my format and output only valid JSON."
                })
                continue
            
            thought = parsed.get("thought", "")
            action = parsed.get("action", "")
            action_input = parsed.get("action_input", {})
            
            LOGGER.info(f"Thought: {thought}")
            LOGGER.info(f"Action: {action}")
            LOGGER.info(f"Action Input: {action_input}")
            
            if action == "finish":
                final_answer = action_input.get("final_answer", "Goal reached successfully.")
                LOGGER.info(f"ReAct Loop finished. Final Answer: {final_answer}")
                return final_answer
            
            # Execute action
            observation = self._execute_tool(action, action_input)
            LOGGER.info(f"Observation: {observation}")
            
            # Formulate self-critique/reflection
            reflection_prompt = f"""
            Goal: {goal}
            Thought: {thought}
            Action: {action}
            Action Input: {json.dumps(action_input)}
            Observation: {observation}
            
            Based on the observation, briefly critique the result. Did this step make progress? What are the key takeaways?
            Return a 1-2 sentence reflection.
            """
            reflection = self.llm.generate(reflection_prompt) or "Observation completed."
            LOGGER.info(f"Reflection: {reflection}")
            
            # Add to history
            self.history.append({
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": observation,
                "reflection": reflection.strip()
            })
            
        LOGGER.warning("ReAct Loop exceeded maximum execution steps.")
        return "Exceeded maximum steps. Task remained incomplete."

if __name__ == "__main__":
    # Test execution
    agent = ReActAgent()
    test_goal = "Search the internet for the latest news on SAP GenAI Hub, and write a summary file named 'sap_genai_summary.txt' inside the datasets folder."
    print("Starting ReAct Agent test...")
    result = agent.run(test_goal)
    print("\n--- Final Result ---")
    print(result)
