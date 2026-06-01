# HyperRAG SAP Agentic System

This repository implements a state-of-the-art Hierarchical Agentic Retrieval-Augmented Generation (RAG) system integrated with a Multi-Agent architecture for enterprise SAP orchestration.

## Architecture

![ReAct Agent System Architecture](datasets/image.png)

The core architecture follows the **Plan → Act → Observe → Reflect Loop** structure, divided into six main functional modules:

### Module 1: Input & Context
This module aggregates all inputs needed to initiate and guide the agentic execution:
* **User Query / Goal**: The primary request or problem defined by the user.
* **System Prompt / Instructions**: The global system guidelines and behavioral boundaries.
* **Workflow State**: Tracks the active step, completed steps, and remaining tasks in the plan.
* **Memory Summary**: Synthesizes short-term execution history and long-term insights.
* **History**: A chronological log of previous thoughts, actions, and observations.
* **Retrieved Context**: Dynamic information injected via local RAG indexes or live web searches.

### Module 2: ReAct Loop (Iterative Reasoning & Acting)
The execution engine of the agent that processes goals through reasoning steps:
1. **Thought** (*What do I know?*): The agent reasons about the current context and progress.
2. **Action Selection** (*Which tool / action?*): The agent decides which tool is best suited for the next task.
3. **Action Input** (*What are the inputs?*): The agent generates the parameters and payloads for the tool.
4. **Execute Action**: Runs the selected tool securely via the Tool Manager.
5. **Observation** (*What is the result?*): The agent receives the tool output or execution status.
6. **Reflect**: Conducts a self-critique step (*Did the action help progress? What should I do next?*) to refine the next Thought.

### Module 3: Memory System
Manages persistent and evolving state across interactions:
* **Memory Store**: A long-term storage facility for preserving historical experiences.
* **Memory Evolution**: Organizes, compresses, and evolves memories sequentially to maintain efficiency.
* **Memory Retrieval**: Utilizes query embedding vector similarity (Top-K) to retrieve relevant historical insights for the current context.

### Module 4: Validation & Recovery
Ensures system reliability and accuracy before returning results:
* **Quality & Schema Validation**: Verifies that tool outputs and structures meet expectations.
* **Success Criteria Checks**: Confirms that the initial user goal has been achieved.
* **Hallucination & Inconsistency Detection**: Flags non-factual or contradictory generation.
* **Failure Recovery**: Dynamically triggers re-planning, parameter retries, or alternative tools in case of errors.

### Module 5: Tools & Action Layer
Controlled by the Tool Manager to interface with systems:
* **MCP Tools**: Custom Model Context Protocol tools parsed dynamically from schemas.
* **SAP APIs**: Integrated interfaces to interact with SAP ERP systems and GenAI Hub.
* **Search**: Real-time Tavily search engine for external knowledge retrieval.
* **File Operations**: Modules to read, write, move, scan, or delete files locally.
* **Code Execution**: Run secure, isolated commands and scripts.

### Module 6: Output / Result
Produces the final outcome to be returned to the user:
* **Final Answer / Artifacts**: Delivers the solution or generated documentation files.
* **Reports & Visualizations**: Compiles results into clean documents (JSON, CSV, PDF, DOCX).
* **Next Steps & Recommendations**: Suggests follow-up steps and actions.
* The output feeds directly back into the Input & Context module for subsequent interactions.

---

## What We Are Building

### 1. RAG Fusion & Retrieval Pipeline
* **Query Decomposition**: Automatically breaks down complex multi-part user questions into simpler, search-optimized sub-queries.
* **Batch Vector Search**: Searches the local FAISS vector index concurrently for all sub-queries.
* **Reciprocal Rank Fusion (RRF)**: Fuses different retrieval lists, promoting documents that appear high across multiple searches.
* **Cohere Re-ranking**: Evaluates retrieved chunks using high-precision re-ranking models before context window insertion.

### 2. Hierarchical Multi-Agent System
* **Manager Agent**: Evaluates requests, generates structured execution plans, assigns tasks to sub-agents, and conducts quality validation checks.
* **Worker Pool**: Spawns isolated workers to run tasks using specialized file, code, or web search tools.
* **MCP Agent**: Leverages tool manifests to discover and resolve SAP APIs and custom external protocols.
* **Token Tracker & Cost Optimization**: Intercepts LLM outputs to track prompt and completion tokens dynamically across Ollama, SAP GenAI Hub (SAPLLM), and NVIDIA NIM (NvidiaLLM) providers.