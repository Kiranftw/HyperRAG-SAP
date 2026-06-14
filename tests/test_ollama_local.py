import logging
import sys
import time

from langchain_ollama import ChatOllama

ollama_model_name = "qwen3:1.7b"
# ollama_model_name = "qwen:0.5b"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
LOGGER = logging.getLogger(__name__)
start_init = time.time()
LOGGER.info(f"Starting initialization for model: {ollama_model_name}")

ollama_model = ChatOllama(
    model=ollama_model_name,
    temperature=0.7,
    verbose=False,
    num_ctx=10000,
    # Strategy: keep_alive="-1" keeps the model in VRAM indefinitely for instant inference
    # Without this, Ollama unloads the model after 5 minutes of inactivity.
    additional_kwargs={"keep_alive": "-1"},
)

init_duration = time.time() - start_init
LOGGER.info(
    f"Model {ollama_model_name} initialized. Keep-alive set to infinite. Duration: {init_duration:.2f}s"
)
# 2. Live inference using streaming with TTFT tracking
prompt = "Explain why a 0.5B parameter model like Qwen-0.5B fails at complex agentic tasks (like tool use or multi-step reasoning) compared to larger models like 1.8B or 7B."

#capturing the thinking tokens from thinking model
start_inference = time.time()
response = ollama_model(
    prompt,
    stream=True,
    additional_kwargs={"stream_thoughts": "true"},
)
inference_duration = time.time() - start_inference
LOGGER.info(f"Inference completed in {inference_duration:.2f}s")
LOGGER.info(f"Response: {response}")
LOGGER.info(f"Response: {response}")
LOGGER.info(f"Response: {response}")