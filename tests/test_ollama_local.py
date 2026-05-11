import logging
import sys
import time

from langchain_ollama import ChatOllama

ollama_model_name = "qwen:1.8b"
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

print(f"\nUser: {prompt}\n")
print("Assistant: ", end="", flush=True)

start_gen = time.time()
first_token_received = False
token_count = 0

try:
    for chunk in ollama_model.stream(prompt):
        if not first_token_received:
            ttft = time.time() - start_gen
            # Log TTFT clearly
            print(
                f"\n\n[DEBUG: Time to First Token (TTFT) = {ttft:.2f}s]\nAssistant (Live): ",
                end="",
                flush=True,
            )
            first_token_received = True

        print(chunk.content, end="", flush=True)
        token_count += 1

    total_gen_time = time.time() - start_gen
    print(f"\n\n[DEBUG: Total Generation Time = {total_gen_time:.2f}s]")
    LOGGER.info(
        f"Generation finished. TTFT: {ttft:.2f}s | Total: {total_gen_time:.2f}s"
    )

except Exception as e:
    LOGGER.error(f"Error during streaming: {str(e)}")
