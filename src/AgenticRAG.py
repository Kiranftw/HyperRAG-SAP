from urllib import response

from click import prompt
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from HyperRAG import HyperRetrivalAugmentedGeneration,LOGGER, ExceptionHandelling, FAISSIndexGeneration
from langchain_ollama import ChatOllama
from google import genai, generativeai
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from gen_ai_hub.proxy import get_proxy_client
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.models.template import Template, TemplateValue
from gen_ai_hub.orchestration.models.message import SystemMessage as OrchestrationSystemMessage, UserMessage as OrchestrationUserMessage
from gen_ai_hub.orchestration.service import OrchestrationService
from langchain_community.document_loaders import TextLoader, JSONLoader, CSVLoader, PyPDFLoader
from PIL import Image
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import SimpleJsonOutputParser
from typing import List, Dict, Any, Optional, Union, Tuple
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import tools
from google.genai import types
from langchain_community.vectorstores import FAISS
from google.genai.errors import ClientError
import requests
import numpy as np
import pytesseract
from dotenv import load_dotenv, find_dotenv
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
import os
import warnings

class AgenticRAG(FAISSIndexGeneration, HyperRetrivalAugmentedGeneration):
    def __init__(self,  ollama_model_name: str = "gpt-oss:120b-cloud", model_name: str = "gpt-4o-mini"):
        super().__init__()
        ignore_warnings = True
        warnings.filterwarnings("ignore") if ignore_warnings else None
        load_dotenv(find_dotenv())
        self.ollama_model = ChatOllama(
            model=ollama_model_name,
            temperature=0.7,
            verbose=True,
            num_ctx=10000,
            # base_url="http://localhost:11434",
        )
        TAVILY_MAX_RESULTS = 20        
        self.SEARCH_ENGINE = TavilySearchResults(
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            max_results=TAVILY_MAX_RESULTS,
            include_answer=True,
            include_raw_content=True,
            include_tables=True,
            include_domains=[
                "help.sap.com",
                "www.sap.com",
                "developers.sap.com",
                "api.sap.com",
                "community.sap.com"
            ],
            include_images=True,
        )
        self.PROXY_CLIENT = get_proxy_client("gen-ai-hub")
        self.ORCHESTRATION_SERVICE = OrchestrationService(proxy_client=self.PROXY_CLIENT)
        self.LLM_MODEL = LLM(name=model_name, parameters={"temperature": 0.7})
        self.parser = SimpleJsonOutputParser()
        LOGGER.info("ROOT DIRECTORY: " + self.DIR)
        print("Initializing Embedding Function...")
        self.embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"})

    @ExceptionHandelling
    def document_handelling(self, documents: List[str]) -> List[str]:
        if isinstance(documents, str):
            documents = [documents]

        processed_documents = []
        for document in documents:
            endswith = os.path.splitext(document)[1].lower()
            if endswith in [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif",
                            ".JPG", ".JPEG", ".PNG", ".TIFF", ".BMP", ".GIF"]:
                image = Image.open(document)
                extracted_text = pytesseract.image_to_string(image)
                processed_documents.append(extracted_text)
            elif endswith in [".TXT", ".txt"]:
                loader = TextLoader(document)
                processed_documents.append(loader.load())
            elif endswith in [".CSV", ".csv"]:
                loader = CSVLoader(document)
                processed_documents.append(loader.load())
            elif endswith in [".md", ".MD"]:
                loader = TextLoader(document)
                processed_documents.append(loader.load())
            elif endswith in [".pdf", ".PDF"]:
                loader = PyPDFLoader(document)
                processed_documents.append(loader.load())
            else:
                raise ValueError(f"UNSUPPORTED FILE TYPE: {endswith}")
        return processed_documents
    
    @ExceptionHandelling
    def retrival_fusion(self, query: str):
        DECOMPOSED_QUERIES_COUNT = 5
        with open(self.DIR + "/prompts/query_decomposition.txt", "r") as file:
            prompt_template = file.read()
        prompt = (
            prompt_template
            .replace("{{ query }}", query)
            .replace("{{ number }}", str(DECOMPOSED_QUERIES_COUNT))
        )
        response = self.ollama_model.invoke([SystemMessage(content=prompt)])
        queries = []
        try:
            content = response.content if isinstance(response.content, str) else str(response.content)
            decomposed = self.parser.parse(content)
            queries = [
                q if isinstance(q, str) else q.get("subquery")
                for q in decomposed
            ] if isinstance(decomposed, list) else []

            queries = [q for q in queries if isinstance(q, str)]
        except Exception as e:
            LOGGER.error(f"QUERY DECOMPOSITION FAILED: {e}")
            return None
        if not queries:
            LOGGER.warning("No queries decomposed. Aborting retrieval.")
            return None
        print("DECOMPOSED QUERIES: ", queries)
        FAISSINDEXPATH = self.DIR + "/faiss_index"
        if not os.path.exists(FAISSINDEXPATH):
            LOGGER.info(f"⚠️ FAISS INDEX NOT FOUND AT '{FAISSINDEXPATH}'. ABORTING RETRIEVAL.")
            return None
        vectorstore = FAISS.load_local(folder_path=FAISSINDEXPATH, embeddings=self.embedding_function, allow_dangerous_deserialization=True
        )
        search_results = []
        query_vectors = self.embedding_function.embed_documents(queries)
        query_matrix = np.array(query_vectors, dtype=np.float32)
        D, I = vectorstore.index.search(query_matrix, k=5)
        for query_idx, query in enumerate(queries):
            for rank, (distance, doc_index) in enumerate(zip(D[query_idx], I[query_idx])):
                if doc_index == -1:
                    continue
                doc_id = vectorstore.index_to_docstore_id[doc_index]
                retrieved_doc = vectorstore.docstore.search(doc_id)
                search_results.append({
                    "query": query,
                    "retrieved_doc": retrieved_doc,
                    "distance": float(distance),
                    "rank": rank + 1
                })
        #sorting results based on rank
        search_results.sort(key=lambda x: x["rank"],)
        return search_results

if __name__ == "__main__":
    agentic_rag = AgenticRAG()
    query = "What are the key insights from the sales data in sales_data.csv and the market trends in market_trends.pdf?"
    response = agentic_rag.retrival_fusion(query)
    print("RESPONSE: ", response)
    print(len(response))
    # response = agentic_rag.ollama_model.invoke([
    #     SystemMessage(content="Decompose the following query into 5 sub-queries: " + query)
    # ])
    # print("Decomposed Queries: ", response)
