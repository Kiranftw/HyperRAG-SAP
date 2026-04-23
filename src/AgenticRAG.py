from urllib import response

from click import prompt
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from faiss_index import HyperRetrivalAugmentedGeneration,LOGGER, ExceptionHandelling, FAISSIndexGeneration
from langchain_ollama import ChatOllama
from google import genai, generativeai
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.retrievers import BM25Retriever
from gen_ai_hub.orchestration.service import OrchestrationService
from langchain_community.document_loaders import TextLoader, JSONLoader, CSVLoader, PyPDFLoader
from PIL import Image
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import SimpleJsonOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
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
import asyncio
import sqlite3
import faiss
import logging
from dotenv import load_dotenv, find_dotenv
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
import os
import warnings
from elasticsearch import Elasticsearch, helpers

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
        self.parser = SimpleJsonOutputParser()
        LOGGER.info("ROOT DIRECTORY: " + self.DIR)
        print("Initializing Embedding Function...")
        self.embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"})
        #connecting elasticsearch through docker continer with a persistant volume
        self.es = Elasticsearch(
            "http://localhost:9200",
            basic_auth=("elastic", "kiranftw"),
            request_timeout=60  # Increased timeout to prevent connection drops
        )
        #loading the vaiss index
        self.vectorstore = self.load_faiss_vectorstore_advanced(
            embeddings=self.embedding_function,
            index_dir="/home/kiranftw/HyperRAG-SAP/faiss_index",
            use_gpu=True,   # this enables GPU
        )
        #connecting cohere account
        self.cohere_reranker = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

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
    
    def data_ingestion_elasticsearch(self) -> None:
        #Denormalized (Join beforehand → Index everything in ES)
        # SELECT 
        #     c.id as chunk_id,
        #     c.content as chunk_content,
        #     d.id as document_id,
        #     d.title as document_title,
        #     d.url as document_url
        # FROM document_chunks c
        # JOIN documents d ON c.document_id = d.id
        if not self.DATASET:
            raise ValueError("NO DATABASE FOUND")
        # Connection to Elasticsearch
        # NOTE: Using credentials from user request
        # Verify connection
        if self.es.ping():
            LOGGER.info("connection with elasticsearch was successful")
        else:
            LOGGER.error("Error connecting to elasticsearch")
            return
        index_name = "sap_knowledge_base"
        # Get the current number of documents and create mapping if it doesn't exist
        try:
            if self.es.indices.exists(index=index_name):
                res = self.es.count(index=index_name)
                start_offset = res['count']
                LOGGER.info(f"FOUND {start_offset} DOCUMENTS IN ELASTICSEARCH. RESUMING...")
            else:
                start_offset = 0
                LOGGER.info("No Index Found. Creating Fresh Index With Mapping.")
                # Define mapping for vector search capabilities
                mapping = {
                    "mappings": {
                        "properties": {
                            "chunk_id": {"type": "integer"},
                            "document_id": {"type": "integer"},
                            "text": {"type": "text"},
                            "title": {"type": "text"},
                            "source": {"type": "keyword"},
                            "embedding": {
                                "type": "dense_vector",
                                "dims": 768,
                                "index": True,
                                "similarity": "cosine"
                            }
                        }
                    }
                }
                self.es.indices.create(index=index_name, body=mapping)
        except Exception as e:
            LOGGER.error(f"ERROR CHECKING/CREATING INDEX: {e}")
            start_offset = 0
        def generate_actions():
            with sqlite3.connect(self.DATASET) as connection:
                connection.row_factory = sqlite3.Row
                cursor = connection.cursor()
                offset = start_offset
                batch_size = 2000
                while True:
                    query = f"""
                    SELECT 
                        c.id as chunk_id,
                        d.id as document_id,
                        c.content as text,
                        d.title,
                        d.url as source,
                        c.embedding
                    FROM document_chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.embedding IS NOT NULL
                    LIMIT {batch_size} OFFSET {offset}
                    """
                    cursor.execute(query)
                    rows = cursor.fetchall()
                    if not rows:
                        break
                    for row in rows:
                        embedding = row["embedding"]
                        # Convert BLOB to list of floats
                        if isinstance(embedding, (bytes, bytearray)):
                            embedding = np.frombuffer(embedding, dtype=np.float32).tolist()
                        elif isinstance(embedding, str):
                            try:
                                embedding = [float(x) for x in embedding.split(",")]
                            except:
                                LOGGER.warning(f"Couldn't parse embedding for row {row['chunk_id']}")
                                continue
                        yield {
                            "_index": index_name,
                            "_id": row["chunk_id"],  # Use chunk_id as the unique identifier
                            "_source": {
                                "chunk_id": row["chunk_id"],
                                "document_id": row["document_id"],
                                "text": row["text"],
                                "title": row["title"],
                                "source": row["source"],
                                "embedding": embedding
                            }
                        }
                    offset += batch_size
                    LOGGER.info(f"INDEXED {offset} DOCUMENTS...")
        try:
            LOGGER.info(f"Starting Bulk Ingestion into Elasticsearch Index: {index_name}")
            success, failed = helpers.bulk(self.es, generate_actions())
            LOGGER.info(f"Successfully Indexed {success} New Documents. Total In Index: {start_offset + success}")
            if failed:
                LOGGER.warning(f"Failed To Index {len(failed)} Documents.")
        except Exception as e:
            LOGGER.error(f"Critical Error During Elasticsearch Ingestion: {e}")
            LOGGER.info("Tip: You Can Run The Script Again To Resume From Where It Stopped.")
    
    def load_faiss_vectorstore_advanced(
        self,
        embeddings: Embeddings,
        index_dir: str,
        use_gpu: bool = False,
        nprobe: int = 10,
        search_k: int = 5,
    ) -> Optional[FAISS]:
        if not os.path.isdir(index_dir):
            raise FileNotFoundError(f"faiss index directory does not exist: {index_dir}")
        self.vectorstore = FAISS.load_local(
            index_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        # Access underlying FAISS index
        index = self.vectorstore.index
        # Apply IVF tuning if supported
        if hasattr(index, "nprobe"):
            index.nprobe = nprobe
        # Move to GPU if requested
        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            self.vectorstore.index = index
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": search_k}
        )
        LOGGER.info(
            "faiss loaded | gpu=%s | nprobe=%s | k=%s",
            use_gpu,
            nprobe,
            search_k
        )
        return self.vectorstore
    
    async def hybrid_search(self, query: str, k: int = 10) -> Optional[List[Document]]:
        KEYWORDS_K: int = 10
        SEMANTIC_SEARCH_K: int = 5

        sparse_results: Any = None
        dense_results: Any = None
        async def run_sparse() -> Any:
            nonlocal sparse_results
            sparse_results = await asyncio.to_thread(
                self.es.search,
                index="sap_knowledge_base",
                query={"match": {"text": query}},
                size=KEYWORDS_K,
                _source_excludes=["embedding"]
            )
        async def run_dense():
            nonlocal dense_results
            dense_results = await self.vectorstore.asimilarity_search_with_score(
                query, k=SEMANTIC_SEARCH_K,
            )
        # Run both concurrently
        await asyncio.gather(
            run_sparse(),
            run_dense()
        )
        # RRF Deduplication
        combined_scores = {}
        unified_results = {}
        RRF_K = 60
        # Process Sparse
        sparse_hits = sparse_results.get("hits", {}).get("hits", []) if isinstance(sparse_results, dict) else getattr(sparse_results, 'body', {}).get("hits", {}).get("hits", [])
        for rank, hit in enumerate(sparse_hits):
            source = hit.get("_source", {})
            chunk_id = source.get("chunk_id")
            if chunk_id is None:
                continue
            if chunk_id not in combined_scores:
                combined_scores[chunk_id] = 0.0
            combined_scores[chunk_id] += 1.0 / (RRF_K + rank + 1)
            unified_results[chunk_id] = {
                "chunk_id": chunk_id,
                "document_id": source.get("document_id"),
                "text": source.get("text"),
                "source": source.get("source"),
                "title": source.get("title")
            }
        # Process Dense
        for rank, (doc, score) in enumerate(dense_results):
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id is None:
                continue
            if chunk_id not in combined_scores:
                combined_scores[chunk_id] = 0.0
            combined_scores[chunk_id] += 1.0 / (RRF_K + rank + 1)
            if chunk_id not in unified_results:
                unified_results[chunk_id] = {
                    "chunk_id": chunk_id,
                    "document_id": doc.metadata.get("document_id"),
                    "text": doc.page_content,
                    "source": doc.metadata.get("url") or doc.metadata.get("source"),
                    "title": doc.metadata.get("title")
                }
        # Sort by combined score
        final_results = []
        for chunk_id, score in sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)[:k]:
            result = unified_results[chunk_id]
            result["rrf_score"] = score
            final_results.append(result)
        with open("combined_results.json", "w") as f:
            json.dump(final_results, f, indent=2)
        return final_results


    def reranking_documents(self, query: str, documents: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        #working with COHERE RERANKING API
        if not documents:
            return [] 
        COHERE_API_KEY = os.getenv("COHERE_API_KEY")
        if not COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY environment variable is not set")
        # We can use the client initialized in __init__ if available, otherwise create a new one
        client = getattr(self, 'cohere_reranker', None) or cohere.ClientV2(COHERE_API_KEY)
        # Call the rerank API
        texts = [doc["text"] for doc in documents]
        response = client.rerank(
            model="rerank-v4.0-pro",
            query=query,
            documents=texts,
            top_n=min(top_n, len(texts))
        )
        # Map the results back to the original documents using the index
        reranked_docs = []
        for item in response.results:
            doc = documents[item.index]
            doc["rerank_score"] = item.relevance_score
            reranked_docs.append(doc)
            
        with open("reranked_results.json", "w") as f:
            json.dump(reranked_docs, f, indent=2)
        return reranked_docs

if __name__ == "__main__":
    agentic_rag = AgenticRAG()
    query="What is SAP S/4HANA?Explain me what is SAP CBC(Central Business Configuration) and how it is different from the traditional SAP S/4HANA configuration?"
    final_results = asyncio.run(agentic_rag.hybrid_search(query))
    final_results = agentic_rag.reranking_documents(
        query=query,
        documents=final_results,
        top_n=5
    )   
    print(final_results)