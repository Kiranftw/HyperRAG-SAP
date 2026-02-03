from copyreg import pickle
import os
import sqlite3
from functools import wraps
import logging
import traceback
from dotenv import load_dotenv, find_dotenv
import google.genai as genai
from chunking import ContentSplitting
import numpy as np
load_dotenv(find_dotenv())
from typing import List, Optional, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
import torch

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
LOGGER = logging.getLogger()
@staticmethod
def ExceptionHandelling(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            LOGGER.info(f"⚠️EXCEPTION IN {func.__name__}: {e}")
            traceback.print_exc()
            return None
    return wrapper

class HyperRetrivalAugmentedGeneration:
    def __init__(self, model_name: str = "gpt-oss:120b-cloud") -> None:
        self.DIR = "/home/kiranftw/HyperRAG-SAP"
        self.DATASET = os.path.join(self.DIR, "datasets", "PRODUCTION.db")
        self.splitting = ContentSplitting()
    
    def vaccume(self) -> None:
        connection: sqlite3.Connection = sqlite3.connect(self.DATASET, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
        cursor: sqlite3.Cursor = connection.cursor()
        cursor.execute("VACUUM;")
        connection.commit()
        connection.close()
        LOGGER.info("DATABASE VACUUM COMPLETED.")
        return None
     
    @ExceptionHandelling
    def document_investigation(self) -> None:
        connection: sqlite3.Connection = sqlite3.connect(self.DATASET, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
        cursor: sqlite3.Cursor = connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        LOGGER.info("TABLES IN THE DATABASE:")
        # deleting the table's from production.db
        tables_to_delete = ["junk_documents", "empty_documents"]
        for table in tables_to_delete:
            cursor.execute(f"DROP TABLE IF EXISTS {table};")
            LOGGER.info(f"TABLE {table} DELETED IF EXISTS.")
        # Convert to set of table names (strings) for proper comparison
        table_names = {t[0] for t in tables}
        tables = [(t,) for t in table_names - set(tables_to_delete)]
        for table in tables:
            LOGGER.info(f"- {table[0]}")
            #printing the length & schema of each table
            cursor.execute(f"PRAGMA table_info({table[0]});")
            columns = cursor.fetchall()
            rows = cursor.execute(f"SELECT COUNT(*) FROM {table[0]};").fetchone()
            LOGGER.info(f" TOTAL ROWS: {rows[0]}")
            LOGGER.info(f" NO OF COLUMNS: {len(columns)}")
            LOGGER.info("  COLUMNS:")
            for column in columns:
                LOGGER.info(f"    - {column[1]} ({column[2]})")
        connection.commit()
        connection.close()
        return None    
    @ExceptionHandelling
    def chunking(self, databasename: str) -> None:
        if not os.path.exists(databasename):
            LOGGER.info(f"⚠️ DATABASE '{databasename}' DOES NOT EXIST. ABORTING CHUNKING.")
            return None
        self.splitting.data_splitting(databasename)
        LOGGER.info("CHUNKING COMPLETED.")
        return None
    
    @ExceptionHandelling
    def embedding_generation(self, EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2") -> None:
        #NOTE:/home/kiranftw/HyperRAG-SAP/HyperRAG.py:117: LangChainDeprecationWarning: The class `HuggingFaceEmbeddings` was deprecated in LangChain 0.2.2 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-huggingface package and should be used instead. To use it run `pip install -U :class:`~langchain-huggingface` and import as `from :class:`~langchain_huggingface import HuggingFaceEmbeddings``.
        if not os.path.exists(self.DATASET):
            LOGGER.info(f"⚠️ DATABASE '{self.DATASET}' DOES NOT EXIST. ABORTING EMBEDDING GENERATION.")
            return None
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        LOGGER.info(f"USING DEVICE: {self.DEVICE.upper()}")
        if self.DEVICE == "cuda":
            LOGGER.info("CLEARING CUDA CACHE... & USING GPU FOR EMBEDDING GENERATION.")
            torch.cuda.empty_cache()
        embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=self.DEVICE)
        if self.DEVICE == "cuda":
            embedding_model.half()
            LOGGER.info("FP16 INFERENCE ENABLED: Compatible with existing docs (Storage remains FP32).")
        # Smaller batch size to avoid CUDA OOM - adjust if needed
        BATCH_SIZE = 256 if self.DEVICE == "cuda" else 128
        with sqlite3.connect(self.DATASET, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES, timeout=5) as connection:
            cursor: sqlite3.Cursor = connection.cursor()
            cursor.execute("PRAGMA journal_mode = WAL;")
            cursor.execute("PRAGMA synchronous = NORMAL;")
            cursor.execute("PRAGMA cache_size = -100000;")
            TABLENAME = "document_chunks"
            EMBEDDING_COLUMN = "embedding"
            # Check if chunks table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (TABLENAME,))
            if cursor.fetchone() is None:
                LOGGER.info(f"⚠️ TABLE '{TABLENAME}' DOES NOT EXIST. ABORTING EMBEDDING GENERATION.")
                return None
            # Add embedding column if it doesn't exist
            cursor.execute(f"PRAGMA table_info({TABLENAME});")
            columns = [info[1] for info in cursor.fetchall()]
            # if EMBEDDING_COLUMN in columns:
            #     cursor.execute(f"ALTER TABLE {TABLENAME} DROP COLUMN {EMBEDDING_COLUMN};")
            #     connection.commit()
            #     LOGGER.info(f"DROPPED EXISTING COLUMN '{EMBEDDING_COLUMN}' FROM")
            #     LOGGER.info(f"TABLE '{TABLENAME}'.")
            if EMBEDDING_COLUMN not in columns:
                cursor.execute(f"ALTER TABLE {TABLENAME} ADD COLUMN {EMBEDDING_COLUMN} BLOB;")
                connection.commit()
                LOGGER.info(f"ADDED COLUMN '{EMBEDDING_COLUMN}' TO TABLE '{TABLENAME}'.")
            # Initialize embedding model
            # embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            # Get total count for progress
            cursor.execute(f"SELECT COUNT(*), MIN(id) FROM {TABLENAME} WHERE {EMBEDDING_COLUMN} IS NULL;")
            row = cursor.fetchone()
            total_remaining = row[0]
            min_id = row[1]
            LOGGER.info(f"TOTAL CHUNKS TO EMBED: {total_remaining}")
            processed = 0
            lastid = (min_id - 1) if min_id is not None else 0
            batch_counter = 0
            while True:
                cursor.execute(
                    f"""
                    SELECT id, content
                    FROM {TABLENAME}
                    WHERE id > ?
                    AND {EMBEDDING_COLUMN} IS NULL
                    ORDER BY id
                    LIMIT ?
                    """,
                    (lastid, BATCH_SIZE),
                )
                rows = cursor.fetchall()
                if not rows:
                    break
                ids = [row[0] for row in rows]
                contents = [row[1] if row[1] else "" for row in rows]
                lastid = ids[-1]
                embeddings = embedding_model.encode(
                    contents,
                    batch_size=BATCH_SIZE,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # L2 normalize for cosine similarity
                )
                batch_updates = [
                    (sqlite3.Binary(emb.astype(np.float32).tobytes()), doc_id)
                    for doc_id, emb in zip(ids, embeddings)
                ]
                cursor.executemany(
                    f"UPDATE {TABLENAME} SET {EMBEDDING_COLUMN} = ? WHERE id = ?;",
                    batch_updates,
                )
                batch_counter += 1
                # Commit every 10 batches to reduce I/O overhead
                if batch_counter % 10 == 0:
                    connection.commit()
                processed += len(rows)
                pct = (processed / total_remaining) * 100
                LOGGER.info(f"✅EMBEDDED {processed}/{total_remaining} ({pct:.1f}%) | Last ID: {lastid}")
        
        LOGGER.info("✅ EMBEDDING GENERATION COMPLETED.")
        return None

class FAISSIndexGeneration(HyperRetrivalAugmentedGeneration):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    def index_generation(self) -> None:
        #NOTE:# Using FAISS IndexHNSWFlat as the primary vector index. This index is chosen
        # for maximum retrieval speed and near-exact recall on large-scale embeddings
        # (~250k+ vectors). Memory usage is intentionally not optimized in favor of
        # low-latency, high-quality semantic search for RAG pipelines. The key features include:
        # - Ultra-fast ANN search (~2–5 ms latency at 250k vectors)
        # - Near-exact recall with no vector compression (high embedding fidelity)
        # - Graph-based HNSW index, ideal for RAG Fusion and multi-query retrieval
        if not os.path.exists(self.DATASET):
            LOGGER.info(f"⚠️ DATABASE '{self.DATASET}' DOES NOT EXIST. ABORTING INDEX GENERATION.")
            return None
        with sqlite3.connect(self.DATASET, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES, timeout=5) as connection:
            cursor: sqlite3.Cursor = connection.cursor()
            TABLENAME = "document_chunks"
            EMBEDDING_COLUMN = "embedding"
            # Check if chunks table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (TABLENAME,))
            if cursor.fetchone() is None:
                LOGGER.info(f"⚠️ TABLE '{TABLENAME}' DOES NOT EXIST. ABORTING INDEX GENERATION.")
                return None
            # Fetch all embeddings and their corresponding chunk IDs
            cursor.execute(f"SELECT id, content, {EMBEDDING_COLUMN} FROM {TABLENAME} WHERE {EMBEDDING_COLUMN} IS NOT NULL;")
            rows = cursor.fetchall()
            if not rows:
                LOGGER.info(f"⚠️ NO EMBEDDINGS FOUND IN TABLE '{TABLENAME}'. ABORTING INDEX GENERATION.")
                return None
            ids = []
            embeddings = []
            docstore_dict = {}
            for row in rows:
                doc_id = row[0]
                content = row[1]
                emb_bytes = row[2]
                ids.append(doc_id)
                embeddings.append(np.frombuffer(emb_bytes, dtype=np.float32))
                docstore_dict[str(doc_id)] = Document(page_content=content if content else "")
            
            # Create FAISS index
            dimension = len(embeddings[0])
            faiss_index = faiss.IndexHNSWFlat(dimension, 32)  # HNSW with 32 neighbors
            faiss_index.hnsw.efConstruction = 200  # Construction parameter for index building
            faiss_index.hnsw.efSearch = 64  # Search parameter for querying
            
            xb = np.vstack(embeddings).astype(np.float32)
            # Add vectors with IDs to enable chunk ID mapping
            index = faiss.IndexIDMap(faiss_index)
            index.add_with_ids(xb, np.array(ids, dtype=np.int64))
            LOGGER.info(f"✅ FAISS INDEX CREATED WITH {index.ntotal} VECTORS OF DIMENSION {dimension}.")
            # Save FAISS index to disk with populated docstore
            docstore = InMemoryDocstore(docstore_dict)
            index_to_docstore_id = {i: str(i) for i in ids}
            vectorstore = FAISS(embedding_function=self.embedding_function, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)
            faiss_index_path = os.path.join(self.DIR, "faiss_index")
            if not os.path.exists(faiss_index_path):
                os.makedirs(faiss_index_path)
            vectorstore.save_local(faiss_index_path)
            LOGGER.info(f"✅ FAISS INDEX GENERATED AND SAVED TO '{faiss_index_path}'.")
        return None
    
if __name__ == "__main__":
    hyper_rag = HyperRetrivalAugmentedGeneration()
    # hyper_rag.vaccume()
    # hyper_rag.document_investigation()
    # hyper_rag.chunking(hyper_rag.DATASET)
    # hyper_rag.embedding_generation()
    faiss_indexer = FAISSIndexGeneration()
    faiss_indexer.index_generation()