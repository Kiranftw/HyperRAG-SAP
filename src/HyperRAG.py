from copyreg import pickle
import os
import shutil
import shutil
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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

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
        # self.embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

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
        
        faiss_index_path = os.path.join(self.DIR, "faiss_index")
        os.makedirs(faiss_index_path, exist_ok=True)
        
        with sqlite3.connect(self.DATASET, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES, timeout=5) as connection:
            cursor: sqlite3.Cursor = connection.cursor()
            TABLENAME = "document_chunks"
            EMBEDDING_COLUMN = "embedding"
            # Check if chunks table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (TABLENAME,))
            if cursor.fetchone() is None:
                LOGGER.info(f"⚠️ TABLE '{TABLENAME}' DOES NOT EXIST. ABORTING INDEX GENERATION.")
                return None
            BATCH_SIZE = 5000
            ids_batch = []
            emb_batch = []
            docstore_dict = {}

            cursor.execute("""
                SELECT 
                    dc.id AS chunk_id,
                    dc.document_id,
                    dc.content,
                    dc.embedding,
                    d.url,
                    d.title,
                    d.images
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE dc.embedding IS NOT NULL;
            """)
            index = None
            dimension = None
            total_vectors = 0

            while True:
                rows = cursor.fetchmany(BATCH_SIZE)
                if not rows:
                    break
                for row in rows:
                    chunk_id = row[0]
                    document_id = row[1]
                    content = row[2]
                    emb_bytes = row[3]
                    url = row[4]
                    title = row[5]
                    images = row[6]

                    vector = np.frombuffer(emb_bytes, dtype=np.float32)
                    if dimension is None:
                        dimension = len(vector)
                        base_index = faiss.IndexHNSWFlat(dimension, 32)
                        base_index.hnsw.efConstruction = 200
                        base_index.hnsw.efSearch = 64
                        index = faiss.IndexIDMap2(base_index)

                    ids_batch.append(chunk_id)
                    emb_batch.append(vector)

                    # Store metadata
                    docstore_dict[str(chunk_id)] = Document(
                        page_content=content or "",
                        metadata={
                            "chunk_id": chunk_id,
                            "document_id": document_id,
                            "url": url,
                            "title": title,
                            "images": images,
                        }
                    )
                xb = np.vstack(emb_batch).astype(np.float32)
                ids_array = np.array(ids_batch, dtype=np.int64)

                index.add_with_ids(xb, ids_array)
                total_vectors += len(ids_batch)
                LOGGER.info(f"ADDED  {total_vectors} VECTORS SO FAR")
                LOGGER.info(f"BATCH OF {len(ids_batch)} VECTORS ADDED LAST CHUNKID {ids_batch[-1]}")
                ids_batch.clear()
                emb_batch.clear()

            # Create vectorstore and save index after all vectors are added
            if index is not None and dimension is not None:
                docstore = InMemoryDocstore(docstore_dict)
                vectorstore = FAISS(embedding_function=self.embedding_function, 
                                   index=index, 
                                   docstore=docstore, 
                                   index_to_docstore_id={i: str(i) for i in range(index.ntotal)})
                vectorstore.save_local(faiss_index_path)
                LOGGER.info(f"✅ FAISS INDEX SAVED TO '{faiss_index_path}'")
                LOGGER.info(f"✅ FAISS INDEX CREATED WITH {index.ntotal} VECTORS | DIMENSION={dimension}")
            else:
                LOGGER.info(f"⚠️ NO VECTORS TO INDEX. INDEX GENERATION SKIPPED.")