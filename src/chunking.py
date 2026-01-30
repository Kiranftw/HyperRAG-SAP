import os
import sqlite3
import pandas as pd
import logging
from typing import List  
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger()
class ContentSplitting:
    def __init__(self) -> None:
        self.DIR = os.path.dirname(os.path.abspath(__file__))
    
    def data_splitting(self, databasename: str) -> None:
        CHUNKSIZE = 3500
        LONG_CHUNKSIZE = 8750
        OVERLAP = 400
        BATCH_SIZE = 1000
        TABLE_NAME = "documents_cleaned_test"
        CONTENT_COLUMN = "content"
        sentence_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNKSIZE,
            chunk_overlap=OVERLAP,
            separators=["\n\n", ". ", "? ", "! ", "\n", " ", ""],
        )
        with sqlite3.connect(databasename, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES, timeout=5) as connection:
            cursor: sqlite3.Cursor = connection.cursor()
            # Check if source table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (TABLE_NAME,))
            if cursor.fetchone() is None:
                LOGGER.info(f"⚠️ TABLE '{TABLE_NAME}' DOES NOT EXIST. ABORTING DATA SPLITTING.")
                return None
            
            # Drop and recreate chunks table
            cursor.execute("DROP TABLE IF EXISTS document_chunks;")
            cursor.execute(f"""
                CREATE TABLE document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER,
                    chunk_index INTEGER,
                    content TEXT,
                    FOREIGN KEY(document_id) REFERENCES {TABLE_NAME}(id) ON DELETE CASCADE
                );
            """)
            connection.commit()
            
            cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE {CONTENT_COLUMN} IS NOT NULL AND {CONTENT_COLUMN} != '';")
            non_empty_count = cursor.fetchone()[0]
            PROCESSED_ROWS = 0
            lastid = 0
            
            while True:
                cursor.execute(
                    f"""
                    SELECT id, {CONTENT_COLUMN}
                    FROM {TABLE_NAME}
                    WHERE id > ? 
                    AND {CONTENT_COLUMN} IS NOT NULL
                    AND {CONTENT_COLUMN} != ''
                    ORDER BY id
                    LIMIT ?
                    """,
                    (lastid, BATCH_SIZE),
                )
                rows = cursor.fetchall()
                if not rows:
                    break
                
                batch_inserts = []
                for document_id, content in rows:
                    lastid = document_id
                    if not content:
                        continue
                    length = len(content)
                    if length <= CHUNKSIZE:
                        chunks = [content]
                    elif length <= LONG_CHUNKSIZE:
                        temp_chunks = sentence_splitter.split_text(content)

                        if len(temp_chunks) == 1:
                            chunks = temp_chunks
                        else:
                            mid = len(temp_chunks) // 2
                            chunks = [
                                " ".join(temp_chunks[:mid]),
                                " ".join(temp_chunks[mid:])
                            ]
                    else:
                        chunks = sentence_splitter.split_text(content)

                    for chunk_index, chunk in enumerate(chunks):
                        batch_inserts.append((document_id, chunk_index, chunk))
                PROCESSED_ROWS += len(rows)
                if batch_inserts:
                    cursor.executemany(
                        """
                        INSERT INTO document_chunks (document_id, chunk_index, content)
                        VALUES (?, ?, ?);
                        """,
                        batch_inserts,
                    )
                    connection.commit()
                    LOGGER.info(f"💾 PROCESSED {PROCESSED_ROWS}/{non_empty_count} ROWS. Inserted {len(batch_inserts)} chunks.")
            
            connection.commit()
        LOGGER.info("DATA SPLITTING COMPLETED.")
        return None

import requests
import json
def main():
    invoice = requests.get("http://127.0.0.1:5000/getdata")
    print(type(invoice.text))
    data = json.loads(invoice.text)
    print(type(data))
    print(data)
                           
if __name__ == "__main__":
    main()