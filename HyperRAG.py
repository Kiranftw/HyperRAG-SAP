import os
import sqlite3
from functools import wraps
import logging
import traceback
from dotenv import load_dotenv, find_dotenv
import google.genai as genai
load_dotenv(find_dotenv())
from langchain_ollama import ChatOllama
from langchain_community.tools import TavilySearchResults

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
        self.DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATASET = os.path.join(self.DIR, "datasets", "PRODUCTION.db")
        self.LLM = ChatOllama(
            model=model_name,
            temperature=0.7,
            streaming=False,
            verbose=True,
            num_ctx=10000,
            base_url="http://localhost:11434"
        )
    
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
        #deleting the table's from production.db
        # tables_to_delete = ["junk_documents", "empty_documents"]
        # for table in tables_to_delete:
        #     cursor.execute(f"DROP TABLE IF EXISTS {table};")
        #     LOGGER.info(f"TABLE {table} DELETED IF EXISTS.")
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
    
        
if __name__ == "__main__":
    hrag = HyperRetrivalAugmentedGeneration()
    hrag.document_investigation()
    hrag.vaccume()