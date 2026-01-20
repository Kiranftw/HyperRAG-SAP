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
    def __init__(self) -> None:
        self.DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATASET = os.path.join(self.DIR, "dataset", "PRODUCTION.db")
        for model in genai.list_models():
            print(model.name)
        self.MODEL = genai.GenerativeModel(
            model_name="models/gemini-2.0-flash",
            generation_config={"response_mime_type": "application/json"},
            safety_settings={},
            tools=None,
            system_instruction=None,
        )
        TAVILY_MAX_RESULTS = 10
        MODELNAME = "gpt-oss-1"
        self.SEARCHTOOL = TavilySearchResults(
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
        self.OLLAMAMODEL = ChatOllama(
            model=MODELNAME,
            temperature=0.7,
            streaming=False,
            verbose=True,
            num_ctx=10000,
            base_url="http://localhost:11434"
        )