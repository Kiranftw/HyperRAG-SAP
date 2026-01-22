import os
import pickle
import pandas as pd
import logging
from typing import List, Dict, Any
import sqlite3
import re
from functools import wraps
import traceback
import os
import sqlite3
import logging
import pandas as pd
import spacy
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

logging.basicConfig(level=logging.INFO)
class DataCleaning(object):
    def __init__(self) -> None:
        self.NLP = spacy.load("en_core_web_sm")
    
    def data_processing(self, chunk: str) -> str:
        _PUA_RANGES = [(0xE000, 0xF8FF), (0xF0000, 0xFFFFD), (0x100000, 0x10FFFD)]
        _PUA_REGEX = re.compile(
            "[" + "".join(
                f"\\u{start:04X}-\\u{end:04X}" if end <= 0xFFFF
                else f"\\U{start:08X}-\\U{end:08X}" for start, end in _PUA_RANGES
            ) + "]"
        )

        BOILERPLATE_PATTERNS = [
            r"^\s*additional\s+languages?\s+available\s*$",
            r"^\s*benefit\s+from\s+machine\s+translations?.*translation\s+hub\s*$",
            r"^\s*use\s+the\s+language\s+menu.*preferred\s+language\s*$",
            r"^\s*don[’']?t\s+show\s+me\s+again\s*$",
            r"^\s*this\s+document\s*$",
            r"^\s*table\s+of\s+contents\s*$",
            r"^\s*pdf\s*$",
            r"^\s*favorite\s*$",
            r"^\s*share\s*$",
            r"^\s*was\s+this\s+page\s+helpful\??\s*$",
            r"^\s*yes\s*$",
            r"^\s*no\s*$",
            r"^\s*privacy\s+policy\s*$",
            r"^\s*terms\s+of\s+use\s*$",
            r"^\s*legal\s+notice\s*$",
            r"^\s*copyright\s+.*$",
            r"^\s*all\s+rights\s+reserved\s*$",
            r"^\s*cookie\s+preferences?\s*$",
            r"^\s*feedback\s*$",
            r"^\s*back\s+to\s+top\s*$",
            r"^\s*version:\s*\d+\s*(latest)?\s*$",
            r"^\s*english\s*$",
            r"^\s*on\s+this\s+page\s*$",
            r"^\s*how\s+we\s+can\s+help\s*$",
            r"^\s*[^\w\s]{1,10}\s*$",
        ]
        KEEP_PHRASES = [r"\bS/4HANA\b", r"\bS4HANA\b", r"\bSAP\b", r"\bFiori\b", r"\bABAP\b", r"\bHANA\b"]
        KEEP_RES = [re.compile(p, re.I) for p in KEEP_PHRASES]
        BOILERPLATE_RES = [re.compile(p, re.I) for p in BOILERPLATE_PATTERNS]

        CUSTOM_STOPWORDS = {"the", "a", "an"}
        ALLOWED_PUNCTUATION = {".", ",", "?", "!", ":", ";", "(", ")", "-", "/"}
        NOISE_REGEX = re.compile(r"[^\w\s.,?!:;()\/\-]")

        chunk = _PUA_REGEX.sub("", chunk)

        lines = chunk.splitlines()
        KEPT = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(p.search(line) for p in KEEP_RES):
                KEPT.append(line)
                continue
            if any(p.match(line) for p in BOILERPLATE_RES):
                continue
            KEPT.append(line)

        chunk = " ".join(KEPT)
        chunk = NOISE_REGEX.sub(" ", chunk)
        chunk = re.sub(r"\.{2,}", ".", chunk)
        chunk = re.sub(r"[?!]{2,}", "?", chunk)
        chunk = re.sub(r"\s+", " ", chunk).strip()

        doc = self.NLP(chunk)
        cleaned_tokens = []
        for token in doc:
            if token.is_space:
                continue
            if token.text.lower() in CUSTOM_STOPWORDS:
                continue
            if token.text in ALLOWED_PUNCTUATION:
                cleaned_tokens.append(token.text)
                continue
            lemma = token.lemma_
            cleaned_tokens.append(lemma)

        return " ".join(cleaned_tokens)
    
    
    def data_cleaning(self, DBPath: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets" ,"PRODUICTION.db")) -> None:
        connection = sqlite3.connect(DBPath)
        cursor = connection.cursor()
        dataframe = pd.read_sql_query("SELECT * FROM documents", connection)

        if "cleaned_content" not in dataframe.columns:
            dataframe["cleaned_content"] = None

        TOTAL = len(dataframe)
        LOGGER.info(f"FOUND {TOTAL} ROWS TO CLEAN")
        BATCH = []
        for index, row in dataframe.iterrows():
            content = row["content"]
            cleaned = self.data_processing(content)
            dataframe.at[index, "cleaned_content"] = cleaned
            BATCH.append(dataframe.iloc[index])

            if (index + 1) % 1000 == 0:
                temp_df = pd.DataFrame(BATCH)
                temp_df.to_sql("documents_cleaned", connection,
                            if_exists="append", index=False)
                connection.commit()
                LOGGER.info(f"💾 SAVED {index + 1}/{TOTAL} ROWS TO DB")
                BATCH.clear()  
            LOGGER.info("CLEANED CONTENT FOR ROW %d", index + 1)
        if BATCH:
            temp_df = pd.DataFrame(BATCH)
            temp_df.to_sql("documents_cleaned", connection,
                        if_exists="append", index=False)
            connection.commit()
            LOGGER.info(f"💾 FINAL SAVED {len(BATCH)} ROWS TO DB")

        connection.close()
        LOGGER.info("✅ DONE CLEANING DATA")
        return None
    