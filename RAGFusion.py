import os
from faiss import IndexFlatL2, read_index, write_index
from HyperRAG import LOGGER, ExceptionHandelling
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.output_parsers import SimpleJsonOutputParser
import sqlite3
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain import agents
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Import HuggingFaceEmbeddings and ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from google import genai
from langchain.tools import tool
import re

IMMUNE_SYSTEM_KNOWLEDGE_BASE = [
    "https://en.wikipedia.org/wiki/Innate_immune_system",
    "https://en.wikipedia.org/wiki/Immune_response",
    "https://en.wikipedia.org/wiki/Immunology",
    "https://en.wikipedia.org/wiki/Adaptive_immune_system",
    "https://en.wikipedia.org/wiki/Antibody",
    "https://en.wikipedia.org/wiki/Vaccine"
    ]

class Fusion:
    """Fusion is about combining multiple retrieved sources into one coherent, unified knowledge representation. Instead of treating retrieved chunks
       independently, fusion reconciles, merges, and integrates information across documents.
       Think of fusion as: “Multiple documents are partially correct — synthesize them into a single, consistent answer.”"""
    @ExceptionHandelling
    def __init__(self) -> None:
        self.DIR = os.path.dirname(os.path.abspath(__file__))
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # pip install sentence-transformers
        #CUDA = 0 if torch.cuda.is_available() else -1 # Use GPU if available 0 = GPU, -1 = CPU
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=1000,
            separators=["\n\n", "\n", " ", ""]
        )
        self.model = ChatOllama(model="mistral")
    
    @ExceptionHandelling
    def make_knowledge_base(self, urls=IMMUNE_SYSTEM_KNOWLEDGE_BASE):
        from newspaper import Article
        texts = []
        for u in urls:
            LOGGER.info(f"SCRAPPING URL: {u}")
            article = Article(u)
            article.download()
            article.parse()
            texts.append(article.text)
        df = pd.DataFrame({"url": urls, "text": texts})
        return df, texts
    
    def datacleaning(self, texts: str):
        if not isinstance(texts, str):
            raise ValueError("Input must be a string.")
        # Remove special characters, symbols, and extra whitespace
        cleaned_text = re.sub(r'[^\w\s]', '', texts) # Remove non-alphanumeric and non-whitespace characters
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text) # Replace multiple whitespaces with a single space
        cleaned_text = cleaned_text.strip() # Remove leading/trailing whitespace
        return cleaned_text.replace('\t', '').replace('\n', '')
    
    def generate_embeddings(self, texts: None = None) -> pd.DataFrame:
        #splitting the texts into chunks
        dataframe: pd.DataFrame = self.make_knowledge_base()[0]
        dataframe["index"] = dataframe.index
        all_chunks = []
        for index, row in dataframe.iterrows():
            LOGGER.info(f"PROCESSING ROW INDEX: {row['index']}")
            cleaned_text = self.datacleaning(row['text'])
            chunks = self.text_splitter.split_text(cleaned_text)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "url": row['url'],
                    "text_chunk": chunk,
                    "chunk_index": i
                })
        
        chunks_df = pd.DataFrame(all_chunks)
        chunks_df["embedding"] = chunks_df["text_chunk"].apply(lambda x: self.embeddings.embed_query(x))
        if not os.path.exists(os.path.join(self.DIR, "DATASETS", "WIKIPEDIA.db")):
            os.makedirs(os.path.join(self.DIR, "DATASETS"), exist_ok=True)
    
        connection = sqlite3.connect(os.path.join(self.DIR, "DATASETS", "WIKIPEDIA.db"), detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
        chunks_df.to_sql("knowledge_chunks", connection, if_exists="replace", index=False)
        connection.close()
        return chunks_df
    
if __name__ == "__main__":
    fusion = Fusion()
    fusion.generate_embeddings()