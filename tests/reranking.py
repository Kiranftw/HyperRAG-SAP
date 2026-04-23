import cohere
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import os
import json
#connect cohere
document_path = "/home/kiranftw/HyperRAG-SAP/src/combined_results.json"
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
with open(document_path, "r") as f:
    documents = json.load(f)

#rerank query and documents
texts = [doc["text"] for doc in documents]
results = co.rerank(
    model="rerank-v4.0-pro",
    query="What is SAP S/4HANA?",
    documents=texts,
    top_n=2,
)

# Display results
for result in results.results:
    print(f"Index: {result.index}")
    print(f"Score: {result.relevance_score}")
    print(f"Document chunk text: {documents[result.index]['text'][:150]}...\n")
