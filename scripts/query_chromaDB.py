import chromadb
import os
from dotenv import load_dotenv

load_dotenv()
CHROMA_DB_PORT = os.getenv("CHROMA_DB_PORT")
CHROMA_DB_HOST = os.getenv("CHROMA_DB_HOST")
CHROMA_DB_COLLECTION = "financial_reports"

client = chromadb.HttpClient(host=CHROMA_DB_HOST, port=CHROMA_DB_PORT)
collection = client.get_or_create_collection(name=CHROMA_DB_COLLECTION)

results = collection.query(
    query_texts=["Que signifie OPCVM ?"],
    n_results=2
)

print(
    results["ids"][0][0], 
    results["metadatas"][0][0]["filename"], 
    results["distances"][0][0], 
    results["documents"][0][0], 
    sep="\n"
)

print(
    results["ids"][0][1], 
    results["metadatas"][0][1]["filename"], 
    results["distances"][0][1], 
    results["documents"][0][1], 
    sep="\n"
)