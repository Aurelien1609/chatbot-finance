import chromadb
import logging
import os
import sys
import subprocess
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

load_dotenv()
CHROMA_DB_PORT = os.getenv("CHROMA_DB_PORT")
CHROMA_DB_HOST = os.getenv("CHROMA_DB_HOST")
CHROMA_DB_COLLECTION = "financial_reports"

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

compose_dir = Path(__file__).resolve().parent.parent
command = ['docker-compose', 'up', '-d']
try:
    subprocess.run(command, cwd=compose_dir, check=True)
    logging.info("✅ ChromaDB successfully launched via Docker Compose.")
except subprocess.CalledProcessError as e:
    logging.info(f"❌ Error launching Docker Compose: {e}")
    sys.exit(1)

data_folder = Path(__file__).resolve().parent.parent / 'data' / 'DIC'
pdf_files = [f.name for f in data_folder.iterdir() if f.is_file()]

client = chromadb.HttpClient(host=CHROMA_DB_HOST, port=CHROMA_DB_PORT)
collection = client.get_or_create_collection(name=CHROMA_DB_COLLECTION)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

for i, pdf_file in enumerate(pdf_files):
    
    try:
        document = PyPDFLoader(os.path.join(data_folder, pdf_file)).load()
        chunks = text_splitter.split_documents(document)
    except Exception as e:
        logging.warning(f"This file has not been processed : {pdf_file}.")
        logging.warning(e)
        continue

    for j, chunk in enumerate(chunks):
        
        embedding = embeddings_model.embed_query(chunk.page_content)
        collection.upsert(
            embeddings=[embedding],
            documents=[chunk.page_content],
            ids=[f"chunk_{i}_{j}"],
            metadatas=[{"filename": pdf_file}]
        )

logging.info("✅ Embeddings successfully stored in ChromaDB.")

