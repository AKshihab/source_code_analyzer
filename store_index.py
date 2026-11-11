import os

from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Chroma
from src.helper import (
    load_embedding,
    repo_ingestion,
    load_repo,
    text_splitter,
)

load_dotenv(find_dotenv())

api_key = os.getenv("EURI_API_KEY")
if not api_key or not api_key.strip():
    raise ValueError("Set EURI_API_KEY in your .env file.")

url = "https://github.com/AKshihab/drinks-quality-prediction-system1"
repo_path = repo_ingestion(url, repo_path="repo/")

documents = load_repo(repo_path)
text_chunks = text_splitter(
    documents, chunk_size=500, chunk_overlap=20
)
embeddings = load_embedding(euri_api_key=api_key)

vectordb = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory="repo_index",
)
vectordb.persist()