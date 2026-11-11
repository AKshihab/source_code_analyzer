from src.helper import load_embedding, repo_ingestion, load_repo, text_splitter
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Chroma
import os

load_dotenv(find_dotenv())

openai_api_key = os.environ.get("EURI_API_KEY")
if not openai_api_key or not openai_api_key.strip():
    raise ValueError("Set EURI_API_KEY in your .env file.")

url = "https://github.com/AKshihab/drinks-quality-prediction-system1"
repo_path = repo_ingestion(url, repo_path="repo/")


documents = load_repo(repo_path)
text_chunks = text_splitter(documents, chunk_size=500, chunk_overlap=20)
embeddings = load_embedding(euri_api_key=openai_api_key)

vectordb = Chroma.from_documents(text_chunks, embeddings, persist_directory="repo_index")
vectordb.persist()
