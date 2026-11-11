"""Repository loading, Python chunking, and Euri embeddings helpers."""

import os
from pathlib import Path
from typing import List, Optional, Union

from dotenv import load_dotenv
from git import InvalidGitRepositoryError, Repo
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def repo_ingestion(
    repo_url: str, repo_path: Union[str, Path] = "repo"
) -> str:
    """Clone a repo, or reuse the same existing clone; return its absolute path.

    Relative destinations are resolved from the current working directory.
    Pass the returned path to ``load_repo`` to avoid loading another folder.
    Existing nonempty folders are never overwritten.
    """
    repo_url = repo_url.strip()
    if not repo_url:
        raise ValueError("repo_url must not be empty.")

    destination = Path(repo_path).expanduser().resolve()
    if destination.exists():
        if not destination.is_dir():
            raise NotADirectoryError(f"Repository path is not a folder: {destination}")
        if any(destination.iterdir()):
            try:
                with Repo(str(destination)) as repository:
                    origins = (
                        list(repository.remotes.origin.urls)
                        if "origin" in repository.remotes
                        else []
                    )
            except InvalidGitRepositoryError as error:
                raise ValueError(
                    f"Folder is nonempty and is not a Git repository: {destination}. "
                    "Choose a different repo_path."
                ) from error

            expected_url = repo_url.replace("\\", "/").rstrip("/").removesuffix(".git")
            if not any(
                url.replace("\\", "/").rstrip("/").removesuffix(".git") == expected_url
                for url in origins
            ):
                raise ValueError(
                    f"A different repository already exists at {destination}. "
                    "Choose a different repo_path."
                )
            return str(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with Repo.clone_from(repo_url, to_path=str(destination)):
        pass
    return str(destination)


def load_repo(repo_path: Union[str, Path]) -> List[Document]:
    """Load Python files recursively with an explicit Python parser."""
    path = Path(repo_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Repository folder does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Repository path is not a folder: {path}")

    loader = GenericLoader.from_filesystem(
        str(path),
        glob="**/*.py",
        suffixes=[".py"],
        parser=LanguageParser(language="python"),
    )
    documents = loader.load()
    if not documents:
        raise ValueError(
            f"No Python documents were loaded from {path}. "
            "Pass the repository path returned by repo_ingestion()."
        )
    return documents


def text_splitter(
    documents: List[Document], chunk_size: int = 2000, chunk_overlap: int = 200
) -> List[Document]:
    """Split Python documents into overlapping chunks, keeping source metadata."""
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size.")

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def load_embedding(
    euri_api_key: Optional[str] = None,
    model: str = "text-embedding-3-small",
    base_url: str = "https://api.euron.one/api/v1/euri",
) -> OpenAIEmbeddings:
    """Configure Euri embeddings for the project's pinned LangChain version.

    Use an explicit key or EURI_API_KEY from the environment/project .env.
    This constructs the client; embedding requests happen when it is used.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = euri_api_key if euri_api_key is not None else os.getenv("EURI_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError("Set EURI_API_KEY in .env or pass euri_api_key explicitly.")

    return OpenAIEmbeddings(
        openai_api_key=api_key.strip(),
        openai_api_base=base_url,
        model=model,
        disallowed_special=(),
        model_kwargs={"encoding_format": "float"},
    )
