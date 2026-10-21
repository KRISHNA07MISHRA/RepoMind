
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from langchain_qdrant import QdrantVectorStore

from embeddings import get_embedding_model

DB_DIRECTORY = Path(__file__).parent / "qdrant_db"
COLLECTION_NAME = "repomind_code"


def create_vector_store():
    return QdrantVectorStore.from_existing_collection(
        embedding=get_embedding_model(),
        path=str(DB_DIRECTORY),
        collection_name=COLLECTION_NAME,
    )


def index_documents(documents, ids=None):
    if not documents:
        raise ValueError("No documents provided for indexing.")

    embedding_model = get_embedding_model()

    # First indexing creates the collection.
    # Later indexing can add or update records in it.
    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embedding_model,
        path=str(DB_DIRECTORY),
        collection_name=COLLECTION_NAME,
        ids=ids,
    )

    return vector_store


def index_repository(github_url, max_files=10):
    from github_client import get_repository_snapshot, process_repository

    snapshot = get_repository_snapshot(
        github_url,
        max_files=max_files,
    )

    documents = process_repository(snapshot)

    if not documents:
        raise ValueError("No code chunks found in this repository.")

    repository = snapshot.get("repository", {})
    repository_name = (
        repository.get("full_name")
        or f"{repository.get('owner', 'unknown')}/"
           f"{repository.get('name', 'unknown')}"
    )
    branch = repository.get("default_branch", "unknown")

    ids = []

    for document in documents:
        file_path = document.metadata["file_path"]
        chunk_index = document.metadata["chunk_index"]

        stable_key = (
            f"{repository_name}|{branch}|"
            f"{file_path}|{chunk_index}"
        )

        stable_id = str(uuid5(NAMESPACE_URL, stable_key))
        ids.append(stable_id)

    vector_store = index_documents(documents, ids=ids)

    return {
        "repository": repository_name,
        "branch": branch,
        "files_fetched": len(snapshot.get("files", {})),
        "chunks_indexed": len(documents),
        "fetch_errors": len(snapshot.get("errors", [])),
        "vector_store": vector_store,
    }
