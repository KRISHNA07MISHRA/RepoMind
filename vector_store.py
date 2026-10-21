
from pathlib import Path

from langchain_qdrant import QdrantVectorStore, RetrievalMode

from embeddings import get_embedding_model

DB_DIRECTORY = Path(__file__).parent / "qdrant_db"
COLLECTION_NAME = "repomind_code"


def create_vector_store():
    embedding_model = get_embedding_model()

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        path=str(DB_DIRECTORY),
        collection_name=COLLECTION_NAME,
        retrieval_mode=RetrievalMode.DENSE,
    )

    return vector_store


def index_documents(documents):
    if not documents:
        raise ValueError("No documents provided for indexing.")

    embedding_model = get_embedding_model()

    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embedding_model,
        path=str(DB_DIRECTORY),
        collection_name=COLLECTION_NAME,
        force_recreate=False,
    )

    return vector_store
