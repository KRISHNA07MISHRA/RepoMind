
from pathlib import PurePosixPath

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)


# Map file extensions to LangChain's supported languages.
LANGUAGE_MAP = {
    ".py": Language.PYTHON,
    ".java": Language.JAVA,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".cpp": Language.CPP,
    ".c": Language.C,
    ".h": Language.C,
    ".hpp": Language.CPP,
    ".html": Language.HTML,
    ".md": Language.MARKDOWN,
    ".php": Language.PHP,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".rb": Language.RUBY,
}


def get_text_splitter(file_path, chunk_size=1000, chunk_overlap=150):
    """Choose a LangChain splitter based on the file extension."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative "
            "and smaller than chunk_size."
        )

    extension = PurePosixPath(file_path).suffix.lower()
    language = LANGUAGE_MAP.get(extension)

    options = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "add_start_index": True,
    }

    if language is not None:
        return RecursiveCharacterTextSplitter.from_language(
            language=language,
            **options,
        )

    # Fallback for JSON, YAML, CSS, text and unsupported extensions.
    return RecursiveCharacterTextSplitter(**options)


def process_file(file_path, content, chunk_size=1000, chunk_overlap=150):
    """
    Convert a repository file into LangChain Documents.

    Each Document stores its file path, chunk index and source line range.
    """

    if not isinstance(content, str):
        raise TypeError("File content must be a string.")

    if not content.strip():
        return []

    splitter = get_text_splitter(
        file_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    documents = splitter.create_documents(
        texts=[content],
        metadatas=[{"file_path": file_path}],
    )

    for index, document in enumerate(documents):
        start_index = document.metadata.get("start_index", 0)

        start_line = content.count("\n", 0, start_index) + 1
        end_line = start_line + document.page_content.count("\n")

        document.metadata.update({
            "chunk_index": index,
            "start_line": start_line,
            "end_line": end_line,
        })

    return documents



if __name__ == "__main__":
    from github_client import get_repository_snapshot, process_repository

    github_url = "https://github.com/psf/requests"

    snapshot = get_repository_snapshot(
        github_url,
        max_files=10,
    )

    documents = process_repository(snapshot)

    print("\nRepository:", snapshot["repository"]["name"])
    print("Fetched files:", len(snapshot["files"]))
    print("Total chunks:", len(documents))
    print("Fetch errors:", len(snapshot.get("errors", [])))

    for document in documents[:3]:
        print("\nMetadata:", document.metadata)
        print(document.page_content[:300])
