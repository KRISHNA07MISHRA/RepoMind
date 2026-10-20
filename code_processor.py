
def chunk_file(file_path, content, chunk_size=40, overlap=5):
    """
    Split a source file into line-based chunks.

    Args:
        file_path: Repository-relative path of the file.
        content: Complete text content of the file.
        chunk_size: Maximum lines in each chunk.
        overlap: Number of lines shared between consecutive chunks.

    Returns:
        A list of chunks with source path and line numbers.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    lines = content.splitlines()
    chunks = []
    start = 0

    while start < len(lines):
        end = min(start + chunk_size, len(lines))

        chunk_text = "\n".join(lines[start:end])

        chunks.append({
            "file_path": file_path,
            "start_line": start + 1,
            "end_line": end,
            "content": chunk_text
        })

        if end == len(lines):
            break

        start = end - overlap

    return chunks
