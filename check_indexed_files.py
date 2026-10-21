
from github_client import get_repository_snapshot

snapshot = get_repository_snapshot(
    "https://github.com/psf/requests",
    max_files=10,
)

print("Files selected for indexing:")
for file_path in snapshot.get("files", {}):
    print(file_path)
