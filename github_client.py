
import os
import base64
import warnings

from pathlib import PurePosixPath
from urllib.parse import urlparse, quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv


# Load optional GitHub credentials from .env
load_dotenv()


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

GITHUB_API_BASE = "https://api.github.com"

MAX_FILE_BYTES = 100_000
MAX_FILES_PER_BATCH = 20
REQUEST_TIMEOUT = 15

ALLOWED_EXTENSIONS = {
    ".py", ".java", ".js", ".jsx", ".ts", ".tsx",
    ".cpp", ".c", ".h", ".hpp",
    ".html", ".css", ".json", ".yaml", ".yml",
    ".toml", ".md", ".txt", ".xml", ".sql",
    ".properties", ".gradle", ".sh"
}

ALLOWED_EXTENSIONLESS_FILES = {
    "dockerfile",
    "makefile",
    "license",
    "readme",
    "procfile"
}

IGNORED_DIRECTORIES = {
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "target",
    ".git",
    ".idea",
    ".vscode",
    "coverage"
}

IGNORED_FILES = {
    ".env",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "composer.lock"
}


# --------------------------------------------------
# 2. CUSTOM EXCEPTION
# --------------------------------------------------

class GitHubClientError(RuntimeError):
    """Raised when a GitHub API operation fails."""


# --------------------------------------------------
# 3. HTTP SESSION AND RETRIES
# --------------------------------------------------

def create_github_session():
    """
    Create a reusable HTTP session.

    Retries temporary failures for GET requests.
    Uses GITHUB_TOKEN if one is configured.
    """

    session = requests.Session()

    retry_policy = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True
    )

    adapter = HTTPAdapter(max_retries=retry_policy)

    session.mount("https://", adapter)

    session.headers.update({
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "RepoMind"
    })

    token = os.getenv("GITHUB_TOKEN")

    if token:
        session.headers["Authorization"] = f"Bearer {token}"

    return session


SESSION = create_github_session()


# --------------------------------------------------
# 4. COMMON API REQUEST FUNCTION
# --------------------------------------------------

def _get_json(endpoint, params=None):
    """
    Send a GET request to GitHub and return JSON.

    Centralizes timeouts, HTTP errors and API error messages.
    """

    url = f"{GITHUB_API_BASE}/{endpoint.lstrip('/')}"

    try:
        response = SESSION.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT
        )

        if response.status_code >= 400:
            message = response.reason

            try:
                message = response.json().get("message", message)
            except ValueError:
                pass

            if response.status_code == 403:
                remaining = response.headers.get(
                    "X-RateLimit-Remaining"
                )

                if remaining == "0":
                    message = (
                        "GitHub API rate limit reached. "
                        "Wait before retrying or configure GITHUB_TOKEN."
                    )

            raise GitHubClientError(
                f"GitHub API error {response.status_code}: {message}"
            )

        try:
            return response.json()
        except ValueError as error:
            raise GitHubClientError(
                "GitHub returned an invalid JSON response."
            ) from error

    except requests.RequestException as error:
        raise GitHubClientError(
            f"Could not connect to GitHub: {error}"
        ) from error


# --------------------------------------------------
# 5. VALIDATE REPOSITORY OWNER AND NAME
# --------------------------------------------------

def _validate_owner_repo(owner, repo):
    """Validate owner and repository name before making requests."""

    for value, label in (
        (owner, "owner"),
        (repo, "repository")
    ):
        if (
            not isinstance(value, str)
            or not value
            or value in {".", ".."}
            or "/" in value
            or "\\" in value
            or not all(
                character.isalnum() or character in "._-"
                for character in value
            )
        ):
            raise ValueError(f"Invalid GitHub {label}.")


# --------------------------------------------------
# 6. PARSE GITHUB URL
# --------------------------------------------------

def parse_github_url(github_url):
    """
    Convert a URL such as
    https://github.com/psf/requests
    into ('psf', 'requests').
    """

    if not isinstance(github_url, str):
        raise ValueError("GitHub URL must be a string.")

    parsed_url = urlparse(github_url.strip())

    if (
        parsed_url.scheme != "https"
        or parsed_url.netloc.lower() != "github.com"
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise ValueError(
            "Provide a valid HTTPS GitHub repository URL."
        )

    parts = parsed_url.path.strip("/").split("/")

    if len(parts) != 2 or not all(parts):
        raise ValueError(
            "URL must look like "
            "https://github.com/owner/repository"
        )

    owner, repo = parts

    if repo.endswith(".git"):
        repo = repo[:-4]

    _validate_owner_repo(owner, repo)

    return owner, repo


# --------------------------------------------------
# 7. GET REPOSITORY METADATA
# --------------------------------------------------

def get_repository_info(owner, repo):
    """
    Return repository metadata, including its default branch.
    """

    _validate_owner_repo(owner, repo)

    data = _get_json(f"repos/{owner}/{repo}")

    return {
        "name": data["name"],
        "owner": data["owner"]["login"],
        "description": data.get("description"),
        "language": data.get("language"),
        "stars": data["stargazers_count"],
        "default_branch": data["default_branch"],
        "html_url": data["html_url"],
        "private": data["private"],
        "size": data["size"],
        "updated_at": data.get("updated_at")
    }


# --------------------------------------------------
# 8. RESOLVE BRANCH
# --------------------------------------------------

def resolve_branch(owner, repo, branch=None):
    """
    Use the supplied branch or discover the repository's
    actual default branch.
    """

    if branch is not None:
        if not isinstance(branch, str) or not branch.strip():
            raise ValueError("Branch must be a non-empty string.")

        return branch.strip()

    info = get_repository_info(owner, repo)

    return info["default_branch"]


# --------------------------------------------------
# 9. GET COMPLETE REPOSITORY TREE RESPONSE
# --------------------------------------------------

def get_repository_tree_data(owner, repo, branch=None):
    """
    Fetch the recursive Git tree.

    Returns both the file entries and GitHub's truncated flag.
    """

    _validate_owner_repo(owner, repo)

    branch = resolve_branch(owner, repo, branch)

    encoded_branch = quote(branch, safe="")

    data = _get_json(
        f"repos/{owner}/{repo}/git/trees/{encoded_branch}",
        params={"recursive": "1"}
    )

    return {
        "tree": data.get("tree", []),
        "truncated": data.get("truncated", False),
        "branch": branch
    }


# --------------------------------------------------
# 10. GET REPOSITORY TREE
# --------------------------------------------------

def get_repository_tree(owner, repo, branch=None):
    """
    Return the list of tree entries.

    Retains the return format of the earlier function.
    """

    result = get_repository_tree_data(owner, repo, branch)

    if result["truncated"]:
        warnings.warn(
            "GitHub truncated the repository tree. "
            "Some files may be missing.",
            RuntimeWarning
        )

    return result["tree"]


# --------------------------------------------------
# 11. CHECK WHETHER A FILE SHOULD BE INCLUDED
# --------------------------------------------------

def should_include_file(file_path):
    """
    Keep relevant source, configuration and documentation files.
    Exclude dependencies, generated output and likely secrets.
    """

    if not isinstance(file_path, str) or not file_path:
        return False

    path = PurePosixPath(file_path)

    if path.is_absolute() or ".." in path.parts:
        return False

    parts = [part.lower() for part in path.parts]
    filename = parts[-1]

    if any(part in IGNORED_DIRECTORIES for part in parts):
        return False

    if filename in IGNORED_FILES:
        return False

    if filename == ".env" or filename.startswith(".env."):
        return False

    if filename.endswith((".pem", ".key", ".p12", ".pfx")):
        return False

    if path.suffix.lower() in ALLOWED_EXTENSIONS:
        return True

    return filename in ALLOWED_EXTENSIONLESS_FILES


# --------------------------------------------------
# 12. GET RELEVANT FILE PATHS
# --------------------------------------------------

def get_relevant_files(owner, repo, branch=None, tree=None):
    """
    Filter the repository tree and return relevant file paths.

    Pass an existing tree to avoid another API request.
    """

    branch = resolve_branch(owner, repo, branch)

    if tree is None:
        tree = get_repository_tree(owner, repo, branch)

    relevant_files = []

    for item in tree:
        if item.get("type") != "blob":
            continue

        file_path = item.get("path", "")

        if should_include_file(file_path):
            relevant_files.append(file_path)

    return relevant_files


# --------------------------------------------------
# 13. GET ONE FILE'S CONTENT
# --------------------------------------------------

def get_file_content(owner, repo, file_path, branch=None):
    """
    Fetch and Base64-decode a single text file.
    """

    _validate_owner_repo(owner, repo)

    if not should_include_file(file_path):
        raise ValueError(
            f"File is excluded by RepoMind's file filter: {file_path}"
        )

    branch = resolve_branch(owner, repo, branch)

    encoded_path = quote(file_path, safe="/")

    data = _get_json(
        f"repos/{owner}/{repo}/contents/{encoded_path}",
        params={"ref": branch}
    )

    if data.get("type") != "file":
        raise ValueError("The requested path is not a file.")

    if data.get("encoding") != "base64":
        raise ValueError(
            "GitHub did not return Base64-encoded file content."
        )

    encoded_content = data.get("content")

    if encoded_content is None:
        raise GitHubClientError(
            "GitHub did not provide the file content."
        )

    try:
        raw_content = base64.b64decode(
            encoded_content,
            validate=False
        )
        return raw_content.decode("utf-8")

    except (ValueError, UnicodeDecodeError) as error:
        raise GitHubClientError(
            f"Could not decode text file: {file_path}"
        ) from error


# --------------------------------------------------
# 14. FETCH MULTIPLE FILES
# --------------------------------------------------

def get_multiple_file_contents(
    owner,
    repo,
    file_paths,
    branch=None,
    max_file_bytes=MAX_FILE_BYTES,
    max_files=MAX_FILES_PER_BATCH,
    tree=None
):
    """
    Fetch multiple files with size limits and per-file errors.

    Returns fetched files, skipped files, errors and unprocessed paths.
    """

    _validate_owner_repo(owner, repo)

    if max_file_bytes < 0 or max_files < 0:
        raise ValueError("File limits cannot be negative.")

    branch = resolve_branch(owner, repo, branch)

    if tree is None:
        tree = get_repository_tree(owner, repo, branch)

    file_sizes = {
        item["path"]: item.get("size")
        for item in tree
        if item.get("type") == "blob"
    }

    unique_paths = list(dict.fromkeys(file_paths))
    selected_paths = unique_paths[:max_files]
    not_processed = unique_paths[max_files:]

    files = {}
    skipped = []
    errors = []

    for file_path in selected_paths:

        if not should_include_file(file_path):
            skipped.append({
                "file": file_path,
                "reason": "Excluded by file filter"
            })
            continue

        size = file_sizes.get(file_path)

        if size is None:
            skipped.append({
                "file": file_path,
                "reason": "File not found in repository tree"
            })
            continue

        if size > max_file_bytes:
            skipped.append({
                "file": file_path,
                "reason": f"File exceeds {max_file_bytes} bytes"
            })
            continue

        try:
            content = get_file_content(
                owner,
                repo,
                file_path,
                branch
            )

            files[file_path] = content

        except (
            GitHubClientError,
            requests.RequestException,
            ValueError,
            KeyError
        ) as error:
            errors.append({
                "file": file_path,
                "error": str(error)
            })

    return {
        "branch": branch,
        "files": files,
        "skipped": skipped,
        "errors": errors,
        "not_processed": not_processed
    }


# --------------------------------------------------
# 15. GET A COMPLETE REPOSITORY SNAPSHOT
# --------------------------------------------------

def get_repository_snapshot(
    github_url,
    max_files=MAX_FILES_PER_BATCH,
    max_file_bytes=MAX_FILE_BYTES
):
    """
    Main ingestion function for RepoMind.

    URL -> metadata -> tree -> relevant paths -> file contents.
    """

    owner, repo = parse_github_url(github_url)

    info = get_repository_info(owner, repo)
    branch = info["default_branch"]

    tree_data = get_repository_tree_data(
        owner,
        repo,
        branch
    )

    tree = tree_data["tree"]

    relevant_paths = get_relevant_files(
        owner,
        repo,
        branch,
        tree
    )

    content_result = get_multiple_file_contents(
        owner,
        repo,
        relevant_paths,
        branch=branch,
        max_file_bytes=max_file_bytes,
        max_files=max_files,
        tree=tree
    )

    return {
        "repository": info,
        "tree_truncated": tree_data["truncated"],
        "relevant_file_paths": relevant_paths,
        "files": content_result["files"],
        "skipped": content_result["skipped"],
        "errors": content_result["errors"],
        "not_processed": content_result["not_processed"]
    }


# --------------------------------------------------
# 16. LOCAL TEST
# --------------------------------------------------

if __name__ == "__main__":

    github_url = "https://github.com/KRISHNA07MISHRA/shopshere-backend"

    try:
        snapshot = get_repository_snapshot(
            github_url,
            max_files=10,
            max_file_bytes=MAX_FILE_BYTES
        )

        repository = snapshot["repository"]

        print("\nREPOSITORY")
        print("Name:", repository["name"])
        print("Owner:", repository["owner"])
        print("Default branch:", repository["default_branch"])
        print("Language:", repository["language"])
        print("Stars:", repository["stars"])

        print("\nINGESTION SUMMARY")
        print("Relevant paths:", len(snapshot["relevant_file_paths"]))
        print("Fetched files:", len(snapshot["files"]))
        print("Skipped files:", len(snapshot["skipped"]))
        print("Errors:", len(snapshot["errors"]))
        print("Not processed:", len(snapshot["not_processed"]))
        print("Tree truncated:", snapshot["tree_truncated"])

        for path, content in snapshot["files"].items():
            print(f"\n--- {path} ---")
            print(content[:300])

        if snapshot["skipped"]:
            print("\nSKIPPED FILES")
            for item in snapshot["skipped"]:
                print(item)

        if snapshot["errors"]:
            print("\nERRORS")
            for item in snapshot["errors"]:
                print(item)

    except (GitHubClientError, ValueError) as error:
        print(f"RepoMind ingestion failed: {error}")
