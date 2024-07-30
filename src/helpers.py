import os
from typing import List, Tuple
from langchain_community.document_loaders import GitHubIssuesLoader
from langchain.schema import Document


def get_repo_config(index: int) -> Tuple[str, str, str]:
    """Get repository configuration from environment variables."""
    owner = os.getenv(f"GITHUB_OWNER_{index}")
    repo = os.getenv(f"GITHUB_REPO_{index}")
    branch = os.getenv(f"GITHUB_BRANCH_{index}", "main")
    return owner, repo, branch


def load_documents_from_repo(github_token: str, owner: str, repo: str, branch: str) -> List[Document]:
    """Load issues from a single GitHub repository."""
    loader = GitHubIssuesLoader(
        repo=f"{owner}/{repo}",
        access_token=github_token,
        state="all",  # This will load both open and closed issues
    )

    try:
        documents = loader.load()
        print(f"Loaded {len(documents)} issues from {owner}/{repo}")
        for i, doc in enumerate(documents):
            print(f"Issue {i + 1} content length: {len(doc.page_content)} characters")
        return documents
    except Exception as e:
        print(f"Error loading issues from {owner}/{repo}: {e}")
        return []