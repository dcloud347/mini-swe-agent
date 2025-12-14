"""Context Retrieval Tool for Prometheus CRA (Context Retrieval Agent)."""

import os
from typing import Any

import requests


class ContextRetrievalError(Exception):
    """Raised when context retrieval fails."""


def _get_cra_retrieval_url() -> str:
    """Get CRA retrieval URL from environment variable."""
    cra_base_url = os.environ.get("CRA_BASE_URL")
    if not cra_base_url:
        raise ContextRetrievalError("CRA_BASE_URL environment variable is not set")
    return f"{cra_base_url}/context/retrieve"


def context_retrieval_tool(
        query: str,
        max_refined_query: int = 3,
) -> dict[str, Any]:
    """
    Call Prometheus Context Retrieval Agent to retrieve relevant context.

    Args:
        query: The search query to find relevant context
        max_refined_query: Maximum number of query refinements (default: 3)

    Returns:
        dict containing:
            - contexts: List of context snippets, each with:
                - relative_path: File path relative to repository root
                - content: Code snippet content
                - start_line_number: Starting line number of the snippet
                - end_line_number: Ending line number of the snippet
            - total_contexts: Total number of contexts retrieved

    Raises:
        ContextRetrievalError: If the request fails or returns an error

    Example:
        >>> result = context_retrieval_tool("authentication implementation")
        >>> print(f"Found {result['total_contexts']} contexts")
        >>> for ctx in result['contexts']:
        ...     print(f"{ctx['relative_path']}:{ctx['start_line_number']}-{ctx['end_line_number']}")
        ...     print(ctx['content'])
    """

    # Get CRA URL (read from environment at call time, not import time)
    cra_url = _get_cra_retrieval_url()

    # Get repository_id from environment
    repository_id = os.environ.get("CRA_REPOSITORY_ID")
    if not repository_id:
        raise ContextRetrievalError("CRA_REPOSITORY_ID environment variable is not set. Repository must be uploaded first.")

    # Prepare request payload
    payload = {
        "query": query,
        "max_refined_query_loop": max_refined_query,
        "repository_id": int(repository_id),
    }

    response = None
    try:
        # Send POST request to CRA
        response = requests.post(
            cra_url,
            json=payload,
            timeout=None,
            headers={
                "Content-Type": "application/json",
            },
        )

        # Check if request was successful
        response.raise_for_status()

        # Parse response
        data = response.json()["data"]

        return data

    except requests.exceptions.ConnectionError as e:
        raise ContextRetrievalError(f"Failed to connect to CRA at {cra_url}: {e}") from e

    except requests.exceptions.HTTPError as e:
        error_msg = f"CRA returned error status {response.status_code if response else 'unknown'}"
        if response:
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg += f": {error_data['error']}"
            except Exception:
                error_msg += f": {response.text}"
        raise ContextRetrievalError(error_msg) from e

    except requests.exceptions.RequestException as e:
        raise ContextRetrievalError(f"CRA request failed: {e}")

    except ValueError as e:
        raise ContextRetrievalError(f"Failed to parse CRA response as JSON: {e}")


# Tool definition for agent registration (OpenAI function calling format)
CONTEXT_RETRIEVAL_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "context_retrieval",
        "description": "Retrieve relevant context from the codebase using Prometheus Context Retrieval Agent. "
                       "Use this to search for code, documentation, or implementation details.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query describing what context you need to retrieve. "
                                   "Be specific about what you're looking for (e.g., 'authentication implementation', "
                                   "'error handling in API routes').",
                },
                "max_refined_query": {
                    "type": "integer",
                    "description": "Maximum number of query refinements to perform for better results. "
                                   "Higher values may provide more rich and accurate context but take longer. "
                                   "Default is 1.",
                    "default": 1,
                    "minimum": 1,
                    "maximum": 1,
                },
            },
            "required": ["query"],
        },
        "examples": [
            {
                "query": "Find ALL self-contained context needed to understand the authentication implementation and how user credentials are validated in the system."
            },
            {
                "query": "Find ALL self-contained context needed to understand how database errors are caught and handled throughout the application."
            },
            {
                "query": "Find ALL self-contained context needed to understand the API routing structure and how endpoints are registered."
            }
        ],
    },
}
