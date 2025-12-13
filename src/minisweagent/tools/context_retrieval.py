"""Context Retrieval Tool for Prometheus CRA (Context Retrieval Agent)."""

import os
from typing import Any

import requests

CRA_BASE_URL = os.environ.get("CRA_BASE_URL")

CRA_RETRIEVAL_URL = f"{CRA_BASE_URL}/context/retrieve"


class ContextRetrievalError(Exception):
    """Raised when context retrieval fails."""


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

    # Prepare request payload
    payload = {
        "query": query,
        "max_refined_query": max_refined_query,
    }

    response = None
    try:
        # Send POST request to CRA
        response = requests.post(
            CRA_RETRIEVAL_URL,
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
        raise ContextRetrievalError(f"Failed to connect to CRA at {CRA_RETRIEVAL_URL}: {e}") from e

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
                                   "Default is 3.",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["query"],
        },
    },
}
