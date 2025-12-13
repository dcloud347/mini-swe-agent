"""Repository management utilities for Prometheus CRA."""

import os
from typing import Any

import requests


class RepositoryError(Exception):
    """Raised when repository operations fail."""


def get_cra_base_url() -> str:
    """Get CRA base URL from environment variable."""
    base_url = os.environ.get("CRA_BASE_URL")
    if not base_url:
        raise RepositoryError("CRA_BASE_URL environment variable is not set")
    return base_url


def upload_repository(
    https_url: str,
    commit_id: str | None = None
) -> dict[str, Any]:
    """
    Upload a repository to Prometheus CRA system.

    Args:
        https_url: HTTPS URL of the git repository (e.g., "https://github.com/user/repo.git")
        commit_id: Specific commit ID to use (optional, defaults to latest)
        timeout: Request timeout in seconds (default: 300 for repo uploads)

    Returns:
        dict containing:
            - repository_id: The ID of the uploaded repository
            - status: Upload status
            - Additional metadata from the server

    Raises:
        RepositoryError: If the upload fails or returns an error

    Example:
        >>> result = upload_repository("https://github.com/user/repo.git")
        >>> print(f"Repository ID: {result['repository_id']}")
    """
    base_url = get_cra_base_url()
    endpoint = f"{base_url}/repository/upload/"

    # Prepare request payload
    payload = {
        "https_url": https_url,
        "commit_id": commit_id,
    }

    try:
        # Send POST request to upload repository
        response = requests.post(
            endpoint,
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

        # Validate response structure
        if "repository_id" not in data:
            raise RepositoryError(f"Invalid upload response: missing 'repository_id' field. Got: {data}")

        return data

    except requests.exceptions.ConnectionError as e:
        raise RepositoryError(f"Failed to connect to CRA at {endpoint}: {e}")

    except requests.exceptions.HTTPError as e:
        error_msg = f"Upload failed with status {response.status_code}"
        try:
            error_data = response.json()
            if "error" in error_data:
                error_msg += f": {error_data['error']}"
            elif "detail" in error_data:
                error_msg += f": {error_data['detail']}"
        except Exception:
            error_msg += f": {response.text}"
        raise RepositoryError(error_msg)

    except requests.exceptions.RequestException as e:
        raise RepositoryError(f"Upload request failed: {e}")

    except ValueError as e:
        raise RepositoryError(f"Failed to parse upload response as JSON: {e}")


def delete_repository(
    repository_id: int,
    force: bool = False,
) -> dict[str, Any]:
    """
    Delete a repository from Prometheus CRA system.

    Args:
        repository_id: The ID of the repository to delete
        force: Force deletion even if there are dependencies (default: False)
        timeout: Request timeout in seconds (default: 60)

    Returns:
        dict containing:
            - status: Deletion status
            - message: Deletion message
            - Additional metadata from the server

    Raises:
        RepositoryError: If the deletion fails or returns an error

    Example:
        >>> delete_repository(repository_id=123, force=False)
        {'status': 'success', 'message': 'Repository deleted'}
    """
    base_url = get_cra_base_url()
    endpoint = f"{base_url}/repository/delete/"

    # Prepare query parameters
    params = {
        "repository_id": repository_id,
        "force": force,
    }

    try:
        # Send DELETE request
        response = requests.delete(
            endpoint,
            params=params,
            timeout=None,
            headers={
                "Content-Type": "application/json",
            },
        )

        # Check if request was successful
        response.raise_for_status()

        # Parse response
        data = response.json()

        return data

    except requests.exceptions.ConnectionError as e:
        raise RepositoryError(f"Failed to connect to CRA at {endpoint}: {e}")

    except requests.exceptions.HTTPError as e:
        error_msg = f"Deletion failed with status {response.status_code}"
        try:
            error_data = response.json()
            if "error" in error_data:
                error_msg += f": {error_data['error']}"
            elif "detail" in error_data:
                error_msg += f": {error_data['detail']}"
        except Exception:
            error_msg += f": {response.text}"
        raise RepositoryError(error_msg)

    except requests.exceptions.RequestException as e:
        raise RepositoryError(f"Deletion request failed: {e}")

    except ValueError as e:
        raise RepositoryError(f"Failed to parse deletion response as JSON: {e}")