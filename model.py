# api_client.py
# A tiny client to call a remote /classify API with (image_path, task_id) -> bool.
# All code and comments are in English as requested.

import os
import time
import base64
import json
from typing import Optional, Dict, Any

import requests

__all__ = ["check_task_complete", "ClassifyAPIClient", "APIClientError"]


class APIClientError(Exception):
    """Raised when the remote API cannot be reached or returns an invalid response."""


class ClassifyAPIClient:
    """
    Simple client for a remote Flask API:
        POST {api_url}/classify
        JSON: { "task_id": int, "image": "<base64 string>" }
        Response: { "complete": bool, ... }

    Usage:
        client = ClassifyAPIClient(api_url="http://host:8000", api_key=None)
        is_done = client.check(image_path, task_id=3)
    """

    def __init__(
        self,
        api_url: str,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 2,
        backoff_sec: float = 0.8,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.backoff_sec = float(backoff_sec)
        self.session = requests.Session()
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            # If the server expects an API key header, set it here (customize the header name if needed).
            self.headers["X-API-Key"] = self.api_key
        if extra_headers:
            self.headers.update(extra_headers)

    def _file_to_base64(self, image_path: str) -> str:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _post_json(self, route: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.api_url}{route}"
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.post(
                    url,
                    data=json.dumps(payload),
                    headers=self.headers,
                    timeout=self.timeout,
                )
                if resp.status_code != 200:
                    # try to extract any server error message
                    try:
                        msg = resp.json()
                    except Exception:
                        msg = resp.text
                    raise APIClientError(
                        f"HTTP {resp.status_code} from server at {url}: {msg}"
                    )
                data = resp.json()
                if not isinstance(data, dict):
                    raise APIClientError("Server returned non-JSON or invalid JSON object.")
                return data
            except (requests.RequestException, APIClientError) as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.backoff_sec * (2 ** attempt))  # simple exponential backoff
                    continue
                raise APIClientError(f"Request failed after retries: {last_err}") from last_err

    def check(self, image_path: str, task_id: int) -> bool:
        """
        Returns True/False for 'complete' according to the server.
        Raises APIClientError on connection/response issues.
        """
        img_b64 = self._file_to_base64(image_path)
        payload = {
            "task_id": int(task_id),
            "image": img_b64,  # plain base64 string; if server prefers data URLs, prepend "data:image/jpeg;base64,"
        }
        data = self._post_json("/classify", payload)

        if "complete" not in data:
            raise APIClientError(f"Missing 'complete' field in server response: {data}")

        complete_val = data["complete"]
        if isinstance(complete_val, bool):
            return complete_val

        # Be tolerant: some servers return "true"/"false" strings or 0/1
        if isinstance(complete_val, str):
            low = complete_val.strip().lower()
            if low in ("true", "1", "yes"):
                return True
            if low in ("false", "0", "no"):
                return False
        if isinstance(complete_val, (int, float)):
            return bool(complete_val)

        raise APIClientError(f"Unsupported 'complete' type: {type(complete_val)} | value={complete_val}")


def check_task_complete(
    image_path: str,
    task_id: int,
    api_url: str,
    *,
    api_key: Optional[str] = None,
    timeout: float = 10.0,
    max_retries: int = 2,
    backoff_sec: float = 0.8,
    extra_headers: Optional[Dict[str, str]] = None,
) -> bool:
    """
    One-shot convenience function: return True/False if task is complete.

    Parameters
    ----------
    image_path : str
        Local path to the image file to send.
    task_id : int
        Stage identifier expected by the server.
    api_url : str
        Base URL of the remote API, e.g. "http://10.162.34.47:8000" or "http://localhost:8000".
    api_key : Optional[str]
        Optional API key to send in header "X-API-Key".
    timeout : float
        Per-request timeout (seconds).
    max_retries : int
        Number of retry attempts on failure (network/5xx).
    backoff_sec : float
        Base seconds for exponential backoff between retries.
    extra_headers : Optional[Dict[str, str]]
        Any additional headers to add.

    Returns
    -------
    bool
        The 'complete' value returned by the server.

    Raises
    ------
    FileNotFoundError
        If the image_path does not exist.
    APIClientError
        If the request fails or the server returns an invalid response.
    """
    client = ClassifyAPIClient(
        api_url=api_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
        extra_headers=extra_headers,
    )
    return client.check(image_path=image_path, task_id=task_id)