"""Earthdata/CMR/AppEEARS helper wrappers.

This module provides robust wrappers for interacting with NASA's Earthdata APIs,
including CMR (granule search) and AppEEARS. It centralizes authentication,
retries, and error handling so downstream scripts (SWOT/ECOSTRESS/OPERA) can
focus on their payloads and output.

Requirements
------------
- An Earthdata token must be available via:
  - `NASA_EARTHDATA_TOKEN` environment variable, OR
  - `erie_remote/erie_remote_data/.earthdata_token` file (as used elsewhere in the repo)

Usage
-----
from scripts.nasa_earthdata_client import EarthdataClient

client = EarthdataClient()
granules = client.search_granules(
    collection_concept_id="C2617126679-POCLOUD",
    bbox=(-82.0, 45.0, -81.0, 46.0),
    start_date="2026-03-01",
    end_date="2026-03-19",
)

"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


_LOG = logging.getLogger(__name__)


class EarthdataAuthError(Exception):
    ...


class EarthdataAPIError(Exception):
    ...


def _get_default_token_file() -> Path:
    # Mirrors the pattern used by other components in this repo.
    return Path(__file__).resolve().parent.parent / "erie_remote" / "erie_remote_data" / ".earthdata_token"


def get_earthdata_token() -> Optional[str]:
    """Retrieve NASA Earthdata token from environment or token file."""
    if "NASA_EARTHDATA_TOKEN" in os.environ and os.environ["NASA_EARTHDATA_TOKEN"].strip():
        return os.environ["NASA_EARTHDATA_TOKEN"].strip()

    token_file = _get_default_token_file()
    if token_file.exists():
        token = token_file.read_text(encoding="utf-8").strip()
        if token:
            return token
    return None


def _build_retry_session(token: str, retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    """Build a requests session with retry behavior and Earthdata auth headers."""
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/json"})

    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST", "PUT", "DELETE"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


class EarthdataClient:
    def __init__(
        self,
        token: Optional[str] = None,
        retries: int = 3,
        backoff_factor: float = 0.5,
    ):
        """Initialize the Earthdata client.

        Args:
            token: Optional explicit token. If not provided, the token will be
                retrieved automatically.
        """
        self.token = token or get_earthdata_token()
        if not self.token:
            raise EarthdataAuthError(
                "Earthdata token missing. Set NASA_EARTHDATA_TOKEN or create a .earthdata_token file."
            )
        self.session = _build_retry_session(self.token, retries=retries, backoff_factor=backoff_factor)

    def _raise_for_status(self, resp: requests.Response, context: str) -> None:
        if resp.status_code >= 400:
            msg = f"{context} failed: {resp.status_code} {resp.reason}"
            try:
                # Prefer JSON errors when available
                payload = resp.json()
                msg += f" - {json.dumps(payload)}"
            except Exception:
                msg += f" - {resp.text.strip()}"
            raise EarthdataAPIError(msg)

    def search_granules(
        self,
        collection_concept_id: str,
        bbox: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        page_size: int = 200,
    ) -> List[Dict[str, Any]]:
        """Search CMR for granules matching the query."""
        url = "https://cmr.earthdata.nasa.gov/search/granules.json"
        params = {
            "collection_concept_id": collection_concept_id,
            "bounding_box": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "temporal": f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
            "page_size": str(page_size),
        }

        resp = self.session.get(url, params=params, timeout=60)
        self._raise_for_status(resp, "CMR granule search")

        data = resp.json()
        return data.get("feed", {}).get("entry", [])

    def extract_download_urls(self, granule_entry: Dict[str, Any]) -> List[str]:
        """Return download URLs for a granule entry.

        CMR granule links use a range of `rel` values (e.g. `http://esipfed.org/ns/fedsearch/1.1/data#`)
        and may not match simple values like "data". This helper attempts to
        find the most likely download URLs based on rel/type and file extensions.
        """
        urls: List[str] = []
        for link in granule_entry.get("links", []):
            rel = (link.get("rel") or "").lower()
            content_type = (link.get("type") or "").lower()
            href = link.get("href")
            if not href:
                continue

            # Prefer explicit data/browse rels and known download content types.
            if (
                "data" in rel
                or "browse" in rel
                or rel in ("http", "https")
                or content_type in (
                    "application/octet-stream",
                    "application/x-hdf",
                    "application/x-netcdf",
                    "image/tiff",
                    "image/png",
                )
            ):
                urls.append(href)
                continue

            # Fallback: include common file extensions.
            if href.lower().endswith((".tif", ".tiff", ".nc", ".hdf", ".zip", ".tar.gz")):
                urls.append(href)
        return urls

    def download_url(self, url: str, output_path: str, chunk_size: int = 1024 * 1024) -> str:
        """Download a URL to disk with robust error handling."""
        resp = self.session.get(url, stream=True, timeout=120)
        self._raise_for_status(resp, "Download")

        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        return output_path


class AppEEARSClient:
    """Minimal AppEEARS API wrapper.

    Note: The AppEEARS API is not formally packaged in this repo, so this
    wrapper uses direct HTTP calls with robust retry and error handling.
    """

    BASE_URL = "https://appeears.earthdatacloud.nasa.gov/api"

    def __init__(self, earthdata_client: EarthdataClient):
        self._earthdata = earthdata_client

    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.BASE_URL.rstrip('/')}/{path.lstrip('/')}"
        resp = self._earthdata.session.request(method, url, timeout=120, **kwargs)
        self._earthdata._raise_for_status(resp, f"AppEEARS {method} {path}")
        return resp.json()

    def create_task(
        self,
        layer: str,
        aoi: Dict[str, Any],
        dates: Dict[str, str],
        output_format: str = "GeoTIFF",
    ) -> Dict[str, Any]:
        """Submit an AppEEARS task and return the response."""
        payload = {
            "layer": layer,
            "aoi": aoi,
            "dates": dates,
            "output": {"format": output_format},
        }
        return self._request("POST", "task", json=payload)

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        return self._request("GET", f"task/{task_id}")

    def download_task_results(self, task_id: str, output_path: str) -> str:
        info = self.get_task_status(task_id)
        if info.get("status") != "complete":
            raise EarthdataAPIError(f"Task {task_id} not complete: {info.get('status')}")
        download_url = info.get("output", {}).get("download_url")
        if not download_url:
            raise EarthdataAPIError(f"No download URL available for task {task_id}")
        return self._earthdata.download_url(download_url, output_path)
