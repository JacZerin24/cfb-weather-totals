from __future__ import annotations

from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .utils import get_required_env, get_settings, load_project_env


CFBD_CONNECT_TIMEOUT_SECONDS = 15
CFBD_READ_TIMEOUT_SECONDS = 60
CFBD_RETRY_TOTAL = 2
CFBD_RETRY_BACKOFF_SECONDS = 1.5
CFBD_RETRY_STATUS_CODES = (429, 500, 502, 503, 504)


def build_cfbd_retry_policy() -> Retry:
    """Retry only idempotent CFBD GETs after transient network/server failures."""
    return Retry(
        total=CFBD_RETRY_TOTAL,
        connect=CFBD_RETRY_TOTAL,
        read=CFBD_RETRY_TOTAL,
        status=CFBD_RETRY_TOTAL,
        backoff_factor=CFBD_RETRY_BACKOFF_SECONDS,
        status_forcelist=CFBD_RETRY_STATUS_CODES,
        allowed_methods=frozenset({'GET'}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )


class CFBDClient:
    def __init__(self) -> None:
        load_project_env()
        settings = get_settings()
        self.base_url = settings['cfbd']['base_url'].rstrip('/')
        token = get_required_env('CFBD_API_KEY')
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {token}'})

        # A single slow CFBD response should not invalidate an otherwise valid
        # scheduled research snapshot. Retry only safe GET requests and keep the
        # retry budget bounded so genuine outages still fail visibly.
        adapter = HTTPAdapter(max_retries=build_cfbd_retry_policy())
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        resp = self.session.get(
            url,
            params=params or {},
            timeout=(CFBD_CONNECT_TIMEOUT_SECONDS, CFBD_READ_TIMEOUT_SECONDS),
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return [data]
        return data

    def get_df(self, endpoint: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        data = self.get(endpoint, params)
        return pd.json_normalize(data)
