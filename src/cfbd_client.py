from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from .utils import get_required_env, get_settings, load_project_env


class CFBDClient:
    def __init__(self) -> None:
        load_project_env()
        settings = get_settings()
        self.base_url = settings['cfbd']['base_url'].rstrip('/')
        token = get_required_env('CFBD_API_KEY')
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {token}'})

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        resp = self.session.get(url, params=params or {}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return [data]
        return data

    def get_df(self, endpoint: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        data = self.get(endpoint, params)
        return pd.json_normalize(data)
