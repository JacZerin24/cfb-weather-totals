from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def load_project_env() -> None:
    load_dotenv(ROOT / '.env')


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(ROOT / path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def get_settings() -> dict[str, Any]:
    return load_yaml('config/settings.yml')


def ensure_dir(path: str | Path) -> Path:
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_df(df: pd.DataFrame, path: str | Path) -> Path:
    out = ROOT / path if not Path(path).is_absolute() else Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == '.parquet':
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    return out


def read_df(path: str | Path) -> pd.DataFrame:
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Missing file: {p}')
    if p.suffix.lower() == '.parquet':
        return pd.read_parquet(p)
    return pd.read_csv(p)


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f'Missing required environment variable: {name}')
    return value
