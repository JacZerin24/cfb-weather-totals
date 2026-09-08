from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ENDPOINT = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
OUTPUT_DIR = Path("outputs/pregame_forecast_realism/source_smoke")

CASES = [
    {"station": "KMSY", "date": "2019-09-07", "label": "new_orleans_2019"},
    {"station": "KSEA", "date": "2021-09-04", "label": "seattle_2021"},
    {"station": "KDEN", "date": "2023-09-02", "label": "denver_2023"},
    {"station": "KBOS", "date": "2025-09-06", "label": "boston_2025"},
]


def _walk_keys(value: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            keys.add(name)
            keys.update(_walk_keys(child, name))
    elif isinstance(value, list):
        for child in value[:5]:
            keys.update(_walk_keys(child, prefix + "[]" if prefix else "[]"))
    return keys


def _count_records(payload: Any) -> int:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("data", "results", "records", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
        return 1 if payload else 0
    return 0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "cfb-weather-totals research source feasibility"})

    for case in CASES:
        start = pd.Timestamp(case["date"], tz="UTC")
        end = start + pd.Timedelta(days=1)
        params = {
            "station": case["station"],
            "model": "NBS",
            "sts": start.strftime("%Y-%m-%dT%H:%MZ"),
            "ets": end.strftime("%Y-%m-%dT%H:%MZ"),
            "format": "json",
        }
        response = session.get(ENDPOINT, params=params, timeout=60)
        text = response.text
        payload: Any = None
        parse_error = ""
        if response.ok:
            try:
                payload = response.json()
            except Exception as exc:  # pragma: no cover - network smoke only
                parse_error = f"{type(exc).__name__}: {exc}"
        raw_path = OUTPUT_DIR / f"{case['label']}.json"
        raw_path.write_text(
            json.dumps(payload, indent=2, default=str) if payload is not None else text,
            encoding="utf-8",
        )
        keys = sorted(_walk_keys(payload)) if payload is not None else []
        rows.append(
            {
                **case,
                "http_status": response.status_code,
                "ok": bool(response.ok),
                "content_type": response.headers.get("content-type", ""),
                "payload_type": type(payload).__name__ if payload is not None else "none",
                "record_count_guess": _count_records(payload),
                "parse_error": parse_error,
                "key_count": len(keys),
                "keys": " | ".join(keys[:150]),
                "response_bytes": len(response.content),
                "request_url": response.url,
            }
        )
        print(
            f"{case['label']}: status={response.status_code} payload={type(payload).__name__} "
            f"records~{_count_records(payload)} bytes={len(response.content)}"
        )
        if keys:
            print("  keys:", ", ".join(keys[:50]))
        if payload is not None:
            preview = json.dumps(payload, default=str)[:2500]
            print("  preview:", preview)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "source_smoke_summary.csv", index=False)
    print(summary[["label", "station", "date", "http_status", "payload_type", "record_count_guess", "key_count"]].to_string(index=False))

    if not summary["ok"].all():
        raise SystemExit("ERROR: At least one NBS archive request failed.")
    if (summary["record_count_guess"] <= 0).any():
        raise SystemExit("ERROR: At least one NBS archive request returned no usable payload records.")


if __name__ == "__main__":
    main()
