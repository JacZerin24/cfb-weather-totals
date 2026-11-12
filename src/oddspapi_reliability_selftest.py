"""Offline regression checks for OddsPapi bookmaker batching and quota reserve."""
from unittest.mock import patch

import pandas as pd

from . import oddspapi_fallback as api


def test_single_book_requests() -> None:
    calls = []
    def fake_get(path, key, **params):
        calls.append((path, params))
        if path == "account":
            return {"subscriptions": [{"is_active": True, "request_limit": 250,
                                      "request_count": 50,
                                      "bookmakers": {x: {} for x in api.NCAA_BOOKMAKER_PRIORITY}}]}, "ok"
        if path == "fixtures":
            return [], "ok"
        if path == "markets":
            return [], "ok"
        if path == "odds-by-tournaments":
            assert len(params["bookmakers"].split(",")) == 1
            return [], "ok"
        raise AssertionError(path)

    with patch.dict("os.environ", {"ODDSPAPI_API_KEY": "test"}), patch.object(api, "_api_get", side_effect=fake_get), patch.object(api.time, "sleep"):
        _, _, _, stats = api.fetch_oddspapi_ncaa(pd.DataFrame())
    assert stats["oddspapi_status"] == "ok", stats
    assert stats["oddspapi_bulk_calls"] == 4, stats
    assert stats["oddspapi_bulk_failures"] == 0, stats
    assert [p["bookmakers"] for path, p in calls if path == "odds-by-tournaments"] == api.NCAA_BOOKMAKER_PRIORITY


def test_quota_reserve() -> None:
    calls = []
    def fake_get(path, key, **params):
        calls.append(path)
        if path == "account":
            return {"subscriptions": [{"is_active": True, "request_limit": 100,
                                      "request_count": 92,
                                      "bookmakers": {x: {} for x in api.NCAA_BOOKMAKER_PRIORITY}}]}, "ok"
        if path in {"fixtures", "markets", "odds-by-tournaments"}:
            return [], "ok"
        raise AssertionError(path)
    with patch.dict("os.environ", {"ODDSPAPI_API_KEY": "test"}), patch.object(api, "_api_get", side_effect=fake_get), patch.object(api.time, "sleep"):
        _, _, _, stats = api.fetch_oddspapi_ncaa(pd.DataFrame())
    assert stats["oddspapi_bulk_calls"] == 1, stats
    assert calls.count("odds-by-tournaments") == 1


if __name__ == "__main__":
    test_single_book_requests()
    test_quota_reserve()
    print("OddsPapi quota and single-book regression tests passed.")
