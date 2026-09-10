from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from .market_line_cache import apply_market_line_cache, cached_market_reason, load_market_cache


def _board(
    total: float | None,
    provider: str = 'DraftKings',
    start_date: str = '2026-09-12T19:00:00Z',
) -> pd.DataFrame:
    return pd.DataFrame([{
        'game_id': 12345,
        'season': 2026,
        'week': 2,
        'start_date': start_date,
        'away_team': 'Away',
        'home_team': 'Home',
        'division_track': 'FBS',
        'closing_total': total,
        'line_provider': provider if total is not None else '',
        'line_source': 'CFBD' if total is not None else '',
        'line_provider_count': 2 if total is not None else None,
        'line_total_median': total,
        'line_total_range': 0.5 if total is not None else None,
    }])


def main() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / 'market_total_cache.csv'

        first, stats = apply_market_line_cache(
            _board(55.5), now='2026-09-09T12:00:00Z', path=path,
        )
        assert stats == {
            'fresh_lines': 1,
            'restored_lines': 0,
            'usable_lines': 1,
            'cached_entries': 1,
            'skipped_rescheduled': 0,
        }
        assert first.loc[0, 'line_cache_status'] == 'LIVE'
        assert not bool(first.loc[0, 'line_is_cached'])

        missing, stats = apply_market_line_cache(
            _board(None), now='2026-09-09T15:30:00Z', path=path,
        )
        assert stats['fresh_lines'] == 0
        assert stats['restored_lines'] == 1
        assert float(missing.loc[0, 'closing_total']) == 55.5
        assert missing.loc[0, 'line_provider'] == 'DraftKings'
        assert missing.loc[0, 'line_source'] == 'CFBD'
        assert missing.loc[0, 'line_cache_status'] == 'CACHED'
        assert bool(missing.loc[0, 'line_is_cached'])
        assert abs(float(missing.loc[0, 'line_age_hours']) - 3.5) < 1e-9
        reason = cached_market_reason(missing.loc[0])
        assert reason is not None and 'fresh market line is required' in reason

        # Reusing a cached line must not refresh its timestamp.
        cache = load_market_cache(path)
        assert str(cache.loc[0, 'line_last_seen_utc']) == '2026-09-09 12:00:00+00:00'

        # A materially changed kickoff is a new market context. The old line
        # must not be restored or allowed to orient the model.
        rescheduled, stats = apply_market_line_cache(
            _board(None, start_date='2026-09-13T15:00:00Z'),
            now='2026-09-09T16:00:00Z',
            path=path,
        )
        assert stats['restored_lines'] == 0
        assert stats['skipped_rescheduled'] == 1
        assert pd.isna(rescheduled.loc[0, 'closing_total'])
        assert rescheduled.loc[0, 'line_cache_status'] == 'NO LINE'

        updated, stats = apply_market_line_cache(
            _board(56.5, 'FanDuel', start_date='2026-09-13T15:00:00Z'),
            now='2026-09-09T16:30:00Z', path=path,
        )
        assert stats['fresh_lines'] == 1
        assert stats['restored_lines'] == 0
        assert float(updated.loc[0, 'closing_total']) == 56.5
        assert updated.loc[0, 'line_cache_status'] == 'LIVE'

        cache = load_market_cache(path)
        assert len(cache) == 1
        assert float(cache.loc[0, 'closing_total']) == 56.5
        assert cache.loc[0, 'line_provider'] == 'FanDuel'
        # Rescheduling resets first-seen because this is a materially new event context.
        assert str(cache.loc[0, 'line_first_seen_utc']) == '2026-09-09T16:30:00+00:00'

    print('Market line cache self-tests passed: live, fallback, timestamp, reschedule guard, and refresh.')


if __name__ == '__main__':
    main()
