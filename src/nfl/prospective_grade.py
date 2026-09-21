from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from ..utils import ROOT, load_yaml, write_df
from .live_week import SCHEDULE_URL, _download_csv, _kickoff_utc
from .prospective_ledger import (
    MONITOR_DIR,
    PROSPECTIVE_ROOT,
    _bool,
    _read_immutable,
    rebuild_derived,
)


EVAL_CONFIG = ROOT / 'config/nfl_prospective_evaluation.yml'
RESULTS_PATH = PROSPECTIVE_ROOT / 'graded_entries.csv'
DECISIONS_PATH = PROSPECTIVE_ROOT / 'graded_decisions.csv'
CALIBRATION_PATH = PROSPECTIVE_ROOT / 'calibration_bins.csv'
STABILITY_PATH = PROSPECTIVE_ROOT / 'signal_stability.csv'
SUMMARY_PATH = PROSPECTIVE_ROOT / 'performance_summary.csv'
STATUS_PATH = PROSPECTIVE_ROOT / 'evaluation_status.json'
REPORT_PATH = PROSPECTIVE_ROOT / 'prospective_evaluation.md'
BREAKEVEN_MINUS110 = 110.0 / 210.0
FINAL_BUFFER_HOURS = 5.0


def _american_profit(odds: float) -> float:
    if not np.isfinite(odds) or odds == 0:
        return np.nan
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def _wilson_interval(
    wins: int,
    trials: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if trials <= 0:
        return np.nan, np.nan
    p = wins / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    spread = (
        z
        * math.sqrt(
            (p * (1.0 - p) / trials)
            + (z * z / (4.0 * trials * trials))
        )
        / denom
    )
    return max(0.0, center - spread), min(1.0, center + spread)


def _load_schedule(now: pd.Timestamp) -> pd.DataFrame:
    schedule = _download_csv(SCHEDULE_URL).copy()
    schedule['season'] = pd.to_numeric(
        schedule.get('season'),
        errors='coerce',
    )
    schedule = schedule[
        schedule['season'].eq(2026)
        & schedule['game_type'].astype(str).str.upper().eq('REG')
    ].copy()
    schedule['game_id'] = schedule['game_id'].astype(str)
    schedule['kickoff_utc'] = [
        _kickoff_utc(day, time)
        for day, time in zip(schedule['gameday'], schedule['gametime'])
    ]
    schedule['home_score'] = pd.to_numeric(
        schedule.get('home_score'),
        errors='coerce',
    )
    schedule['away_score'] = pd.to_numeric(
        schedule.get('away_score'),
        errors='coerce',
    )
    schedule['actual_total_points'] = (
        schedule['home_score'] + schedule['away_score']
    )
    schedule['score_available'] = (
        schedule['home_score'].notna()
        & schedule['away_score'].notna()
    )
    schedule['past_final_buffer'] = (
        schedule['kickoff_utc'].notna()
        & schedule['kickoff_utc'].le(
            now - pd.Timedelta(hours=FINAL_BUFFER_HOURS)
        )
    )
    schedule['gradeable_final'] = (
        schedule['score_available']
        & schedule['past_final_buffer']
    )
    keep = [
        c for c in [
            'game_id',
            'season',
            'week',
            'gameday',
            'gametime',
            'kickoff_utc',
            'away_team',
            'home_team',
            'away_score',
            'home_score',
            'actual_total_points',
            'gradeable_final',
        ]
        if c in schedule.columns
    ]
    return schedule[keep].copy()


def _grade_decisions(
    decisions: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()

    work = decisions.copy()
    work['game_id'] = work['game_id'].astype(str)
    merged = work.merge(
        schedule,
        on='game_id',
        how='left',
        suffixes=('', '_result'),
    )
    merged['decision_total'] = pd.to_numeric(
        merged.get('closing_total'),
        errors='coerce',
    )
    merged['decision_probability_over'] = pd.to_numeric(
        merged.get('over_probability'),
        errors='coerce',
    )
    merged['actual_total_points'] = pd.to_numeric(
        merged.get('actual_total_points'),
        errors='coerce',
    )
    merged['gradeable_final'] = merged.get(
        'gradeable_final',
        pd.Series(False, index=merged.index),
    ).map(_bool)

    diff = merged['actual_total_points'] - merged['decision_total']
    merged['decision_market_result'] = np.select(
        [
            merged['gradeable_final'] & diff.gt(0),
            merged['gradeable_final'] & diff.lt(0),
            merged['gradeable_final'] & diff.eq(0),
        ],
        ['OVER', 'UNDER', 'PUSH'],
        default='PENDING',
    )
    merged['over_binary'] = np.where(
        merged['decision_market_result'].eq('OVER'),
        1.0,
        np.where(
            merged['decision_market_result'].eq('UNDER'),
            0.0,
            np.nan,
        ),
    )
    merged['brier_component'] = np.where(
        merged['over_binary'].notna()
        & merged['decision_probability_over'].notna(),
        (
            merged['decision_probability_over']
            - merged['over_binary']
        ) ** 2,
        np.nan,
    )
    return merged


def _grade_entries(
    entries: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    if entries.empty:
        return pd.DataFrame()

    work = entries.copy()
    work['game_id'] = work['game_id'].astype(str)
    merged = work.merge(
        schedule,
        on='game_id',
        how='left',
        suffixes=('', '_result'),
    )
    merged['entry_total'] = pd.to_numeric(
        merged.get('entry_total'),
        errors='coerce',
    )
    merged['entry_over_price'] = pd.to_numeric(
        merged.get('entry_over_price'),
        errors='coerce',
    )
    merged['actual_total_points'] = pd.to_numeric(
        merged.get('actual_total_points'),
        errors='coerce',
    )
    merged['gradeable_final'] = merged.get(
        'gradeable_final',
        pd.Series(False, index=merged.index),
    ).map(_bool)

    diff = merged['actual_total_points'] - merged['entry_total']
    merged['entry_result'] = np.select(
        [
            merged['gradeable_final'] & diff.gt(0),
            merged['gradeable_final'] & diff.lt(0),
            merged['gradeable_final'] & diff.eq(0),
        ],
        ['WIN', 'LOSS', 'PUSH'],
        default='PENDING',
    )

    units: list[float] = []
    flat_units: list[float] = []
    for _, row in merged.iterrows():
        result = str(row['entry_result'])
        price = row['entry_over_price']
        if result == 'WIN' and pd.notna(price):
            units.append(_american_profit(float(price)))
            flat_units.append(100.0 / 110.0)
        elif result == 'LOSS':
            units.append(-1.0)
            flat_units.append(-1.0)
        elif result == 'PUSH':
            units.append(0.0)
            flat_units.append(0.0)
        else:
            units.append(np.nan)
            flat_units.append(np.nan)
    merged['paper_units'] = units
    merged['flat_minus110_units'] = flat_units
    return merged


def _calibration_bins(
    graded_decisions: pd.DataFrame,
    edges: list[float],
) -> pd.DataFrame:
    if graded_decisions.empty:
        return pd.DataFrame()

    work = graded_decisions[
        graded_decisions['over_binary'].notna()
        & graded_decisions['decision_probability_over'].notna()
    ].copy()
    if work.empty:
        return pd.DataFrame()

    bins = sorted({float(v) for v in edges})
    if bins[0] > 0:
        bins.insert(0, 0.0)
    if bins[-1] < 1:
        bins.append(1.0)

    work['probability_bin'] = pd.cut(
        work['decision_probability_over'],
        bins=bins,
        include_lowest=True,
        right=True,
        duplicates='drop',
    )
    rows = []
    for interval, group in work.groupby(
        'probability_bin',
        observed=True,
    ):
        rows.append({
            'probability_bin': str(interval),
            'decisions': int(len(group)),
            'mean_predicted_over_probability': float(
                group['decision_probability_over'].mean()
            ),
            'observed_over_rate': float(group['over_binary'].mean()),
            'brier_score': float(group['brier_component'].mean()),
        })
    return pd.DataFrame(rows)


def _signal_stability(
    decisions: pd.DataFrame,
    monitors: pd.DataFrame,
) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()

    decision_work = decisions.copy()
    decision_work['game_id'] = decision_work['game_id'].astype(str)
    decision_work['official_snapshot_utc'] = pd.to_datetime(
        decision_work.get('snapshot_timestamp_utc'),
        utc=True,
        errors='coerce',
    )
    decision_work['kickoff_utc'] = pd.to_datetime(
        decision_work.get('kickoff_utc'),
        utc=True,
        errors='coerce',
    )

    if monitors.empty:
        monitors = pd.DataFrame(columns=[
            'game_id',
            'snapshot_timestamp_utc',
            'model_signal',
        ])
    monitor_work = monitors.copy()
    if not monitor_work.empty:
        monitor_work['game_id'] = monitor_work['game_id'].astype(str)
        monitor_work['snapshot_timestamp_utc'] = pd.to_datetime(
            monitor_work.get('snapshot_timestamp_utc'),
            utc=True,
            errors='coerce',
        )
        monitor_work['github_run_attempt'] = pd.to_numeric(
            monitor_work.get('github_run_attempt'),
            errors='coerce',
        ).fillna(1).astype(int)
        eligible = (
            monitor_work.get(
                'official_eligible',
                pd.Series(False, index=monitor_work.index),
            ).map(_bool)
            & monitor_work['github_run_attempt'].eq(1)
            & monitor_work.get(
                'record_kind',
                pd.Series('', index=monitor_work.index),
            ).astype(str).eq('monitor_snapshot')
        )
        monitor_work = monitor_work[eligible].copy()

    rows = []
    qualifying = {'QUALIFIES', 'STRONG'}
    for _, decision in decision_work.iterrows():
        game_id = str(decision['game_id'])
        official_signal = str(decision.get('model_signal') or 'NO PLAY')
        game_monitors = monitor_work[
            monitor_work.get(
                'game_id',
                pd.Series('', index=monitor_work.index),
            ).astype(str).eq(game_id)
        ].copy()
        if not game_monitors.empty:
            game_monitors = game_monitors[
                game_monitors['snapshot_timestamp_utc'].gt(
                    decision['official_snapshot_utc']
                )
                & game_monitors['snapshot_timestamp_utc'].lt(
                    decision['kickoff_utc']
                )
            ].sort_values('snapshot_timestamp_utc')

        signals = (
            game_monitors.get(
                'model_signal',
                pd.Series(dtype=str),
            ).fillna('NO PLAY').astype(str).tolist()
            if not game_monitors.empty else []
        )
        later_qualifier = any(signal in qualifying for signal in signals)
        later_no_play = any(signal == 'NO PLAY' for signal in signals)
        latest_signal = signals[-1] if signals else ''
        rows.append({
            'game_id': game_id,
            'week': decision.get('week'),
            'away_team': decision.get('away_team'),
            'home_team': decision.get('home_team'),
            'official_signal': official_signal,
            'official_snapshot_utc': decision.get('snapshot_timestamp_utc'),
            'monitor_count': int(len(game_monitors)),
            'latest_pre_kickoff_signal': latest_signal,
            'late_qualifier_after_freeze': bool(
                official_signal not in qualifying and later_qualifier
            ),
            'official_entry_faded': bool(
                official_signal in qualifying and later_no_play
            ),
            'tier_upgraded_later': bool(
                official_signal == 'QUALIFIES'
                and 'STRONG' in signals
            ),
            'tier_downgraded_later': bool(
                official_signal == 'STRONG'
                and 'QUALIFIES' in signals
            ),
        })
    return pd.DataFrame(rows)


def _summary(
    graded_decisions: pd.DataFrame,
    graded_entries: pd.DataFrame,
    entries_with_clv: pd.DataFrame,
    stability: pd.DataFrame,
) -> dict[str, Any]:
    final_decisions = graded_decisions[
        graded_decisions.get(
            'decision_market_result',
            pd.Series(dtype=str),
        ).isin(['OVER', 'UNDER', 'PUSH'])
    ].copy() if not graded_decisions.empty else pd.DataFrame()

    final_entries = graded_entries[
        graded_entries.get(
            'entry_result',
            pd.Series(dtype=str),
        ).isin(['WIN', 'LOSS', 'PUSH'])
    ].copy() if not graded_entries.empty else pd.DataFrame()

    wins = int(final_entries.get(
        'entry_result',
        pd.Series(dtype=str),
    ).eq('WIN').sum())
    losses = int(final_entries.get(
        'entry_result',
        pd.Series(dtype=str),
    ).eq('LOSS').sum())
    pushes = int(final_entries.get(
        'entry_result',
        pd.Series(dtype=str),
    ).eq('PUSH').sum())
    settled = wins + losses + pushes
    decisive = wins + losses
    paper_units = float(
        pd.to_numeric(
            final_entries.get('paper_units'),
            errors='coerce',
        ).sum()
    ) if settled else 0.0
    flat_units = float(
        pd.to_numeric(
            final_entries.get('flat_minus110_units'),
            errors='coerce',
        ).sum()
    ) if settled else 0.0

    hit_rate = wins / decisive if decisive else np.nan
    wilson_low, wilson_high = _wilson_interval(wins, decisive)
    p_value = (
        float(
            binomtest(
                wins,
                decisive,
                p=BREAKEVEN_MINUS110,
                alternative='greater',
            ).pvalue
        )
        if decisive else np.nan
    )

    calibrated = final_decisions[
        final_decisions.get(
            'over_binary',
            pd.Series(dtype=float),
        ).notna()
        & final_decisions.get(
            'decision_probability_over',
            pd.Series(dtype=float),
        ).notna()
    ].copy()
    brier = (
        float(calibrated['brier_component'].mean())
        if not calibrated.empty else np.nan
    )
    if not calibrated.empty:
        base_rate = float(calibrated['over_binary'].mean())
        base_brier = float(
            ((calibrated['over_binary'] - base_rate) ** 2).mean()
        )
    else:
        base_rate = np.nan
        base_brier = np.nan

    clv_frame = entries_with_clv.copy()
    if not clv_frame.empty:
        clv_frame['clv_points'] = pd.to_numeric(
            clv_frame.get('clv_points'),
            errors='coerce',
        )
        clv_frame = clv_frame[clv_frame['clv_points'].notna()].copy()
    clv_count = int(len(clv_frame))
    mean_clv = (
        float(clv_frame['clv_points'].mean())
        if clv_count else np.nan
    )
    median_clv = (
        float(clv_frame['clv_points'].median())
        if clv_count else np.nan
    )
    positive_clv_fraction = (
        float(clv_frame['clv_points'].gt(0).mean())
        if clv_count else np.nan
    )

    distinct_weeks = int(
        pd.to_numeric(
            final_entries.get('week'),
            errors='coerce',
        ).dropna().nunique()
    ) if not final_entries.empty else 0

    monitored = (
        int(stability['monitor_count'].gt(0).sum())
        if not stability.empty else 0
    )
    monitor_coverage = (
        monitored / len(stability)
        if len(stability) else np.nan
    )

    return {
        'graded_official_decisions': int(len(final_decisions)),
        'graded_entries': int(settled),
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'decisive_entries': decisive,
        'hit_rate': hit_rate,
        'hit_rate_wilson95_low': wilson_low,
        'hit_rate_wilson95_high': wilson_high,
        'one_sided_p_vs_minus110': p_value,
        'paper_units': paper_units,
        'captured_price_roi': (
            paper_units / settled if settled else np.nan
        ),
        'flat_minus110_units': flat_units,
        'flat_minus110_roi': (
            flat_units / settled if settled else np.nan
        ),
        'entries_with_clv': clv_count,
        'mean_clv_points': mean_clv,
        'median_clv_points': median_clv,
        'positive_clv_fraction': positive_clv_fraction,
        'distinct_entry_weeks': distinct_weeks,
        'calibration_decisions': int(len(calibrated)),
        'mean_predicted_over_probability': (
            float(calibrated['decision_probability_over'].mean())
            if not calibrated.empty else np.nan
        ),
        'observed_over_rate': (
            float(calibrated['over_binary'].mean())
            if not calibrated.empty else np.nan
        ),
        'brier_score': brier,
        'empirical_base_rate': base_rate,
        'empirical_base_rate_brier': base_brier,
        'monitored_decisions': monitored,
        'monitor_coverage_fraction': monitor_coverage,
        'late_qualifiers_after_freeze': (
            int(stability['late_qualifier_after_freeze'].sum())
            if not stability.empty else 0
        ),
        'official_entries_faded': (
            int(stability['official_entry_faded'].sum())
            if not stability.empty else 0
        ),
        'tier_upgrades_later': (
            int(stability['tier_upgraded_later'].sum())
            if not stability.empty else 0
        ),
        'tier_downgrades_later': (
            int(stability['tier_downgraded_later'].sum())
            if not stability.empty else 0
        ),
    }


def _status(
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> tuple[str, dict[str, bool]]:
    sample = config['sample_gate']
    review = config['performance_review']
    statuses = config['reporting']['statuses']

    def finite(value: Any) -> bool:
        try:
            return bool(np.isfinite(float(value)))
        except Exception:
            return False

    sample_checks = {
        'graded_entries': (
            metrics['graded_entries']
            >= int(sample['minimum_graded_entries'])
        ),
        'graded_official_decisions': (
            metrics['graded_official_decisions']
            >= int(sample['minimum_graded_official_decisions'])
        ),
        'entries_with_clv': (
            metrics['entries_with_clv']
            >= int(sample['minimum_entries_with_clv'])
        ),
        'distinct_entry_weeks': (
            metrics['distinct_entry_weeks']
            >= int(sample['minimum_distinct_entry_weeks'])
        ),
        'monitor_coverage': (
            finite(metrics['monitor_coverage_fraction'])
            and float(metrics['monitor_coverage_fraction'])
            >= float(sample['minimum_monitor_coverage_fraction'])
        ),
    }

    performance_checks = {
        'positive_captured_price_roi': (
            finite(metrics['captured_price_roi'])
            and (
                float(metrics['captured_price_roi']) > 0
                if review['require_positive_captured_price_roi']
                else True
            )
        ),
        'positive_mean_clv': (
            finite(metrics['mean_clv_points'])
            and float(metrics['mean_clv_points'])
            > float(review['require_mean_clv_points_greater_than'])
        ),
        'nonnegative_median_clv': (
            finite(metrics['median_clv_points'])
            and float(metrics['median_clv_points'])
            >= float(review['require_median_clv_points_at_least'])
        ),
        'positive_clv_fraction': (
            finite(metrics['positive_clv_fraction'])
            and float(metrics['positive_clv_fraction'])
            >= float(review['require_positive_clv_fraction_at_least'])
        ),
        'brier_at_most_threshold': (
            finite(metrics['brier_score'])
            and float(metrics['brier_score'])
            <= float(review['require_brier_score_at_most'])
        ),
        'brier_not_worse_than_empirical_base_rate': (
            finite(metrics['brier_score'])
            and finite(metrics['empirical_base_rate_brier'])
            and (
                float(metrics['brier_score'])
                <= float(metrics['empirical_base_rate_brier'])
                if review[
                    'require_brier_not_worse_than_empirical_base_rate'
                ]
                else True
            )
        ),
    }

    checks = {
        **{f'sample_{k}': v for k, v in sample_checks.items()},
        **{
            f'performance_{k}': v
            for k, v in performance_checks.items()
        },
    }
    if not all(sample_checks.values()):
        return str(statuses['insufficient']), checks
    if all(performance_checks.values()):
        return str(statuses['review']), checks
    return str(statuses['do_not_promote']), checks


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def evaluate(
    now: pd.Timestamp | None = None,
) -> dict[str, Any]:
    now = now or pd.Timestamp.now(tz='UTC')
    config = load_yaml('config/nfl_prospective_evaluation.yml')
    derived = rebuild_derived()
    decisions = derived['decisions']
    entries = derived['entries_with_clv']
    schedule = _load_schedule(now)

    graded_decisions = _grade_decisions(decisions, schedule)
    graded_entries = _grade_entries(entries, schedule)
    monitors = _read_immutable(MONITOR_DIR)
    stability = _signal_stability(decisions, monitors)
    calibration = _calibration_bins(
        graded_decisions,
        config['reporting']['calibration_bins'],
    )

    metrics = _summary(
        graded_decisions,
        graded_entries,
        entries,
        stability,
    )
    status, checks = _status(metrics, config)

    write_df(graded_decisions, DECISIONS_PATH)
    write_df(graded_entries, RESULTS_PATH)
    write_df(calibration, CALIBRATION_PATH)
    write_df(stability, STABILITY_PATH)
    write_df(pd.DataFrame([metrics]), SUMMARY_PATH)

    payload = {
        'generated_at_utc': now.isoformat(),
        'evaluation_id': config['evaluation']['id'],
        'protocol_id': config['evaluation']['protocol_id'],
        'status': status,
        'auto_validate_for_wagering': False,
        'human_review_required': True,
        'metrics': metrics,
        'checks': checks,
        'sample_gate': config['sample_gate'],
        'performance_review': config['performance_review'],
    }
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False),
        encoding='utf-8',
    )

    def fmt(value: Any, pct: bool = False) -> str:
        try:
            if not np.isfinite(float(value)):
                return '—'
            if pct:
                return f'{100 * float(value):.2f}%'
            return f'{float(value):.3f}'
        except Exception:
            return '—'

    failed = [name for name, passed in checks.items() if not passed]
    lines = [
        '# NFL 2026 Prospective Evaluation',
        '',
        f'**Status: {status}**',
        '',
        'This status never authorizes wagering automatically. '
        'REVIEW_ELIGIBLE means only that the pre-registered evidence '
        'thresholds have been met and the frozen system is ready for '
        'human review.',
        '',
        '## Prospective performance',
        '',
        f'- Official decisions graded: {metrics["graded_official_decisions"]}',
        f'- Official entries graded: {metrics["graded_entries"]}',
        (
            f'- Record: {metrics["wins"]}-{metrics["losses"]}'
            f'-{metrics["pushes"]} pushes'
        ),
        f'- Win rate: {fmt(metrics["hit_rate"], pct=True)}',
        (
            '- 95% Wilson interval: '
            f'{fmt(metrics["hit_rate_wilson95_low"], pct=True)} to '
            f'{fmt(metrics["hit_rate_wilson95_high"], pct=True)}'
        ),
        (
            '- Captured-price ROI: '
            f'{fmt(metrics["captured_price_roi"], pct=True)} '
            f'({fmt(metrics["paper_units"])} units)'
        ),
        (
            '- Flat -110 reference ROI: '
            f'{fmt(metrics["flat_minus110_roi"], pct=True)}'
        ),
        '',
        '## Closing-line value',
        '',
        f'- Entries with CLV: {metrics["entries_with_clv"]}',
        f'- Mean CLV: {fmt(metrics["mean_clv_points"])} points',
        f'- Median CLV: {fmt(metrics["median_clv_points"])} points',
        (
            '- Positive CLV fraction: '
            f'{fmt(metrics["positive_clv_fraction"], pct=True)}'
        ),
        '',
        '## Calibration',
        '',
        f'- Calibration decisions: {metrics["calibration_decisions"]}',
        f'- Brier score: {fmt(metrics["brier_score"])}',
        (
            '- Empirical base-rate Brier: '
            f'{fmt(metrics["empirical_base_rate_brier"])}'
        ),
        (
            '- Mean predicted P(OVER): '
            f'{fmt(metrics["mean_predicted_over_probability"], pct=True)}'
        ),
        (
            '- Observed OVER rate: '
            f'{fmt(metrics["observed_over_rate"], pct=True)}'
        ),
        '',
        '## Frozen-signal stability',
        '',
        (
            '- Monitor coverage: '
            f'{fmt(metrics["monitor_coverage_fraction"], pct=True)}'
        ),
        (
            '- Official NO PLAYs that qualified later: '
            f'{metrics["late_qualifiers_after_freeze"]}'
        ),
        (
            '- Official entries that faded to NO PLAY later: '
            f'{metrics["official_entries_faded"]}'
        ),
        f'- Later tier upgrades: {metrics["tier_upgrades_later"]}',
        f'- Later tier downgrades: {metrics["tier_downgrades_later"]}',
        '',
        '## Pre-registered gate',
        '',
        (
            '- Unmet checks: '
            + (', '.join(failed) if failed else 'none')
        ),
        '',
        'The frozen v1 protocol and its historical backtests are not '
        're-optimized from these results.',
    ]
    REPORT_PATH.write_text('\n'.join(lines), encoding='utf-8')
    print(
        f'NFL prospective evaluation: {status}; '
        f'{metrics["graded_entries"]} graded entries, '
        f'{metrics["graded_official_decisions"]} graded decisions.'
    )
    return payload


def main() -> None:
    evaluate()


if __name__ == '__main__':
    main()
