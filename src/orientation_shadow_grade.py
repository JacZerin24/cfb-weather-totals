from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cfbd_client import CFBDClient
from .prospective_ledger import CLOSE_DIR, _read_immutable, select_benchmark_closes
from .utils import ROOT, ensure_dir, load_yaml

PROTOCOL_PATH = ROOT / 'config/orientation_evaluation_protocol_2026.yml'
SHADOW_DIR = ROOT / 'outputs/orientation_shadow/2026'
EVAL_DIR = SHADOW_DIR / 'evaluation'
SNAPSHOT_GLOB = 'shadow_*.csv'


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def load_protocol() -> dict[str, Any]:
    protocol = load_yaml('config/orientation_evaluation_protocol_2026.yml')
    if int(protocol.get('season', 0)) != 2026:
        raise RuntimeError('Orientation evaluation protocol must remain scoped to 2026.')
    return protocol


def protocol_sha256() -> str:
    return hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def validate_protocol() -> None:
    protocol = load_protocol()
    production = load_yaml('config/prospective_protocol_2026.yml')
    frozen = protocol['official_entry_policy']
    live = production['official_entry_policy']

    checks = [
        ('eligible event', str(frozen['eligible_event_name']), str(live['eligible_event_name'])),
        ('minimum lead', int(frozen['minimum_lead_minutes']), int(live['minimum_lead_minutes'])),
        ('eligible crons', sorted(map(str, frozen['eligible_crons'])), sorted(map(str, live['eligible_crons']))),
    ]
    drift = [f'{name}: orientation={expected!r}, production={actual!r}' for name, expected, actual in checks if expected != actual]
    if drift:
        raise RuntimeError('Orientation evaluation entry timing drifted from the official prospective protocol: ' + '; '.join(drift))

    challenger = load_yaml('config/orientation_challenger_2026.yml')
    if str(challenger.get('version')) != str(protocol['challenger_version']):
        raise RuntimeError('Frozen orientation evaluation challenger version no longer matches challenger config.')
    if str(challenger.get('status')) != 'research_only_shadow':
        raise RuntimeError('Orientation challenger is no longer marked research-only shadow.')


def _read_shadow_snapshots() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in sorted(SHADOW_DIR.glob(SNAPSHOT_GLOB)):
        if path.name in {'latest.csv', 'manifest.csv'}:
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame['_shadow_file'] = str(path.relative_to(ROOT)).replace('\\', '/')
        frame['_shadow_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def select_official_shadow_entries(snapshots: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame()
    protocol = load_protocol()
    policy = protocol['official_entry_policy']
    frozen_at = pd.Timestamp(protocol['frozen_at_utc'])
    if frozen_at.tzinfo is None:
        frozen_at = frozen_at.tz_localize('UTC')
    else:
        frozen_at = frozen_at.tz_convert('UTC')

    work = snapshots.copy()
    work['captured_at_utc'] = pd.to_datetime(work.get('captured_at_utc'), utc=True, errors='coerce')
    work['start_date'] = pd.to_datetime(work.get('start_date'), utc=True, errors='coerce')
    work['github_run_attempt'] = pd.to_numeric(work.get('github_run_attempt'), errors='coerce').fillna(1).astype(int)
    work['game_id'] = pd.to_numeric(work.get('game_id'), errors='coerce')
    work['entry_lead_minutes'] = (work['start_date'] - work['captured_at_utc']).dt.total_seconds() / 60.0

    event_name = work.get('github_event_name', pd.Series('', index=work.index)).astype(str)
    event_schedule = work.get('github_event_schedule', pd.Series('', index=work.index)).astype(str)
    eligible = (
        event_name.eq(str(policy['eligible_event_name']))
        & event_schedule.isin({str(v) for v in policy.get('eligible_crons', [])})
    )
    if 'official_evaluation_eligible' in work.columns:
        eligible &= work['official_evaluation_eligible'].map(as_bool)

    orientation_ready = work.get('orientation_ready', pd.Series(False, index=work.index)).map(as_bool)
    if not orientation_ready.any():
        orientation_ready = (
            pd.to_numeric(work.get('crosswind_mph'), errors='coerce').notna()
            & pd.to_numeric(work.get('baseline_pred_market_residual'), errors='coerce').notna()
            & pd.to_numeric(work.get('challenger_pred_market_residual'), errors='coerce').notna()
        )

    selected = work[
        work['captured_at_utc'].notna()
        & work['start_date'].notna()
        & work['captured_at_utc'].ge(frozen_at)
        & work['entry_lead_minutes'].ge(float(policy['minimum_lead_minutes']))
        & eligible
        & orientation_ready
        & work.get('division_track', pd.Series('', index=work.index)).astype(str).str.upper().eq(str(protocol['scope']['division_track']).upper())
        & work.get('challenger_version', pd.Series('', index=work.index)).astype(str).eq(str(protocol['challenger_version']))
    ].copy()
    if selected.empty:
        return selected

    selected = selected.sort_values(['github_run_id', 'game_id', 'github_run_attempt', 'captured_at_utc'])
    selected = selected.drop_duplicates(['github_run_id', 'game_id'], keep='first')
    selected = selected.sort_values(['game_id', 'captured_at_utc'])
    selected = selected.groupby('game_id', as_index=False, sort=False).tail(1).copy()
    selected['official_orientation_entry'] = True
    return selected.sort_values(['start_date', 'game_id']).reset_index(drop=True)


def final_results(client: CFBDClient | None = None) -> pd.DataFrame:
    protocol = load_protocol()
    client = client or CFBDClient()
    records = client.get('/games', {'year': int(protocol['season']), 'seasonType': 'regular'})
    games = pd.json_normalize(records)
    if games.empty:
        return pd.DataFrame(columns=['game_id', 'actual_total_points'])
    games = games.rename(columns={
        'id': 'game_id',
        'homePoints': 'final_home_points',
        'awayPoints': 'final_away_points',
    })
    games['game_id'] = pd.to_numeric(games.get('game_id'), errors='coerce')
    games['final_home_points'] = pd.to_numeric(games.get('final_home_points'), errors='coerce')
    games['final_away_points'] = pd.to_numeric(games.get('final_away_points'), errors='coerce')
    completed = games.get('completed', pd.Series(False, index=games.index)).map(as_bool)
    games['actual_total_points'] = games['final_home_points'] + games['final_away_points']
    games = games[completed & games['actual_total_points'].notna()].copy()
    keep = ['game_id', 'final_home_points', 'final_away_points', 'actual_total_points']
    return games[keep].drop_duplicates('game_id')


def result_vs_total(actual: float, total: float) -> str:
    if not np.isfinite(actual) or not np.isfinite(total):
        return ''
    if actual < total:
        return 'under'
    if actual > total:
        return 'over'
    return 'push'


def profit_units(result: str, protocol: dict[str, Any]) -> float:
    econ = protocol['qualifier_economics']
    if result == 'under':
        return float(econ['win_profit_units'])
    if result == 'over':
        return float(econ['loss_profit_units'])
    if result == 'push':
        return float(econ['push_profit_units'])
    return np.nan


def bootstrap_mae_delta(differences: np.ndarray, protocol: dict[str, Any]) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 5:
        return np.nan, np.nan
    spec = protocol['primary_metric']
    reps = int(spec['bootstrap_reps'])
    rng = np.random.default_rng(int(spec['bootstrap_seed']))
    means = np.empty(reps, dtype=float)
    for start in range(0, reps, 1000):
        n = min(1000, reps - start)
        idx = rng.integers(0, len(values), size=(n, len(values)))
        means[start:start + n] = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def grade_entries(entries: pd.DataFrame, client: CFBDClient | None = None) -> pd.DataFrame:
    if entries.empty:
        return pd.DataFrame()
    protocol = load_protocol()
    finals = final_results(client)
    closes = select_benchmark_closes(_read_immutable(CLOSE_DIR))

    out = entries.copy()
    if not finals.empty:
        out = out.merge(finals, on='game_id', how='left')
    if not closes.empty:
        out = out.merge(closes, on='game_id', how='left')

    numeric = [
        'closing_total', 'baseline_pred_market_residual', 'challenger_pred_market_residual',
        'actual_total_points', 'benchmark_close_total', 'crosswind_mph', 'wind_mph',
    ]
    for col in numeric:
        out[col] = pd.to_numeric(out.get(col), errors='coerce')

    out = out[out['actual_total_points'].notna()].copy()
    if out.empty:
        return out

    out['entry_total'] = out['closing_total']
    out['observed_market_residual'] = out['actual_total_points'] - out['entry_total']
    out['baseline_projected_total'] = out['entry_total'] + out['baseline_pred_market_residual']
    out['challenger_projected_total'] = out['entry_total'] + out['challenger_pred_market_residual']
    out['baseline_projection_error'] = out['baseline_projected_total'] - out['actual_total_points']
    out['challenger_projection_error'] = out['challenger_projected_total'] - out['actual_total_points']
    out['baseline_abs_error'] = out['baseline_projection_error'].abs()
    out['challenger_abs_error'] = out['challenger_projection_error'].abs()
    out['challenger_minus_baseline_abs_error'] = out['challenger_abs_error'] - out['baseline_abs_error']
    out['baseline_sq_error'] = out['baseline_projection_error'] ** 2
    out['challenger_sq_error'] = out['challenger_projection_error'] ** 2
    out['result_vs_entry_total'] = [result_vs_total(a, t) for a, t in zip(out['actual_total_points'], out['entry_total'])]

    strength = {str(k): int(v) for k, v in protocol['classification_evaluation']['strength_order'].items()}
    out['baseline_strength'] = out['baseline_status'].astype(str).map(strength).fillna(0).astype(int)
    out['challenger_strength'] = out['challenger_status'].astype(str).map(strength).fillna(0).astype(int)
    out['status_migration'] = out['challenger_strength'] - out['baseline_strength']
    out['status_disagreement'] = out['challenger_status'].astype(str) != out['baseline_status'].astype(str)
    out['qualifier_disagreement'] = out['challenger_status'].eq('QUALIFIES') ^ out['baseline_status'].eq('QUALIFIES')
    out['migration_correct'] = np.nan
    toward_under = out['status_migration'] > 0
    away_under = out['status_migration'] < 0
    under = out['result_vs_entry_total'].eq('under')
    over = out['result_vs_entry_total'].eq('over')
    out.loc[toward_under & (under | over), 'migration_correct'] = under[toward_under & (under | over)]
    out.loc[away_under & (under | over), 'migration_correct'] = over[away_under & (under | over)]

    for prefix, status_col in [('baseline', 'baseline_status'), ('challenger', 'challenger_status')]:
        qualifier = out[status_col].astype(str).eq('QUALIFIES')
        out[f'{prefix}_qualifier'] = qualifier
        out[f'{prefix}_qualifier_result'] = np.where(qualifier, out['result_vs_entry_total'], '')
        out[f'{prefix}_qualifier_units_1u'] = [
            profit_units(result, protocol) if q else np.nan
            for result, q in zip(out['result_vs_entry_total'], qualifier)
        ]
        out[f'{prefix}_qualifier_clv_points'] = np.where(
            qualifier & out['benchmark_close_total'].notna(),
            out['entry_total'] - out['benchmark_close_total'],
            np.nan,
        )

    return out.sort_values(['start_date', 'game_id']).reset_index(drop=True)


def qualifier_summary(graded: pd.DataFrame, prefix: str) -> dict[str, Any]:
    q = graded[graded[f'{prefix}_qualifier']].copy() if not graded.empty else pd.DataFrame()
    if q.empty:
        return {
            f'{prefix}_qualifiers': 0,
            f'{prefix}_wins': 0,
            f'{prefix}_losses': 0,
            f'{prefix}_pushes': 0,
            f'{prefix}_hit_rate': np.nan,
            f'{prefix}_roi_per_1u': np.nan,
            f'{prefix}_avg_clv_points': np.nan,
        }
    wins = int(q['result_vs_entry_total'].eq('under').sum())
    losses = int(q['result_vs_entry_total'].eq('over').sum())
    pushes = int(q['result_vs_entry_total'].eq('push').sum())
    decisions = wins + losses
    units = pd.to_numeric(q[f'{prefix}_qualifier_units_1u'], errors='coerce').sum(min_count=1)
    return {
        f'{prefix}_qualifiers': len(q),
        f'{prefix}_wins': wins,
        f'{prefix}_losses': losses,
        f'{prefix}_pushes': pushes,
        f'{prefix}_hit_rate': wins / decisions if decisions else np.nan,
        f'{prefix}_roi_per_1u': float(units / len(q)) if len(q) and pd.notna(units) else np.nan,
        f'{prefix}_avg_clv_points': float(pd.to_numeric(q[f'{prefix}_qualifier_clv_points'], errors='coerce').mean()),
    }


def build_summary(graded: pd.DataFrame) -> pd.DataFrame:
    protocol = load_protocol()
    if graded.empty:
        row = {
            'protocol_version': protocol['protocol_version'],
            'protocol_sha256': protocol_sha256(),
            'challenger_version': protocol['challenger_version'],
            'graded_paired_games': 0,
        }
        row.update(qualifier_summary(graded, 'baseline'))
        row.update(qualifier_summary(graded, 'challenger'))
        return pd.DataFrame([row])

    base_mae = float(graded['baseline_abs_error'].mean())
    chall_mae = float(graded['challenger_abs_error'].mean())
    delta = chall_mae - base_mae
    ci_low, ci_high = bootstrap_mae_delta(graded['challenger_minus_baseline_abs_error'].to_numpy(float), protocol)
    migrations = graded[graded['status_migration'].ne(0) & graded['migration_correct'].notna()].copy()
    row = {
        'protocol_version': protocol['protocol_version'],
        'protocol_sha256': protocol_sha256(),
        'challenger_version': protocol['challenger_version'],
        'graded_paired_games': len(graded),
        'through_week': int(pd.to_numeric(graded['week'], errors='coerce').max()) if graded['week'].notna().any() else np.nan,
        'baseline_mae': base_mae,
        'challenger_mae': chall_mae,
        'challenger_minus_baseline_mae': delta,
        'mae_delta_bootstrap_ci_low': ci_low,
        'mae_delta_bootstrap_ci_high': ci_high,
        'baseline_rmse': float(math.sqrt(graded['baseline_sq_error'].mean())),
        'challenger_rmse': float(math.sqrt(graded['challenger_sq_error'].mean())),
        'baseline_mean_signed_error': float(graded['baseline_projection_error'].mean()),
        'challenger_mean_signed_error': float(graded['challenger_projection_error'].mean()),
        'status_disagreements': int(graded['status_disagreement'].sum()),
        'qualifier_disagreements': int(graded['qualifier_disagreement'].sum()),
        'scorable_status_migrations': len(migrations),
        'correct_status_migrations': int(migrations['migration_correct'].astype(bool).sum()) if len(migrations) else 0,
        'status_migration_accuracy': float(migrations['migration_correct'].astype(float).mean()) if len(migrations) else np.nan,
        'mean_crosswind_mph': float(graded['crosswind_mph'].mean()),
        'median_crosswind_mph': float(graded['crosswind_mph'].median()),
    }
    row.update(qualifier_summary(graded, 'baseline'))
    row.update(qualifier_summary(graded, 'challenger'))
    return pd.DataFrame([row])


def write_outputs(entries: pd.DataFrame, graded: pd.DataFrame, summary: pd.DataFrame) -> None:
    ensure_dir(EVAL_DIR)
    entries.to_csv(EVAL_DIR / 'official_entries.csv', index=False)
    graded.to_csv(EVAL_DIR / 'graded_games.csv', index=False)
    summary.to_csv(EVAL_DIR / 'summary.csv', index=False)

    disagreements = graded[graded.get('status_disagreement', pd.Series(False, index=graded.index)).fillna(False)].copy() if not graded.empty else pd.DataFrame()
    disagreements.to_csv(EVAL_DIR / 'status_disagreements.csv', index=False)

    if graded.empty:
        weekly = pd.DataFrame(columns=['week', 'games', 'baseline_mae', 'challenger_mae', 'challenger_minus_baseline_mae'])
    else:
        weekly = graded.groupby('week', as_index=False).agg(
            games=('game_id', 'size'),
            baseline_mae=('baseline_abs_error', 'mean'),
            challenger_mae=('challenger_abs_error', 'mean'),
        )
        weekly['challenger_minus_baseline_mae'] = weekly['challenger_mae'] - weekly['baseline_mae']
    weekly.to_csv(EVAL_DIR / 'weekly_summary.csv', index=False)

    s = summary.iloc[0].to_dict()
    n = int(s.get('graded_paired_games', 0) or 0)
    lines = [
        '# 2026 Baseline vs Orientation Shadow Evaluation',
        '',
        '**Research only. This report cannot alter the operational weekly board or official prospective ledger.**',
        '',
        f"Protocol: `{s.get('protocol_version', '')}`  ",
        f"Challenger: `{s.get('challenger_version', '')}`  ",
        f"Graded paired orientation-ready games: **{n}**",
        '',
    ]
    if n:
        lines += [
            '## Primary paired model metric',
            '',
            f"- Baseline MAE: **{s.get('baseline_mae', np.nan):.3f}** points",
            f"- Challenger MAE: **{s.get('challenger_mae', np.nan):.3f}** points",
            f"- Challenger minus baseline MAE: **{s.get('challenger_minus_baseline_mae', np.nan):+.3f}** points (negative is better)",
            f"- 95% paired bootstrap interval: **[{s.get('mae_delta_bootstrap_ci_low', np.nan):+.3f}, {s.get('mae_delta_bootstrap_ci_high', np.nan):+.3f}]**",
            '',
            '## Decision-support diagnostics',
            '',
            f"- Status disagreements: **{int(s.get('status_disagreements', 0) or 0)}**",
            f"- Qualifier disagreements: **{int(s.get('qualifier_disagreements', 0) or 0)}**",
            f"- Scorable status migrations: **{int(s.get('scorable_status_migrations', 0) or 0)}**",
            f"- Status-migration accuracy: **{s.get('status_migration_accuracy', np.nan):.3f}**",
            '',
            '## Qualifier economics (supporting evidence)',
            '',
            f"- Baseline: {int(s.get('baseline_wins', 0) or 0)}-{int(s.get('baseline_losses', 0) or 0)}-{int(s.get('baseline_pushes', 0) or 0)}, ROI {s.get('baseline_roi_per_1u', np.nan):+.3f}, avg CLV {s.get('baseline_avg_clv_points', np.nan):+.3f}",
            f"- Challenger: {int(s.get('challenger_wins', 0) or 0)}-{int(s.get('challenger_losses', 0) or 0)}-{int(s.get('challenger_pushes', 0) or 0)}, ROI {s.get('challenger_roi_per_1u', np.nan):+.3f}, avg CLV {s.get('challenger_avg_clv_points', np.nan):+.3f}",
            '',
        ]
    else:
        lines += ['No completed official orientation-shadow entries are available yet.', '']
    lines += [
        '## Interpretation guardrail',
        '',
        'The paired MAE comparison is primary. ROI, hit rate, CLV, and individual disagreement games are secondary evidence and cannot alone justify a model promotion. Formal promotion decisions occur only at the predeclared review points and require a separate versioned decision.',
    ]
    (EVAL_DIR / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')


def build(client: CFBDClient | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_protocol()
    snapshots = _read_shadow_snapshots()
    entries = select_official_shadow_entries(snapshots)
    graded = grade_entries(entries, client)
    summary = build_summary(graded)
    write_outputs(entries, graded, summary)
    print(summary.to_string(index=False))
    return entries, graded, summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Frozen 2026 baseline-vs-orientation prospective evaluator.')
    parser.add_argument('command', nargs='?', choices=['validate-protocol', 'build'], default='build')
    args = parser.parse_args()
    if args.command == 'validate-protocol':
        validate_protocol()
        print(f'Orientation evaluation protocol valid: {protocol_sha256()}')
    else:
        build()


if __name__ == '__main__':
    main()
