from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .stadium_wind_orientation_research import build_research_frame
from .utils import ensure_dir, write_df

PRIMARY_LOW = 10.0
PRIMARY_HIGH = 15.0
PARALLEL_MAX = 30.0
CROSS_MIN = 60.0
REPS = 10_000
SEED = 20260831


def as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({'true', '1', 'yes', 'y'})


def angle_to_axis(wind_direction: np.ndarray, axis: np.ndarray) -> np.ndarray:
    wind_axis = np.mod(wind_direction, 180.0)
    field_axis = np.mod(axis, 180.0)
    delta = np.abs(wind_axis - field_axis)
    return np.minimum(delta, 180.0 - delta)


def controls(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    numeric = ['wind_mph', 'temperature_f', 'humidity', 'precipitation', 'dewpoint_f', 'pressure', 'closing_total']
    for col in numeric:
        out[col] = pd.to_numeric(out.get(col), errors='coerce')
    out['neutral_site_num'] = as_bool(out.get('neutral_site', pd.Series(False, index=out.index))).astype(float)
    out['venue_fe'] = pd.to_numeric(out['venue_id'], errors='coerce').astype('Int64').astype(str)
    out['season_fe'] = pd.to_numeric(out['season'], errors='coerce').astype('Int64').astype(str)
    out['provider_fe'] = out.get('line_provider', pd.Series('missing', index=out.index)).astype(str).fillna('missing')
    theta = np.deg2rad(pd.to_numeric(out['wind_direction_degrees'], errors='coerce').to_numpy(float))
    out['wind_dir_sin'] = np.sin(theta)
    out['wind_dir_cos'] = np.cos(theta)
    out['wind_dir_sin2'] = np.sin(2 * theta)
    out['wind_dir_cos2'] = np.cos(2 * theta)
    return out.dropna(subset=['pressure']).reset_index(drop=True)


def design(frame: pd.DataFrame, include_cross: bool = False, direction_harmonics: bool = False) -> tuple[np.ndarray, list[str]]:
    numeric = ['wind_mph', 'temperature_f', 'humidity', 'precipitation', 'dewpoint_f', 'pressure', 'closing_total', 'neutral_site_num']
    if direction_harmonics:
        numeric += ['wind_dir_sin', 'wind_dir_cos', 'wind_dir_sin2', 'wind_dir_cos2']
    x = frame[numeric].astype(float).copy()
    if include_cross:
        x.insert(0, 'is_cross', frame['is_cross'].astype(float).to_numpy())
    dummies = pd.get_dummies(frame[['venue_fe', 'season_fe', 'provider_fe']], drop_first=True, dtype=float)
    x = pd.concat([x.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    x.insert(0, 'intercept', 1.0)
    return x.to_numpy(float), list(x.columns)


def ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return beta, y - x @ beta


def cluster_se(x: np.ndarray, residual: np.ndarray, cluster: np.ndarray) -> np.ndarray:
    xtx_inv = np.linalg.pinv(x.T @ x)
    meat = np.zeros((x.shape[1], x.shape[1]))
    groups = pd.unique(cluster)
    for group in groups:
        idx = cluster == group
        score = x[idx].T @ residual[idx]
        meat += np.outer(score, score)
    n, k = x.shape
    g = len(groups)
    correction = (g / (g - 1)) * ((n - 1) / max(1, n - k)) if g > 1 else 1.0
    cov = correction * xtx_inv @ meat @ xtx_inv
    return np.sqrt(np.maximum(np.diag(cov), 0))


def normal_p(z: float) -> float:
    return math.erfc(abs(float(z)) / math.sqrt(2.0))


def fixed_effect_cross(frame: pd.DataFrame, direction_harmonics: bool = False) -> dict[str, float]:
    angle = angle_to_axis(frame['wind_direction_degrees'].to_numpy(float), frame['field_axis_deg'].to_numpy(float))
    cp = frame[(angle <= PARALLEL_MAX) | (angle >= CROSS_MIN)].copy().reset_index(drop=True)
    angle = angle_to_axis(cp['wind_direction_degrees'].to_numpy(float), cp['field_axis_deg'].to_numpy(float))
    cp['is_cross'] = (angle >= CROSS_MIN).astype(float)
    x, names = design(cp, include_cross=True, direction_harmonics=direction_harmonics)
    y = cp['market_residual'].to_numpy(float)
    beta, residual = ols(x, y)
    se = cluster_se(x, residual, cp['venue_fe'].to_numpy())
    idx = names.index('is_cross')
    z = beta[idx] / se[idx] if se[idx] else np.nan
    return {
        'fe_cross_minus_parallel_points': float(beta[idx]),
        'fe_cluster_se': float(se[idx]),
        'fe_normal_p_two_sided': float(normal_p(z)) if np.isfinite(z) else np.nan,
        'fe_ci_low': float(beta[idx] - 1.96 * se[idx]),
        'fe_ci_high': float(beta[idx] + 1.96 * se[idx]),
    }


def nuisance_residual(frame: pd.DataFrame) -> np.ndarray:
    x, _ = design(frame)
    return ols(x, frame['market_residual'].to_numpy(float))[1]


def observed_stat(frame: pd.DataFrame, residual: np.ndarray, axes: np.ndarray | None = None) -> tuple[float, int, int]:
    if axes is None:
        axes = frame['field_axis_deg'].to_numpy(float)
    angle = angle_to_axis(frame['wind_direction_degrees'].to_numpy(float), axes)
    cross = angle >= CROSS_MIN
    parallel = angle <= PARALLEL_MAX
    stat = residual[cross].mean() - residual[parallel].mean()
    return float(stat), int(cross.sum()), int(parallel.sum())


def axis_permutation(frame: pd.DataFrame, residual: np.ndarray, mode: str, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    venues = np.array(sorted(frame['venue_id'].dropna().unique()))
    venue_map = {v: i for i, v in enumerate(venues)}
    game_idx = np.array([venue_map[v] for v in frame['venue_id'].to_numpy()])
    actual = np.array([frame.loc[frame['venue_id'].eq(v), 'field_axis_deg'].iloc[0] for v in venues])
    wind_axis = np.mod(frame['wind_direction_degrees'].to_numpy(float), 180.0)[None, :]
    y = residual[None, :]
    output = []
    for start in range(0, reps, 500):
        n = min(500, reps - start)
        if mode == 'uniform':
            axes = rng.uniform(0, 180, size=(n, len(venues)))
        elif mode == 'shuffle':
            axes = np.vstack([rng.permutation(actual) for _ in range(n)])
        else:
            raise ValueError(mode)
        game_axes = axes[:, game_idx]
        delta = np.abs(wind_axis - game_axes)
        angle = np.minimum(delta, 180 - delta)
        cross = angle >= CROSS_MIN
        parallel = angle <= PARALLEL_MAX
        output.append((cross * y).sum(1) / cross.sum(1) - (parallel * y).sum(1) / parallel.sum(1))
    return np.concatenate(output)


def stratified_angle_permutation(frame: pd.DataFrame, residual: np.ndarray, strata: list[str], reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    observed_angle = angle_to_axis(frame['wind_direction_degrees'].to_numpy(float), frame['field_axis_deg'].to_numpy(float))
    groups = [np.asarray(idx, dtype=int) for idx in frame.groupby(strata, sort=False).indices.values()]
    output = np.empty(reps)
    for rep in range(reps):
        angle = observed_angle.copy()
        for idx in groups:
            if len(idx) > 1:
                angle[idx] = rng.permutation(angle[idx])
        cross = angle >= CROSS_MIN
        parallel = angle <= PARALLEL_MAX
        output[rep] = residual[cross].mean() - residual[parallel].mean()
    return output


def empirical_p(null: np.ndarray, observed: float) -> tuple[float, float]:
    left = (np.sum(null <= observed) + 1) / (len(null) + 1)
    right = (np.sum(null >= observed) + 1) / (len(null) + 1)
    return float(left), float(min(1.0, 2 * min(left, right)))


def jitter(frame: pd.DataFrame, residual: np.ndarray, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    venues = np.array(sorted(frame['venue_id'].dropna().unique()))
    venue_map = {v: i for i, v in enumerate(venues)}
    game_idx = np.array([venue_map[v] for v in frame['venue_id'].to_numpy()])
    axis = np.array([frame.loc[frame['venue_id'].eq(v), 'field_axis_deg'].iloc[0] for v in venues])
    uncertainty = np.array([
        pd.to_numeric(frame.loc[frame['venue_id'].eq(v), 'axis_uncertainty_deg'], errors='coerce').dropna().iloc[0]
        if frame.loc[frame['venue_id'].eq(v), 'axis_uncertainty_deg'].notna().any() else 2.0
        for v in venues
    ])
    wind_axis = np.mod(frame['wind_direction_degrees'].to_numpy(float), 180.0)[None, :]
    y = residual[None, :]
    output = []
    for start in range(0, reps, 500):
        n = min(500, reps - start)
        axes = (axis[None, :] + rng.normal(0, uncertainty, size=(n, len(venues)))) % 180
        game_axes = axes[:, game_idx]
        delta = np.abs(wind_axis - game_axes)
        angle = np.minimum(delta, 180 - delta)
        cross = angle >= CROSS_MIN
        parallel = angle <= PARALLEL_MAX
        output.append((cross * y).sum(1) / cross.sum(1) - (parallel * y).sum(1) / parallel.sum(1))
    return np.concatenate(output)


def sign_test(negative: int, n: int) -> float:
    return float(sum(math.comb(n, k) for k in range(negative, n + 1)) / 2**n) if n else np.nan


def within_venue(frame: pd.DataFrame) -> pd.DataFrame:
    angle = angle_to_axis(frame['wind_direction_degrees'].to_numpy(float), frame['field_axis_deg'].to_numpy(float))
    cp = frame.copy()
    cp['alignment_test'] = np.select([angle <= PARALLEL_MAX, angle >= CROSS_MIN], ['parallel', 'cross'], default='oblique')
    cp = cp[cp['alignment_test'].isin(['parallel', 'cross'])]
    grouped = cp.groupby(['venue_id', 'alignment_test']).agg(n=('game_id', 'size'), mean_residual=('market_residual', 'mean')).reset_index()
    counts = grouped.pivot(index='venue_id', columns='alignment_test', values='n')
    means = grouped.pivot(index='venue_id', columns='alignment_test', values='mean_residual')
    rows = []
    for minimum in [1, 3, 5]:
        eligible = counts.dropna(subset=['cross', 'parallel'])
        eligible = eligible[(eligible['cross'] >= minimum) & (eligible['parallel'] >= minimum)]
        diff = (means.loc[eligible.index, 'cross'] - means.loc[eligible.index, 'parallel']).dropna()
        negative = int((diff < 0).sum())
        rows.append({
            'minimum_games_each_direction_per_venue': minimum,
            'venues': len(diff),
            'venues_cross_lower': negative,
            'fraction_cross_lower': negative / len(diff) if len(diff) else np.nan,
            'mean_venue_difference_points': float(diff.mean()) if len(diff) else np.nan,
            'median_venue_difference_points': float(diff.median()) if len(diff) else np.nan,
            'exact_sign_test_p_one_sided': sign_test(negative, len(diff)),
        })
    return pd.DataFrame(rows)


def run_bin(common: pd.DataFrame, label: str, low: float, high: float, inclusive_low: bool, seed: int) -> dict:
    if inclusive_low:
        sub = common[common['wind_mph'].between(low, high, inclusive='both')]
    else:
        sub = common[(common['wind_mph'] > low) & (common['wind_mph'] <= high)]
    sub = controls(sub)
    residual = nuisance_residual(sub)
    angle = angle_to_axis(sub['wind_direction_degrees'].to_numpy(float), sub['field_axis_deg'].to_numpy(float))
    cross = angle >= CROSS_MIN
    parallel = angle <= PARALLEL_MAX
    raw = sub.loc[cross, 'market_residual'].mean() - sub.loc[parallel, 'market_residual'].mean()
    adjusted = residual[cross].mean() - residual[parallel].mean()
    fe = fixed_effect_cross(sub)
    null = axis_permutation(sub, residual, 'uniform', REPS // 2, seed)
    p_one, p_two = empirical_p(null, adjusted)
    return {
        'wind_bin': label, 'games': len(sub), 'cross_games': int(cross.sum()), 'parallel_games': int(parallel.sum()),
        'raw_cross_minus_parallel_points': float(raw), 'adjusted_cross_minus_parallel_points': float(adjusted),
        **fe, 'uniform_axis_perm_p_one_sided_cross_lower': p_one, 'uniform_axis_perm_p_two_sided': p_two,
    }


def period(common: pd.DataFrame, start: int, end: int, seed: int) -> dict:
    sub = common[(common['season'] >= start) & (common['season'] <= end) & (common['wind_mph'] > PRIMARY_LOW) & (common['wind_mph'] <= PRIMARY_HIGH)]
    sub = controls(sub)
    residual = nuisance_residual(sub)
    stat, cross_n, parallel_n = observed_stat(sub, residual)
    fe = fixed_effect_cross(sub)
    null = axis_permutation(sub, residual, 'uniform', REPS // 2, seed)
    p_one, p_two = empirical_p(null, stat)
    return {'period': f'{start}-{end}', 'games': len(sub), 'cross_games': cross_n, 'parallel_games': parallel_n,
            'adjusted_cross_minus_parallel_points': stat, **fe,
            'uniform_axis_perm_p_one_sided_cross_lower': p_one, 'uniform_axis_perm_p_two_sided': p_two}


def main() -> None:
    merged, _ = build_research_frame()
    common = merged[(merged['division_track'].eq('FBS')) & merged['outdoor']].copy()
    common = common.dropna(subset=['field_axis_deg', 'wind_mph', 'wind_direction_degrees']).copy()

    bins = pd.DataFrame([
        run_bin(common, '0-5', 0, 5, True, SEED + 1),
        run_bin(common, '5-10', 5, 10, False, SEED + 2),
        run_bin(common, '10-15', 10, 15, False, SEED + 3),
        run_bin(common, '15-20', 15, 20, False, SEED + 4),
        run_bin(common, '20+', 20, np.inf, False, SEED + 5),
    ])

    primary = controls(common[(common['wind_mph'] > PRIMARY_LOW) & (common['wind_mph'] <= PRIMARY_HIGH)])
    residual = nuisance_residual(primary)
    observed, cross_n, parallel_n = observed_stat(primary, residual)
    uniform = axis_permutation(primary, residual, 'uniform', REPS, SEED + 17)
    shuffled = axis_permutation(primary, residual, 'shuffle', REPS, SEED + 29)
    venue_perm = stratified_angle_permutation(primary, residual, ['venue_id'], REPS, SEED + 53)
    venue_season_perm = stratified_angle_permutation(primary, residual, ['venue_id', 'season'], REPS, SEED + 59)
    jittered = jitter(primary, residual, REPS, SEED + 41)
    standard_fe = fixed_effect_cross(primary)
    direction_fe = fixed_effect_cross(primary, direction_harmonics=True)

    u1, u2 = empirical_p(uniform, observed)
    s1, s2 = empirical_p(shuffled, observed)
    v1, v2 = empirical_p(venue_perm, observed)
    vs1, vs2 = empirical_p(venue_season_perm, observed)
    primary_row = pd.DataFrame([{
        'games': len(primary), 'cross_games': cross_n, 'parallel_games': parallel_n,
        'adjusted_cross_minus_parallel_points': observed,
        **standard_fe,
        'uniform_axis_perm_p_one_sided': u1, 'uniform_axis_perm_p_two_sided': u2,
        'shuffle_axis_perm_p_one_sided': s1, 'shuffle_axis_perm_p_two_sided': s2,
        'within_venue_angle_perm_p_one_sided': v1, 'within_venue_angle_perm_p_two_sided': v2,
        'within_venue_season_angle_perm_p_one_sided': vs1, 'within_venue_season_angle_perm_p_two_sided': vs2,
        'direction_harmonic_fe_cross_minus_parallel_points': direction_fe['fe_cross_minus_parallel_points'],
        'direction_harmonic_fe_p_two_sided': direction_fe['fe_normal_p_two_sided'],
        'jitter_median_points': float(np.median(jittered)),
        'jitter_q025': float(np.quantile(jittered, .025)), 'jitter_q975': float(np.quantile(jittered, .975)),
        'jitter_fraction_negative': float((jittered < 0).mean()),
    }])

    periods = pd.DataFrame([period(common, 2014, 2020, SEED + 101), period(common, 2021, 2025, SEED + 102), period(common, 2023, 2025, SEED + 103)])
    within = within_venue(primary)

    primary_for_loo = primary.copy()
    primary_for_loo['_residual'] = residual
    loo = []
    for season in sorted(primary_for_loo['season'].unique()):
        sub = primary_for_loo[primary_for_loo['season'] != season]
        stat, c, p = observed_stat(sub, sub['_residual'].to_numpy(float))
        loo.append({'season': int(season), 'adjusted_cross_minus_parallel_points': stat, 'cross_games': c, 'parallel_games': p})

    quality = []
    for season, group in common.groupby('season'):
        direction = pd.to_numeric(group['wind_direction_degrees'], errors='coerce').dropna().to_numpy(float)
        quality.append({'season': int(season), 'rows_with_direction': len(direction), 'unique_direction_values': int(pd.Series(direction).nunique()),
                        'fraction_exact_multiple_10deg': float(np.isclose(np.mod(direction, 10), 0, atol=1e-9).mean())})

    out = ensure_dir('outputs/orientation_research')
    write_df(bins, out / 'falsification_wind_bins.csv')
    write_df(primary_row, out / 'falsification_primary_permutation.csv')
    write_df(periods, out / 'falsification_period_sensitivity.csv')
    write_df(within, out / 'falsification_within_venue.csv')
    write_df(pd.DataFrame(loo), out / 'falsification_leave_one_season_out.csv')
    write_df(pd.DataFrame(quality), out / 'falsification_direction_quality.csv')
    print('Wrote research-only stadium orientation falsification outputs.')


if __name__ == '__main__':
    main()
