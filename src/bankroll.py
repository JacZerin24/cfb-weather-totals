from __future__ import annotations

from .utils import load_yaml


def suggested_units(edge_points: float) -> float:
    cfg = load_yaml('config/bankroll.yml')['sizing']
    edge = abs(float(edge_points))
    if edge < cfg['no_play_edge_points']:
        return 0.0
    if edge < cfg['small_play_edge_points']:
        return cfg['small_play_units']
    if edge < cfg['normal_play_edge_points']:
        return cfg['normal_play_units']
    return min(cfg['strong_play_units'], load_yaml('config/bankroll.yml')['bankroll']['max_single_play_units'])


def unit_dollars() -> float:
    cfg = load_yaml('config/bankroll.yml')['bankroll']
    return float(cfg['starting_bankroll']) * float(cfg['unit_percent']) / 100.0
