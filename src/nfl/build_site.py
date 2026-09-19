from __future__ import annotations

import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from ..utils import ROOT, ensure_dir


CT = ZoneInfo('America/Chicago')
LIVE_BOARD = ROOT / 'outputs/nfl/live/weekly_board.csv'
PROSPECTIVE_ROOT = ROOT / 'outputs/nfl/prospective/2026'
DOCS_NFL = ROOT / 'docs/nfl'


STATUS_CLASS = {
    'STRONG': 'strong',
    'QUALIFIES': 'qualifies',
    'NO PLAY': 'no-play',
    'EARLY LOOK': 'early',
    'PAST WINDOW': 'past',
    'WAITING': 'waiting',
    'NO LINE': 'no-line',
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _text(value: Any, fallback: str = '—') -> str:
    try:
        if value is None or pd.isna(value):
            return fallback
    except Exception:
        pass
    value = str(value).strip()
    return value if value and value.lower() != 'nan' else fallback


def _num(value: Any, decimals: int = 1, fallback: str = '—') -> str:
    try:
        if value is None or pd.isna(value):
            return fallback
        return f'{float(value):.{decimals}f}'
    except Exception:
        return fallback


def _pct(value: Any, fallback: str = '—') -> str:
    try:
        if value is None or pd.isna(value):
            return fallback
        return f'{100 * float(value):.1f}%'
    except Exception:
        return fallback


def _price(value: Any, fallback: str = '—') -> str:
    try:
        if value is None or pd.isna(value):
            return fallback
        number = int(round(float(value)))
        return f'+{number}' if number > 0 else str(number)
    except Exception:
        return fallback


def _kickoff(value: Any) -> str:
    try:
        ts = pd.to_datetime(value, utc=True)
        return ts.tz_convert(CT).strftime('%a %b %-d · %-I:%M %p CT')
    except Exception:
        return 'Kickoff TBD'


def _generated(board: pd.DataFrame) -> str:
    if not board.empty and 'generated_at_utc' in board.columns:
        values = board['generated_at_utc'].dropna()
        if not values.empty:
            try:
                ts = pd.to_datetime(values.iloc[0], utc=True)
                return ts.tz_convert(CT).strftime('%b %-d, %Y · %-I:%M %p CT')
            except Exception:
                pass
    return datetime.now(timezone.utc).astimezone(CT).strftime(
        '%b %-d, %Y · %-I:%M %p CT'
    )


def _roof_weather(row: pd.Series) -> str:
    roof = _text(row.get('decision_roof'), 'unknown').lower()
    if roof == 'dome':
        return 'Dome · weather withheld by protocol'
    if roof == 'retractable':
        return 'Retractable roof · weather withheld by protocol'
    if roof != 'outdoors':
        return 'Venue weather unavailable'

    temp = _num(row.get('forecast_temp_24h'), 0)
    wind = _num(row.get('forecast_wind_24h'), 0)
    if temp == '—' and wind == '—':
        return '24h JMA forecast unavailable'
    return f'24h JMA · {temp}°F · wind {wind} mph'


def _market_line(row: pd.Series) -> str:
    total = _num(row.get('closing_total'))
    if total == '—':
        return 'No current total'
    price = _price(row.get('consensus_over_price'))
    book = _text(row.get('consensus_over_sportsbook'), '')
    suffix = f' · {price}' if price != '—' else ''
    if book:
        suffix += f' · {book}'
    return f'OVER {total}{suffix}'


def _best_line(row: pd.Series) -> str:
    total = _num(row.get('closing_total'))
    price = _price(row.get('best_over_price_same_line'))
    book = _text(row.get('best_over_sportsbook_same_line'), '')
    if total == '—' or price == '—':
        return 'Best same-line price unavailable'
    return f'Best at {total}: {price}' + (f' · {book}' if book else '')


def _model_line(row: pd.Series) -> str:
    edge = _num(row.get('pred_market_residual'))
    prob = _pct(row.get('over_probability'))
    projected = _num(row.get('model_projected_total'))
    if edge == '—':
        return 'Model not scored'
    return f'+{edge} pts · P(OVER) {prob} · model total {projected}'


def _official_map(decisions: pd.DataFrame) -> dict[str, pd.Series]:
    if decisions.empty or 'game_id' not in decisions.columns:
        return {}
    return {
        str(row['game_id']): row
        for _, row in decisions.iterrows()
    }


def _entry_map(entries: pd.DataFrame) -> dict[str, pd.Series]:
    if entries.empty or 'game_id' not in entries.columns:
        return {}
    return {
        str(row['game_id']): row
        for _, row in entries.iterrows()
    }


def _card(
    row: pd.Series,
    decision: pd.Series | None,
    entry: pd.Series | None,
) -> str:
    status = _text(row.get('status'), 'NO PLAY')
    status_css = STATUS_CLASS.get(status, 'no-play')
    signal = _text(row.get('model_signal'), 'NO PLAY')
    state = _text(row.get('decision_state'), '')
    matchup = (
        f'{escape(_text(row.get("away_team"), "Away"))} '
        f'<span>@</span> '
        f'{escape(_text(row.get("home_team"), "Home"))}'
    )
    venue = escape(_text(row.get('stadium'), 'Venue TBD'))

    if decision is None:
        official = (
            '<div class="official pending">'
            '<span class="official-label">Official 24h decision</span>'
            '<strong>Not frozen yet</strong>'
            '</div>'
        )
    else:
        frozen_signal = _text(decision.get('model_signal'), 'NO PLAY')
        frozen_css = STATUS_CLASS.get(frozen_signal, 'no-play')
        frozen_at = _text(decision.get('snapshot_timestamp_utc'), '')
        try:
            frozen_time = pd.to_datetime(
                frozen_at, utc=True
            ).tz_convert(CT).strftime('%b %-d · %-I:%M %p CT')
        except Exception:
            frozen_time = 'time unavailable'
        official = (
            f'<div class="official {frozen_css}">'
            '<span class="official-label">Official 24h decision</span>'
            f'<strong>{escape(frozen_signal)}</strong>'
            f'<small>Frozen {escape(frozen_time)}</small>'
            '</div>'
        )

    entry_note = ''
    if entry is not None:
        tier = escape(_text(entry.get('entry_tier'), 'QUALIFIES'))
        entry_total = _num(entry.get('entry_total'))
        entry_price = _price(entry.get('entry_over_price'))
        entry_book = escape(_text(entry.get('entry_over_sportsbook'), ''))
        clv = _num(entry.get('clv_points'))
        clv_html = ''
        if clv != '—':
            clv_html = f'<span>CLV {clv} pts</span>'
        entry_note = (
            '<div class="entry-strip">'
            f'<strong>{tier} PAPER ENTRY</strong>'
            f'<span>OVER {entry_total} {entry_price}</span>'
            + (f'<span>{entry_book}</span>' if entry_book else '')
            + clv_html
            + '</div>'
        )

    hours = _num(row.get('hours_to_kickoff'))
    breadth = _text(row.get('sportsbooks_in_snapshot'), '—')
    market_range = _num(row.get('market_total_range'))

    return f'''
      <article class="game-card" data-status="{escape(status)}" data-signal="{escape(signal)}" data-state="{escape(state)}">
        <div class="card-top">
          <span class="badge {status_css}">{escape(status)}</span>
          <span class="time-to-kick">{hours}h to kickoff</span>
        </div>
        <h3>{matchup}</h3>
        <div class="kickoff">{escape(_kickoff(row.get('kickoff_utc')))} · {venue}</div>
        {entry_note}
        <div class="pick">{escape(_market_line(row))}</div>
        <div class="best-price">{escape(_best_line(row))}</div>
        <div class="grid">
          <div><span>Frozen model</span><strong>{escape(_model_line(row))}</strong></div>
          <div><span>Weather input</span><strong>{escape(_roof_weather(row))}</strong></div>
          <div><span>Market breadth</span><strong>{escape(breadth)} books · range {market_range} pts</strong></div>
          <div><span>Decision state</span><strong>{escape(state.replace('_', ' '))}</strong></div>
        </div>
        {official}
      </article>
    '''


def _decision_rows(decisions: pd.DataFrame) -> str:
    if decisions.empty:
        return (
            '<tr><td colspan="6" class="empty-cell">'
            'No official 24-hour decisions have been frozen yet.'
            '</td></tr>'
        )

    rows: list[str] = []
    ordered = decisions.copy()
    if 'kickoff_utc' in ordered.columns:
        ordered['_kick'] = pd.to_datetime(
            ordered['kickoff_utc'], utc=True, errors='coerce'
        )
        ordered = ordered.sort_values('_kick')

    for _, row in ordered.iterrows():
        signal = _text(row.get('model_signal'), 'NO PLAY')
        css = STATUS_CLASS.get(signal, 'no-play')
        rows.append(
            '<tr>'
            f'<td><span class="badge {css}">{escape(signal)}</span></td>'
            f'<td><strong>{escape(_text(row.get("away_team"), ""))} @ '
            f'{escape(_text(row.get("home_team"), ""))}</strong><br>'
            f'<small>{escape(_kickoff(row.get("kickoff_utc")))}</small></td>'
            f'<td>{escape(_market_line(row))}</td>'
            f'<td>{escape(_model_line(row))}</td>'
            f'<td>{escape(_roof_weather(row))}</td>'
            f'<td>{escape(_text(row.get("snapshot_timestamp_utc"), "—"))}</td>'
            '</tr>'
        )
    return ''.join(rows)


def _load_evaluation() -> dict[str, Any]:
    path = PROSPECTIVE_ROOT / 'evaluation_status.json'
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _evaluation_panel(evaluation: dict[str, Any]) -> str:
    status = escape(str(evaluation.get('status') or 'INSUFFICIENT_SAMPLE'))
    metrics = evaluation.get('metrics') or {}
    checks = evaluation.get('checks') or {}

    def metric(name: str, kind: str = 'number') -> str:
        value = metrics.get(name)
        try:
            if value is None or pd.isna(value):
                return '—'
        except Exception:
            pass
        try:
            if kind == 'pct':
                return f'{100 * float(value):.1f}%'
            if kind == 'one':
                return f'{float(value):.1f}'
            if kind == 'two':
                return f'{float(value):.2f}'
            if kind == 'three':
                return f'{float(value):.3f}'
            return str(int(value))
        except Exception:
            return escape(str(value))

    unmet = [name for name, passed in checks.items() if not passed]
    unmet_text = (
        ', '.join(unmet[:5]) + ('…' if len(unmet) > 5 else '')
        if unmet else 'none'
    )
    return f'''
      <div class="evaluation-status">
        <div>
          <span class="eyebrow">Prospective evidence status</span>
          <h3>{status}</h3>
          <p>Never an automatic wagering authorization. REVIEW_ELIGIBLE means only that the pre-registered paper-evidence gate is ready for human review.</p>
        </div>
        <div class="evaluation-grid">
          <div><span>Record</span><strong>{metric('wins')}-{metric('losses')}-{metric('pushes')}</strong></div>
          <div><span>Captured-price ROI</span><strong>{metric('captured_price_roi','pct')}</strong></div>
          <div><span>Mean CLV</span><strong>{metric('mean_clv_points','two')} pts</strong></div>
          <div><span>Brier score</span><strong>{metric('brier_score','three')}</strong></div>
          <div><span>Graded entries</span><strong>{metric('graded_entries')}</strong></div>
          <div><span>Graded decisions</span><strong>{metric('graded_official_decisions')}</strong></div>
        </div>
        <div class="evaluation-note"><strong>Unmet pre-registered checks:</strong> {escape(unmet_text)}</div>
      </div>
    '''


def build_html(
    board: pd.DataFrame,
    decisions: pd.DataFrame,
    entries: pd.DataFrame,
    evaluation: dict[str, Any],
) -> str:
    generated = _generated(board)
    season = (
        int(pd.to_numeric(board['season'], errors='coerce').dropna().iloc[0])
        if not board.empty and 'season' in board.columns
        and pd.to_numeric(board['season'], errors='coerce').notna().any()
        else 2026
    )
    week = (
        int(pd.to_numeric(board['week'], errors='coerce').dropna().iloc[0])
        if not board.empty and 'week' in board.columns
        and pd.to_numeric(board['week'], errors='coerce').notna().any()
        else '—'
    )

    decisions_by_game = _official_map(decisions)
    entries_by_game = _entry_map(entries)
    cards = []
    for _, row in board.iterrows():
        key = str(row.get('game_id'))
        cards.append(
            _card(
                row,
                decisions_by_game.get(key),
                entries_by_game.get(key),
            )
        )

    if not cards:
        cards_html = (
            '<div class="empty-state">'
            '<strong>No active regular-season NFL slate.</strong><br>'
            'The live pipeline will populate this page when games are within '
            'the operational window.'
            '</div>'
        )
    else:
        cards_html = ''.join(cards)

    lines_available = (
        int(pd.to_numeric(board.get('closing_total'), errors='coerce').notna().sum())
        if not board.empty and 'closing_total' in board.columns else 0
    )
    evaluation_html = _evaluation_panel(evaluation)

    current_qualifiers = (
        int(board.get('model_signal', pd.Series(dtype=str)).isin(
            ['QUALIFIES', 'STRONG']
        ).sum())
        if not board.empty else 0
    )

    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>NFL Weather Totals · Live Paper Board</title>
  <style>
    :root {{
      --bg:#07101c; --panel:#101c2e; --panel2:#16253b; --text:#f4f8ff;
      --muted:#9cb0cb; --line:rgba(255,255,255,.11); --blue:#60a5fa;
      --cyan:#67e8f9; --green:#22c55e; --lime:#a3e635; --red:#ef4444;
      --amber:#f59e0b; --gray:#64748b; --purple:#a78bfa;
    }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{
      margin:0; color:var(--text);
      font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      background:
        radial-gradient(circle at 12% 0%,rgba(96,165,250,.18),transparent 28%),
        radial-gradient(circle at 92% 8%,rgba(34,197,94,.10),transparent 24%),
        linear-gradient(145deg,#050a12,#091322 48%,#0c1728);
      line-height:1.45;
    }}
    a {{ color:inherit; }}
    .shell {{ width:min(1320px,calc(100vw - 28px)); margin:0 auto; }}
    .sport-switch {{
      display:inline-flex; gap:4px; padding:4px; border-radius:999px;
      border:1px solid var(--line); background:rgba(6,12,22,.82);
      box-shadow:0 10px 30px rgba(0,0,0,.24);
    }}
    .sport-switch a {{
      min-width:72px; text-align:center; text-decoration:none; font-weight:900;
      font-size:12px; letter-spacing:.05em; padding:8px 12px; border-radius:999px;
      color:var(--muted);
    }}
    .sport-switch a.active {{
      color:#06101d; background:linear-gradient(135deg,#93c5fd,#67e8f9);
    }}
    header {{ padding:26px 0 18px; }}
    .topline {{ display:flex; justify-content:space-between; align-items:center; gap:14px; margin-bottom:16px; }}
    .mode {{
      display:inline-flex; align-items:center; gap:8px; color:#bbf7d0;
      font-size:11px; font-weight:950; letter-spacing:.09em; text-transform:uppercase;
    }}
    .mode::before {{ content:""; width:9px; height:9px; border-radius:50%; background:var(--green); box-shadow:0 0 16px rgba(34,197,94,.8); }}
    .hero {{ display:grid; grid-template-columns:1.35fr .65fr; gap:16px; }}
    .hero-card,.metric,.game-card,.panel,.empty-state {{
      border:1px solid var(--line);
      background:linear-gradient(180deg,rgba(17,30,50,.97),rgba(10,20,35,.96));
      box-shadow:0 18px 44px rgba(0,0,0,.28);
      border-radius:20px;
    }}
    .hero-card {{ padding:26px; }}
    .eyebrow {{ color:#bfdbfe; text-transform:uppercase; letter-spacing:.13em; font-size:11px; font-weight:950; }}
    h1 {{ font-size:clamp(36px,5.4vw,66px); line-height:.96; margin:10px 0 14px; }}
    h2 {{ font-size:26px; margin:0; }}
    h3 {{ margin:0; font-size:22px; }}
    h3 span {{ color:var(--muted); font-weight:500; }}
    p,.muted {{ color:var(--muted); }}
    .hero-card p {{ margin:0; max-width:800px; }}
    .hero-side {{ display:grid; gap:9px; }}
    .hero-side > div {{ padding:12px 14px; border:1px solid var(--line); background:rgba(255,255,255,.035); border-radius:13px; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:16px 0 0; }}
    .metric {{ padding:15px; }}
    .metric span {{ color:var(--muted); font-size:10px; text-transform:uppercase; letter-spacing:.09em; font-weight:900; }}
    .metric strong {{ display:block; font-size:28px; margin-top:2px; }}
    nav {{
      position:sticky; top:0; z-index:100; backdrop-filter:blur(14px);
      background:rgba(5,10,18,.84); border-block:1px solid var(--line);
    }}
    nav .shell {{ display:flex; gap:18px; align-items:center; overflow:auto; padding:10px 0; }}
    nav a {{ white-space:nowrap; text-decoration:none; font-size:13px; font-weight:850; color:#dbeafe; }}
    main {{ padding:22px 0 56px; }}
    section {{ margin-bottom:26px; }}
    .section-head {{ display:flex; justify-content:space-between; align-items:end; gap:14px; margin-bottom:12px; }}
    .section-head p {{ margin:0; max-width:640px; }}
    .toolbar {{
      display:flex; flex-wrap:wrap; gap:8px; padding:12px; margin-bottom:12px;
      border:1px solid var(--line); border-radius:15px; background:rgba(10,20,35,.86);
    }}
    input,select {{
      min-height:40px; background:#0b1728; color:var(--text); border:1px solid var(--line);
      border-radius:10px; padding:8px 11px;
    }}
    input {{ flex:1 1 230px; }}
    .cards {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:13px; }}
    .game-card {{ padding:18px; }}
    .card-top {{ display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:11px; }}
    .time-to-kick,.kickoff,.best-price,small {{ color:var(--muted); font-size:12px; }}
    .pick {{ font-size:27px; font-weight:950; margin:16px 0 2px; }}
    .best-price {{ margin-bottom:13px; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
    .grid > div {{
      padding:10px; min-height:74px; background:rgba(255,255,255,.035);
      border:1px solid rgba(255,255,255,.075); border-radius:11px;
    }}
    .grid span,.official-label {{
      display:block; color:var(--muted); font-size:9px; text-transform:uppercase;
      letter-spacing:.085em; font-weight:900; margin-bottom:4px;
    }}
    .grid strong {{ font-size:12px; line-height:1.35; }}
    .badge {{
      display:inline-block; padding:5px 9px; border-radius:999px; font-size:10px;
      font-weight:950; letter-spacing:.045em;
    }}
    .strong {{ color:#ecfccb; background:rgba(163,230,53,.14); border:1px solid rgba(163,230,53,.38); }}
    .qualifies {{ color:#bbf7d0; background:rgba(34,197,94,.13); border:1px solid rgba(34,197,94,.34); }}
    .no-play {{ color:#fecaca; background:rgba(239,68,68,.11); border:1px solid rgba(239,68,68,.28); }}
    .early {{ color:#dbeafe; background:rgba(96,165,250,.13); border:1px solid rgba(96,165,250,.32); }}
    .past {{ color:#cbd5e1; background:rgba(100,116,139,.16); border:1px solid rgba(148,163,184,.28); }}
    .waiting {{ color:#fde68a; background:rgba(245,158,11,.12); border:1px solid rgba(245,158,11,.32); }}
    .no-line {{ color:#ddd6fe; background:rgba(167,139,250,.13); border:1px solid rgba(167,139,250,.32); }}
    .official {{
      margin-top:13px; padding:10px 11px; border-radius:11px;
      display:flex; align-items:center; gap:9px; flex-wrap:wrap;
    }}
    .official strong {{ font-size:12px; }}
    .official small {{ margin-left:auto; }}
    .official.pending {{ background:rgba(255,255,255,.025); border:1px dashed var(--line); }}
    .entry-strip {{
      display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:13px;
      padding:9px 10px; border-radius:11px; border:1px solid rgba(34,197,94,.30);
      background:rgba(34,197,94,.075); color:#dcfce7; font-size:11px;
    }}
    .entry-strip strong {{ color:#bbf7d0; }}
    .panel {{ overflow:hidden; }}
    .table-wrap {{ overflow:auto; }}
    table {{ width:100%; min-width:1080px; border-collapse:collapse; }}
    th,td {{ text-align:left; vertical-align:top; padding:10px 11px; border-bottom:1px solid rgba(255,255,255,.07); font-size:12px; }}
    th {{ background:#101c2f; color:#dbeafe; position:sticky; top:0; }}
    td {{ color:#dce7f7; }}
    .empty-state,.empty-cell {{ padding:22px; color:var(--muted); }}
    .evaluation-status {{
      padding:18px; border:1px solid var(--line); border-radius:18px;
      background:linear-gradient(180deg,rgba(17,30,50,.97),rgba(10,20,35,.96));
      box-shadow:0 18px 44px rgba(0,0,0,.22);
    }}
    .evaluation-status h3 {{ margin-top:5px; font-size:24px; }}
    .evaluation-status p {{ margin:5px 0 14px; }}
    .evaluation-grid {{ display:grid; grid-template-columns:repeat(6,1fr); gap:8px; }}
    .evaluation-grid > div {{
      padding:10px; border:1px solid rgba(255,255,255,.08);
      border-radius:11px; background:rgba(255,255,255,.03);
    }}
    .evaluation-grid span {{ display:block; color:var(--muted); font-size:9px; text-transform:uppercase; letter-spacing:.08em; font-weight:900; }}
    .evaluation-grid strong {{ display:block; margin-top:4px; font-size:15px; }}
    .evaluation-note {{ margin-top:11px; color:var(--muted); font-size:11px; }}
    .protocol {{
      display:grid; grid-template-columns:repeat(4,1fr); gap:10px;
    }}
    .protocol > div {{ padding:14px; background:rgba(255,255,255,.035); border:1px solid var(--line); border-radius:13px; }}
    .protocol span {{ display:block; color:var(--muted); font-size:10px; text-transform:uppercase; font-weight:900; letter-spacing:.08em; }}
    .protocol strong {{ display:block; margin-top:4px; font-size:14px; }}
    footer {{ color:var(--muted); font-size:11px; padding-bottom:38px; }}
    @media(max-width:900px) {{
      .hero,.metrics,.cards,.protocol,.evaluation-grid {{ grid-template-columns:1fr; }}
      .section-head {{ align-items:start; flex-direction:column; }}
      .grid {{ grid-template-columns:1fr; }}
      .topline {{ align-items:flex-start; }}
    }}
  </style>
</head>
<body>
  <header class="shell">
    <div class="topline">
      <div class="sport-switch" aria-label="Sport mode">
        <a href="../">NCAA</a>
        <a class="active" href="./" aria-current="page">NFL</a>
      </div>
      <div class="mode">Paper mode · frozen protocol</div>
    </div>
    <div class="hero">
      <div class="hero-card">
        <div class="eyebrow">Live 2026 NFL totals · 24-hour decision protocol</div>
        <h1>NFL Weather Totals</h1>
        <p>The live board uses current sportsbook totals, fixed-lead JMA forecasts, venue/roof policy, and the frozen RF regression + RF classifier ensemble. Official entries are prospective paper records only.</p>
      </div>
      <div class="hero-card hero-side">
        <div><span class="eyebrow">Slate</span><br><strong>{season} · Week {week}</strong></div>
        <div><span class="eyebrow">Updated</span><br><strong>{escape(generated)}</strong></div>
        <div><span class="eyebrow">Official horizon</span><br><strong>24 hours before kickoff</strong></div>
        <div><span class="eyebrow">Side</span><br><strong>OVER only</strong></div>
      </div>
    </div>
    <div class="metrics">
      <div class="metric"><span>Games</span><strong>{len(board)}</strong></div>
      <div class="metric"><span>Current totals</span><strong>{lines_available}</strong></div>
      <div class="metric"><span>Official decisions</span><strong>{len(decisions)}</strong></div>
      <div class="metric"><span>Official entries</span><strong>{len(entries)}</strong></div>
    </div>
  </header>

  <nav><div class="shell">
    <a href="#live">Live cards</a>
    <a href="#ledger">Frozen decisions</a>
    <a href="#performance">Prospective performance</a>
    <a href="#protocol">Protocol</a>
    <a href="../">NCAA board →</a>
  </div></nav>

  <main class="shell">
    <section id="live">
      <div class="section-head">
        <div><div class="eyebrow">Current market + model state</div><h2>Live weekly board</h2></div>
        <p>Current signals may move before the official decision. Only the first eligible scheduled snapshot inside the 24-hour window becomes permanent.</p>
      </div>
      <div class="toolbar">
        <input id="gameSearch" type="search" placeholder="Search team, venue, sportsbook…">
        <select id="statusFilter">
          <option value="ALL">All live statuses</option>
          <option>STRONG</option><option>QUALIFIES</option><option>NO PLAY</option>
          <option>EARLY LOOK</option><option>PAST WINDOW</option><option>WAITING</option><option>NO LINE</option>
        </select>
        <select id="signalFilter">
          <option value="ALL">All model signals</option>
          <option>STRONG</option><option>QUALIFIES</option><option>NO PLAY</option>
        </select>
      </div>
      <div class="cards" id="gameCards">{cards_html}</div>
    </section>

    <section id="ledger">
      <div class="section-head">
        <div><div class="eyebrow">Immutable prospective record</div><h2>Frozen 24-hour decisions</h2></div>
        <p>NO PLAY decisions stay visible so later model movement cannot rewrite the historical record.</p>
      </div>
      <div class="panel"><div class="table-wrap">
        <table>
          <thead><tr><th>Decision</th><th>Game</th><th>Captured market</th><th>Frozen model</th><th>Weather policy</th><th>Snapshot UTC</th></tr></thead>
          <tbody>{_decision_rows(decisions)}</tbody>
        </table>
      </div></div>
    </section>

    <section id="performance">
      <div class="section-head">
        <div><div class="eyebrow">Untouched 2026 paper evidence</div><h2>Prospective performance</h2></div>
        <p>ROI, CLV, calibration, and post-freeze signal movement are evaluated against criteria frozen before prospective outcomes accumulate.</p>
      </div>
      {evaluation_html}
    </section>

    <section id="protocol">
      <div class="section-head">
        <div><div class="eyebrow">nfl_totals_paper_v1</div><h2>Frozen protocol</h2></div>
        <p>These rules are versioned and cannot be retuned from 2026 outcomes.</p>
      </div>
      <div class="protocol">
        <div><span>QUALIFIES</span><strong>OVER edge ≥ +3.0 · P(OVER) ≥ 60%</strong></div>
        <div><span>STRONG</span><strong>OVER edge ≥ +4.0 · P(OVER) ≥ 60%</strong></div>
        <div><span>Weather</span><strong>JMA GSM fixed-lead · outdoor only</strong></div>
        <div><span>Tracking</span><strong>1u paper entries · captured price + CLV</strong></div>
      </div>
    </section>
  </main>

  <footer class="shell">
    Research and paper tracking only. Historical and prospective paper results do not guarantee future wagering performance. The NFL protocol remains production-disabled for real-money use.
  </footer>

  <script>
    const search = document.getElementById('gameSearch');
    const statusFilter = document.getElementById('statusFilter');
    const signalFilter = document.getElementById('signalFilter');
    const cards = [...document.querySelectorAll('.game-card')];

    function filterCards() {{
      const q = (search.value || '').toLowerCase();
      const status = statusFilter.value;
      const signal = signalFilter.value;
      cards.forEach(function(card) {{
        const textMatch = card.innerText.toLowerCase().includes(q);
        const statusMatch = status === 'ALL' || card.dataset.status === status;
        const signalMatch = signal === 'ALL' || card.dataset.signal === signal;
        card.style.display = textMatch && statusMatch && signalMatch ? '' : 'none';
      }});
    }}

    search.addEventListener('input', filterCards);
    statusFilter.addEventListener('change', filterCards);
    signalFilter.addEventListener('change', filterCards);
  </script>
</body>
</html>
'''


def main() -> None:
    board = _read_csv(LIVE_BOARD)
    decisions = _read_csv(PROSPECTIVE_ROOT / 'official_decisions.csv')
    entries_with_clv = _read_csv(PROSPECTIVE_ROOT / 'entries_with_clv.csv')
    entries = (
        entries_with_clv
        if not entries_with_clv.empty
        else _read_csv(PROSPECTIVE_ROOT / 'official_entries.csv')
    )
    evaluation = _load_evaluation()

    ensure_dir(DOCS_NFL)
    html = build_html(board, decisions, entries, evaluation)
    output = DOCS_NFL / 'index.html'
    output.write_text(html, encoding='utf-8')
    print(
        f'Wrote {output.relative_to(ROOT)}: '
        f'{len(board)} live games, {len(decisions)} official decisions, '
        f'{len(entries)} official entries.'
    )


if __name__ == '__main__':
    main()
