from __future__ import annotations

import json
import re
from html import escape
from typing import Any

import pandas as pd

from .utils import ROOT


def _text(value: Any, fallback: str = '') -> str:
    try:
        if value is None or pd.isna(value):
            return fallback
    except Exception:
        pass
    text = str(value).strip()
    return text if text and text.lower() != 'nan' else fallback


def _fmt_total(value: Any) -> str:
    try:
        return f'{float(value):.1f}'
    except Exception:
        return '—'


def _qualifier_records(board: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if board.empty or 'status' not in board.columns:
        return [], [], []

    qualifying = board[board['status'].astype(str).eq('QUALIFIES')].copy()
    if 'abs_pred_edge' in qualifying.columns:
        qualifying = qualifying.sort_values('abs_pred_edge', ascending=False)

    if 'division_track' in qualifying.columns:
        is_fcs = qualifying['division_track'].astype(str).str.upper().eq('FCS')
    else:
        is_fcs = pd.Series(False, index=qualifying.index)

    general = qualifying[~is_fcs].copy()
    fcs = qualifying[is_fcs].copy()

    def records(frame: pd.DataFrame, track: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            out.append({
                'game_id': _text(row.get('game_id')),
                'matchup': f"{_text(row.get('away_team'), 'Away')} @ {_text(row.get('home_team'), 'Home')}",
                'total': _fmt_total(row.get('closing_total')),
                'edge': _fmt_total(row.get('abs_pred_edge')),
                'track': track,
            })
        return out

    general_records = records(general, 'GENERAL')
    fcs_records = records(fcs, 'FCS RESEARCH')
    card_records = general_records[:2] if len(general_records) >= 2 else []
    return general_records, fcs_records, card_records


def _section_html(general_count: int, fcs_count: int, has_card: bool) -> str:
    card_text = 'one top 2-leg card is available' if has_card else 'no validated 2-leg card is available'
    return f'''
    <section id="bankroll">
      <div class="section-head">
        <div><div class="eyebrow">Flat-risk planning · no forced action</div><h2>Bankroll & weekly plan</h2></div>
        <p>Turns the live board into a staking checklist while keeping the general and FCS evidence levels separate.</p>
      </div>
      <div class="bankroll-grid">
        <div class="bankroll-panel">
          <div class="bankroll-controls">
            <label><span>Bankroll</span><input id="bankrollAmount" type="number" min="0" step="50" inputmode="decimal" placeholder="5000"></label>
            <label><span>Risk profile</span>
              <select id="bankrollProfile">
                <option value="conservative" selected>Prospective conservative</option>
                <option value="standard">Standard sizing</option>
                <option value="custom">Custom</option>
              </select>
            </label>
            <label><span>General straight %</span><input id="straightRiskPct" type="number" min="0" max="5" step="0.05" value="0.50"></label>
            <label><span>FCS research %</span><input id="fcsRiskPct" type="number" min="0" max="5" step="0.05" value="0.25"></label>
            <label><span>2-leg card %</span><input id="cardRiskPct" type="number" min="0" max="5" step="0.05" value="0.25"></label>
          </div>
          <div class="bankroll-metrics">
            <div><span>General straight</span><strong id="straightStake">—</strong></div>
            <div><span>FCS research</span><strong id="fcsStake">—</strong></div>
            <div><span>2-leg card</span><strong id="cardStake">—</strong></div>
            <div><span>This snapshot max risk</span><strong id="weeklyRisk">—</strong></div>
            <div><span>Bankroll exposure</span><strong id="weeklyExposure">—</strong></div>
          </div>
          <div id="bankrollWarning" class="bankroll-warning" hidden></div>
          <div class="bankroll-note">The FCS percentage is intentionally smaller because its selected historical screen was encouraging but its stricter nested validation was weaker. Set FCS risk to 0% anytime you want to keep that track paper-only. These are flat-risk planning suggestions, not Kelly sizing. Your bankroll is stored only in this browser.</div>
        </div>
        <div class="bankroll-panel guide-panel">
          <div class="eyebrow">What the current research supports</div>
          <h3>What should I be in?</h3>
          <ol>
            <li><strong>General QUALIFIES:</strong> if following the historical straight-bet method, take each one once as a straight UNDER at the same flat stake. Do not cherry-pick after qualification.</li>
            <li><strong>FCS RESEARCH QUALIFIES:</strong> optional reduced straight stake. The default is half the general stake because this track has more validation uncertainty. It remains excluded from the 2-leg card.</li>
            <li><strong>2-leg card:</strong> add only the single top card shown by the site when two eligible general qualifiers exist. Do not create extra combinations from the same slate.</li>
            <li><strong>LEAN / WATCH / NO PLAY:</strong> no bankroll stake under the current method.</li>
            <li><strong>Site refreshes:</strong> the same game appearing again Thursday, Friday, or Saturday is not a new wager. Stake it once unless a separate re-entry rule is tested later.</li>
          </ol>
          <p class="small">Current snapshot: <strong>{general_count}</strong> general qualifier(s), <strong>{fcs_count}</strong> FCS research qualifier(s), and {escape(card_text)}.</p>
        </div>
      </div>
      <div class="bankroll-panel plan-panel">
        <div class="plan-head"><div><div class="eyebrow">Current-board translation</div><h3>Suggested staking checklist</h3></div><span class="small">The card overlaps two general straight-bet exposures, so its default stake is smaller.</span></div>
        <div id="bankrollPlan"></div>
      </div>
    </section>
    '''


BANKROLL_CSS = r'''
    .bankroll-grid { display:grid; grid-template-columns:1.05fr .95fr; gap:14px; }
    .bankroll-panel { border:1px solid var(--line); background:linear-gradient(180deg,rgba(20,32,55,.96),rgba(13,24,43,.94)); border-radius:20px; box-shadow:0 18px 44px rgba(0,0,0,.22); padding:20px; }
    .bankroll-controls { display:grid; grid-template-columns:1.1fr 1.25fr .8fr .8fr .8fr; gap:10px; align-items:end; }
    .bankroll-controls label { display:grid; gap:6px; color:var(--muted); font-size:11px; font-weight:900; text-transform:uppercase; letter-spacing:.06em; }
    .bankroll-controls input,.bankroll-controls select { width:100%; min-height:42px; }
    .bankroll-metrics { display:grid; grid-template-columns:repeat(5,1fr); gap:9px; margin-top:14px; }
    .bankroll-metrics div { border:1px solid var(--line); background:rgba(255,255,255,.035); border-radius:13px; padding:11px; }
    .bankroll-metrics span { display:block; color:var(--muted); font-size:10px; text-transform:uppercase; letter-spacing:.06em; font-weight:900; }
    .bankroll-metrics strong { display:block; margin-top:3px; font-size:20px; }
    .bankroll-note { margin-top:12px; color:var(--muted); font-size:11px; }
    .bankroll-warning { margin-top:12px; border:1px solid rgba(245,158,11,.38); background:rgba(245,158,11,.10); color:#fde68a; border-radius:12px; padding:10px 12px; font-size:12px; }
    .guide-panel h3,.plan-panel h3 { margin:5px 0 10px; font-size:21px; }
    .guide-panel ol { margin:8px 0 0 20px; padding:0; color:#dce7f7; }
    .guide-panel li { margin:8px 0; padding-left:3px; }
    .plan-panel { margin-top:14px; }
    .plan-head { display:flex; justify-content:space-between; gap:14px; align-items:end; margin-bottom:10px; }
    .stake-table { width:100%; border-collapse:collapse; min-width:0; }
    .stake-table th,.stake-table td { position:static; background:transparent; padding:10px 8px; font-size:12px; }
    .stake-kind { font-weight:950; font-size:10px; letter-spacing:.06em; border-radius:999px; padding:5px 7px; display:inline-block; white-space:nowrap; }
    .stake-straight { color:#bbf7d0; border:1px solid rgba(34,197,94,.35); background:rgba(34,197,94,.12); }
    .stake-card { color:#bae6fd; border:1px solid rgba(125,211,252,.35); background:rgba(125,211,252,.10); }
    .stake-fcs { color:#ddd6fe; border:1px solid rgba(139,92,246,.45); background:rgba(139,92,246,.14); }
    .stake-empty { color:var(--muted); border:1px dashed var(--line); border-radius:13px; padding:15px; }
    @media(max-width:1100px) { .bankroll-grid { grid-template-columns:1fr; } .bankroll-controls { grid-template-columns:repeat(2,1fr); } .bankroll-metrics { grid-template-columns:repeat(3,1fr); } }
    @media(max-width:650px) { .bankroll-controls,.bankroll-metrics { grid-template-columns:1fr; } .plan-head { align-items:start; flex-direction:column; } .stake-table { min-width:640px; } .plan-panel { overflow:auto; } }
'''


def _script(general: list[dict[str, Any]], fcs: list[dict[str, Any]], card: list[dict[str, Any]]) -> str:
    payload = json.dumps({'general': general, 'fcs': fcs, 'card': card}, ensure_ascii=False).replace('<', '\\u003c')
    return f'''
  <script>
    (function() {{
      const bankrollData = {payload};
      const amountInput = document.getElementById('bankrollAmount');
      const profile = document.getElementById('bankrollProfile');
      const straightPctInput = document.getElementById('straightRiskPct');
      const fcsPctInput = document.getElementById('fcsRiskPct');
      const cardPctInput = document.getElementById('cardRiskPct');
      const straightStakeEl = document.getElementById('straightStake');
      const fcsStakeEl = document.getElementById('fcsStake');
      const cardStakeEl = document.getElementById('cardStake');
      const weeklyRiskEl = document.getElementById('weeklyRisk');
      const weeklyExposureEl = document.getElementById('weeklyExposure');
      const planEl = document.getElementById('bankrollPlan');
      const warningEl = document.getElementById('bankrollWarning');

      if (!amountInput || !planEl) return;

      const profiles = {{
        conservative: {{ straight:0.50, fcs:0.25, card:0.25 }},
        standard: {{ straight:1.00, fcs:0.50, card:0.50 }}
      }};

      const saved = localStorage.getItem('cfbBankrollAmount');
      if (saved && Number(saved) > 0) amountInput.value = saved;

      function money(value) {{
        if (!Number.isFinite(value)) return '—';
        return value.toLocaleString(undefined, {{ style:'currency', currency:'USD', minimumFractionDigits:2, maximumFractionDigits:2 }});
      }}

      function esc(value) {{
        return String(value == null ? '' : value).replace(/[&<>'\"]/g, function(ch) {{
          return {{ '&':'&amp;', '<':'&lt;', '>':'&gt;', "'":'&#39;', '\"':'&quot;' }}[ch];
        }});
      }}

      function setProfile() {{
        const selected = profiles[profile.value];
        if (!selected) return;
        straightPctInput.value = selected.straight.toFixed(2);
        fcsPctInput.value = selected.fcs.toFixed(2);
        cardPctInput.value = selected.card.toFixed(2);
        render();
      }}

      function render() {{
        const bankroll = Math.max(0, Number(amountInput.value) || 0);
        const straightPct = Math.max(0, Number(straightPctInput.value) || 0);
        const fcsPct = Math.max(0, Number(fcsPctInput.value) || 0);
        const cardPct = Math.max(0, Number(cardPctInput.value) || 0);
        if (bankroll > 0) localStorage.setItem('cfbBankrollAmount', String(bankroll));

        const straightStake = bankroll * straightPct / 100;
        const fcsStake = bankroll * fcsPct / 100;
        const cardStake = bankroll * cardPct / 100;
        const hasCard = bankrollData.card.length === 2;
        const totalRisk = straightStake * bankrollData.general.length + fcsStake * bankrollData.fcs.length + (hasCard ? cardStake : 0);
        const exposure = bankroll > 0 ? totalRisk / bankroll * 100 : 0;

        straightStakeEl.textContent = bankroll ? money(straightStake) : 'Enter bankroll';
        fcsStakeEl.textContent = bankroll ? money(fcsStake) : 'Enter bankroll';
        cardStakeEl.textContent = bankroll ? money(cardStake) : 'Enter bankroll';
        weeklyRiskEl.textContent = bankroll ? money(totalRisk) : 'Enter bankroll';
        weeklyExposureEl.textContent = bankroll ? exposure.toFixed(2) + '%' : '—';

        warningEl.hidden = true;
        if (bankroll && exposure > 7.5) {{
          warningEl.textContent = 'This snapshot would put ' + exposure.toFixed(1) + '% of the bankroll at risk. Consider the conservative profile or smaller custom percentages rather than increasing exposure just because the slate is busy.';
          warningEl.hidden = false;
        }} else if (straightPct > 1.5 || fcsPct > 0.75 || cardPct > 0.75) {{
          warningEl.textContent = 'One or more custom percentages are above the site’s normal planning range. The prospective evidence is not strong enough to justify aggressive sizing by default.';
          warningEl.hidden = false;
        }}

        const rows = [];
        bankrollData.general.forEach(function(game) {{
          rows.push('<tr><td><span class="stake-kind stake-straight">GENERAL STRAIGHT</span></td><td><strong>' + esc(game.matchup) + '</strong><br><span class="small">UNDER ' + esc(game.total) + ' · edge ' + esc(game.edge) + '</span></td><td>' + (bankroll ? '<strong>' + money(straightStake) + '</strong>' : 'Enter bankroll') + '</td><td class="small">One flat stake. Do not scale up solely because the edge number is larger.</td></tr>');
        }});

        bankrollData.fcs.forEach(function(game) {{
          rows.push('<tr><td><span class="stake-kind stake-fcs">FCS RESEARCH</span></td><td><strong>' + esc(game.matchup) + '</strong><br><span class="small">UNDER ' + esc(game.total) + ' · FCS edge ' + esc(game.edge) + '</span></td><td>' + (bankroll ? '<strong>' + money(fcsStake) + '</strong>' : 'Enter bankroll') + '</td><td class="small">Optional reduced straight stake because the FCS screen has weaker robustness. Set FCS risk to 0% to paper-track only. Never add it to the legacy 2-leg card.</td></tr>');
        }});

        if (hasCard) {{
          const a = bankrollData.card[0];
          const b = bankrollData.card[1];
          rows.push('<tr><td><span class="stake-kind stake-card">2-LEG CARD</span></td><td><strong>UNDER ' + esc(a.total) + ' ' + esc(a.matchup) + '</strong><br><strong>UNDER ' + esc(b.total) + ' ' + esc(b.matchup) + '</strong></td><td>' + (bankroll ? '<strong>' + money(cardStake) + '</strong>' : 'Enter bankroll') + '</td><td class="small">The one site-selected general card only. It overlaps two straight positions, so the stake is smaller.</td></tr>');
        }}

        if (!rows.length) {{
          planEl.innerHTML = '<div class="stake-empty"><strong>No staking actions from this snapshot.</strong><br>There are no qualifying general or FCS research unders and no valid two-leg card right now.</div>';
        }} else {{
          planEl.innerHTML = '<div class="table-wrap"><table class="stake-table"><thead><tr><th>Type</th><th>Play</th><th>Suggested risk</th><th>Method note</th></tr></thead><tbody>' + rows.join('') + '</tbody></table></div>';
        }}
      }}

      profile.addEventListener('change', setProfile);
      amountInput.addEventListener('input', render);
      straightPctInput.addEventListener('input', function() {{ profile.value = 'custom'; render(); }});
      fcsPctInput.addEventListener('input', function() {{ profile.value = 'custom'; render(); }});
      cardPctInput.addEventListener('input', function() {{ profile.value = 'custom'; render(); }});
      render();
    }})();
  </script>
'''


def _inject(html: str, section: str, script: str) -> str:
    if 'id="bankroll"' in html:
        html = re.sub(r'\s*<section id="bankroll">.*?</section>\s*', '\n', html, count=1, flags=re.S)

    if '.bankroll-grid {' not in html:
        html = html.replace('</style>', BANKROLL_CSS + '\n  </style>', 1)
    if 'href="#bankroll"' not in html:
        html = html.replace('<a href="#card">2-leg card</a>', '<a href="#bankroll">Bankroll plan</a><a href="#card">2-leg card</a>', 1)
    marker = '<section id="card">'
    if marker not in html:
        raise RuntimeError('Could not locate card section for bankroll-helper insertion.')
    html = html.replace(marker, section + '\n    ' + marker, 1)
    html = html.replace('</body>', script + '\n</body>', 1)
    return html


def main() -> None:
    board_path = ROOT / 'outputs/weekly_board.csv'
    if not board_path.exists():
        raise RuntimeError('weekly_board.csv is required before building the bankroll helper.')
    board = pd.read_csv(board_path)
    general, fcs, card = _qualifier_records(board)
    section = _section_html(len(general), len(fcs), len(card) == 2)
    script = _script(general, fcs, card)

    updated_count = 0
    for relative in ['docs/index.html', 'outputs/live_dashboard.html']:
        path = ROOT / relative
        if not path.exists():
            continue
        html = path.read_text(encoding='utf-8')
        path.write_text(_inject(html, section, script), encoding='utf-8')
        updated_count += 1

    if not updated_count:
        raise RuntimeError('No live dashboard HTML files were available for bankroll-helper injection.')
    print(f'Added bankroll helper: {len(general)} general qualifier(s), {len(fcs)} FCS research qualifier(s), card={len(card) == 2}.')


if __name__ == '__main__':
    main()
