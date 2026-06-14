from __future__ import annotations

from pathlib import Path

from .utils import ROOT


TAB_MARKER = '<button class="tab active" data-target="overview">Overview</button>'
SECTION_MARKER = '<section class="section" id="straight">'

EXPLAINER_TAB = '''<button class="tab" data-target="howitworks">How It Works</button>'''

EXPLAINER_SECTION = r'''
    <section class="section" id="howitworks">
      <div class="panel">
        <h2>How this research came together</h2>
        <p>This page is meant to be transparent enough that someone can understand the whole process without reading the code. The project is not trying to predict the exact final score. It is trying to learn whether certain games are more likely to finish above or below the market total after accounting for the closing total.</p>
        <div class="grid3">
          <div class="rule"><strong>1. Pull historical games</strong><span>Collect college football games, scores, teams, locations, lines/totals, weather, and team-context fields.</span></div>
          <div class="rule"><strong>2. Build the target</strong><span>Calculate market residual: actual total points minus the closing total. Negative means the game finished under the market total.</span></div>
          <div class="rule"><strong>3. Test weather/context</strong><span>Check simple rules first, such as wind, cold, rain, total buckets, indoor/outdoor games, and high-total games.</span></div>
          <div class="rule"><strong>4. Train models</strong><span>Train multiple model types using historical seasons and compare whether their predicted edge beats the market.</span></div>
          <div class="rule"><strong>5. Validate thresholds</strong><span>Split results by side, edge threshold, recent period, total bin, line provider, and season to see what survived.</span></div>
          <div class="rule"><strong>6. Build weekly-card logic</strong><span>Limit the strategy to a realistic number of weekly targets instead of grading every possible combo.</span></div>
        </div>
      </div>

      <div class="grid">
        <div class="panel">
          <h2>Key terms</h2>
          <div class="rule-list">
            <div class="rule"><strong>Closing total</strong><span>The market's final over/under number used for the historical backtest. Live use will need the current total at decision time.</span></div>
            <div class="rule"><strong>Actual total points</strong><span>The final combined score of both teams.</span></div>
            <div class="rule"><strong>Market residual</strong><span>Actual total points minus closing total. Example: if the closing total was 58.5 and the game finished 52, the residual is -6.5.</span></div>
            <div class="rule"><strong>Model edge</strong><span>How far the model thinks the game is from the market total. A negative predicted residual points toward an under; a positive predicted residual points toward an over.</span></div>
            <div class="rule"><strong>3.5+ edge threshold</strong><span>The current minimum historical threshold. Below 3.5 points, the signal has not been strong enough to treat as a production target.</span></div>
            <div class="rule"><strong>High-total screen</strong><span>A filter for games with totals in the 56+ range. This is where the under signal and weekly-card results looked cleanest.</span></div>
          </div>
        </div>
        <div class="panel">
          <h2>Model glossary</h2>
          <div class="rule-list">
            <div class="rule"><strong>HGB / HistGradientBoosting</strong><span>A tree-based machine-learning model that can learn nonlinear interactions. In this project, it can combine things like total level, wind, temperature, precipitation, provider, and team context. It was the most useful model in the current results.</span></div>
            <div class="rule"><strong>Ridge</strong><span>A regularized linear model. It is easier to interpret, but it did not validate as strongly as HGB for the current edge strategy.</span></div>
            <div class="rule"><strong>ElasticNet</strong><span>Another regularized linear model that can shrink or remove weak predictors. It showed some pockets of signal but was less stable.</span></div>
            <div class="rule"><strong>Random Forest / Extra Trees</strong><span>Tree-ensemble models. They can catch interactions, but some strong-looking results were thinner or less stable.</span></div>
            <div class="rule"><strong>Logistic regression</strong><span>A classification model that tries to estimate over/under probability directly. It did not become the main strategy because the residual-based HGB approach was stronger.</span></div>
          </div>
        </div>
      </div>

      <div class="panel">
        <h2>What the current model is saying</h2>
        <div class="grid3">
          <div class="rule"><strong class="good">Best validated direction</strong><span>Unders, not overs. Overs are shown for transparency, but they are not the default production target.</span></div>
          <div class="rule"><strong class="good">Best model</strong><span>HistGradientBoosting, abbreviated HGB throughout the dashboard.</span></div>
          <div class="rule"><strong class="good">Best threshold</strong><span>3.5+ predicted point edge, with 5.0+ treated as stricter but thinner.</span></div>
          <div class="rule"><strong class="good">Best weekly combo profile</strong><span>Top 1 weekly 2-leg combo from HGB under 3.5+ high-total games.</span></div>
          <div class="rule"><strong class="warn">Still needs live tracking</strong><span>Historical testing used closing totals and historical weather. Production use must use current lines and forecast weather.</span></div>
          <div class="rule"><strong class="bad">No-play by design</strong><span>Below-threshold edges, over-only leans, missing line/weather confidence, or extra parlays should be shown as no-play or paper-only.</span></div>
        </div>
      </div>

      <div class="panel">
        <h2>Why this is not just a pick page</h2>
        <p>The goal is to show both sides of the research: what worked and what failed. A trustworthy weekly page should show specific games only when they meet the historical threshold. It should also clearly show why other games are no-plays. The number of games and combos should vary by week because the model should not force action on weak slates.</p>
        <div class="callout">Simple summary: HGB means HistGradientBoosting. The model predicts how far a game may finish from the market total. The current historical strategy is selective unders only, especially high-total games with at least a 3.5-point model edge.</div>
      </div>
    </section>
'''


def enrich_html(html: str) -> str:
    if 'data-target="howitworks"' not in html:
        html = html.replace(TAB_MARKER, TAB_MARKER + '\n      ' + EXPLAINER_TAB)
    if 'id="howitworks"' not in html:
        html = html.replace('    ' + SECTION_MARKER, EXPLAINER_SECTION + '\n    ' + SECTION_MARKER)
    return html


def enrich_file(path: Path) -> None:
    if not path.exists():
        return
    html = path.read_text(encoding='utf-8')
    path.write_text(enrich_html(html), encoding='utf-8')


def main() -> None:
    enrich_file(ROOT / 'docs' / 'index.html')
    enrich_file(ROOT / 'outputs' / 'research_dashboard.html')
    print('Added modeling explainer section to research dashboard HTML')


if __name__ == '__main__':
    main()
