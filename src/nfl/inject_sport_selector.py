from __future__ import annotations

import re

from ..utils import ROOT


INDEX = ROOT / 'docs/index.html'
START = '<!-- SPORT_MODE_SELECTOR_START -->'
END = '<!-- SPORT_MODE_SELECTOR_END -->'

BLOCK = r'''<!-- SPORT_MODE_SELECTOR_START -->
<style id="sportModeSelectorStyle">
  #sportModeSelector {
    position:fixed;
    right:16px;
    bottom:16px;
    z-index:10000;
    display:inline-flex;
    gap:4px;
    padding:4px;
    border:1px solid rgba(255,255,255,.16);
    border-radius:999px;
    background:rgba(6,11,20,.90);
    backdrop-filter:blur(14px);
    box-shadow:0 14px 36px rgba(0,0,0,.34);
    font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  }
  #sportModeSelector a {
    min-width:70px;
    padding:8px 12px;
    border-radius:999px;
    text-align:center;
    text-decoration:none;
    font-size:12px;
    line-height:1.2;
    font-weight:900;
    letter-spacing:.04em;
    color:#9fb0ce;
  }
  #sportModeSelector a.active {
    color:#06101d;
    background:linear-gradient(135deg,#93c5fd,#67e8f9);
  }
  #sportModeSelector a:focus-visible {
    outline:2px solid #7dd3fc;
    outline-offset:2px;
  }
  @media(max-width:600px) {
    #sportModeSelector {
      right:10px;
      bottom:10px;
    }
    #sportModeSelector a {
      min-width:60px;
      padding:7px 9px;
    }
  }
</style>
<div id="sportModeSelector" aria-label="Sport mode">
  <a class="active" href="./" aria-current="page">NCAA</a>
  <a href="nfl/">NFL</a>
</div>
<!-- SPORT_MODE_SELECTOR_END -->'''


def inject(html: str) -> str:
    if START in html and END in html:
        pattern = re.compile(
            re.escape(START) + r'.*?' + re.escape(END),
            flags=re.DOTALL,
        )
        return pattern.sub(BLOCK, html, count=1)

    closing = html.lower().rfind('</body>')
    if closing < 0:
        raise RuntimeError('docs/index.html has no </body> insertion point.')
    return html[:closing] + BLOCK + '\n' + html[closing:]


def main() -> None:
    if not INDEX.exists():
        raise FileNotFoundError(f'Missing NCAA site: {INDEX}')

    before = INDEX.read_text(encoding='utf-8')
    after = inject(before)
    INDEX.write_text(after, encoding='utf-8')

    count = after.count(START)
    if count != 1:
        raise RuntimeError(
            f'Sport selector injection is not idempotent: marker count={count}'
        )
    print('Injected NCAA/NFL sport selector without altering dashboard logic.')


if __name__ == '__main__':
    main()
