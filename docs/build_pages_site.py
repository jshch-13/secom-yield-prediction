"""Build the optional GitHub Pages site under docs/.

Copies the four SECOM-based interactive Plotly HTML files from
``outputs/interactive/`` (real SECOM data, produced by
``dashboard/build_static_reports.py``) into ``docs/`` and writes an
``index.html`` that links to all four.

This script does NOT push to GitHub or change repository settings -- see
the README for how to enable Pages manually.

Run as a script:

    python docs/build_pages_site.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
INTERACTIVE_DIR = PROJECT_ROOT / "outputs" / "interactive"

SECOM_HTML_FILES = (
    "interactive_feature_distribution.html",
    "interactive_scatter_matrix.html",
    "prediction_probability_dashboard.html",
    "feature_importance.html",
)


def copy_secom_htmls() -> list[str]:
    """Copy the real-SECOM interactive HTMLs from outputs/interactive/ into docs/.

    Returns:
        List of filenames actually copied (missing source files are skipped
        with a warning printed, so a partial ``outputs/interactive/`` still
        produces a usable, if incomplete, docs/ site).
    """
    copied = []
    for name in SECOM_HTML_FILES:
        src = INTERACTIVE_DIR / name
        if not src.exists():
            print(f"warning: {src} not found; run dashboard/build_static_reports.py first. Skipping.")
            continue
        shutil.copy2(src, DOCS_DIR / name)
        copied.append(name)
    return copied


def build_index_html(secom_files: list[str]) -> Path:
    """Write ``docs/index.html`` linking to every generated page.

    Args:
        secom_files: Filenames of the real-SECOM HTML pages actually present.

    Returns:
        The path to the written ``index.html``.
    """
    secom_links = "\n".join(
        f'<li><a href="{f}">{f}</a></li>' for f in secom_files
    ) or "<li><em>Not yet generated -- run dashboard/build_static_reports.py</em></li>"

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SECOM Yield Prediction - Interactive Reports</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; color: #1F2430; }}
h1 {{ font-size: 22px; }}
.notice {{ background: #FFF4E5; border: 1px solid #F0B429; border-radius: 8px; padding: 12px 16px; margin: 16px 0; font-size: 13px; }}
.section {{ margin-top: 28px; }}
a {{ color: #2F6FED; }}
footer {{ margin-top: 40px; font-size: 12px; color: #5B6472; }}
</style>
</head>
<body>
<h1>SECOM Yield Prediction &mdash; Interactive Reports</h1>
<p>공개·익명화된 UCI/Kaggle SECOM 데이터셋 기반 반도체 공정 센서 데이터 불량 예측 프로젝트의
인터랙티브 산출물 모음입니다. 아래 리포트는 전부 실제 SECOM 데이터와 학습된 모델 결과를 사용합니다.</p>

<div class="notice">
센서 ID는 익명화되어 있으며, 실제 Fab 장비/공정/챔버를 특정하지 않습니다.
</div>

<div class="section">
<h2>SECOM Analysis</h2>
<ul>
{secom_links}
</ul>
</div>

<div class="section">
<h2>Dash Dashboard</h2>
<p>Plotly Dash 기반 대화형 대시보드(<code>dashboard/app.py</code>)는 GitHub Pages 같은 정적 호스팅으로는
제공되지 않습니다. 로컬 실행(<code>python dashboard/app.py</code>) 또는 별도의 서버 배포가 필요합니다.</p>
</div>

<footer>
Repository: <code>&lt;GITHUB_REPOSITORY_URL_PLACEHOLDER&gt;</code><br>
Public anonymized SECOM dataset. Sensor IDs are anonymized; this project does not identify real Fab equipment,
chambers, or processes.
</footer>
</body>
</html>
"""
    index_path = DOCS_DIR / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path


def main() -> None:
    """Build the full docs/ GitHub Pages site (copy SECOM HTMLs + index)."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    secom_files = copy_secom_htmls()
    index_path = build_index_html(secom_files)
    print(f"wrote {index_path}")
    for f in secom_files:
        print(f"copied {DOCS_DIR / f}")
    print(
        "\nGitHub Pages was NOT enabled and nothing was pushed. "
        "See README.md for how to enable Pages manually (Settings > Pages > "
        "Deploy from branch > /docs)."
    )


if __name__ == "__main__":
    main()
