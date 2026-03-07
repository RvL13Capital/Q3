"""
Export latest DB tables to CSV snapshots so the dashboard can be served
from Streamlit Community Cloud (which has no access to the DuckDB file).

Usage (run after pipeline):
  python src/reporting/export_snapshot.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.db import get_connection, get_latest_portfolio, get_latest_signal_scores

SNAPSHOT_DIR = Path(__file__).parent.parent.parent / "snapshots"


def export_snapshot() -> None:
    SNAPSHOT_DIR.mkdir(exist_ok=True)

    conn = get_connection()

    portfolio = get_latest_portfolio(conn)
    scores    = get_latest_signal_scores(conn)

    portfolio.to_csv(SNAPSHOT_DIR / "latest_portfolio.csv", index=False)
    scores.to_csv(SNAPSHOT_DIR / "latest_scores.csv", index=False)

    conn.close()

    print(f"Exported: {len(portfolio)} portfolio rows, {len(scores)} score rows → {SNAPSHOT_DIR}")


if __name__ == "__main__":
    export_snapshot()
