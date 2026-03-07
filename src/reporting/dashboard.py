"""
Streamlit dashboard — 4 tabs:
  1. Portfolio Overview
  2. Signal Deep Dive (per-stock)
  3. Universe Scanner
  4. Exit Monitor / Alerts

Run: streamlit run src/reporting/dashboard.py
"""
import sys
from pathlib import Path

# Make sure src/ is on the path when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from src.data.db       import get_connection, get_latest_portfolio, get_latest_signal_scores
from src.data.universe import load_universe

DB_PATH       = Path(__file__).parent.parent.parent / "data" / "q3.duckdb"
SNAPSHOT_DIR  = Path(__file__).parent.parent.parent / "snapshots"

# Single source of truth for exit/watch thresholds used by Tab 4
# (matches params["signals"]["crowding_exit_threshold"] and
#  params["reporting"]["watch_crowding_threshold"])
_CROWD_EXIT = 0.75
_WATCH_THR  = 0.55

_SCORE_COLS = ["ticker", "composite_score", "quality_score", "crowding_score", "physical_norm"]


@st.cache_resource
def get_conn():
    return get_connection(DB_PATH)


@st.cache_data(ttl=300)
def load_data():
    """Load portfolio and scores from DuckDB if available, else fall back to CSV snapshots."""
    if DB_PATH.exists():
        conn      = get_conn()
        portfolio = get_latest_portfolio(conn)
        scores    = get_latest_signal_scores(conn)
    else:
        port_csv   = SNAPSHOT_DIR / "latest_portfolio.csv"
        scores_csv = SNAPSHOT_DIR / "latest_scores.csv"
        portfolio  = pd.read_csv(port_csv)   if port_csv.exists()   else pd.DataFrame()
        scores     = pd.read_csv(scores_csv) if scores_csv.exists() else pd.DataFrame()

    universe = load_universe()
    return portfolio, scores, universe


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

def _color_crowding(val):
    if pd.isna(val):
        return ""
    if val >= 0.75:
        return "background-color: #ffcccc"
    if val >= 0.55:
        return "background-color: #fff3cc"
    return "background-color: #ccffcc"


def _color_composite(val):
    if pd.isna(val):
        return ""
    if val >= 0.65:
        return "background-color: #ccffcc"
    if val >= 0.55:
        return "background-color: #e6f7ff"
    return ""


# ---------------------------------------------------------------------------
# Pure data-transformation helpers (testable without Streamlit)
# ---------------------------------------------------------------------------

def status_label(crowd, crowd_exit: float = _CROWD_EXIT, watch_thr: float = _WATCH_THR) -> str:
    """Return an emoji status string for a crowding score."""
    if pd.isna(crowd):
        return "⚪ No data"
    if crowd >= crowd_exit:
        return "🔴 EXIT"
    if crowd >= watch_thr:
        return "🟡 WATCH"
    return "🟢 OK"


def _fmt_sizing(val) -> str:
    """Format a sizing metric (mu/sigma/kelly) as percentage, or '—' if missing."""
    return f"{val:.1%}" if pd.notna(val) else "—"


def build_portfolio_display(portfolio: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """
    Merge live signal scores into the portfolio table.

    portfolio.composite_score is the stale value saved at construction time;
    scores has the current value. Drop the stale column before merging so the
    result has a single unambiguous composite_score column (no _x/_y suffixes).
    """
    if scores.empty:
        merged = portfolio.copy()
    else:
        port = portfolio.drop(columns=["composite_score"], errors="ignore")
        merged = port.merge(scores[_SCORE_COLS], on="ticker", how="left")

    display_cols = ["ticker", "weight", "composite_score", "quality_score", "crowding_score"]
    display_cols = [c for c in display_cols if c in merged.columns]
    display_df = merged[display_cols].copy()
    display_df["weight"] = display_df["weight"].map(lambda x: f"{x:.1%}")

    if "composite_score" in display_df.columns:
        display_df = display_df.sort_values("composite_score", ascending=False)

    return display_df.reset_index(drop=True)


def build_scanner_display(scores: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Merge scores with universe metadata for the Universe Scanner tab."""
    merged = scores.merge(
        universe[["ticker", "region", "primary_bucket"]],
        on="ticker", how="left",
    )
    if "composite_score" in merged.columns:
        merged = merged.sort_values("composite_score", ascending=False)

    display_cols = ["ticker", "region", "primary_bucket",
                    "composite_score", "physical_norm",
                    "quality_score", "crowding_score", "entry_signal"]
    display_cols = [c for c in display_cols if c in merged.columns]
    return merged[display_cols].reset_index(drop=True)


def build_exit_monitor_display(
    portfolio: pd.DataFrame,
    scores: pd.DataFrame,
    crowd_exit: float = _CROWD_EXIT,
    watch_thr: float  = _WATCH_THR,
) -> tuple[pd.DataFrame, int, int]:
    """
    Build the Exit Monitor display table.

    Returns (display_df, n_red, n_yellow) where n_red/n_yellow are alert counts.
    Returns an empty DataFrame and zero counts when portfolio or scores is empty.
    """
    if portfolio.empty or scores.empty:
        return pd.DataFrame(), 0, 0

    port_tickers = portfolio["ticker"].tolist()
    port_scores  = scores[scores["ticker"].isin(port_tickers)].copy()

    if port_scores.empty:
        return pd.DataFrame(), 0, 0

    port_scores["status"] = port_scores["crowding_score"].apply(
        lambda c: status_label(c, crowd_exit, watch_thr)
    )
    display = port_scores[
        ["ticker", "crowding_score", "quality_score", "composite_score", "status"]
    ].copy().sort_values("crowding_score", ascending=False).reset_index(drop=True)

    n_red = int((port_scores["crowding_score"] >= crowd_exit).sum())
    n_yel = int(
        ((port_scores["crowding_score"] >= watch_thr) &
         (port_scores["crowding_score"] <  crowd_exit)).sum()
    )
    return display, n_red, n_yel


# ---------------------------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------------------------

def run_dashboard():
    st.set_page_config(
        page_title="EARKE Quant 3.0",
        page_icon="📊",
        layout="wide",
    )
    st.title("EARKE Quant 3.0 — Megatrend Monitor")

    portfolio, scores, universe = load_data()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Portfolio", "🔍 Signal Deep Dive", "🌍 Universe Scanner", "🚨 Exit Monitor"
    ])

    # ────────────────────────────────────────────────────────────────────────
    # TAB 1: Portfolio Overview
    # ────────────────────────────────────────────────────────────────────────
    with tab1:
        if portfolio.empty:
            st.info("No portfolio snapshot yet. Run `python src/main.py` to generate one.")
        else:
            invested = portfolio["weight"].sum()
            cash     = 1.0 - invested
            n_pos    = len(portfolio)

            col1, col2, col3 = st.columns(3)
            col1.metric("Positions", n_pos)
            col2.metric("Invested", f"{invested:.1%}")
            col3.metric("Cash", f"{cash:.1%}")

            st.dataframe(
                build_portfolio_display(portfolio, scores),
                use_container_width=True,
            )

            # Bucket allocation chart
            bucket_col = ("primary_bucket" if "primary_bucket" in portfolio.columns
                          else "bucket_id" if "bucket_id" in portfolio.columns
                          else None)
            if bucket_col:
                bucket_agg = portfolio.groupby(bucket_col)["weight"].sum().reset_index()
                st.subheader("Allocation by Megatrend Bucket")
                try:
                    import altair as alt
                    chart = alt.Chart(bucket_agg).mark_arc().encode(
                        theta=alt.Theta("weight:Q"),
                        color=alt.Color(f"{bucket_col}:N"),
                        tooltip=[bucket_col, alt.Tooltip("weight:Q", format=".1%")],
                    )
                    st.altair_chart(chart, use_container_width=True)
                except ImportError:
                    st.bar_chart(bucket_agg.set_index(bucket_col)["weight"])

    # ────────────────────────────────────────────────────────────────────────
    # TAB 2: Signal Deep Dive
    # ────────────────────────────────────────────────────────────────────────
    with tab2:
        if scores.empty:
            st.info("No signal scores yet.")
        else:
            all_tickers = scores["ticker"].tolist()
            selected = st.selectbox("Select ticker", all_tickers)

            if selected:
                row = scores[scores["ticker"] == selected].iloc[0]

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Composite", f"{row.get('composite_score', 0):.3f}")
                col2.metric("Physical",  f"{row.get('physical_norm', 0):.3f}")
                col3.metric("Quality",   f"{row.get('quality_score', 0):.3f}")
                col4.metric("Crowding",  f"{row.get('crowding_score', 0):.3f}")

                st.subheader("Quality Breakdown")
                st.table(pd.DataFrame({
                    "Sub-score": ["ROIC−WACC Spread", "Margin SNR", "Inflation Convexity"],
                    "Value": [
                        row.get("roic_wacc_spread"),
                        row.get("margin_snr"),
                        row.get("inflation_convexity"),
                    ],
                }))

                st.subheader("Crowding Breakdown")
                st.table(pd.DataFrame({
                    "Component": ["ETF Correlation (60d)", "Trends (normalized)", "Short Interest"],
                    "Score": [
                        row.get("etf_correlation"),
                        row.get("trends_norm"),
                        row.get("short_pct"),
                    ],
                }))

                st.subheader("Sizing")
                st.table(pd.DataFrame({
                    "Metric": ["μ estimate", "σ estimate", "Kelly 25%"],
                    "Value": [
                        _fmt_sizing(row.get("mu_estimate")),
                        _fmt_sizing(row.get("sigma_estimate")),
                        _fmt_sizing(row.get("kelly_25pct")),
                    ],
                }))

    # ────────────────────────────────────────────────────────────────────────
    # TAB 3: Universe Scanner
    # ────────────────────────────────────────────────────────────────────────
    with tab3:
        if scores.empty:
            st.info("No signal scores yet.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                region_filter = st.multiselect(
                    "Region", ["EU", "US", "CA"],
                    default=["EU", "US", "CA"]
                )
            with col2:
                bucket_filter = st.multiselect(
                    "Bucket",
                    ["grid", "nuclear", "defense", "water", "critical_materials", "ai_infra"],
                    default=["grid", "nuclear", "defense", "water", "critical_materials", "ai_infra"]
                )

            scanner_df = build_scanner_display(scores, universe)
            filtered   = scanner_df[
                scanner_df["region"].isin(region_filter)
                & scanner_df["primary_bucket"].isin(bucket_filter)
            ]
            st.dataframe(filtered, use_container_width=True)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 4: Exit Monitor
    # ────────────────────────────────────────────────────────────────────────
    with tab4:
        display, n_red, n_yel = build_exit_monitor_display(portfolio, scores)

        if display.empty:
            st.info("No portfolio or signal data yet.")
        else:
            st.dataframe(display, use_container_width=True)

            st.subheader("Crowding Score Distribution (Held Positions)")
            crowd_data = display[["ticker", "crowding_score"]].dropna()
            if not crowd_data.empty:
                st.bar_chart(crowd_data.set_index("ticker")["crowding_score"])

            if n_red:
                st.error(f"🔴 {n_red} position(s) have triggered the EXIT threshold!")
            if n_yel:
                st.warning(f"🟡 {n_yel} position(s) are approaching the exit threshold.")


if __name__ == "__main__":
    run_dashboard()
