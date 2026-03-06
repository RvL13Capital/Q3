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

DB_PATH = Path(__file__).parent.parent.parent / "data" / "q3.duckdb"


@st.cache_resource
def get_conn():
    return get_connection(DB_PATH)


@st.cache_data(ttl=300)
def load_data():
    conn = get_conn()
    portfolio = get_latest_portfolio(conn)
    scores    = get_latest_signal_scores(conn)
    universe  = load_universe()
    return portfolio, scores, universe


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

            # Merge signal scores
            if not scores.empty:
                merged = portfolio.merge(
                    scores[["ticker", "composite_score", "quality_score",
                            "crowding_score", "physical_norm"]],
                    on="ticker", how="left"
                )
            else:
                merged = portfolio.copy()

            display_cols = ["ticker", "weight", "composite_score",
                            "quality_score", "crowding_score"]
            display_cols = [c for c in display_cols if c in merged.columns]
            display_df = merged[display_cols].copy()
            display_df["weight"] = display_df["weight"].map(lambda x: f"{x:.1%}")

            st.dataframe(
                display_df.sort_values("composite_score", ascending=False)
                if "composite_score" in display_df.columns
                else display_df,
                use_container_width=True,
            )

            # Bucket allocation pie
            if "primary_bucket" in portfolio.columns or "bucket_id" in portfolio.columns:
                bucket_col = "primary_bucket" if "primary_bucket" in portfolio.columns else "bucket_id"
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
                q_data = {
                    "Sub-score": ["ROIC−WACC Spread", "Margin SNR", "Inflation Convexity"],
                    "Value": [
                        row.get("roic_wacc_spread"),
                        row.get("margin_snr"),
                        row.get("inflation_convexity"),
                    ],
                }
                st.table(pd.DataFrame(q_data))

                st.subheader("Crowding Breakdown")
                c_data = {
                    "Component": ["ETF Correlation (60d)", "Trends (normalized)", "Short Interest"],
                    "Score": [
                        row.get("etf_correlation"),
                        row.get("trends_norm"),
                        row.get("short_pct"),
                    ],
                }
                st.table(pd.DataFrame(c_data))

                st.subheader("Sizing")
                s_data = {
                    "Metric": ["μ estimate", "σ estimate", "Kelly 25%"],
                    "Value": [
                        f"{row.get('mu_estimate', 0):.1%}" if row.get("mu_estimate") else "—",
                        f"{row.get('sigma_estimate', 0):.1%}" if row.get("sigma_estimate") else "—",
                        f"{row.get('kelly_25pct', 0):.1%}" if row.get("kelly_25pct") else "—",
                    ],
                }
                st.table(pd.DataFrame(s_data))

    # ────────────────────────────────────────────────────────────────────────
    # TAB 3: Universe Scanner
    # ────────────────────────────────────────────────────────────────────────
    with tab3:
        if scores.empty:
            st.info("No signal scores yet.")
        else:
            # Merge with universe for region/bucket filter
            merged = scores.merge(
                universe[["ticker", "region", "primary_bucket"]],
                on="ticker", how="left"
            )

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

            filtered = merged[
                merged["region"].isin(region_filter)
                & merged["primary_bucket"].isin(bucket_filter)
            ].copy()

            if "composite_score" in filtered.columns:
                filtered = filtered.sort_values("composite_score", ascending=False)

            display_cols = ["ticker", "region", "primary_bucket",
                            "composite_score", "physical_norm",
                            "quality_score", "crowding_score", "entry_signal"]
            display_cols = [c for c in display_cols if c in filtered.columns]

            st.dataframe(
                filtered[display_cols].reset_index(drop=True),
                use_container_width=True,
            )

    # ────────────────────────────────────────────────────────────────────────
    # TAB 4: Exit Monitor
    # ────────────────────────────────────────────────────────────────────────
    with tab4:
        if portfolio.empty or scores.empty:
            st.info("No portfolio or signal data yet.")
        else:
            port_tickers = portfolio["ticker"].tolist()
            port_scores  = scores[scores["ticker"].isin(port_tickers)].copy()

            crowd_exit = 0.75
            watch_thr  = 0.55

            def status_label(crowd):
                if pd.isna(crowd):
                    return "⚪ No data"
                if crowd >= crowd_exit:
                    return "🔴 EXIT"
                if crowd >= watch_thr:
                    return "🟡 WATCH"
                return "🟢 OK"

            if not port_scores.empty:
                port_scores["status"] = port_scores["crowding_score"].apply(status_label)
                display = port_scores[["ticker", "crowding_score", "quality_score",
                                       "composite_score", "status"]].copy()
                st.dataframe(display.sort_values("crowding_score", ascending=False),
                             use_container_width=True)

                st.subheader("Crowding Score Distribution (Held Positions)")
                crowd_data = port_scores[["ticker", "crowding_score"]].dropna()
                if not crowd_data.empty:
                    st.bar_chart(crowd_data.set_index("ticker")["crowding_score"])

                n_red  = (port_scores["crowding_score"] >= crowd_exit).sum()
                n_yel  = ((port_scores["crowding_score"] >= watch_thr) & (port_scores["crowding_score"] < crowd_exit)).sum()
                if n_red:
                    st.error(f"🔴 {n_red} position(s) have triggered the EXIT threshold!")
                if n_yel:
                    st.warning(f"🟡 {n_yel} position(s) are approaching the exit threshold.")


if __name__ == "__main__":
    run_dashboard()
