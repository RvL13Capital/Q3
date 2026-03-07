"""
Streamlit dashboard — futuristic command-center layout:
  Sidebar nav → OVERVIEW · SCANNER · SIGNALS · ALERTS

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

_SECTIONS: dict[str, str] = {
    "OVERVIEW": "⬡  OVERVIEW",
    "SCANNER":  "◈  SCANNER",
    "SIGNALS":  "◉  SIGNALS",
    "ALERTS":   "⚠  ALERTS",
}


@st.cache_resource
def get_conn():
    return get_connection(DB_PATH)


@st.cache_data(ttl=300)
def load_data():
    """Load portfolio and scores from DuckDB if available, else fall back to CSV snapshots.

    Returns (portfolio, scores, universe, is_mock) where is_mock=True means the
    data comes from the committed seed CSVs, not a live pipeline run.
    """
    if DB_PATH.exists():
        conn      = get_conn()
        portfolio = get_latest_portfolio(conn)
        scores    = get_latest_signal_scores(conn)

    if not DB_PATH.exists() or portfolio.empty or scores.empty:
        port_csv   = SNAPSHOT_DIR / "latest_portfolio.csv"
        scores_csv = SNAPSHOT_DIR / "latest_scores.csv"
        portfolio  = pd.read_csv(port_csv)   if port_csv.exists()   else pd.DataFrame()
        scores     = pd.read_csv(scores_csv) if scores_csv.exists() else pd.DataFrame()
        is_mock    = True
    else:
        is_mock    = False

    universe = load_universe()
    return portfolio, scores, universe, is_mock


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
# CSS — futuristic command-center theme
# ---------------------------------------------------------------------------

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

/* ── Base ────────────────────────────────────────────────────── */
html, body, .stApp {
    background: #04080f !important;
    color: #a8c4e0;
    font-family: 'Rajdhani', sans-serif;
}
*, *::before, *::after { box-sizing: border-box; }

/* ── Sidebar ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #060b14 !important;
    border-right: 1px solid #0d2a4a !important;
}
[data-testid="stSidebar"] * { color: #7aaccc !important; }
[data-testid="stSidebarNav"] { display: none; }

/* ── Radio nav ───────────────────────────────────────────────── */
div[role="radiogroup"] { gap: 6px; }
div[role="radiogroup"] label {
    display: block;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a8aba !important;
    padding: 8px 14px !important;
    border: 1px solid #0d2a4a;
    border-radius: 2px;
    cursor: pointer;
    transition: all 0.15s;
    margin-bottom: 2px;
}
div[role="radiogroup"] label:hover {
    border-color: #00e5ff !important;
    color: #00e5ff !important;
    background: #00e5ff0a;
}
div[role="radiogroup"] label[data-baseweb="radio"] { background: #00e5ff12; }

/* ── Hide default streamlit chrome ──────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem !important; padding-bottom: 2rem !important; }

/* ── Typography ──────────────────────────────────────────────── */
h1 {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700; font-size: 1.5rem !important;
    color: #00e5ff !important;
    text-transform: uppercase; letter-spacing: 0.2em;
    text-shadow: 0 0 24px #00e5ff50;
    border-bottom: 1px solid #0d2a4a;
    padding-bottom: 8px; margin-bottom: 0 !important;
}
h2 {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600; font-size: 0.8rem !important;
    color: #5a9abb !important;
    text-transform: uppercase; letter-spacing: 0.25em;
    margin: 18px 0 6px !important;
}
h3 {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #4a8aac !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin: 14px 0 4px !important;
}

/* ── Metrics ─────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #070d1a !important;
    border: 1px solid #0d2a4a !important;
    border-top: 2px solid #00e5ff !important;
    padding: 10px 14px !important;
    border-radius: 2px;
}
[data-testid="stMetricLabel"] p {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.62rem !important;
    color: #5a9abb !important;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1.5rem !important;
    color: #00e5ff !important;
}
[data-testid="stMetricDelta"] { font-size: 0.7rem !important; }

/* ── Dataframe / table ───────────────────────────────────────── */
[data-testid="stDataFrame"] iframe,
[data-testid="stDataFrame"] div {
    background: #04080f !important;
    color: #a8c4e0 !important;
}
.stDataFrame { border: 1px solid #0d2a4a !important; }

/* ── Selectbox / multiselect ─────────────────────────────────── */
[data-baseweb="select"] > div {
    background: #070d1a !important;
    border: 1px solid #0d2a4a !important;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
}
[data-baseweb="select"] span { color: #7aacc0 !important; }

/* ── Alerts ──────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 2px !important;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
}

/* ── Divider ─────────────────────────────────────────────────── */
hr { border-color: #0d2a4a !important; margin: 10px 0 !important; }

/* ── Bar chart axis ──────────────────────────────────────────── */
[data-testid="stVegaLiteChart"] { border: 1px solid #0d2a4a; }

/* ── Top nav strip ───────────────────────────────────────────── */
.q3-nav {
    display: flex;
    gap: 4px;
    margin: 0 0 18px;
    border-bottom: 1px solid #0d2a4a;
    padding-bottom: 8px;
}
.q3-nav-item {
    flex: 1;
    text-align: center;
    padding: 7px 4px;
    background: transparent;
    border: 1px solid #0d2a4a;
    border-radius: 2px;
    color: #5a9abb;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-decoration: none;
    text-transform: uppercase;
    transition: all 0.15s;
}
.q3-nav-item:hover {
    border-color: #00e5ff;
    color: #00e5ff;
    background: #00e5ff0a;
}
.q3-nav-active {
    border-color: #00e5ff !important;
    border-top: 2px solid #00e5ff !important;
    color: #00e5ff !important;
    background: #00e5ff12 !important;
}

/* ── Scrollbar ───────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #04080f; }
::-webkit-scrollbar-thumb { background: #0d2a4a; border-radius: 2px; }
</style>
"""

# ---------------------------------------------------------------------------
# HTML component helpers
# ---------------------------------------------------------------------------

def _stat_card(label: str, value: str, sub: str = "", accent: str = "#00e5ff") -> str:
    return (
        f'<div style="background:#070d1a;border:1px solid #0d2a4a;border-top:2px solid {accent};'
        f'padding:12px 16px;border-radius:2px;min-width:120px">'
        f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.6rem;'
        f'letter-spacing:0.2em;text-transform:uppercase;color:#5a9abb">{label}</div>'
        f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:1.55rem;'
        f'color:{accent};margin:4px 0 2px">{value}</div>'
        f'<div style="font-size:0.7rem;color:#4a8aac">{sub}</div>'
        f'</div>'
    )


def _section_header(title: str, subtitle: str = "") -> str:
    return (
        f'<div style="border-left:3px solid #00e5ff;padding:4px 0 4px 12px;margin-bottom:14px">'
        f'<span style="font-family:\'Rajdhani\',sans-serif;font-weight:700;font-size:0.95rem;'
        f'color:#00e5ff;text-transform:uppercase;letter-spacing:0.2em">{title}</span>'
        + (f'<span style="font-size:0.72rem;color:#5a9abb;margin-left:12px">{subtitle}</span>' if subtitle else "")
        + '</div>'
    )


def _score_bar(label: str, value: float, max_val: float = 1.0, color: str = "#00e5ff") -> str:
    pct = max(0.0, min(1.0, value / max_val if max_val else 0)) * 100
    return (
        f'<div style="margin-bottom:7px">'
        f'<div style="display:flex;justify-content:space-between;'
        f'font-family:\'Share Tech Mono\',monospace;font-size:0.68rem;margin-bottom:3px">'
        f'<span style="color:#6aabcb">{label}</span>'
        f'<span style="color:{color}">{value:.3f}</span></div>'
        f'<div style="background:#0a1828;border-radius:1px;height:4px">'
        f'<div style="width:{pct:.1f}%;height:100%;background:{color};'
        f'box-shadow:0 0 6px {color}80;border-radius:1px"></div>'
        f'</div></div>'
    )


def _mock_banner() -> str:
    return (
        '<div style="background:#0a0f04;border:1px solid #4a7a0040;border-left:3px solid #8aba20;'
        'padding:8px 14px;font-family:\'Share Tech Mono\',monospace;font-size:0.7rem;'
        'color:#6a9a30;letter-spacing:0.08em;margin-bottom:10px">'
        '⚠ MOCK DATA — randomly generated placeholders · will be replaced after first pipeline run'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Layout — sidebar navigation
# ---------------------------------------------------------------------------

def run_dashboard():
    st.set_page_config(
        page_title="EARKE Q3 // COMMAND CENTER",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    portfolio, scores, universe, is_mock = load_data()

    # ── pre-compute shared aggregates ────────────────────────────────────
    invested   = portfolio["weight"].sum() if not portfolio.empty else 0.0
    n_pos      = len(portfolio)
    n_universe = len(scores)
    n_entry    = int(scores["entry_signal"].sum()) if "entry_signal" in scores.columns else 0
    avg_comp   = scores["composite_score"].mean() if not scores.empty else 0.0
    avg_crowd  = scores["crowding_score"].mean()  if not scores.empty else 0.0
    display_alerts, n_red, n_yel = build_exit_monitor_display(portfolio, scores)

    # ── sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div style="font-family:\'Share Tech Mono\',monospace;font-size:1.05rem;'
            'color:#00e5ff;letter-spacing:0.2em;text-shadow:0 0 12px #00e5ff60;'
            'padding:8px 0 4px">◈ EARKE Q3</div>'
            '<div style="font-size:0.62rem;color:#4a8aac;letter-spacing:0.25em;'
            'margin-bottom:16px">MEGATREND COMMAND CENTER</div>',
            unsafe_allow_html=True,
        )

        data_src = "MOCK DATA" if is_mock else "LIVE · DB"
        src_color = "#8aba20" if is_mock else "#00ff88"
        st.markdown(
            f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.62rem;'
            f'color:{src_color};letter-spacing:0.15em;margin-bottom:14px">'
            f'● DATA SOURCE: {data_src}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        _tab = st.query_params.get("tab", "OVERVIEW")
        if _tab not in _SECTIONS:
            _tab = "OVERVIEW"
        _sidebar_pick = st.radio(
            "NAVIGATION",
            list(_SECTIONS.values()),
            index=list(_SECTIONS.keys()).index(_tab),
            label_visibility="collapsed",
        )
        if _sidebar_pick != _SECTIONS[_tab]:
            _new_key = {v: k for k, v in _SECTIONS.items()}[_sidebar_pick]
            st.query_params["tab"] = _new_key
            st.rerun()
        section = _SECTIONS[_tab]
        st.markdown("---")

        # sidebar system stats
        st.markdown(
            '<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.62rem;'
            'letter-spacing:0.15em;color:#4a8aac">SYSTEM STATS</div>',
            unsafe_allow_html=True,
        )
        for lbl, val in [
            ("UNIVERSE", f"{n_universe} tickers"),
            ("POSITIONS", f"{n_pos} held"),
            ("INVESTED",  f"{invested:.1%}"),
            ("CASH",      f"{1-invested:.1%}"),
            ("ENTRIES",   f"{n_entry} signals"),
            ("EXIT ALERTS", f"{n_red} red / {n_yel} amber"),
        ]:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-family:\'Share Tech Mono\',monospace;font-size:0.65rem;'
                f'padding:3px 0;border-bottom:1px solid #0a1828">'
                f'<span style="color:#5a9abb">{lbl}</span>'
                f'<span style="color:#4a9aba">{val}</span></div>',
                unsafe_allow_html=True,
            )

    # ── global HUD strip (always rendered) ───────────────────────────────
    st.markdown(
        '<div style="font-family:\'Rajdhani\',sans-serif;font-weight:700;font-size:1.35rem;'
        'color:#00e5ff;text-transform:uppercase;letter-spacing:0.25em;'
        'text-shadow:0 0 20px #00e5ff40;margin-bottom:8px">'
        'EARKE QUANT 3.0 — MEGATREND MONITOR</div>',
        unsafe_allow_html=True,
    )

    if is_mock:
        st.markdown(_mock_banner(), unsafe_allow_html=True)

    # top metric strip
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    accent_map = ["#00e5ff", "#00ff88", "#00ff88", "#ffb800", "#ff6040", "#ff3060"]
    for col, lbl, val, acc in zip(
        [c1, c2, c3, c4, c5, c6],
        ["POSITIONS", "INVESTED", "AVG COMPOSITE", "AVG CROWDING", "EXIT ALERTS", "ENTRY SIGNALS"],
        [n_pos, f"{invested:.1%}", f"{avg_comp:.3f}", f"{avg_crowd:.3f}", n_red, n_entry],
        accent_map,
    ):
        with col:
            st.markdown(_stat_card(lbl, str(val), accent=acc), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── persistent top nav strip ──────────────────────────────────────────
    _nav_html = '<div class="q3-nav">'
    for _key, _label in _SECTIONS.items():
        _cls = "q3-nav-item q3-nav-active" if _key == _tab else "q3-nav-item"
        _nav_html += f'<a href="?tab={_key}" class="{_cls}">{_label}</a>'
    _nav_html += '</div>'
    st.markdown(_nav_html, unsafe_allow_html=True)

    # ── OVERVIEW ─────────────────────────────────────────────────────────
    if section == "⬡  OVERVIEW":
        st.markdown(_section_header("PORTFOLIO OVERVIEW", f"{n_pos} positions · {invested:.1%} deployed"), unsafe_allow_html=True)

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown('<div style="font-size:0.65rem;font-family:\'Share Tech Mono\',monospace;'
                        'color:#4a8aac;letter-spacing:0.2em;margin-bottom:4px">HOLDINGS</div>',
                        unsafe_allow_html=True)
            if portfolio.empty:
                st.info("No portfolio data.")
            else:
                st.dataframe(build_portfolio_display(portfolio, scores), use_container_width=True, height=360)

        with col_right:
            st.markdown('<div style="font-size:0.65rem;font-family:\'Share Tech Mono\',monospace;'
                        'color:#4a8aac;letter-spacing:0.2em;margin-bottom:4px">BUCKET ALLOCATION</div>',
                        unsafe_allow_html=True)
            bucket_col = ("primary_bucket" if "primary_bucket" in portfolio.columns
                          else "bucket_id" if "bucket_id" in portfolio.columns else None)
            if bucket_col and not portfolio.empty:
                bucket_agg = portfolio.groupby(bucket_col)["weight"].sum().reset_index()
                st.bar_chart(bucket_agg.set_index(bucket_col)["weight"], use_container_width=True, height=200)

                # per-bucket stats table
                if not scores.empty:
                    sc_uni = scores.merge(universe[["ticker", "primary_bucket"]], on="ticker", how="left")
                    bkt_stats = (sc_uni.groupby("primary_bucket")
                                 .agg(count=("ticker","count"),
                                      avg_comp=("composite_score","mean"),
                                      avg_crowd=("crowding_score","mean"))
                                 .round(3).reset_index())
                    st.markdown('<div style="font-size:0.65rem;font-family:\'Share Tech Mono\',monospace;'
                                'color:#4a8aac;letter-spacing:0.2em;margin:10px 0 4px">BUCKET SCORES</div>',
                                unsafe_allow_html=True)
                    st.dataframe(bkt_stats, use_container_width=True, hide_index=True)

        # score distribution
        if not scores.empty:
            st.markdown("---")
            st.markdown(_section_header("SCORE DISTRIBUTION", f"{n_universe} universe tickers"), unsafe_allow_html=True)
            dc1, dc2 = st.columns(2)
            with dc1:
                st.markdown('<div style="font-size:0.65rem;font-family:\'Share Tech Mono\',monospace;color:#4a8aac;letter-spacing:0.2em;margin-bottom:4px">COMPOSITE SCORE</div>', unsafe_allow_html=True)
                hist_comp = scores["composite_score"].value_counts(bins=15, sort=False).sort_index()
                st.bar_chart(hist_comp, use_container_width=True, height=130)
            with dc2:
                st.markdown('<div style="font-size:0.65rem;font-family:\'Share Tech Mono\',monospace;color:#4a8aac;letter-spacing:0.2em;margin-bottom:4px">CROWDING SCORE</div>', unsafe_allow_html=True)
                hist_crowd = scores["crowding_score"].value_counts(bins=15, sort=False).sort_index()
                st.bar_chart(hist_crowd, use_container_width=True, height=130)

    # ── SCANNER ──────────────────────────────────────────────────────────
    elif section == "◈  SCANNER":
        st.markdown(_section_header("UNIVERSE SCANNER", f"{n_universe} instruments tracked"), unsafe_allow_html=True)

        f1, f2, f3 = st.columns([1, 2, 1])
        with f1:
            region_filter = st.multiselect("REGION", ["EU", "US", "CA"], default=["EU", "US", "CA"])
        with f2:
            bucket_filter = st.multiselect(
                "BUCKET",
                ["grid", "nuclear", "defense", "water", "critical_materials", "ai_infra"],
                default=["grid", "nuclear", "defense", "water", "critical_materials", "ai_infra"],
            )
        with f3:
            min_comp = st.slider("MIN COMPOSITE", 0.0, 1.0, 0.0, 0.05)

        if scores.empty:
            st.info("No signal scores.")
        else:
            scanner_df = build_scanner_display(scores, universe)
            mask = (
                scanner_df["region"].isin(region_filter)
                & scanner_df["primary_bucket"].isin(bucket_filter)
                & (scanner_df["composite_score"] >= min_comp)
            )
            filtered = scanner_df[mask]

            # summary row
            sc1, sc2, sc3, sc4 = st.columns(4)
            for col, lbl, val, acc in zip(
                [sc1, sc2, sc3, sc4],
                ["SHOWING", "AVG COMPOSITE", "ENTRY SIGNALS", "HIGH CROWD ≥0.75"],
                [
                    len(filtered),
                    f'{filtered["composite_score"].mean():.3f}' if not filtered.empty else "—",
                    int(filtered["entry_signal"].sum()) if "entry_signal" in filtered.columns else "—",
                    int((filtered["crowding_score"] >= 0.75).sum()) if not filtered.empty else "—",
                ],
                ["#00e5ff", "#00ff88", "#ffb800", "#ff3060"],
            ):
                col.markdown(_stat_card(lbl, str(val), accent=acc), unsafe_allow_html=True)

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.dataframe(filtered, use_container_width=True, height=480)

            # top 10 entry signals callout
            if "entry_signal" in filtered.columns:
                top_entries = filtered[filtered["entry_signal"] == True].nlargest(10, "composite_score")
                if not top_entries.empty:
                    st.markdown("---")
                    st.markdown(_section_header("TOP ENTRY SIGNALS"), unsafe_allow_html=True)
                    cols = st.columns(min(5, len(top_entries)))
                    for i, (_, r) in enumerate(top_entries.head(5).iterrows()):
                        crowd_c = "#ff3060" if r["crowding_score"] >= 0.75 else "#ffb800" if r["crowding_score"] >= 0.55 else "#00ff88"
                        cols[i].markdown(
                            _stat_card(r["ticker"], f'{r["composite_score"]:.3f}',
                                       sub=f'crowd {r["crowding_score"]:.2f}', accent="#00ff88"),
                            unsafe_allow_html=True,
                        )

    # ── SIGNALS ──────────────────────────────────────────────────────────
    elif section == "◉  SIGNALS":
        st.markdown(_section_header("SIGNAL DEEP DIVE", "per-instrument breakdown"), unsafe_allow_html=True)

        if scores.empty:
            st.info("No signal scores.")
        else:
            in_port = portfolio["ticker"].tolist() if not portfolio.empty else []
            all_tickers = scores["ticker"].tolist()
            selected = st.selectbox("SELECT INSTRUMENT", all_tickers,
                                    format_func=lambda t: f"◆ {t}" if t in in_port else t)

            if selected:
                row = scores[scores["ticker"] == selected].iloc[0]
                comp  = row.get("composite_score", 0) or 0
                phys  = row.get("physical_norm",   0) or 0
                qual  = row.get("quality_score",   0) or 0
                crowd = row.get("crowding_score",  0) or 0
                crowd_acc = "#ff3060" if crowd >= 0.75 else "#ffb800" if crowd >= 0.55 else "#00ff88"

                st.markdown("---")
                mc1, mc2, mc3, mc4 = st.columns(4)
                for col, lbl, val, acc in zip(
                    [mc1, mc2, mc3, mc4],
                    ["COMPOSITE", "PHYSICAL", "QUALITY", "CROWDING"],
                    [f"{comp:.3f}", f"{phys:.3f}", f"{qual:.3f}", f"{crowd:.3f}"],
                    ["#00e5ff", "#00e5ff", "#00ff88", crowd_acc],
                ):
                    col.markdown(_stat_card(lbl, val, accent=acc), unsafe_allow_html=True)

                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

                left, mid, right = st.columns(3)

                with left:
                    st.markdown(_section_header("SCORE PROFILE"), unsafe_allow_html=True)
                    bars_html = "".join([
                        _score_bar("Composite",  comp,  color="#00e5ff"),
                        _score_bar("Physical",   phys,  color="#00c8ff"),
                        _score_bar("Quality",    qual,  color="#00ff88"),
                        _score_bar("Crowding",   crowd, color=crowd_acc),
                    ])
                    st.markdown(
                        f'<div style="background:#070d1a;border:1px solid #0d2a4a;padding:14px;border-radius:2px">{bars_html}</div>',
                        unsafe_allow_html=True,
                    )

                with mid:
                    st.markdown(_section_header("QUALITY"), unsafe_allow_html=True)
                    for lbl, key in [
                        ("ROIC − WACC Spread",   "roic_wacc_spread"),
                        ("Margin SNR",            "margin_snr"),
                        ("Inflation Convexity",   "inflation_convexity"),
                    ]:
                        v = row.get(key)
                        v_str = f"{v:.4f}" if pd.notna(v) else "—"
                        st.markdown(
                            f'<div style="display:flex;justify-content:space-between;'
                            f'font-family:\'Share Tech Mono\',monospace;font-size:0.7rem;'
                            f'padding:6px 0;border-bottom:1px solid #0a1828">'
                            f'<span style="color:#5a9abb">{lbl}</span>'
                            f'<span style="color:#00ff88">{v_str}</span></div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown(_section_header("CROWDING"), unsafe_allow_html=True)
                    for lbl, key in [
                        ("ETF Corr (60d)",   "etf_correlation"),
                        ("Trends Norm",      "trends_norm"),
                        ("Short Interest",   "short_pct"),
                    ]:
                        v = row.get(key)
                        v_str = f"{v:.4f}" if pd.notna(v) else "—"
                        c = "#ff3060" if (pd.notna(v) and v >= 0.75) else "#ffb800" if (pd.notna(v) and v >= 0.55) else "#7aacc0"
                        st.markdown(
                            f'<div style="display:flex;justify-content:space-between;'
                            f'font-family:\'Share Tech Mono\',monospace;font-size:0.7rem;'
                            f'padding:6px 0;border-bottom:1px solid #0a1828">'
                            f'<span style="color:#5a9abb">{lbl}</span>'
                            f'<span style="color:{c}">{v_str}</span></div>',
                            unsafe_allow_html=True,
                        )

                with right:
                    st.markdown(_section_header("SIZING"), unsafe_allow_html=True)
                    for lbl, key, fmt in [
                        ("μ Return Est",  "mu_estimate",    _fmt_sizing),
                        ("σ Volatility",  "sigma_estimate", _fmt_sizing),
                        ("Kelly 25%",     "kelly_25pct",    _fmt_sizing),
                    ]:
                        v = row.get(key)
                        v_str = fmt(v)
                        st.markdown(
                            f'<div style="display:flex;justify-content:space-between;'
                            f'font-family:\'Share Tech Mono\',monospace;font-size:0.7rem;'
                            f'padding:6px 0;border-bottom:1px solid #0a1828">'
                            f'<span style="color:#5a9abb">{lbl}</span>'
                            f'<span style="color:#ffb800">{v_str}</span></div>',
                            unsafe_allow_html=True,
                        )

                    in_portfolio = selected in in_port
                    status_txt = "IN PORTFOLIO" if in_portfolio else "NOT HELD"
                    status_c   = "#00ff88" if in_portfolio else "#5a9abb"
                    entry_sig  = row.get("entry_signal", False)
                    signal_txt = "ENTRY SIGNAL ●" if entry_sig else "NO SIGNAL ○"
                    signal_c   = "#00ff88" if entry_sig else "#5a9abb"
                    st.markdown(
                        f'<div style="background:#070d1a;border:1px solid #0d2a4a;border-radius:2px;'
                        f'padding:14px;margin-top:14px">'
                        f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.65rem;'
                        f'color:#4a8aac;letter-spacing:0.2em;margin-bottom:8px">STATUS</div>'
                        f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.85rem;'
                        f'color:{status_c};margin-bottom:6px">{status_txt}</div>'
                        f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.85rem;'
                        f'color:{signal_c}">{signal_txt}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # ── ALERTS ───────────────────────────────────────────────────────────
    elif section == "⚠  ALERTS":
        st.markdown(_section_header("EXIT MONITOR", "crowding alert system"), unsafe_allow_html=True)

        if display_alerts.empty:
            st.info("No portfolio or signal data.")
        else:
            a1, a2, a3 = st.columns(3)
            a1.markdown(_stat_card("EXIT TRIGGERED", str(n_red), sub="crowding ≥ 0.75", accent="#ff3060"), unsafe_allow_html=True)
            a2.markdown(_stat_card("WATCH ZONE",     str(n_yel), sub="crowding 0.55–0.75", accent="#ffb800"), unsafe_allow_html=True)
            a3.markdown(_stat_card("CLEAR",
                                   str(len(display_alerts) - n_red - n_yel),
                                   sub="crowding < 0.55", accent="#00ff88"), unsafe_allow_html=True)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            al, ar = st.columns([3, 2])
            with al:
                st.markdown(_section_header("POSITION CROWDING TABLE"), unsafe_allow_html=True)
                st.dataframe(display_alerts, use_container_width=True, height=400)

            with ar:
                st.markdown(_section_header("CROWDING DISTRIBUTION"), unsafe_allow_html=True)
                crowd_data = display_alerts[["ticker", "crowding_score"]].dropna().set_index("ticker")
                st.bar_chart(crowd_data, use_container_width=True, height=200)

                # threshold lines as text
                st.markdown(
                    '<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.65rem;'
                    'margin-top:10px">'
                    '<div style="color:#ff3060;margin-bottom:4px">▬ EXIT threshold: 0.75</div>'
                    '<div style="color:#ffb800;margin-bottom:4px">▬ WATCH threshold: 0.55</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

                if n_red:
                    st.markdown(
                        f'<div style="background:#1a0408;border:1px solid #ff306040;border-left:3px solid #ff3060;'
                        f'padding:10px 14px;font-family:\'Share Tech Mono\',monospace;font-size:0.72rem;'
                        f'color:#ff6080;margin-top:10px">⚠ {n_red} POSITION(S) TRIGGERED EXIT</div>',
                        unsafe_allow_html=True,
                    )
                if n_yel:
                    st.markdown(
                        f'<div style="background:#1a1204;border:1px solid #ffb80040;border-left:3px solid #ffb800;'
                        f'padding:10px 14px;font-family:\'Share Tech Mono\',monospace;font-size:0.72rem;'
                        f'color:#d09020;margin-top:6px">● {n_yel} POSITION(S) APPROACHING EXIT</div>',
                        unsafe_allow_html=True,
                    )


if __name__ == "__main__":
    run_dashboard()
