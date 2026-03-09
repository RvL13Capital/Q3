"""
Generates a Markdown weekly report summarizing pipeline run results.
File: reports/YYYY-MM-DD_weekly.md
"""
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _fmt_pct(val, decimals: int = 1) -> str:
    if val is None or pd.isna(val):
        return "—"
    return f"{val * 100:.{decimals}f}%"


def _fmt_score(val, decimals: int = 3) -> str:
    if val is None or pd.isna(val):
        return "—"
    return f"{val:.{decimals}f}"


def _fmt_ret(val, decimals: int = 2) -> str:
    if val is None or pd.isna(val):
        return "—"
    return f"{val * 100:+.{decimals}f}%"


def _traffic_light(crowding_score: Optional[float], params: dict) -> str:
    if crowding_score is None or pd.isna(crowding_score):
        return "⚪"
    watch_thr = params["reporting"].get("watch_crowding_threshold", 0.55)
    exit_thr  = params["signals"]["crowding_exit_threshold"]
    if crowding_score >= exit_thr:
        return "🔴"
    if crowding_score >= watch_thr:
        return "🟡"
    return "🟢"


def _flow_arrow(score: Optional[float]) -> str:
    if score is None or pd.isna(score):
        return "⚪"
    if score > 0.30:
        return "🟢▲"
    if score > 0.10:
        return "🟢"
    if score < -0.30:
        return "🔴▼"
    if score < -0.10:
        return "🔴"
    return "⚪"


def _fmt_spread(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{float(v):.2%}"


def _fmt_snr(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{float(v):.1f}×"


def _fmt_autocorr(v) -> str:
    """Autocorrelation delta as dimensionless coefficient (Δρ₁), 3 dp — no % sign."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{float(v):+.3f}"


def _fmt_abs(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{float(v):+.4f}"


def generate_weekly_report(
    conn,
    actions: dict,
    scored_df: pd.DataFrame,
    params: dict,
    as_of_date: str,
    flow_result: Optional[dict] = None,
    output_dir: str = "reports",
) -> str:
    """
    Generate Markdown weekly report.
    Returns path to the created file.
    """
    from src.data.db import get_latest_portfolio
    from src.data.macro import get_risk_free_rate

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fname = output_path / f"{as_of_date}_weekly.md"

    portfolio  = actions.get("new_portfolio", pd.DataFrame())
    exits      = actions.get("exits", pd.DataFrame())
    entries    = actions.get("entries", pd.DataFrame())
    watch_list = actions.get("watch_list", pd.DataFrame())
    diff       = actions.get("diff", pd.DataFrame())

    rf_us = get_risk_free_rate(conn, "US", as_of_date, params)
    rf_eu = get_risk_free_rate(conn, "EU", as_of_date, params)

    lines = [
        "# EARKE Quant 3.0 — Weekly Report",
        f"**Date:** {as_of_date}  |  **RF US:** {rf_us:.2%}  |  **RF EU:** {rf_eu:.2%}",
        "",
    ]

    # ── 1. Executive Summary ─────────────────────────────────────────────────
    lines += ["## Summary"]
    n_pos    = len(portfolio) if not portfolio.empty else 0
    invested = portfolio["weight"].sum() if not portfolio.empty else 0.0
    cash_w   = 1.0 - invested

    lines.append("⚡ **Action required this week.**" if actions.get("any_action")
                 else "✅ **No action required.** Portfolio unchanged.")
    lines += [
        "",
        f"- Positions: **{n_pos}**  |  Invested: **{_fmt_pct(invested)}**  |  Cash: **{_fmt_pct(cash_w)}**",
        f"- Exit triggers: **{exits['exit_triggered'].sum() if not exits.empty else 0}**",
        f"- New entry candidates: **{len(entries) if not entries.empty else 0}**",
        f"- Watch list (crowding elevated): **{len(watch_list) if not watch_list.empty else 0}**",
        "",
    ]

    # ── 2. Actions ────────────────────────────────────────────────────────────
    lines += ["## This Week's Actions", ""]

    if not exits.empty:
        triggered = exits[exits["exit_triggered"]]
        if not triggered.empty:
            lines += ["### ❌ Positions to Exit", ""]
            lines += ["| Ticker | Crowding | Quality | Reason |"]
            lines += ["|--------|----------|---------|--------|"]
            for _, r in triggered.iterrows():
                reasons_str = "; ".join(r.get("exit_reasons") or [])
                lines.append(
                    f"| {r['ticker']} | {_fmt_score(r.get('crowding_score'))} "
                    f"| {_fmt_score(r.get('quality_score'))} | {reasons_str} |"
                )
            lines.append("")

    if not entries.empty:
        # Kelly% column only if at least one value is non-null
        has_kelly = entries["kelly_25pct"].notna().any() if "kelly_25pct" in entries.columns else False
        lines += ["### ✅ New Entry Candidates", ""]
        if has_kelly:
            lines += ["| Ticker | Composite | Quality | Crowding | μ_base | Kelly 25% |"]
            lines += ["|--------|-----------|---------|----------|--------|-----------|"]
        else:
            lines += ["| Ticker | Composite | Quality | Crowding | μ_base |"]
            lines += ["|--------|-----------|---------|----------|--------|"]
        for _, r in entries.head(params["reporting"].get("top_candidates_count", 10)).iterrows():
            row_str = (
                f"| {r['ticker']} | {_fmt_score(r.get('composite_score'))} "
                f"| {_fmt_score(r.get('quality_score'))} "
                f"| {_fmt_score(r.get('crowding_score'))} "
                f"| {_fmt_pct(r.get('mu_estimate'))} |"
            )
            if has_kelly:
                row_str = row_str[:-1] + f" {_fmt_pct(r.get('kelly_25pct'))} |"
            lines.append(row_str)
        lines.append("")

    if not diff.empty:
        rebalances = diff[diff["action"].isin(["increase", "decrease"])]
        if not rebalances.empty:
            lines += ["### 🔄 Rebalances", ""]
            lines += ["| Ticker | Old Weight | New Weight | Action |"]
            lines += ["|--------|-----------|-----------|--------|"]
            for _, r in rebalances.iterrows():
                lines.append(
                    f"| {r['ticker']} | {_fmt_pct(r['old_weight'])} "
                    f"| {_fmt_pct(r['new_weight'])} | {r['action']} |"
                )
            lines.append("")

    # ── 3. Full Portfolio ─────────────────────────────────────────────────────
    if not portfolio.empty:
        lines += ["## Current Portfolio", ""]
        lines += ["| Ticker | Weight | Composite | Quality | Crowding | Status |"]
        lines += ["|--------|--------|-----------|---------|----------|--------|"]

        score_map = {}
        if not scored_df.empty:
            score_map = scored_df.set_index("ticker").to_dict("index")

        for _, row in portfolio.sort_values("weight", ascending=False).iterrows():
            t     = row["ticker"]
            s     = score_map.get(t, {})
            crowd = s.get("crowding_score")
            lines.append(
                f"| {t} | {_fmt_pct(row['weight'])} "
                f"| {_fmt_score(row.get('composite_score'))} "
                f"| {_fmt_score(s.get('quality_score'))} "
                f"| {_fmt_score(crowd)} "
                f"| {_traffic_light(crowd, params)} |"
            )
        lines += ["", f"*Cash reserve: {_fmt_pct(cash_w)}*", ""]

    # ── 4. Watch List ─────────────────────────────────────────────────────────
    if not watch_list.empty:
        lines += ["## ⚠️ Watch List (Crowding Elevated)", ""]
        lines += ["| Ticker | Crowding | Exit Threshold | Quality |"]
        lines += ["|--------|----------|---------------|---------|"]
        crowd_exit = params["signals"]["crowding_exit_threshold"]
        for _, r in watch_list.iterrows():
            lines.append(
                f"| {r['ticker']} | {_fmt_score(r.get('crowding_score'))} "
                f"| {crowd_exit} "
                f"| {_fmt_score(r.get('quality_score'))} |"
            )
        lines.append("")

    # ── 5. Universe Scanner (top 15 by composite) ────────────────────────────
    # Replace physical_norm (constant across all stocks) with mu_estimate (stock-specific)
    top15_tickers: set = set()
    lines += ["## Universe Top Candidates", ""]
    if not scored_df.empty:
        top = (scored_df.dropna(subset=["composite_score"])
               .sort_values("composite_score", ascending=False)
               .head(15))
        top15_tickers = set(top["ticker"].tolist())
        lines += ["| Ticker | Composite | μ_base | Quality | Crowding | Entry? |"]
        lines += ["|--------|-----------|--------|---------|----------|--------|"]
        for _, r in top.iterrows():
            entry = "✅" if r.get("entry_signal") else "—"
            lines.append(
                f"| {r['ticker']} | {_fmt_score(r.get('composite_score'))} "
                f"| {_fmt_pct(r.get('mu_estimate'))} "
                f"| {_fmt_score(r.get('quality_score'))} "
                f"| {_fmt_score(r.get('crowding_score'))} "
                f"| {entry} |"
            )
        lines.append("")

    # ── 5a. Near Misses (exclude tickers already in top-15) ──────────────────
    if not scored_df.empty:
        sig       = params.get("signals", {})
        min_conf  = sig.get("min_composite_confidence", 0.40)
        entry_thr = sig.get("entry_threshold", 0.30)
        crowd_max = sig.get("crowding_entry_max", 0.40)

        nm = scored_df.dropna(subset=["composite_score"]).copy()
        has_conf  = "composite_confidence" in nm.columns
        conf_mask = (nm["composite_confidence"] >= min_conf) if has_conf \
                    else pd.Series(True, index=nm.index)
        near_miss = nm[
            conf_mask &
            (~nm["ticker"].isin(top15_tickers)) &   # ← no duplicate with top-15
            (~nm["entry_signal"].fillna(False)) &
            (nm["composite_score"] >= entry_thr * 0.5)
        ].sort_values("composite_score", ascending=False).head(8)

        if not near_miss.empty:
            lines += ["## Near Misses — Confidence Passed, Entry Blocked", ""]
            lines += ["| Ticker | Composite | Quality | Crowding | Blocker |"]
            lines += ["|--------|-----------|---------|----------|---------|"]
            for _, r in near_miss.iterrows():
                reasons = []
                if (r.get("crowding_score") or 0) > crowd_max:
                    reasons.append("Crowding")
                if (r.get("composite_score") or 0) < entry_thr:
                    reasons.append("Score")
                lines.append(
                    f"| {r['ticker']} | {_fmt_score(r.get('composite_score'))} "
                    f"| {_fmt_score(r.get('quality_score'))} "
                    f"| {_fmt_score(r.get('crowding_score'))} "
                    f"| {' + '.join(reasons) or '—'} |"
                )
            lines.append("")

    # ── 5b. Signal Sub-Components (portfolio stocks) ──────────────────────────
    # Autocorr Δ and Abs.Δ are market-wide CSD indicators (same value per run)
    if not portfolio.empty and not scored_df.empty:
        score_map_full = scored_df.set_index("ticker").to_dict("index")
        port_tickers_sc = portfolio.sort_values("weight", ascending=False)["ticker"].tolist()
        if any(score_map_full.get(t, {}).get("roic_wacc_spread") is not None
               for t in port_tickers_sc):
            lines += ["## Signal Sub-Components — Portfolio Holdings", ""]
            lines += [
                "*Autocorr Δρ₁ and Abs.Δ are market-wide CSD indicators — "
                "same across all holdings. Inf.Conv. = ∂GM/∂PPI (requires PPI YoY data).*",
                "",
                "| Ticker | ROIC−WACC | Margin SNR | Inf.Conv. | Δρ₁ (CSD) | Abs.Δ (CSD) |",
                "|--------|-----------|------------|-----------|-----------|-------------|",
            ]
            for t in port_tickers_sc:
                s = score_map_full.get(t, {})
                lines.append(
                    f"| {t} "
                    f"| {_fmt_spread(s.get('roic_wacc_spread'))} "
                    f"| {_fmt_snr(s.get('margin_snr'))} "
                    f"| {_fmt_score(s.get('inflation_convexity'))} "
                    f"| {_fmt_autocorr(s.get('autocorr_delta'))} "
                    f"| {_fmt_abs(s.get('absorption_delta'))} |"
                )
            lines.append("")

    # ── 6. Macro Context ─────────────────────────────────────────────────────
    from src.data.macro import get_risk_free_rate
    from src.data.db import get_macro_series
    import math as _math_wr

    try:
        rf_ca = get_risk_free_rate(conn, "CA", as_of_date, params)
    except Exception:
        rf_ca = None

    # Compute X_E from canonical physical module to guarantee consistency
    try:
        from src.signals.physical import compute_ecs_x_e
        _x_e, _pct_rank = compute_ecs_x_e(conn, params, as_of_date)
        if _math_wr.isnan(_x_e):
            _x_e, _pct_rank = None, None
        # Get raw PPIENG value for reference
        _phys      = params.get("physical", {})
        _fetch_st  = (date.fromisoformat(as_of_date) - timedelta(days=62*31)).isoformat()
        _ppi_s     = get_macro_series(
            conn, _phys.get("us_energy_ppi_series", "US_PPIENG"), _fetch_st, as_of_date
        )
        _ppi_val   = float(_ppi_s.iloc[-1]) if not _ppi_s.empty else None
        _win_months = len(_ppi_s)
        _spec_months = _phys.get("ecs_lookback_months", 60)
    except Exception:
        _x_e, _pct_rank, _ppi_val, _win_months, _spec_months = None, None, None, None, 60

    lines += ["## Macro Context", ""]
    ecs_caveat = (
        f"⚠ {_win_months}-month window (spec: {_spec_months}m — wider history needed for full precision)"
        if _win_months is not None and _win_months < _spec_months
        else "Low = EROEI-favourable"
    )
    lines += [
        "| Series | Value | Note |",
        "|--------|-------|------|",
        f"| US 10Y | {rf_us:.2%} | WACC baseline US |",
        f"| EU 10Y | {rf_eu:.2%} | WACC baseline EU |",
        f"| CA 10Y | {f'{rf_ca:.2%}' if rf_ca is not None else '—'} | WACC baseline CA |",
        f"| PPIENG index | {f'{_ppi_val:.1f}' if _ppi_val is not None else '—'} | US Energy PPI (level) |",
        f"| PPIENG ECS rank | {f'{_pct_rank:.1%}' if _pct_rank is not None else '—'} | {ecs_caveat} |",
        f"| **X_E (EROEI signal)** | **{f'{_x_e:.3f}' if _x_e is not None else '—'}** | 0=cliff · 1=max advantage |",
        "",
    ]

    # ── 7. FX & Capital Flows ────────────────────────────────────────────────
    if flow_result:
        lines += ["## FX & Capital Flows", ""]

        usd_trend = flow_result.get("usd_trend", "neutral")
        usd_index = flow_result.get("usd_index", 0.0)
        trend_icon = {"strengthening": "📈", "weakening": "📉", "neutral": "➡️"}.get(usd_trend, "➡️")
        lines += [f"**USD Trend:** {trend_icon} {usd_trend.capitalize()} (index = {usd_index:+.2f})", ""]

        fx_df = flow_result.get("fx_momentum", pd.DataFrame())
        if not fx_df.empty:
            lines += ["### FX Rates — Momentum", ""]
            lines += ["| Pair | 1M | 3M | 12M |"]
            lines += ["|------|----|----|----|"]
            for _, row in fx_df.iterrows():
                lines.append(
                    f"| {row['pair']} "
                    f"| {_fmt_ret(row.get('return_1m'))} "
                    f"| {_fmt_ret(row.get('return_3m'))} "
                    f"| {_fmt_ret(row.get('return_12m'))} |"
                )
            lines.append("")

        country_flows = flow_result.get("country_flows", pd.DataFrame())
        if not country_flows.empty:
            lines += ["### Country Capital Flows (EUR-adjusted)", ""]
            lines += ["| Region | Index | 4W EUR | 13W EUR | Flow |"]
            lines += ["|--------|-------|--------|---------|------|"]
            for _, row in country_flows.iterrows():
                lines.append(
                    f"| {row['region']} "
                    f"| {row.get('broad_index', '—')} "
                    f"| {_fmt_ret(row.get('eur_ret_4w'))} "
                    f"| {_fmt_ret(row.get('eur_ret_13w'))} "
                    f"| {_flow_arrow(row.get('flow_score'))} {row.get('flow_direction', '—')} |"
                )
            lines.append("")

        sector_flows = flow_result.get("sector_flows", pd.DataFrame())
        if not sector_flows.empty:
            # Drop rows with no price data — flow_score can be 0.0 even when returns are null
            sector_flows = sector_flows[sector_flows["local_ret_4w"].notna()]
        if not sector_flows.empty:
            lines += ["### Sector Capital Flows", ""]
            lines += ["| Region | Sector | ETF | Local 4W | EUR 4W | FX Contrib | vs Mkt 4W | Flow |"]
            lines += ["|--------|--------|-----|----------|--------|------------|-----------|------|"]
            for _, row in sector_flows.sort_values("flow_score", ascending=False).iterrows():
                lines.append(
                    f"| {row['region']} "
                    f"| {row['bucket']} "
                    f"| {row['etf_ticker']} "
                    f"| {_fmt_ret(row.get('local_ret_4w'))} "
                    f"| {_fmt_ret(row.get('usd_ret_4w'))} "
                    f"| {_fmt_ret(row.get('fx_contrib_4w'))} "
                    f"| {_fmt_ret(row.get('vs_market_4w'))} "
                    f"| {_flow_arrow(row.get('flow_score'))} |"
                )
            lines.append("")

        implications = flow_result.get("implications", [])
        if implications:
            lines += ["### Portfolio Implications", ""]
            for impl in implications:
                lines.append(f"- {impl}")
            lines.append("")

    # ── Footer ────────────────────────────────────────────────────────────────
    lines += [
        "---",
        f"*Generated by EARKE Quant 3.0 on {as_of_date}. "
        "This is a quantitative scan — all decisions require human review.*",
    ]

    report_text = "\n".join(lines)
    fname.write_text(report_text, encoding="utf-8")
    logger.info(f"Weekly report saved to {fname}")
    return str(fname)
