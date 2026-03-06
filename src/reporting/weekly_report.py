"""
Generates a Markdown weekly report summarizing pipeline run results.
File: reports/YYYY-MM-DD_weekly.md
"""
import os
import logging
from datetime import date
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


def _traffic_light(crowding_score: Optional[float], params: dict) -> str:
    """Emoji traffic light for crowding level."""
    if crowding_score is None or pd.isna(crowding_score):
        return "⚪"
    watch_thr = params["reporting"].get("watch_crowding_threshold", 0.55)
    exit_thr  = params["signals"]["crowding_exit_threshold"]
    if crowding_score >= exit_thr:
        return "🔴"
    if crowding_score >= watch_thr:
        return "🟡"
    return "🟢"


def generate_weekly_report(
    conn,
    actions: dict,
    scored_df: pd.DataFrame,
    params: dict,
    as_of_date: str,
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

    portfolio   = actions.get("new_portfolio", pd.DataFrame())
    exits       = actions.get("exits", pd.DataFrame())
    entries     = actions.get("entries", pd.DataFrame())
    watch_list  = actions.get("watch_list", pd.DataFrame())
    diff        = actions.get("diff", pd.DataFrame())

    # Risk-free rates for context
    rf_us = get_risk_free_rate(conn, "US", as_of_date, params)
    rf_eu = get_risk_free_rate(conn, "EU", as_of_date, params)

    lines = [
        f"# EARKE Quant 3.0 — Weekly Report",
        f"**Date:** {as_of_date}  |  **RF US:** {rf_us:.2%}  |  **RF EU:** {rf_eu:.2%}",
        "",
    ]

    # ── 1. Executive Summary ─────────────────────────────────────────────────
    lines += ["## Summary"]
    n_pos = len(portfolio) if not portfolio.empty else 0
    invested = portfolio["weight"].sum() if not portfolio.empty else 0.0
    cash_w   = 1.0 - invested

    if actions.get("any_action"):
        lines.append(f"⚡ **Action required this week.**")
    else:
        lines.append(f"✅ **No action required.** Portfolio unchanged.")

    lines += [
        f"",
        f"- Positions: **{n_pos}**  |  Invested: **{_fmt_pct(invested)}**  |  Cash: **{_fmt_pct(cash_w)}**",
        f"- Exit triggers: **{exits['exit_triggered'].sum() if not exits.empty else 0}**",
        f"- New entry candidates: **{len(entries) if not entries.empty else 0}**",
        f"- Watch list (crowding elevated): **{len(watch_list) if not watch_list.empty else 0}**",
        "",
    ]

    # ── 2. Actions ────────────────────────────────────────────────────────────
    lines += ["## This Week's Actions", ""]

    # Exits
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

    # Entries
    if not entries.empty:
        lines += ["### ✅ New Entry Candidates", ""]
        lines += ["| Ticker | Composite | Quality | Crowding | Kelly% |"]
        lines += ["|--------|-----------|---------|----------|--------|"]
        for _, r in entries.head(params["reporting"].get("top_candidates_count", 10)).iterrows():
            lines.append(
                f"| {r['ticker']} | {_fmt_score(r.get('composite_score'))} "
                f"| {_fmt_score(r.get('quality_score'))} "
                f"| {_fmt_score(r.get('crowding_score'))} "
                f"| {_fmt_pct(r.get('kelly_25pct'))} |"
            )
        lines.append("")

    # Rebalances from diff
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

        port_tickers = portfolio["ticker"].tolist()
        score_map = {}
        if not scored_df.empty:
            score_map = scored_df.set_index("ticker").to_dict("index")

        for _, row in portfolio.sort_values("weight", ascending=False).iterrows():
            t = row["ticker"]
            s = score_map.get(t, {})
            crowd = s.get("crowding_score")
            status = _traffic_light(crowd, params)
            lines.append(
                f"| {t} | {_fmt_pct(row['weight'])} "
                f"| {_fmt_score(row.get('composite_score'))} "
                f"| {_fmt_score(s.get('quality_score'))} "
                f"| {_fmt_score(crowd)} "
                f"| {status} |"
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
    lines += ["## Universe Top Candidates", ""]
    if not scored_df.empty:
        top = scored_df.dropna(subset=["composite_score"]).sort_values(
            "composite_score", ascending=False
        ).head(15)
        lines += ["| Ticker | Composite | Physical | Quality | Crowding | Entry? |"]
        lines += ["|--------|-----------|----------|---------|----------|--------|"]
        for _, r in top.iterrows():
            entry = "✅" if r.get("entry_signal") else "—"
            lines.append(
                f"| {r['ticker']} | {_fmt_score(r.get('composite_score'))} "
                f"| {_fmt_score(r.get('physical_norm'))} "
                f"| {_fmt_score(r.get('quality_score'))} "
                f"| {_fmt_score(r.get('crowding_score'))} "
                f"| {entry} |"
            )
        lines.append("")

    # ── 6. Macro Context ─────────────────────────────────────────────────────
    lines += ["## Macro Context", ""]
    lines += [
        f"| Region | 10Y Rate |",
        f"|--------|----------|",
        f"| US     | {rf_us:.2%} |",
        f"| EU     | {rf_eu:.2%} |",
        "",
    ]

    # ── Footer ────────────────────────────────────────────────────────────────
    lines += [
        "---",
        f"*Generated by EARKE Quant 3.0 on {as_of_date}. "
        f"This is a quantitative scan — all decisions require human review.*",
    ]

    report_text = "\n".join(lines)
    fname.write_text(report_text, encoding="utf-8")
    logger.info(f"Weekly report saved to {fname}")
    return str(fname)
