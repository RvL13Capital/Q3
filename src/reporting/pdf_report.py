"""
EARKE Quant 3.0 — PDF Weekly Report
Von Linck Capital Management

Generates a fully graphical, multi-page PDF research memorandum.
Entry point: generate_pdf_report(...)  →  returns path to saved PDF.

Design language: dark command-center palette, cyan/purple accents,
gradient score bars, radar charts, heatmaps.
"""
from __future__ import annotations

import io
import logging
import math
from datetime import date as _date
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from reportlab.lib import colors as rl_colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)
from reportlab.platypus.flowables import Flowable
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":          "#0A0F1E",
    "surface":     "#111827",
    "surface2":    "#1C2333",
    "border":      "#1E2D45",
    "cyan":        "#00D4FF",
    "purple":      "#7B61FF",
    "green":       "#00E676",
    "amber":       "#FFB300",
    "red":         "#FF4444",
    "text":        "#E8ECF0",
    "muted":       "#6B7A8D",
    "white":       "#FFFFFF",
}

def _hex(key: str):
    return rl_colors.HexColor(C[key])


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib helpers
# ─────────────────────────────────────────────────────────────────────────────
MPL_PARAMS = {
    "figure.facecolor":  C["surface"],
    "axes.facecolor":    C["surface2"],
    "axes.edgecolor":    C["border"],
    "axes.labelcolor":   C["text"],
    "xtick.color":       C["muted"],
    "ytick.color":       C["muted"],
    "text.color":        C["text"],
    "grid.color":        C["border"],
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "DejaVu Sans",
}

def _apply_mpl():
    plt.rcParams.update(MPL_PARAMS)


def _fig_to_image(fig, width_cm: float = 17.0) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    img = Image(buf)
    aspect = img.imageHeight / img.imageWidth
    w = width_cm * cm
    img.drawWidth  = w
    img.drawHeight = w * aspect
    return img


def _score_color(v: float | None) -> str:
    """Map [0,1] score to hex color: red → amber → green."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return C["muted"]
    v = max(0.0, min(1.0, float(v)))
    if v >= 0.6:
        return C["green"]
    if v >= 0.35:
        return C["amber"]
    return C["red"]


def _rl_score_color(v: float | None):
    return rl_colors.HexColor(_score_color(v))


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────

def _chart_bucket_allocation(portfolio: pd.DataFrame) -> Image:
    _apply_mpl()
    buckets = (
        portfolio.groupby("primary_bucket")["weight"]
        .sum()
        .sort_values(ascending=True)
    )
    if buckets.empty:
        buckets = pd.Series({"No data": 1.0})

    palette = [C["cyan"], C["purple"], C["green"], C["amber"], C["red"], "#FF6EC7"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.barh(buckets.index, buckets.values * 100,
                   color=palette[:len(buckets)], height=0.55, edgecolor="none")
    ax.set_xlabel("Weight (%)", fontsize=9)
    ax.set_title("Bucket Allocation", color=C["cyan"], fontsize=11, fontweight="bold", pad=8)
    ax.axvline(35, color=C["red"], linestyle="--", linewidth=0.8, alpha=0.7, label="35% cap")
    for bar, val in zip(bars, buckets.values):
        ax.text(val * 100 + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", fontsize=8, color=C["text"])
    ax.legend(fontsize=7, framealpha=0.3)
    ax.set_xlim(0, max(40, buckets.max() * 110))
    fig.tight_layout()
    return _fig_to_image(fig, width_cm=8.5)


def _chart_score_distribution(scored_df: pd.DataFrame) -> Image:
    _apply_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    cols = [
        ("composite_score", "Composite", C["cyan"]),
        ("quality_score",   "Quality",   C["purple"]),
        ("crowding_score",  "Crowding",  C["amber"]),
    ]
    for ax, (col, label, color) in zip(axes, cols):
        data = scored_df[col].dropna() if col in scored_df.columns else pd.Series([], dtype=float)
        if data.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color=C["muted"])
        else:
            ax.hist(data, bins=20, color=color, alpha=0.75, edgecolor="none")
            ax.axvline(data.median(), color=C["white"], linewidth=1.2,
                       linestyle="--", label=f"med {data.median():.2f}")
            ax.legend(fontsize=7, framealpha=0.3)
        ax.set_title(label, color=color, fontsize=9, fontweight="bold")
        ax.set_xlabel("Score", fontsize=8)
    fig.suptitle("Score Distributions — Universe", color=C["text"], fontsize=10, y=1.02)
    fig.tight_layout()
    return _fig_to_image(fig, width_cm=17)


def _chart_radar(row: pd.Series, ticker: str) -> Image:
    _apply_mpl()
    labels   = ["Physical\n(EROEI)", "Quality\n(Moat)", "Anti-\nCrowding",
                 "Composite\nScore", "Confidence"]
    values   = [
        _safe(row, "physical_norm"),
        _safe(row, "quality_score"),
        1.0 - _safe(row, "crowding_score"),
        _safe(row, "composite_score"),
        _safe(row, "composite_confidence"),
    ]
    n = len(labels)
    angles = [i * 2 * math.pi / n for i in range(n)] + [0]
    values_plot = values + [values[0]]

    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))
    ax.set_facecolor(C["surface2"])
    fig.patch.set_facecolor(C["surface"])
    ax.plot(angles, values_plot, color=C["cyan"], linewidth=1.8)
    ax.fill(angles, values_plot, color=C["cyan"], alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=7, color=C["text"])
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=6, color=C["muted"])
    ax.grid(color=C["border"], linewidth=0.6)
    ax.spines["polar"].set_color(C["border"])
    ax.set_title(ticker, color=C["cyan"], fontsize=10, fontweight="bold", pad=10)
    fig.tight_layout()
    return _fig_to_image(fig, width_cm=5.5)


def _chart_fx_bar(fx_df: pd.DataFrame) -> Image:
    _apply_mpl()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    pairs = fx_df["pair"].tolist() if "pair" in fx_df.columns else []
    r1m   = (fx_df["return_1m"].fillna(0) * 100).tolist() if "return_1m" in fx_df.columns else []
    r3m   = (fx_df["return_3m"].fillna(0) * 100).tolist() if "return_3m" in fx_df.columns else []

    x = np.arange(len(pairs))
    w = 0.35
    ax.bar(x - w/2, r1m, width=w, label="1M",  color=C["cyan"],   alpha=0.85, edgecolor="none")
    ax.bar(x + w/2, r3m, width=w, label="3M",  color=C["purple"], alpha=0.85, edgecolor="none")
    ax.axhline(0, color=C["muted"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Return (%)", fontsize=8)
    ax.set_title("FX Momentum", color=C["cyan"], fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.3)
    fig.tight_layout()
    return _fig_to_image(fig, width_cm=10)


def _chart_sector_heatmap(sector_df: pd.DataFrame) -> Image:
    _apply_mpl()
    if sector_df.empty or "flow_score" not in sector_df.columns:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No sector flow data", ha="center", va="center",
                transform=ax.transAxes, color=C["muted"])
        return _fig_to_image(fig, width_cm=10)

    pivot = sector_df.pivot_table(
        index="bucket", columns="region", values="flow_score", aggfunc="mean"
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(max(4, len(pivot.columns) * 2), max(2.5, len(pivot) * 0.55)))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "earke", [C["red"], C["surface2"], C["green"]]
    )
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Sector Flow Scores by Region", color=C["cyan"], fontsize=9, fontweight="bold")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=7, color=C["white"] if abs(v) > 0.3 else C["text"])

    fig.colorbar(im, ax=ax, fraction=0.03, label="Flow Score")
    fig.tight_layout()
    return _fig_to_image(fig, width_cm=10)


def _chart_kelly_bars(portfolio: pd.DataFrame, scored_df: pd.DataFrame) -> Image:
    _apply_mpl()
    if portfolio.empty:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No portfolio data", ha="center", va="center",
                transform=ax.transAxes, color=C["muted"])
        return _fig_to_image(fig, width_cm=10)

    merged = portfolio.copy()
    if not scored_df.empty and "kelly_25pct" in scored_df.columns:
        merged = merged.merge(
            scored_df[["ticker", "kelly_25pct", "kelly_raw"]].rename(
                columns={"kelly_25pct": "k25_score", "kelly_raw": "k_raw_score"}
            ),
            on="ticker", how="left"
        )

    tickers = merged["ticker"].tolist()
    weights = (merged["weight"] * 100).tolist()
    k25     = (merged.get("k25_score", pd.Series([None]*len(merged))).fillna(0) * 100).tolist()

    x = np.arange(len(tickers))
    fig, ax = plt.subplots(figsize=(max(8, len(tickers) * 0.55), 3.5))
    ax.bar(x - 0.2, weights, 0.35, label="Actual Weight", color=C["cyan"], alpha=0.85)
    ax.bar(x + 0.2, k25,     0.35, label="Kelly 25%",     color=C["purple"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, fontsize=7, rotation=35, ha="right")
    ax.set_ylabel("Weight / Kelly %", fontsize=8)
    ax.set_title("Position Weights vs Kelly Fractions", color=C["cyan"],
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.3)
    ax.axhline(8, color=C["red"], linewidth=0.8, linestyle="--", alpha=0.6, label="8% cap")
    fig.tight_layout()
    return _fig_to_image(fig, width_cm=17)


def _chart_usd_gauge(usd_index: float, usd_trend: str) -> Image:
    _apply_mpl()
    fig, ax = plt.subplots(figsize=(3.2, 2.4), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor(C["surface"])
    ax.set_facecolor(C["surface"])

    # Semicircle gauge
    theta = np.linspace(np.pi, 0, 200)
    for i, (start, end, color) in enumerate([
        (np.pi, np.pi * 2/3, C["green"]),
        (np.pi * 2/3, np.pi / 3, C["amber"]),
        (np.pi / 3, 0, C["red"]),
    ]):
        t = np.linspace(start, end, 50)
        ax.fill_between(np.cos(t), np.sin(t) * 0.65, np.sin(t), color=color, alpha=0.7)

    # Needle: map usd_index [-1,1] → angle [π, 0]
    clamped = max(-1.0, min(1.0, usd_index))
    angle   = np.pi * (1 - (clamped + 1) / 2)
    ax.annotate("", xy=(np.cos(angle) * 0.85, np.sin(angle) * 0.85),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C["white"], lw=2))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.15, 1.1)
    ax.axis("off")
    ax.text(0, -0.12, f"USD {usd_trend.upper()}  ({usd_index:+.2f})",
            ha="center", fontsize=7.5, color=C["text"])
    ax.set_title("USD Strength", color=C["cyan"], fontsize=9, fontweight="bold")
    return _fig_to_image(fig, width_cm=5.5)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _safe(row, col, default=0.0):
    v = row.get(col, default)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    return float(v)


def _fmt_pct(v, decimals=1):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{float(v)*100:.{decimals}f}%"


def _fmt_score(v, decimals=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{float(v):.{decimals}f}"


def _fmt_ret(v, decimals=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{float(v)*100:+.{decimals}f}%"


# ─────────────────────────────────────────────────────────────────────────────
# ReportLab styles
# ─────────────────────────────────────────────────────────────────────────────

def _build_styles():
    base = getSampleStyleSheet()
    s = {}

    def ps(name, font="Helvetica", size=8.5, leading=12, color="text",
           bold=False, align=TA_LEFT, space_before=0, space_after=0):
        return ParagraphStyle(
            name,
            fontName="Helvetica-Bold" if bold else font,
            fontSize=size,
            leading=leading,
            textColor=_hex(color),
            alignment=align,
            spaceBefore=space_before,
            spaceAfter=space_after,
        )

    s["h1"]         = ps("h1",  size=22, leading=26, color="cyan",  bold=True,  align=TA_LEFT,   space_after=4)
    s["h2"]         = ps("h2",  size=13, leading=17, color="cyan",  bold=True,  align=TA_LEFT,   space_before=10, space_after=4)
    s["h3"]         = ps("h3",  size=10, leading=13, color="purple",bold=True,  align=TA_LEFT,   space_before=6,  space_after=2)
    s["body"]       = ps("body",size=8.5,leading=12, color="text")
    s["muted"]      = ps("muted",size=7.5,leading=11,color="muted")
    s["center"]     = ps("center",size=8.5,leading=12,color="text", align=TA_CENTER)
    s["kpi_label"]  = ps("kpi_label",size=7,leading=9,color="muted",align=TA_CENTER)
    s["kpi_value"]  = ps("kpi_value",size=18,leading=22,color="cyan",bold=True,align=TA_CENTER)
    s["cover_fund"] = ps("cover_fund",size=11,leading=14,color="muted",align=TA_CENTER)
    s["cover_title"]= ps("cover_title",size=30,leading=36,color="white",bold=True,align=TA_CENTER)
    s["cover_sub"]  = ps("cover_sub",size=13,leading=16,color="cyan",align=TA_CENTER)
    s["th"]         = ps("th",size=7.5,leading=10,color="cyan",bold=True,align=TA_CENTER)
    s["td"]         = ps("td",size=7.5,leading=10,color="text",align=TA_CENTER)
    s["td_left"]    = ps("td_left",size=7.5,leading=10,color="text",align=TA_LEFT)
    s["tag_green"]  = ps("tag_green",size=7,leading=9,color="green",bold=True,align=TA_CENTER)
    s["tag_red"]    = ps("tag_red",size=7,leading=9,color="red",bold=True,align=TA_CENTER)
    s["tag_amber"]  = ps("tag_amber",size=7,leading=9,color="amber",bold=True,align=TA_CENTER)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Page templates
# ─────────────────────────────────────────────────────────────────────────────
W, H = A4  # 595 × 842 pt

MARGIN = 1.8 * cm
CONTENT_W = W - 2 * MARGIN
CONTENT_H = H - 2 * MARGIN - 1.5 * cm  # space for header/footer


def _draw_cover_background(canvas, doc):
    canvas.saveState()
    # Full-page dark bg
    canvas.setFillColor(rl_colors.HexColor(C["bg"]))
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    # Cyan accent bar top
    canvas.setFillColor(_hex("cyan"))
    canvas.rect(0, H - 0.8 * cm, W, 0.8 * cm, fill=1, stroke=0)
    # Purple accent bar bottom
    canvas.setFillColor(_hex("purple"))
    canvas.rect(0, 0, W, 0.5 * cm, fill=1, stroke=0)
    # Decorative grid lines
    canvas.setStrokeColor(rl_colors.HexColor(C["border"]))
    canvas.setLineWidth(0.4)
    for i in range(0, int(W) + 1, 40):
        canvas.line(i, 0, i, H)
    for i in range(0, int(H) + 1, 40):
        canvas.line(0, i, W, i)
    canvas.restoreState()


def _draw_page_background(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(rl_colors.HexColor(C["bg"]))
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    # Top header bar
    canvas.setFillColor(rl_colors.HexColor(C["surface"]))
    canvas.rect(0, H - 1.1 * cm, W, 1.1 * cm, fill=1, stroke=0)
    # Header separator
    canvas.setStrokeColor(_hex("cyan"))
    canvas.setLineWidth(1.0)
    canvas.line(MARGIN, H - 1.15 * cm, W - MARGIN, H - 1.15 * cm)
    # Header text
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(_hex("cyan"))
    canvas.drawString(MARGIN, H - 0.78 * cm, "EARKE QUANT 3.0")
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(_hex("muted"))
    canvas.drawRightString(W - MARGIN, H - 0.78 * cm,
                           f"Von Linck Capital Management  ·  {doc._earke_date}")
    # Footer
    canvas.setFillColor(rl_colors.HexColor(C["surface"]))
    canvas.rect(0, 0, W, 0.9 * cm, fill=1, stroke=0)
    canvas.setStrokeColor(_hex("border"))
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 0.95 * cm, W - MARGIN, 0.95 * cm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(_hex("muted"))
    canvas.drawString(MARGIN, 0.28 * cm,
                      "PRIVATE & CONFIDENTIAL — For authorised recipients only. "
                      "Quantitative scan — all decisions require human review.")
    canvas.drawRightString(W - MARGIN, 0.28 * cm,
                           f"Page {doc.page}")
    canvas.restoreState()


# ─────────────────────────────────────────────────────────────────────────────
# Table builders
# ─────────────────────────────────────────────────────────────────────────────

_BASE_TABLE_STYLE = [
    ("BACKGROUND",    (0, 0), (-1, 0),  rl_colors.HexColor(C["surface"])),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
     [rl_colors.HexColor(C["surface2"]), rl_colors.HexColor(C["surface"])]),
    ("GRID",          (0, 0), (-1, -1), 0.3, rl_colors.HexColor(C["border"])),
    ("TOPPADDING",    (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LEFTPADDING",   (0, 0), (-1, -1), 5),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
    ("FONTSIZE",      (0, 0), (-1, -1), 7.5),
    ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
    ("TEXTCOLOR",     (0, 0), (-1, 0),  _hex("cyan")),
    ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
]


def _build_portfolio_table(portfolio: pd.DataFrame, scored_df: pd.DataFrame,
                            params: dict, styles: dict) -> Table:
    crowd_exit  = params.get("signals", {}).get("crowding_exit_threshold", 0.75)
    watch_thr   = params.get("reporting", {}).get("watch_crowding_threshold", 0.55)

    score_map = {}
    if not scored_df.empty:
        score_map = scored_df.set_index("ticker").to_dict("index")

    headers = ["Ticker", "Bucket", "Weight", "Composite", "Physical",
               "Quality", "Crowding", "Kelly 25%", "Status"]
    rows = [headers]
    style_cmds = list(_BASE_TABLE_STYLE)

    for ri, (_, row) in enumerate(
        portfolio.sort_values("weight", ascending=False).iterrows(), start=1
    ):
        t = row["ticker"]
        s = score_map.get(t, {})
        crowd = s.get("crowding_score")
        comp  = row.get("composite_score", s.get("composite_score"))
        qual  = s.get("quality_score")
        phys  = s.get("physical_norm")
        kelly = s.get("kelly_25pct")
        wt    = row.get("weight")
        bucket = row.get("primary_bucket", "—")

        if crowd is not None and not math.isnan(float(crowd)):
            if float(crowd) >= crowd_exit:
                status, sc = "EXIT", C["red"]
            elif float(crowd) >= watch_thr:
                status, sc = "WATCH", C["amber"]
            else:
                status, sc = "HOLD", C["green"]
        else:
            status, sc = "—", C["muted"]

        rows.append([t, bucket, _fmt_pct(wt), _fmt_score(comp), _fmt_score(phys),
                     _fmt_score(qual), _fmt_score(crowd), _fmt_pct(kelly), status])

        # Color score cells
        for ci, val in [(3, comp), (4, phys), (5, qual)]:
            style_cmds.append(("TEXTCOLOR", (ci, ri), (ci, ri), _rl_score_color(val)))
        # Crowding inverse
        if crowd is not None and not math.isnan(float(crowd)):
            inv = 1.0 - float(crowd)
            style_cmds.append(("TEXTCOLOR", (6, ri), (6, ri), _rl_score_color(inv)))
        # Status colour
        style_cmds.append(("TEXTCOLOR", (8, ri), (8, ri), rl_colors.HexColor(sc)))
        style_cmds.append(("FONTNAME",  (8, ri), (8, ri), "Helvetica-Bold"))
        # Ticker bold left
        style_cmds.append(("FONTNAME", (0, ri), (0, ri), "Helvetica-Bold"))
        style_cmds.append(("ALIGN",    (0, ri), (0, ri), "LEFT"))
        style_cmds.append(("TEXTCOLOR", (0, ri), (0, ri), _hex("text")))

    col_widths = [
        1.5*cm, 2.5*cm, 1.4*cm, 1.7*cm, 1.7*cm, 1.7*cm, 1.7*cm, 1.6*cm, 1.4*cm
    ]
    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def _build_universe_table(scored_df: pd.DataFrame, params: dict, n: int = 15) -> Table:
    top = (scored_df.dropna(subset=["composite_score"])
           .sort_values("composite_score", ascending=False)
           .head(n))

    headers = ["#", "Ticker", "Composite", "Physical", "Quality",
               "Crowding", "mu est.", "Kelly 25%", "Entry"]
    rows = [headers]
    style_cmds = list(_BASE_TABLE_STYLE)

    for ri, (_, r) in enumerate(top.iterrows(), start=1):
        entry = r.get("entry_signal", False)
        rows.append([
            str(ri),
            r["ticker"],
            _fmt_score(r.get("composite_score")),
            _fmt_score(r.get("physical_norm")),
            _fmt_score(r.get("quality_score")),
            _fmt_score(r.get("crowding_score")),
            _fmt_pct(r.get("mu_estimate")),
            _fmt_pct(r.get("kelly_25pct")),
            "YES" if entry else "—",
        ])
        for ci, col in [(2, "composite_score"), (3, "physical_norm"),
                        (4, "quality_score")]:
            style_cmds.append(
                ("TEXTCOLOR", (ci, ri), (ci, ri), _rl_score_color(r.get(col)))
            )
        crowd = r.get("crowding_score")
        if crowd is not None:
            style_cmds.append(
                ("TEXTCOLOR", (5, ri), (5, ri), _rl_score_color(1.0 - float(crowd)))
            )
        if entry:
            style_cmds.append(("TEXTCOLOR", (8, ri), (8, ri), _hex("green")))
            style_cmds.append(("FONTNAME",  (8, ri), (8, ri), "Helvetica-Bold"))
        style_cmds.append(("FONTNAME", (1, ri), (1, ri), "Helvetica-Bold"))
        style_cmds.append(("ALIGN",    (1, ri), (1, ri), "LEFT"))
        style_cmds.append(("TEXTCOLOR", (1, ri), (1, ri), _hex("text")))

    col_widths = [0.6*cm, 1.7*cm, 1.7*cm, 1.7*cm, 1.7*cm,
                  1.7*cm, 1.6*cm, 1.6*cm, 1.2*cm]
    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def _build_exits_table(exits: pd.DataFrame) -> Table | None:
    triggered = exits[exits.get("exit_triggered", pd.Series([], dtype=bool))] \
        if "exit_triggered" in exits.columns else exits
    if triggered.empty:
        return None

    headers = ["Ticker", "Crowding", "Quality", "Composite", "Reason(s)"]
    rows = [headers]
    style_cmds = list(_BASE_TABLE_STYLE)

    for ri, (_, r) in enumerate(triggered.iterrows(), start=1):
        reasons = "; ".join(r.get("exit_reasons") or [])
        rows.append([
            r["ticker"],
            _fmt_score(r.get("crowding_score")),
            _fmt_score(r.get("quality_score")),
            _fmt_score(r.get("composite_score")),
            reasons,
        ])
        style_cmds.append(("TEXTCOLOR", (0, ri), (0, ri), _hex("red")))
        style_cmds.append(("FONTNAME",  (0, ri), (0, ri), "Helvetica-Bold"))
        style_cmds.append(("ALIGN",     (4, ri), (4, ri), "LEFT"))

    col_widths = [1.5*cm, 1.8*cm, 1.8*cm, 1.8*cm, 9.6*cm]
    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def _build_entries_table(entries: pd.DataFrame, n: int = 10) -> Table | None:
    if entries.empty:
        return None

    headers = ["Ticker", "Composite", "Quality", "Crowding", "mu est.", "Kelly 25%"]
    rows = [headers]
    style_cmds = list(_BASE_TABLE_STYLE)

    for ri, (_, r) in enumerate(entries.head(n).iterrows(), start=1):
        rows.append([
            r["ticker"],
            _fmt_score(r.get("composite_score")),
            _fmt_score(r.get("quality_score")),
            _fmt_score(r.get("crowding_score")),
            _fmt_pct(r.get("mu_estimate")),
            _fmt_pct(r.get("kelly_25pct")),
        ])
        style_cmds.append(("TEXTCOLOR", (0, ri), (0, ri), _hex("green")))
        style_cmds.append(("FONTNAME",  (0, ri), (0, ri), "Helvetica-Bold"))
        for ci, col in [(1, "composite_score"), (2, "quality_score")]:
            style_cmds.append(
                ("TEXTCOLOR", (ci, ri), (ci, ri), _rl_score_color(r.get(col)))
            )

    col_widths = [2.0*cm, 2.0*cm, 2.0*cm, 2.0*cm, 2.0*cm, 2.0*cm]
    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def _build_watch_table(watch_list: pd.DataFrame, params: dict) -> Table | None:
    if watch_list.empty:
        return None

    crowd_exit = params.get("signals", {}).get("crowding_exit_threshold", 0.75)
    headers = ["Ticker", "Crowding", "Exit Threshold", "Quality", "Composite"]
    rows = [headers]
    style_cmds = list(_BASE_TABLE_STYLE)

    for ri, (_, r) in enumerate(watch_list.iterrows(), start=1):
        rows.append([
            r["ticker"],
            _fmt_score(r.get("crowding_score")),
            f"{crowd_exit:.2f}",
            _fmt_score(r.get("quality_score")),
            _fmt_score(r.get("composite_score")),
        ])
        style_cmds.append(("TEXTCOLOR", (0, ri), (0, ri), _hex("amber")))
        style_cmds.append(("FONTNAME",  (0, ri), (0, ri), "Helvetica-Bold"))

    col_widths = [2.0*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm]
    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def _build_fx_table(fx_df: pd.DataFrame) -> Table | None:
    if fx_df.empty:
        return None

    headers = ["Pair", "1M Return", "3M Return", "12M Return"]
    rows = [headers]
    style_cmds = list(_BASE_TABLE_STYLE)

    for ri, (_, r) in enumerate(fx_df.iterrows(), start=1):
        r1m = r.get("return_1m", 0) or 0
        r3m = r.get("return_3m", 0) or 0
        rows.append([
            r["pair"],
            _fmt_ret(r.get("return_1m")),
            _fmt_ret(r.get("return_3m")),
            _fmt_ret(r.get("return_12m")),
        ])
        for ci, val in [(1, r1m), (2, r3m)]:
            c = C["green"] if val > 0.005 else (C["red"] if val < -0.005 else C["muted"])
            style_cmds.append(("TEXTCOLOR", (ci, ri), (ci, ri), rl_colors.HexColor(c)))

    col_widths = [3.5*cm, 3.0*cm, 3.0*cm, 3.0*cm]
    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def _build_macro_table(rf_us: float, rf_eu: float) -> Table:
    headers = ["Region", "10Y Risk-Free Rate"]
    rows = [
        headers,
        ["US",  f"{rf_us:.2%}"],
        ["EU",  f"{rf_eu:.2%}"],
    ]
    style_cmds = list(_BASE_TABLE_STYLE)
    col_widths = [4.0*cm, 5.0*cm]
    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# KPI card flowable
# ─────────────────────────────────────────────────────────────────────────────

class KPICard(Flowable):
    """Draws a single metric card: label + value in a bordered box."""
    def __init__(self, label: str, value: str, color: str = C["cyan"],
                 width: float = 3.8*cm, height: float = 2.0*cm):
        super().__init__()
        self.label  = label
        self.value  = value
        self.color  = color
        self.width  = width
        self.height = height

    def wrap(self, *args):
        return self.width, self.height

    def draw(self):
        c = self.canv
        c.saveState()
        # Background
        c.setFillColor(rl_colors.HexColor(C["surface2"]))
        c.roundRect(0, 0, self.width, self.height, radius=4, fill=1, stroke=0)
        # Border
        c.setStrokeColor(rl_colors.HexColor(self.color))
        c.setLineWidth(1.2)
        c.roundRect(0, 0, self.width, self.height, radius=4, fill=0, stroke=1)
        # Value
        c.setFillColor(rl_colors.HexColor(self.color))
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(self.width / 2, self.height * 0.42, self.value)
        # Label
        c.setFillColor(rl_colors.HexColor(C["muted"]))
        c.setFont("Helvetica", 7)
        c.drawCentredString(self.width / 2, self.height * 0.15, self.label)
        c.restoreState()


def _kpi_row(cards: list[KPICard], gap: float = 0.4*cm) -> Table:
    """Pack KPI cards side-by-side in a single-row Table."""
    col_widths = [card.width + gap for card in cards]
    tbl = Table([[cards]], colWidths=[sum(col_widths)])
    tbl.setStyle(TableStyle([
        ("ALIGN",   (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",  (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
    ]))
    return tbl


def _side_by_side(*items, col_widths=None, gap=0.5*cm) -> Table:
    if col_widths is None:
        col_widths = [(CONTENT_W / len(items))] * len(items)
    tbl = Table([list(items)], colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("ALIGN",   (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",  (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
    ]))
    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# Section divider helper
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str, styles: dict) -> list:
    return [
        Spacer(1, 0.3*cm),
        Paragraph(title, styles["h2"]),
        HRFlowable(width=CONTENT_W, thickness=0.5, color=_hex("border"),
                   spaceAfter=4, spaceBefore=2),
    ]


def _subsection(title: str, styles: dict) -> list:
    return [Paragraph(title, styles["h3"])]


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(
    conn,
    actions: dict,
    scored_df: pd.DataFrame,
    params: dict,
    as_of_date: str,
    flow_result: Optional[dict] = None,
    output_dir: str = "reports",
    aum_eur: Optional[float] = None,
) -> str:
    """
    Generate the graphical PDF weekly report.
    Returns path to the created PDF file.
    """
    from src.data.macro import get_risk_free_rate

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fname = str(output_path / f"{as_of_date}_weekly.pdf")

    portfolio   = actions.get("new_portfolio", pd.DataFrame())
    exits       = actions.get("exits",         pd.DataFrame())
    entries     = actions.get("entries",       pd.DataFrame())
    watch_list  = actions.get("watch_list",    pd.DataFrame())
    diff        = actions.get("diff",          pd.DataFrame())

    rf_us = get_risk_free_rate(conn, "US", as_of_date, params)
    rf_eu = get_risk_free_rate(conn, "EU", as_of_date, params)
    aum   = aum_eur or params.get("kelly", {}).get("aum_eur", 0)

    styles = _build_styles()

    # ── Document setup ───────────────────────────────────────────────────────
    doc = BaseDocTemplate(
        fname,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=1.5*cm, bottomMargin=1.3*cm,
        title=f"EARKE Quant 3.0 — Weekly Report {as_of_date}",
        author="Von Linck Capital Management",
    )
    doc._earke_date = as_of_date  # available in page callbacks

    cover_frame = Frame(0, 0, W, H, id="cover")
    body_frame  = Frame(MARGIN, 1.2*cm, CONTENT_W, H - 2.6*cm, id="body")

    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[cover_frame],
                     onPage=_draw_cover_background),
        PageTemplate(id="body",  frames=[body_frame],
                     onPage=_draw_page_background),
    ])

    story = []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PAGE 1 — Cover
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(NextPageTemplate("cover"))

    story.append(Spacer(1, 6.0*cm))
    story.append(Paragraph("EARKE QUANT 3.0", styles["cover_title"]))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Weekly Investment Report", styles["cover_sub"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Von Linck Capital Management · Private Research Memorandum",
                            styles["cover_fund"]))
    story.append(Spacer(1, 1.8*cm))

    # Cover KPIs
    n_pos    = len(portfolio) if not portfolio.empty else 0
    invested = portfolio["weight"].sum() if not portfolio.empty else 0.0
    n_entry  = len(entries) if not entries.empty else 0
    n_exit   = int(exits["exit_triggered"].sum()) if (
        not exits.empty and "exit_triggered" in exits.columns) else 0

    cover_cards = [
        KPICard("AS OF DATE",   as_of_date,           C["cyan"],   width=4.2*cm, height=2.2*cm),
        KPICard("POSITIONS",    str(n_pos),            C["purple"], width=3.2*cm, height=2.2*cm),
        KPICard("INVESTED",     _fmt_pct(invested),    C["cyan"],   width=3.2*cm, height=2.2*cm),
        KPICard("ENTRY ALERTS", str(n_entry),          C["green"],  width=3.2*cm, height=2.2*cm),
        KPICard("EXIT ALERTS",  str(n_exit),
                C["red"] if n_exit else C["muted"],    width=3.2*cm, height=2.2*cm),
    ]
    total_w = sum(c.width + 0.3*cm for c in cover_cards)
    left_pad = (W - total_w) / 2

    cover_row_items = [[Spacer(left_pad, 1)] + cover_cards]
    cover_tbl = Table(cover_row_items,
                      colWidths=[left_pad] + [c.width + 0.3*cm for c in cover_cards])
    cover_tbl.setStyle(TableStyle([
        ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
    ]))
    story.append(cover_tbl)

    story.append(Spacer(1, 1.5*cm))

    # AUM badge
    if aum:
        aum_str = f"AUM  €{aum/1e6:.0f}M" if aum >= 1e6 else f"AUM  €{aum:,.0f}"
        story.append(Paragraph(aum_str, styles["cover_sub"]))

    story.append(Spacer(1, 2.0*cm))
    action_label = "⚡ ACTION REQUIRED THIS WEEK" if actions.get("any_action") \
        else "✓ NO ACTION REQUIRED — PORTFOLIO UNCHANGED"
    action_color = C["red"] if actions.get("any_action") else C["green"]
    story.append(Paragraph(
        f'<font color="{action_color}">{action_label}</font>',
        ParagraphStyle("action", fontName="Helvetica-Bold", fontSize=11,
                       textColor=rl_colors.HexColor(action_color),
                       alignment=TA_CENTER)
    ))

    story.append(Spacer(1, 3.0*cm))
    story.append(Paragraph(
        f"RF US: {rf_us:.2%}  ·  RF EU: {rf_eu:.2%}",
        styles["center"]
    ))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "CONFIDENTIAL — For authorised recipients only. "
        "All decisions require human review.",
        styles["muted"]
    ))

    story.append(NextPageTemplate("body"))
    story.append(PageBreak())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PAGE 2 — Executive Dashboard
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story += _section("Executive Dashboard", styles)

    dash_cards = [
        KPICard("TOTAL POSITIONS", str(n_pos),           C["cyan"]),
        KPICard("INVESTED",        _fmt_pct(invested),   C["purple"]),
        KPICard("CASH",            _fmt_pct(1-invested), C["cyan"]),
        KPICard("ENTRY SIGNALS",   str(n_entry),         C["green"]),
    ]
    story.append(
        Table([[dash_cards]], colWidths=[CONTENT_W],
              style=[("ALIGN",(0,0),(-1,-1),"CENTER"),
                     ("VALIGN",(0,0),(-1,-1),"MIDDLE")])
    )
    story.append(Spacer(1, 0.5*cm))

    # Side-by-side: bucket allocation + score distributions
    chart_bucket = _chart_bucket_allocation(portfolio) if not portfolio.empty \
        else Spacer(1, 4*cm)
    chart_scores = _chart_score_distribution(scored_df)

    story.append(_side_by_side(
        chart_bucket, chart_scores,
        col_widths=[9.0*cm, CONTENT_W - 9.0*cm]
    ))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PAGE 3 — Portfolio Holdings
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(PageBreak())
    story += _section("Current Portfolio Holdings", styles)

    if not portfolio.empty:
        story.append(
            _build_portfolio_table(portfolio, scored_df, params, styles)
        )
        story.append(Spacer(1, 0.3*cm))
        cash_w = 1.0 - invested
        story.append(Paragraph(
            f"Cash reserve: <b>{_fmt_pct(cash_w)}</b>  ·  "
            f"Max per position: 8%  ·  Max per bucket: 35%",
            styles["muted"]
        ))
        story.append(Spacer(1, 0.6*cm))
        story.append(_chart_kelly_bars(portfolio, scored_df))
    else:
        story.append(Paragraph("No active portfolio positions.", styles["muted"]))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PAGE 4 — Signal Analysis (radar charts for top holdings)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(PageBreak())
    story += _section("Signal Deep Dive — Portfolio Holdings", styles)

    story.append(Paragraph(
        "Radar charts decompose each position into its five EARKE signal dimensions. "
        "All axes: 0 (worst) → 1 (best). Anti-Crowding = 1 − X_C.",
        styles["muted"]
    ))
    story.append(Spacer(1, 0.3*cm))

    if not portfolio.empty and not scored_df.empty:
        score_map = scored_df.set_index("ticker").to_dict("index")
        held = portfolio.sort_values("weight", ascending=False)["ticker"].tolist()
        # Radar charts in rows of 3
        for i in range(0, len(held), 3):
            batch = held[i:i+3]
            radars = []
            for t in batch:
                row = score_map.get(t, {})
                row["ticker"] = t
                radars.append(_chart_radar(pd.Series(row), t))
            while len(radars) < 3:
                radars.append(Spacer(1, 1))
            story.append(
                _side_by_side(*radars, col_widths=[CONTENT_W/3]*3)
            )
            story.append(Spacer(1, 0.2*cm))
    else:
        story.append(Paragraph("No portfolio data for signal charts.", styles["muted"]))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PAGE 5 — Actions (Exits / Entries / Rebalances)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(PageBreak())
    story += _section("This Week's Actions", styles)

    # Exits
    story += _subsection("Positions to Exit", styles)
    exit_tbl = _build_exits_table(exits)
    if exit_tbl:
        story.append(exit_tbl)
    else:
        story.append(Paragraph("No exit signals triggered.", styles["muted"]))

    story.append(Spacer(1, 0.5*cm))

    # Entries
    story += _subsection("New Entry Candidates", styles)
    entry_tbl = _build_entries_table(entries)
    if entry_tbl:
        story.append(entry_tbl)
    else:
        story.append(Paragraph("No new entry candidates.", styles["muted"]))

    story.append(Spacer(1, 0.5*cm))

    # Watch list
    story += _subsection("Watch List — Crowding Elevated", styles)
    watch_tbl = _build_watch_table(watch_list, params)
    if watch_tbl:
        story.append(watch_tbl)
    else:
        story.append(Paragraph("Watch list is clear.", styles["muted"]))

    # Rebalances
    if not diff.empty:
        reb = diff[diff["action"].isin(["increase", "decrease"])]
        if not reb.empty:
            story.append(Spacer(1, 0.5*cm))
            story += _subsection("Rebalances", styles)
            reb_rows = [["Ticker", "Old Weight", "New Weight", "Action"]]
            reb_style = list(_BASE_TABLE_STYLE)
            for ri, (_, r) in enumerate(reb.iterrows(), start=1):
                reb_rows.append([
                    r["ticker"],
                    _fmt_pct(r.get("old_weight")),
                    _fmt_pct(r.get("new_weight")),
                    r.get("action", "—"),
                ])
                color = C["green"] if r.get("action") == "increase" else C["amber"]
                reb_style.append(("TEXTCOLOR", (3, ri), (3, ri), rl_colors.HexColor(color)))
            reb_tbl = Table(reb_rows, colWidths=[3*cm, 3*cm, 3*cm, 3*cm], repeatRows=1)
            reb_tbl.setStyle(TableStyle(reb_style))
            story.append(reb_tbl)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PAGE 6 — Universe Scanner
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(PageBreak())
    story += _section("Universe Scanner — Top 15 Candidates", styles)
    story.append(Paragraph(
        "Ranked by composite score (X_E × X_P × (1−X_C)). "
        "Entry = YES requires composite ≥ 0.30 and crowding ≤ 0.40.",
        styles["muted"]
    ))
    story.append(Spacer(1, 0.25*cm))

    if not scored_df.empty:
        story.append(_build_universe_table(scored_df, params))
    else:
        story.append(Paragraph("No scored universe data available.", styles["muted"]))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PAGE 7 — FX & Capital Flows
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(PageBreak())
    story += _section("FX & Capital Flows", styles)

    if flow_result:
        usd_index = flow_result.get("usd_index", 0.0) or 0.0
        usd_trend = flow_result.get("usd_trend", "neutral")
        fx_df     = flow_result.get("fx_momentum",   pd.DataFrame())
        sect_df   = flow_result.get("sector_flows",  pd.DataFrame())
        country_df= flow_result.get("country_flows", pd.DataFrame())
        implications = flow_result.get("implications", [])

        # USD gauge + FX bar side by side
        usd_gauge = _chart_usd_gauge(usd_index, usd_trend)
        fx_bar    = _chart_fx_bar(fx_df) if not fx_df.empty else Spacer(1, 1)
        story.append(_side_by_side(
            usd_gauge, fx_bar,
            col_widths=[6*cm, CONTENT_W - 6*cm]
        ))
        story.append(Spacer(1, 0.4*cm))

        # FX table
        story += _subsection("FX Pair Returns", styles)
        fx_tbl = _build_fx_table(fx_df)
        if fx_tbl:
            story.append(fx_tbl)

        story.append(Spacer(1, 0.4*cm))

        # Sector heatmap
        story += _subsection("Sector Flow Heatmap", styles)
        story.append(_chart_sector_heatmap(sect_df))

        # Country flows
        if not country_df.empty:
            story.append(Spacer(1, 0.3*cm))
            story += _subsection("Country Capital Flows (EUR-adjusted)", styles)
            c_rows = [["Region", "Index", "4W EUR", "13W EUR", "Flow Direction"]]
            c_style = list(_BASE_TABLE_STYLE)
            for ri, (_, r) in enumerate(country_df.iterrows(), start=1):
                direction = r.get("flow_direction", "—")
                c_rows.append([
                    r.get("region", "—"),
                    r.get("broad_index", "—"),
                    _fmt_ret(r.get("eur_ret_4w")),
                    _fmt_ret(r.get("eur_ret_13w")),
                    direction,
                ])
                fc = r.get("flow_score", 0) or 0
                col = C["green"] if fc > 0.1 else (C["red"] if fc < -0.1 else C["muted"])
                c_style.append(("TEXTCOLOR", (4, ri), (4, ri), rl_colors.HexColor(col)))
            c_tbl = Table(c_rows, colWidths=[2.5*cm, 4*cm, 2.5*cm, 2.5*cm, 3*cm], repeatRows=1)
            c_tbl.setStyle(TableStyle(c_style))
            story.append(c_tbl)

        # Implications
        if implications:
            story.append(Spacer(1, 0.4*cm))
            story += _subsection("Portfolio Implications", styles)
            for imp in implications:
                story.append(Paragraph(f"• {imp}", styles["body"]))
    else:
        story.append(Paragraph("Capital flow data not available for this run.", styles["muted"]))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PAGE 8 — Macro + Risk Reference
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(PageBreak())
    story += _section("Macro Context & Risk Parameters", styles)

    story += _subsection("Risk-Free Rates", styles)
    story.append(_build_macro_table(rf_us, rf_eu))

    story.append(Spacer(1, 0.5*cm))
    story += _subsection("Kelly & Constraint Parameters", styles)

    kp = params.get("kelly", {})
    sp = params.get("signals", {})
    param_rows = [
        ["Parameter", "Value", "Description"],
        ["Kelly fraction",      f"{kp.get('fraction', 0.25):.0%}",
         "Fractional Kelly multiplier (epistemic discount)"],
        ["Max position",        f"{kp.get('max_position', 0.08):.0%}",
         "Per-stock cap — Goodhart immunity"],
        ["Max bucket",          f"{kp.get('max_bucket', 0.35):.0%}",
         "Megatrend concentration cap"],
        ["Min position",        f"{kp.get('min_position', 0.02):.0%}",
         "Below this → position dropped"],
        ["Cash reserve",        f"{kp.get('cash_reserve', 0.10):.0%}",
         "Minimum liquidity buffer"],
        ["Entry threshold",     f"{sp.get('entry_threshold', 0.30):.2f}",
         "Composite score required for entry"],
        ["Crowding entry max",  f"{sp.get('crowding_entry_max', 0.40):.2f}",
         "Maximum X_C at entry (crowding gate)"],
        ["Crowding exit",       f"{sp.get('crowding_exit_threshold', 0.75):.2f}",
         "X_C ≥ this → mandatory exit"],
        ["Quality exit",        f"{sp.get('quality_exit_threshold', 0.25):.2f}",
         "X_P ≤ this → mandatory exit"],
        ["Impact scaling c",    f"{kp.get('impact_scaling', 1.0):.1f}",
         "Universal Square-Root Law constant (Bouchaud)"],
    ]
    p_style = list(_BASE_TABLE_STYLE) + [("ALIGN", (2, 0), (2, -1), "LEFT")]
    p_tbl = Table(param_rows, colWidths=[4.5*cm, 2.5*cm, CONTENT_W - 7.0*cm], repeatRows=1)
    p_tbl.setStyle(TableStyle(p_style))
    story.append(p_tbl)

    story.append(Spacer(1, 0.5*cm))
    story += _subsection("EARKE Module Implementation Status", styles)
    mod_rows = [
        ["Module", "Spec Component", "Status"],
        ["I — Kausales Gedächtnis", "CD-NKBF · bitemporale t_e/t_k · Ghost States", "PARTIAL"],
        ["II — Deterministischer Anker", "X_E · X_P · X_C · μ_base (Eq. 4–7)", "COMPLETE"],
        ["III — Hybrider Motor", "UDE · Deep Ensembles · σ²_epist (Eq. 8–9)", "DEFERRED"],
        ["IV — Kontrafaktisches Labor", "PI-SBDM · Lévy diffusion · minimax λ* (Eq. 10)", "DEFERRED"],
        ["V — Hard-Capacity Kelly", "Full Eq. 12 · Not-Aus · AUM-Asymptote", "PARTIAL"],
    ]
    m_style = list(_BASE_TABLE_STYLE)
    status_colors = {"COMPLETE": C["green"], "PARTIAL": C["amber"], "DEFERRED": C["red"]}
    for ri in range(1, len(mod_rows)):
        status = mod_rows[ri][2]
        m_style.append(("TEXTCOLOR", (2, ri), (2, ri),
                        rl_colors.HexColor(status_colors.get(status, C["muted"]))))
        m_style.append(("FONTNAME",  (2, ri), (2, ri), "Helvetica-Bold"))
        m_style.append(("ALIGN",     (1, ri), (1, ri), "LEFT"))
    m_tbl = Table(mod_rows, colWidths=[4.5*cm, 9.5*cm, 2.5*cm], repeatRows=1)
    m_tbl.setStyle(TableStyle(m_style))
    story.append(m_tbl)

    # ─ Footer note
    story.append(Spacer(1, 1.0*cm))
    story.append(HRFlowable(width=CONTENT_W, thickness=0.5,
                            color=_hex("border"), spaceBefore=4, spaceAfter=6))
    story.append(Paragraph(
        f"Generated by EARKE Quant 3.0 on {as_of_date}. "
        "Paradigm: Minimax-Robust Control under Non-Ergodicity. "
        "P(Ruin) = 0 is a hard constraint. "
        "This document is PRIVATE & CONFIDENTIAL.",
        styles["muted"]
    ))

    # ── Build PDF ─────────────────────────────────────────────────────────
    doc.build(story)
    logger.info(f"PDF report saved to {fname}")
    return fname
