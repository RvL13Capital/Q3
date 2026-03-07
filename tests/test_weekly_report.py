"""
Tests for src/reporting/weekly_report.py.

generate_weekly_report writes a Markdown file to disk. We use tmp_path
to avoid touching the real filesystem and patch get_risk_free_rate so no
live macro data is needed.
"""
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.reporting.weekly_report import (
    _fmt_pct,
    _fmt_score,
    _traffic_light,
    generate_weekly_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AS_OF = "2025-03-07"

_RF_PATCH = "src.data.macro.get_risk_free_rate"


def _base_params():
    return {
        "signals": {
            "crowding_exit_threshold": 0.75,
        },
        "reporting": {
            "watch_crowding_threshold": 0.55,
            "top_candidates_count": 10,
        },
    }


def _make_portfolio(*tickers, weight=0.08):
    return pd.DataFrame([
        {
            "ticker":          t,
            "weight":          weight,
            "composite_score": 0.65,
            "kelly_25pct":     0.06,
            "bucket_id":       "grid",
        }
        for t in tickers
    ])


def _make_scored(*tickers, composite=0.70, quality=0.55,
                 crowding=0.25, physical=0.65, entry=True):
    return pd.DataFrame([
        {
            "ticker":          t,
            "composite_score": composite,
            "quality_score":   quality,
            "crowding_score":  crowding,
            "physical_norm":   physical,
            "entry_signal":    entry,
            "kelly_25pct":     0.06,
        }
        for t in tickers
    ])


def _make_exits(*tickers, exit_triggered=True, crowding=0.80, quality=0.30):
    return pd.DataFrame([
        {
            "ticker":         t,
            "exit_triggered": exit_triggered,
            "crowding_score": crowding,
            "quality_score":  quality,
            "exit_reasons":   ["crowding_exit"],
        }
        for t in tickers
    ])


def _run_report(tmp_path, actions=None, scored_df=None, params=None, conn=None):
    if actions is None:
        actions = {"any_action": False, "new_portfolio": pd.DataFrame()}
    if scored_df is None:
        scored_df = pd.DataFrame()
    if params is None:
        params = _base_params()
    with patch(_RF_PATCH, return_value=0.043):
        path = generate_weekly_report(
            conn=conn or object(),
            actions=actions,
            scored_df=scored_df,
            params=params,
            as_of_date=AS_OF,
            output_dir=str(tmp_path),
        )
    return path, Path(path).read_text()


# ===========================================================================
# 1. _fmt_pct
# ===========================================================================

class TestFmtPct:
    @pytest.mark.parametrize("val,expected", [
        (0.075,  "7.5%"),
        (0.0,    "0.0%"),
        (1.0,    "100.0%"),
        (-0.05,  "-5.0%"),
    ])
    def test_numeric_values(self, val, expected):
        assert _fmt_pct(val) == expected

    @pytest.mark.parametrize("val", [None, float("nan")])
    def test_missing_returns_dash(self, val):
        assert _fmt_pct(val) == "—"

    def test_custom_decimals(self):
        assert _fmt_pct(0.12345, decimals=2) == "12.35%"


# ===========================================================================
# 2. _fmt_score
# ===========================================================================

class TestFmtScore:
    @pytest.mark.parametrize("val,expected", [
        (0.75, "0.750"),
        (0.0,  "0.000"),
        (1.0,  "1.000"),
    ])
    def test_numeric_values(self, val, expected):
        assert _fmt_score(val) == expected

    @pytest.mark.parametrize("val", [None, float("nan")])
    def test_missing_returns_dash(self, val):
        assert _fmt_score(val) == "—"

    def test_custom_decimals(self):
        assert _fmt_score(0.12345, decimals=2) == "0.12"

    def test_rounds_at_3dp(self):
        # Python's float f-string rounding at 3dp — accept either banker's round
        assert _fmt_score(0.1235) in ("0.123", "0.124")


# ===========================================================================
# 3. _traffic_light
# ===========================================================================

class TestTrafficLight:
    def _p(self, watch=0.55, exit_thr=0.75):
        return {
            "reporting": {"watch_crowding_threshold": watch},
            "signals":   {"crowding_exit_threshold": exit_thr},
        }

    @pytest.mark.parametrize("val", [None, float("nan")])
    def test_missing_returns_white(self, val):
        assert _traffic_light(val, self._p()) == "⚪"

    @pytest.mark.parametrize("val", [0.00, 0.30, 0.54])
    def test_below_watch_is_green(self, val):
        assert _traffic_light(val, self._p()) == "🟢"

    @pytest.mark.parametrize("val", [0.55, 0.60, 0.74])
    def test_watch_zone_is_yellow(self, val):
        assert _traffic_light(val, self._p()) == "🟡"

    @pytest.mark.parametrize("val", [0.75, 0.90, 1.00])
    def test_exit_zone_is_red(self, val):
        assert _traffic_light(val, self._p()) == "🔴"

    def test_exact_boundary_watch(self):
        assert _traffic_light(0.55, self._p()) == "🟡"
        assert _traffic_light(0.54, self._p()) == "🟢"

    def test_exact_boundary_exit(self):
        assert _traffic_light(0.75, self._p()) == "🔴"
        assert _traffic_light(0.74, self._p()) == "🟡"

    def test_custom_thresholds(self):
        p = self._p(watch=0.40, exit_thr=0.60)
        assert _traffic_light(0.35, p) == "🟢"
        assert _traffic_light(0.45, p) == "🟡"
        assert _traffic_light(0.65, p) == "🔴"


# ===========================================================================
# 4. generate_weekly_report — file output
# ===========================================================================

class TestGenerateWeeklyReportFile:
    def test_creates_file_at_correct_path(self, tmp_path):
        path, _ = _run_report(tmp_path)
        assert Path(path).exists()
        assert Path(path).name == f"{AS_OF}_weekly.md"

    def test_returns_string_path(self, tmp_path):
        path, _ = _run_report(tmp_path)
        assert isinstance(path, str)

    def test_output_dir_created_if_missing(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        assert not nested.exists()
        with patch(_RF_PATCH, return_value=0.04):
            generate_weekly_report(
                conn=object(),
                actions={"any_action": False, "new_portfolio": pd.DataFrame()},
                scored_df=pd.DataFrame(),
                params=_base_params(),
                as_of_date=AS_OF,
                output_dir=str(nested),
            )
        assert nested.exists()


# ===========================================================================
# 5. generate_weekly_report — header / metadata
# ===========================================================================

class TestReportHeader:
    def test_date_in_header(self, tmp_path):
        _, text = _run_report(tmp_path)
        assert AS_OF in text

    def test_rf_rates_in_header(self, tmp_path):
        _, text = _run_report(tmp_path)
        assert "4.30%" in text   # 0.043 formatted as 4.30%

    def test_title_present(self, tmp_path):
        _, text = _run_report(tmp_path)
        assert "EARKE Quant 3.0" in text

    def test_generated_footer_present(self, tmp_path):
        _, text = _run_report(tmp_path)
        assert "quantitative scan" in text
        assert "human review" in text


# ===========================================================================
# 6. generate_weekly_report — summary section
# ===========================================================================

class TestReportSummary:
    def test_no_action_message(self, tmp_path):
        _, text = _run_report(tmp_path, actions={
            "any_action": False,
            "new_portfolio": pd.DataFrame(),
        })
        assert "No action required" in text

    def test_action_required_message(self, tmp_path):
        _, text = _run_report(tmp_path, actions={
            "any_action": True,
            "new_portfolio": _make_portfolio("ETN"),
        })
        assert "Action required" in text

    def test_position_count(self, tmp_path):
        portfolio = _make_portfolio("ETN", "CCJ")
        _, text = _run_report(tmp_path, actions={
            "any_action": False,
            "new_portfolio": portfolio,
        })
        assert "Positions: **2**" in text

    def test_invested_and_cash_weights(self, tmp_path):
        portfolio = _make_portfolio("ETN", "CCJ", weight=0.10)  # 20% invested
        _, text = _run_report(tmp_path, actions={
            "any_action": False,
            "new_portfolio": portfolio,
        })
        assert "Invested: **20.0%**" in text
        assert "Cash: **80.0%**" in text

    def test_exit_trigger_count(self, tmp_path):
        exits = pd.concat([
            _make_exits("ETN", exit_triggered=True),
            _make_exits("CCJ", exit_triggered=False),
        ], ignore_index=True)
        _, text = _run_report(tmp_path, actions={
            "any_action": True,
            "new_portfolio": pd.DataFrame(),
            "exits": exits,
        })
        assert "Exit triggers: **1**" in text

    def test_entry_candidates_count(self, tmp_path):
        entries = _make_scored("ETN", "NVDA", "AWK")
        _, text = _run_report(tmp_path, actions={
            "any_action": True,
            "new_portfolio": pd.DataFrame(),
            "entries": entries,
        })
        assert "New entry candidates: **3**" in text

    def test_watch_list_count(self, tmp_path):
        watch = _make_scored("RHM.DE", crowding=0.62)
        _, text = _run_report(tmp_path, actions={
            "any_action": False,
            "new_portfolio": pd.DataFrame(),
            "watch_list": watch,
        })
        assert "Watch list (crowding elevated): **1**" in text


# ===========================================================================
# 7. generate_weekly_report — actions section
# ===========================================================================

class TestReportActions:
    def test_exit_table_rendered(self, tmp_path):
        exits = _make_exits("ETN")
        _, text = _run_report(tmp_path, actions={
            "any_action": True,
            "new_portfolio": pd.DataFrame(),
            "exits": exits,
        })
        assert "Positions to Exit" in text
        assert "ETN" in text
        assert "crowding_exit" in text

    def test_untriggered_exits_not_shown(self, tmp_path):
        exits = _make_exits("ETN", exit_triggered=False)
        _, text = _run_report(tmp_path, actions={
            "any_action": False,
            "new_portfolio": pd.DataFrame(),
            "exits": exits,
        })
        assert "Positions to Exit" not in text

    def test_multiple_exit_reasons_joined(self, tmp_path):
        exits = pd.DataFrame([{
            "ticker": "ETN", "exit_triggered": True,
            "crowding_score": 0.80, "quality_score": 0.20,
            "exit_reasons": ["crowding_exit", "quality_exit"],
        }])
        _, text = _run_report(tmp_path, actions={
            "any_action": True,
            "new_portfolio": pd.DataFrame(),
            "exits": exits,
        })
        assert "crowding_exit; quality_exit" in text

    def test_entry_table_rendered(self, tmp_path):
        entries = _make_scored("CCJ", "NVDA", composite=0.72)
        _, text = _run_report(tmp_path, actions={
            "any_action": True,
            "new_portfolio": pd.DataFrame(),
            "entries": entries,
        })
        assert "New Entry Candidates" in text
        assert "CCJ" in text
        assert "NVDA" in text

    def test_entry_table_capped_at_top_candidates_count(self, tmp_path):
        params  = _base_params()
        params["reporting"]["top_candidates_count"] = 3
        tickers = [f"T{i}" for i in range(10)]
        entries = _make_scored(*tickers)
        _, text = _run_report(tmp_path, actions={
            "any_action": True,
            "new_portfolio": pd.DataFrame(),
            "entries": entries,
        }, params=params)
        # First 3 tickers appear; the 4th (T3) must not
        assert "| T0 |" in text
        assert "| T2 |" in text
        assert "| T3 |" not in text

    def test_rebalance_table_rendered(self, tmp_path):
        diff = pd.DataFrame([
            {"ticker": "ETN", "old_weight": 0.10, "new_weight": 0.08, "action": "decrease"},
            {"ticker": "CCJ", "old_weight": 0.06, "new_weight": 0.09, "action": "increase"},
        ])
        _, text = _run_report(tmp_path, actions={
            "any_action": True,
            "new_portfolio": pd.DataFrame(),
            "diff": diff,
        })
        assert "Rebalances" in text
        assert "ETN" in text
        assert "decrease" in text
        assert "increase" in text

    def test_no_rebalance_section_when_only_new_add(self, tmp_path):
        diff = pd.DataFrame([
            {"ticker": "NVDA", "old_weight": 0.0, "new_weight": 0.08, "action": "add"},
        ])
        _, text = _run_report(tmp_path, actions={
            "any_action": True,
            "new_portfolio": pd.DataFrame(),
            "diff": diff,
        })
        assert "Rebalances" not in text


# ===========================================================================
# 8. generate_weekly_report — portfolio section
# ===========================================================================

class TestReportPortfolio:
    def test_portfolio_section_present_when_held(self, tmp_path):
        portfolio = _make_portfolio("ETN", "CCJ")
        scored    = _make_scored("ETN", "CCJ")
        _, text = _run_report(tmp_path,
                              actions={"any_action": False, "new_portfolio": portfolio},
                              scored_df=scored)
        assert "Current Portfolio" in text
        assert "ETN" in text

    def test_portfolio_section_absent_when_empty(self, tmp_path):
        _, text = _run_report(tmp_path)
        assert "Current Portfolio" not in text

    def test_traffic_light_red_in_portfolio_row(self, tmp_path):
        portfolio = _make_portfolio("ETN")
        scored    = _make_scored("ETN", crowding=0.80)
        _, text = _run_report(tmp_path,
                              actions={"any_action": False, "new_portfolio": portfolio},
                              scored_df=scored)
        # crowding 0.80 ≥ exit threshold 0.75 → 🔴
        assert "🔴" in text

    def test_traffic_light_green_when_crowding_low(self, tmp_path):
        portfolio = _make_portfolio("ETN")
        scored    = _make_scored("ETN", crowding=0.20)
        _, text = _run_report(tmp_path,
                              actions={"any_action": False, "new_portfolio": portfolio},
                              scored_df=scored)
        assert "🟢" in text

    def test_cash_reserve_line(self, tmp_path):
        portfolio = _make_portfolio("ETN", weight=0.10)
        _, text = _run_report(tmp_path,
                              actions={"any_action": False, "new_portfolio": portfolio},
                              scored_df=_make_scored("ETN"))
        assert "*Cash reserve: 90.0%*" in text

    def test_portfolio_sorted_by_weight_desc(self, tmp_path):
        portfolio = pd.DataFrame([
            {"ticker": "ETN", "weight": 0.05, "composite_score": 0.65,
             "kelly_25pct": 0.06, "bucket_id": "grid"},
            {"ticker": "CCJ", "weight": 0.12, "composite_score": 0.70,
             "kelly_25pct": 0.08, "bucket_id": "nuclear"},
        ])
        _, text = _run_report(tmp_path,
                              actions={"any_action": False, "new_portfolio": portfolio},
                              scored_df=_make_scored("ETN", "CCJ"))
        assert text.index("| CCJ |") < text.index("| ETN |")  # CCJ (12%) before ETN (5%)

    def test_portfolio_ticker_missing_from_scores_shows_dashes(self, tmp_path):
        """Score columns render '—' when the ticker has no signal data."""
        portfolio = _make_portfolio("ORPHAN")
        scored    = _make_scored("ETN")           # ORPHAN has no score
        _, text = _run_report(tmp_path,
                              actions={"any_action": False, "new_portfolio": portfolio},
                              scored_df=scored)
        assert "ORPHAN" in text
        assert "—" in text                         # quality/crowding shown as dash


# ===========================================================================
# 9. generate_weekly_report — universe top candidates
# ===========================================================================

class TestReportUniverseTop:
    def test_top_candidates_section_present(self, tmp_path):
        scored = _make_scored("ETN", "CCJ", "NVDA")
        _, text = _run_report(tmp_path, scored_df=scored)
        assert "Universe Top Candidates" in text

    def test_entry_signal_rendered_as_checkmark(self, tmp_path):
        scored = _make_scored("ETN", entry=True)
        _, text = _run_report(tmp_path, scored_df=scored)
        assert "✅" in text

    def test_no_entry_signal_rendered_as_dash(self, tmp_path):
        scored = _make_scored("CCJ", entry=False)
        _, text = _run_report(tmp_path, scored_df=scored)
        # At least one cell should show "—" for the entry column
        assert "—" in text

    def test_candidates_capped_at_15(self, tmp_path):
        tickers = [f"T{i:02d}" for i in range(25)]
        scored  = _make_scored(*tickers)
        _, text = _run_report(tmp_path, scored_df=scored)
        uc_idx    = text.index("Universe Top Candidates")
        uc_block  = text[uc_idx:]
        data_rows = [l for l in uc_block.splitlines()
                     if l.startswith("| T") and "Composite" not in l]
        assert len(data_rows) == 15

    def test_nan_composite_excluded(self, tmp_path):
        scored  = _make_scored("ETN")
        scored2 = pd.DataFrame([{
            "ticker": "GHOST", "composite_score": float("nan"),
            "quality_score": 0.5, "crowding_score": 0.2,
            "physical_norm": 0.6, "entry_signal": False, "kelly_25pct": 0.0,
        }])
        all_scored = pd.concat([scored, scored2], ignore_index=True)
        _, text = _run_report(tmp_path, scored_df=all_scored)
        assert "GHOST" not in text

    def test_sorted_by_composite_descending(self, tmp_path):
        scored = pd.DataFrame([
            {"ticker": "LOW",  "composite_score": 0.60, "quality_score": 0.5,
             "crowding_score": 0.2, "physical_norm": 0.6, "entry_signal": False, "kelly_25pct": 0.04},
            {"ticker": "HIGH", "composite_score": 0.85, "quality_score": 0.7,
             "crowding_score": 0.1, "physical_norm": 0.8, "entry_signal": True,  "kelly_25pct": 0.08},
        ])
        _, text = _run_report(tmp_path, scored_df=scored)
        uc_idx  = text.index("Universe Top Candidates")
        uc_text = text[uc_idx:]
        assert uc_text.index("HIGH") < uc_text.index("LOW")


# ===========================================================================
# 10. generate_weekly_report — watch list section
# ===========================================================================

class TestReportWatchList:
    def test_watch_list_section_rendered(self, tmp_path):
        watch = _make_scored("RHM.DE", crowding=0.62)
        _, text = _run_report(tmp_path, actions={
            "any_action": False,
            "new_portfolio": pd.DataFrame(),
            "watch_list": watch,
        })
        assert "Watch List" in text
        assert "RHM.DE" in text

    def test_watch_list_absent_when_empty(self, tmp_path):
        _, text = _run_report(tmp_path)
        assert "Watch List" not in text

    def test_exit_threshold_shown_in_watch_table(self, tmp_path):
        watch = _make_scored("ETN", crowding=0.60)
        _, text = _run_report(tmp_path, actions={
            "any_action": False,
            "new_portfolio": pd.DataFrame(),
            "watch_list": watch,
        })
        assert "0.75" in text   # crowding_exit_threshold from _base_params


# ===========================================================================
# 11. generate_weekly_report — macro context
# ===========================================================================

class TestReportMacro:
    def test_macro_section_present(self, tmp_path):
        _, text = _run_report(tmp_path)
        assert "Macro Context" in text

    def test_us_and_eu_rates_shown(self, tmp_path):
        with patch(_RF_PATCH, side_effect=[0.043, 0.028]):
            path = generate_weekly_report(
                conn=object(),
                actions={"any_action": False, "new_portfolio": pd.DataFrame()},
                scored_df=pd.DataFrame(),
                params=_base_params(),
                as_of_date=AS_OF,
                output_dir=str(tmp_path),
            )
        text = Path(path).read_text()
        assert "4.30%" in text
        assert "2.80%" in text
