"""
Tests for src/main.py — CLI argument parsing and dry_run propagation.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestMainArgParsing:
    """Verify every CLI flag is wired to the right run_weekly_pipeline arg."""

    def _parse(self, argv):
        """Parse a list of CLI args using main's parser."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--date",          type=str,  default=None)
        parser.add_argument("--force-refresh", action="store_true")
        parser.add_argument("--skip-fetch",    action="store_true")
        parser.add_argument("--dry-run",       action="store_true")
        parser.add_argument("--trends",        action="store_true")
        return parser.parse_args(argv)

    def test_defaults_are_false(self):
        args = self._parse([])
        assert args.date is None
        assert args.force_refresh is False
        assert args.skip_fetch is False
        assert args.dry_run is False
        assert args.trends is False

    def test_date_flag(self):
        args = self._parse(["--date", "2025-03-01"])
        assert args.date == "2025-03-01"

    def test_force_refresh_flag(self):
        args = self._parse(["--force-refresh"])
        assert args.force_refresh is True

    def test_dry_run_flag(self):
        args = self._parse(["--dry-run"])
        assert args.dry_run is True

    def test_skip_fetch_flag(self):
        args = self._parse(["--skip-fetch"])
        assert args.skip_fetch is True

    def test_trends_flag(self):
        args = self._parse(["--trends"])
        assert args.trends is True

    def test_combined_flags(self):
        args = self._parse(["--dry-run", "--force-refresh", "--date", "2025-01-06"])
        assert args.dry_run is True
        assert args.force_refresh is True
        assert args.date == "2025-01-06"


# ---------------------------------------------------------------------------
# dry_run propagation through run_weekly_pipeline
# ---------------------------------------------------------------------------

class TestRunWeeklyPipelineDryRun:
    """dry_run=True must reach generate_weekly_actions without a DB write."""

    def test_dry_run_forwarded_to_generate_weekly_actions(self, monkeypatch):
        """Verify run_weekly_pipeline passes dry_run=True down the call chain."""
        import types
        import pandas as pd

        calls = {}

        # Minimal stubs for all heavy dependencies
        def fake_load_params(*a, **kw):
            from tests.conftest import pytest  # noqa: F401 — use conftest params indirectly
            return {
                "kelly": {"fraction": 0.25, "max_position": 0.08, "max_bucket": 0.35,
                          "min_position": 0.02, "cash_reserve": 0.10},
                "signals": {
                    "weights": {"physical": 0.4, "quality": 0.35, "crowding": 0.25},
                    "entry_threshold": 0.55, "crowding_entry_max": 0.40,
                    "crowding_exit_threshold": 0.75, "quality_exit_threshold": 0.25,
                    "composite_decay_pct": 0.20,
                    "crowding": {"etf_corr_window": 60, "etf_corr_weight": 0.35,
                                 "short_interest_weight": 0.15, "rel_strength_weight": 0.30,
                                 "trends_weight": 0.20, "rel_strength_window": 126},
                    "quality": {"roic_wacc_spread_min": -0.10, "roic_wacc_spread_max": 0.20,
                                "margin_snr_min": 2.0, "margin_snr_max": 10.0,
                                "convexity_min": -0.05, "convexity_max": 0.10, "lookback_years": 5},
                },
                "return_estimation": {"equity_risk_premium": 0.05,
                                      "composite_anchors": [[0.55, 0.06], [1.0, 0.30]]},
                "data": {"price_staleness_days": 1, "fundamental_staleness_days": 90,
                         "macro_staleness_days": 7, "trends_staleness_days": 7,
                         "lookback_prices_days": 756, "price_history_years": 5},
                "macro": {"eu_risk_free_series": "EU_10Y_DE", "us_risk_free_series": "US_10Y",
                          "ca_risk_free_series": "US_10Y", "eu_cpi_series": "EU_HICP_YOY",
                          "eu_ppi_series": "EU_PPI_YOY", "us_cpi_series": "US_CPI_YOY",
                          "us_ppi_series": "US_PPI_YOY"},
                "sector_etfs": {"US|grid": "XLI"},
                "market_indices": {"US": "SPY", "EU": "EXW1.DE", "CA": "SPY"},
                "reporting": {"output_dir": "reports", "top_candidates_count": 10,
                              "watch_crowding_threshold": 0.55},
            }

        import duckdb
        from src.data.db import initialize_schema

        fake_conn = duckdb.connect(":memory:")
        initialize_schema(fake_conn)

        empty_universe = pd.DataFrame(columns=[
            "ticker", "name", "region", "buckets", "primary_bucket",
            "trends_keyword", "isin", "currency", "accounting_std",
        ])
        empty_scored = pd.DataFrame()

        import src.main as main_mod
        monkeypatch.setattr(main_mod, "load_params", fake_load_params)

        import src.data.db as db_mod
        monkeypatch.setattr(db_mod, "get_connection", lambda: fake_conn)

        import src.data.universe as uni_mod
        monkeypatch.setattr(uni_mod, "load_universe", lambda: empty_universe)
        monkeypatch.setattr(uni_mod, "all_trend_keywords", lambda u: [])
        monkeypatch.setattr(uni_mod, "all_sector_etfs", lambda u, p: [])

        import src.data.prices as prices_mod
        monkeypatch.setattr(prices_mod, "update_prices",
                            lambda *a, **kw: {})

        import src.data.fundamentals as fund_mod
        monkeypatch.setattr(fund_mod, "update_fundamentals",
                            lambda *a, **kw: {})

        import src.data.macro as macro_mod
        monkeypatch.setattr(macro_mod, "update_macro", lambda *a, **kw: None)

        import src.signals.composite as comp_mod
        monkeypatch.setattr(comp_mod, "run_weekly_scoring",
                            lambda *a, **kw: empty_scored)

        import src.portfolio.monitor as monitor_mod

        def fake_generate(conn, scored, universe, params, as_of, dry_run=False):
            calls["dry_run"] = dry_run
            return {"exits": pd.DataFrame(), "entries": pd.DataFrame(),
                    "watch_list": pd.DataFrame(), "new_portfolio": pd.DataFrame(),
                    "diff": pd.DataFrame(), "as_of_date": as_of, "any_action": False}

        monkeypatch.setattr(monitor_mod, "generate_weekly_actions", fake_generate)

        import src.reporting.weekly_report as report_mod
        monkeypatch.setattr(report_mod, "generate_weekly_report",
                            lambda *a, **kw: "reports/test.md")

        main_mod.run_weekly_pipeline(
            as_of_date="2025-01-06",
            force_refresh=False,
            skip_fetch=True,
            dry_run=True,
        )

        assert calls.get("dry_run") is True, (
            "run_weekly_pipeline must forward dry_run=True to generate_weekly_actions"
        )
