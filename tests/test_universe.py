"""
Tests for src/data/universe.py.

load_universe uses a temp YAML file written by the fixture — no
dependency on the real config/universe.yaml.
All other functions are pure (take a DataFrame) and tested directly.
"""
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from src.data.universe import (
    load_universe,
    get_sector_etf,
    get_market_index,
    get_bucket_tickers,
    get_universe_by_region,
    get_stock,
    all_trend_keywords,
    all_sector_etfs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

YAML_CONTENT = textwrap.dedent("""\
    stocks:
      - ticker: "ETN"
        name: "Eaton Corp"
        exchange: NYSE
        region: US
        currency: USD
        accounting_std: GAAP
        buckets: [grid, ai_infra]
        primary_bucket: grid
        trends_keyword: "electrical grid"

      - ticker: "RHM.DE"
        eodhd_ticker: "RHM.XETRA"
        name: "Rheinmetall"
        exchange: XETRA
        region: EU
        currency: EUR
        accounting_std: IFRS
        buckets: [defense]
        primary_bucket: defense
        isin: "DE0007030009"
        trends_keyword: "Rheinmetall defense"

      - ticker: "CCJ"
        name: "Cameco"
        exchange: NYSE
        region: US
        currency: USD
        accounting_std: GAAP
        buckets: [nuclear, critical_materials]
        primary_bucket: nuclear
        trends_keyword: "uranium nuclear"

      - ticker: "AWK"
        name: "American Water Works"
        exchange: NYSE
        region: US
        currency: USD
        accounting_std: GAAP
        buckets: [water]
        primary_bucket: water
        trends_keyword: "water infrastructure"

      - ticker: "ETN"
        name: "Eaton Corp duplicate"
        exchange: NYSE
        region: US
        currency: USD
        accounting_std: GAAP
        buckets: [grid]
        primary_bucket: grid
        trends_keyword: "eaton duplicate"
""")

_PARAMS = {
    "sector_etfs": {
        "US|grid":     "XLI",
        "EU|defense":  "DFEN.L",
        "US|nuclear":  "URA",
        "US|water":    "PHO",
    },
    "market_indices": {"US": "SPY", "EU": "EXW1.DE", "CA": "SPY"},
}


@pytest.fixture
def yaml_path(tmp_path) -> Path:
    p = tmp_path / "universe.yaml"
    p.write_text(YAML_CONTENT)
    return p


@pytest.fixture
def universe(yaml_path) -> pd.DataFrame:
    return load_universe(yaml_path)


# ===========================================================================
# 1. load_universe
# ===========================================================================

class TestLoadUniverse:
    def test_returns_dataframe(self, universe):
        assert isinstance(universe, pd.DataFrame)

    def test_expected_columns(self, universe):
        for col in ["ticker", "name", "exchange", "region", "currency",
                    "accounting_std", "buckets", "primary_bucket",
                    "eodhd_ticker", "isin", "trends_keyword"]:
            assert col in universe.columns

    def test_correct_row_count(self, universe):
        # YAML has 5 entries but ETN appears twice — dedup → 4 rows
        assert len(universe) == 4

    def test_duplicate_ticker_deduplicated(self, universe):
        # First occurrence kept; second silently dropped
        assert universe["ticker"].value_counts()["ETN"] == 1

    def test_duplicate_retains_first_occurrence(self, universe):
        etn = universe[universe["ticker"] == "ETN"].iloc[0]
        assert etn["name"] == "Eaton Corp"   # not "Eaton Corp duplicate"

    def test_buckets_is_list(self, universe):
        for _, row in universe.iterrows():
            assert isinstance(row["buckets"], list)

    def test_multi_bucket_stock(self, universe):
        etn = universe[universe["ticker"] == "ETN"].iloc[0]
        assert "grid" in etn["buckets"]
        assert "ai_infra" in etn["buckets"]

    def test_optional_fields_default_to_none(self, universe):
        etn = universe[universe["ticker"] == "ETN"].iloc[0]
        # pandas stores absent optional fields as NaN in mixed-type columns
        assert pd.isna(etn["isin"])
        assert pd.isna(etn["eodhd_ticker"])

    def test_optional_fields_populated_when_present(self, universe):
        rhm = universe[universe["ticker"] == "RHM.DE"].iloc[0]
        assert rhm["eodhd_ticker"] == "RHM.XETRA"
        assert rhm["isin"] == "DE0007030009"

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_universe(tmp_path / "nonexistent.yaml")


# ===========================================================================
# 2. get_sector_etf
# ===========================================================================

class TestGetSectorEtf:
    def test_known_bucket_region(self):
        assert get_sector_etf("grid", "US", _PARAMS) == "XLI"
        assert get_sector_etf("defense", "EU", _PARAMS) == "DFEN.L"
        assert get_sector_etf("nuclear", "US", _PARAMS) == "URA"

    def test_unknown_bucket_returns_none(self):
        assert get_sector_etf("ai_infra", "US", _PARAMS) is None

    def test_unknown_region_returns_none(self):
        assert get_sector_etf("grid", "CA", _PARAMS) is None

    def test_empty_params_returns_none(self):
        assert get_sector_etf("grid", "US", {}) is None


# ===========================================================================
# 3. get_market_index
# ===========================================================================

class TestGetMarketIndex:
    def test_known_regions(self):
        assert get_market_index("US", _PARAMS) == "SPY"
        assert get_market_index("EU", _PARAMS) == "EXW1.DE"
        assert get_market_index("CA", _PARAMS) == "SPY"

    def test_unknown_region_defaults_to_spy(self):
        assert get_market_index("JP", _PARAMS) == "SPY"

    def test_empty_params_returns_spy(self):
        assert get_market_index("US", {}) == "SPY"


# ===========================================================================
# 4. get_bucket_tickers
# ===========================================================================

class TestGetBucketTickers:
    def test_single_bucket_match(self, universe):
        tickers = get_bucket_tickers(universe, "defense")
        assert tickers == ["RHM.DE"]

    def test_multi_stock_bucket(self, universe):
        tickers = get_bucket_tickers(universe, "grid")
        assert "ETN" in tickers

    def test_multi_bucket_stock_included(self, universe):
        # ETN is in both grid and ai_infra
        assert "ETN" in get_bucket_tickers(universe, "ai_infra")

    def test_cross_bucket_stock_in_both(self, universe):
        # CCJ is in both nuclear and critical_materials
        assert "CCJ" in get_bucket_tickers(universe, "nuclear")
        assert "CCJ" in get_bucket_tickers(universe, "critical_materials")

    def test_empty_bucket_returns_empty_list(self, universe):
        assert get_bucket_tickers(universe, "nonexistent_bucket") == []

    def test_returns_list(self, universe):
        result = get_bucket_tickers(universe, "grid")
        assert isinstance(result, list)


# ===========================================================================
# 5. get_universe_by_region
# ===========================================================================

class TestGetUniverseByRegion:
    def test_us_region(self, universe):
        us = get_universe_by_region(universe, "US")
        assert set(us["ticker"]) == {"ETN", "CCJ", "AWK"}

    def test_eu_region(self, universe):
        eu = get_universe_by_region(universe, "EU")
        assert set(eu["ticker"]) == {"RHM.DE"}

    def test_unknown_region_returns_empty(self, universe):
        ca = get_universe_by_region(universe, "CA")
        assert ca.empty

    def test_returns_copy_not_view(self, universe):
        us = get_universe_by_region(universe, "US")
        us["region"] = "MODIFIED"
        assert (universe["region"] != "MODIFIED").all()


# ===========================================================================
# 6. get_stock
# ===========================================================================

class TestGetStock:
    def test_known_ticker_returns_dict(self, universe):
        stock = get_stock(universe, "ETN")
        assert isinstance(stock, dict)
        assert stock["ticker"] == "ETN"
        assert stock["primary_bucket"] == "grid"

    def test_unknown_ticker_returns_none(self, universe):
        assert get_stock(universe, "GHOST") is None

    def test_returned_dict_has_expected_keys(self, universe):
        stock = get_stock(universe, "RHM.DE")
        assert "name" in stock
        assert "region" in stock
        assert "buckets" in stock


# ===========================================================================
# 7. all_trend_keywords
# ===========================================================================

class TestAllTrendKeywords:
    def test_returns_list(self, universe):
        kws = all_trend_keywords(universe)
        assert isinstance(kws, list)

    def test_contains_expected_keywords(self, universe):
        kws = all_trend_keywords(universe)
        assert "electrical grid" in kws
        assert "Rheinmetall defense" in kws
        assert "uranium nuclear" in kws

    def test_no_duplicates(self, universe):
        kws = all_trend_keywords(universe)
        assert len(kws) == len(set(kws))

    def test_none_keywords_excluded(self, tmp_path):
        yaml = textwrap.dedent("""\
            stocks:
              - ticker: "X"
                name: "No keyword"
                exchange: NYSE
                region: US
                currency: USD
                accounting_std: GAAP
                buckets: [grid]
                primary_bucket: grid
        """)
        p = tmp_path / "u.yaml"
        p.write_text(yaml)
        df  = load_universe(p)
        kws = all_trend_keywords(df)
        assert kws == []


# ===========================================================================
# 8. all_sector_etfs
# ===========================================================================

class TestAllSectorEtfs:
    def test_returns_sorted_list(self, universe):
        etfs = all_sector_etfs(universe, _PARAMS)
        assert etfs == sorted(etfs)

    def test_contains_known_etfs(self, universe):
        etfs = all_sector_etfs(universe, _PARAMS)
        assert "XLI"     in etfs   # US|grid
        assert "DFEN.L"  in etfs   # EU|defense
        assert "URA"     in etfs   # US|nuclear
        assert "PHO"     in etfs   # US|water

    def test_includes_market_indices(self, universe):
        etfs = all_sector_etfs(universe, _PARAMS)
        assert "SPY"      in etfs
        assert "EXW1.DE"  in etfs

    def test_no_duplicates(self, universe):
        etfs = all_sector_etfs(universe, _PARAMS)
        assert len(etfs) == len(set(etfs))

    def test_unmapped_bucket_excluded(self, universe):
        # ai_infra has no sector_etf in _PARAMS — shouldn't appear
        etfs = all_sector_etfs(universe, _PARAMS)
        assert "XLK" not in etfs   # no ai_infra ETF defined

    def test_empty_params_returns_only_indices(self, universe):
        params_no_etfs = {"sector_etfs": {}, "market_indices": {"US": "SPY"}}
        etfs = all_sector_etfs(universe, params_no_etfs)
        assert etfs == ["SPY"]
