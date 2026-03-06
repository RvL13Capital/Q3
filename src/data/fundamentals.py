"""
Fundamental data fetcher: SEC EDGAR (US/GAAP), EODHD (EU/IFRS), yfinance (fallback).
Computes ROIC with IFRS 16 adjustment for EU stocks.
"""
import logging
import time
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Default tax rates when not reported
DEFAULT_TAX_RATE = {"GAAP": 0.21, "IFRS": 0.25}

# ---------------------------------------------------------------------------
# Invested capital computation (IFRS 16 adjusted)
# ---------------------------------------------------------------------------

def compute_invested_capital(row: pd.Series) -> float:
    """
    GAAP: total_equity + total_debt - cash - goodwill
    IFRS: same, minus right_of_use_assets, plus lease_liabilities (IFRS 16 strip).
    Returns NaN if required fields are missing.
    """
    equity = row.get("total_equity")
    debt   = row.get("total_debt")
    cash   = row.get("cash")
    gw     = row.get("goodwill") or 0.0

    if any(pd.isna(v) for v in [equity, debt, cash]):
        return float("nan")

    ic = equity + debt - cash - gw

    if row.get("accounting_std") == "IFRS":
        rou = row.get("right_of_use_assets") or 0.0
        ll  = row.get("lease_liabilities") or 0.0
        ic = ic - rou + ll

    return ic if ic > 0 else float("nan")


def compute_nopat(row: pd.Series) -> float:
    """NOPAT = EBIT * (1 - effective_tax_rate)."""
    ebit = row.get("ebit")
    if pd.isna(ebit):
        return float("nan")
    tax_rate = row.get("effective_tax_rate")
    if pd.isna(tax_rate) or tax_rate is None:
        tax_rate = DEFAULT_TAX_RATE.get(row.get("accounting_std", "GAAP"), 0.21)
    return ebit * (1.0 - tax_rate)


def compute_roic(row: pd.Series) -> float:
    """NOPAT / invested_capital. Returns NaN if denominator <= 0."""
    nopat = row.get("nopat")
    ic    = row.get("invested_capital")
    if pd.isna(nopat) or pd.isna(ic) or ic <= 0:
        return float("nan")
    return nopat / ic


def _enrich_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived columns: gross_margin, invested_capital, nopat, roic."""
    df = df.copy()
    if "gross_margin" not in df.columns or df["gross_margin"].isna().all():
        df["gross_margin"] = df["gross_profit"] / df["revenue"].replace(0, float("nan"))

    if "effective_tax_rate" not in df.columns:
        df["effective_tax_rate"] = None
    df["effective_tax_rate"] = df.apply(
        lambda r: (r.get("tax_expense") / r.get("ebit"))
                  if (not pd.isna(r.get("tax_expense", float("nan")))
                      and not pd.isna(r.get("ebit", float("nan")))
                      and r.get("ebit", 0) != 0)
                  else DEFAULT_TAX_RATE.get(r.get("accounting_std", "GAAP"), 0.21),
        axis=1,
    )

    df["invested_capital"] = df.apply(compute_invested_capital, axis=1)
    df["nopat"]            = df.apply(compute_nopat, axis=1)
    df["roic"]             = df.apply(compute_roic, axis=1)
    return df


# ---------------------------------------------------------------------------
# SEC EDGAR fetch (US GAAP)
# ---------------------------------------------------------------------------

EDGAR_BASE = "https://data.sec.gov/api/xbrl/companyfacts"
EDGAR_CIK_LOOKUP = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={}&type=10-K&dateb=&owner=include&count=10&output=atom"

# XBRL concept → our column mapping
GAAP_CONCEPTS = {
    "Revenues":                              "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "GrossProfit":                           "gross_profit",
    "OperatingIncomeLoss":                   "ebit",
    "NetIncomeLoss":                         "net_income",
    "InterestExpense":                       "interest_expense",
    "IncomeTaxExpense":                      "tax_expense",
    "Assets":                                "total_assets",
    "StockholdersEquity":                    "total_equity",
    "LongTermDebt":                          "total_debt",
    "CashAndCashEquivalentsAtCarryingValue": "cash",
    "Goodwill":                              "goodwill",
    "FiniteLivedIntangibleAssetsNet":        "intangible_assets",
    "PaymentsToAcquirePropertyPlantAndEquipment": "capex",
}


def _get_cik_for_ticker(ticker: str) -> Optional[str]:
    """Look up SEC CIK number for a ticker using the SEC company tickers JSON."""
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            timeout=15,
            headers={"User-Agent": "EARKE-Quant research@example.com"},
        )
        resp.raise_for_status()
        data = resp.json()
        for _, info in data.items():
            if info.get("ticker", "").upper() == ticker.upper():
                return str(info["cik_str"]).zfill(10)
    except Exception as e:
        logger.warning(f"CIK lookup failed for {ticker}: {e}")
    return None


def fetch_fundamentals_edgar(
    ticker: str,
    lookback_years: int = 5,
) -> pd.DataFrame:
    """
    Fetch annual fundamental data from SEC EDGAR XBRL API.
    Returns DataFrame matching fundamentals_annual schema, or empty DF.
    """
    cik = _get_cik_for_ticker(ticker)
    if not cik:
        logger.warning(f"No CIK found for {ticker}")
        return pd.DataFrame()

    try:
        url = f"{EDGAR_BASE}/CIK{cik}.json"
        resp = requests.get(
            url, timeout=30,
            headers={"User-Agent": "EARKE-Quant research@example.com"},
        )
        resp.raise_for_status()
        facts = resp.json()
    except Exception as e:
        logger.warning(f"EDGAR fetch failed for {ticker} (CIK {cik}): {e}")
        return pd.DataFrame()

    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    cutoff_year = date.today().year - lookback_years

    # Collect annual data points per concept
    concept_data: dict[str, dict[int, float]] = {}
    for concept, col in GAAP_CONCEPTS.items():
        if concept not in us_gaap:
            continue
        units = us_gaap[concept].get("units", {})
        # Try USD first, then USD/shares
        for unit_key in ["USD", "shares"]:
            if unit_key not in units:
                continue
            entries = units[unit_key]
            for entry in entries:
                # Annual 10-K filings: form=10-K, frame has no Q
                if entry.get("form") not in ("10-K", "10-K/A"):
                    continue
                end_date = entry.get("end")
                if not end_date:
                    continue
                try:
                    period_end = datetime.strptime(end_date, "%Y-%m-%d").date()
                except ValueError:
                    continue
                fiscal_year = period_end.year
                if fiscal_year < cutoff_year:
                    continue
                val = entry.get("val")
                if val is None:
                    continue
                concept_data.setdefault(col, {})[fiscal_year] = val
            break  # found a valid unit, stop checking others

    if not concept_data:
        return pd.DataFrame()

    # Build one row per fiscal year
    years = sorted(set(y for vals in concept_data.values() for y in vals), reverse=True)
    rows = []
    for fy in years:
        row = {
            "ticker":         ticker,
            "fiscal_year":    fy,
            "report_date":    None,
            "currency":       "USD",
            "accounting_std": "GAAP",
            "source":         "edgar",
        }
        for col, year_vals in concept_data.items():
            row[col] = year_vals.get(fy)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = _ensure_annual_columns(df)
    df = _enrich_fundamentals(df)
    return df


# ---------------------------------------------------------------------------
# EODHD fetch (EU IFRS + US fallback)
# ---------------------------------------------------------------------------

def fetch_fundamentals_eodhd(
    ticker: str,
    eodhd_ticker: str,
    api_key: str,
    accounting_std: str,
    currency: str,
    lookback_years: int = 5,
) -> pd.DataFrame:
    """
    Fetch fundamentals from EODHD /fundamentals endpoint.
    Handles IFRS right-of-use assets for EU stocks.
    """
    url = f"https://eodhd.com/api/fundamentals/{eodhd_ticker}"
    params = {
        "api_token": api_key,
        "fmt": "json",
        "filter": "Financials",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"EODHD fundamentals fetch failed for {eodhd_ticker}: {e}")
        return pd.DataFrame()

    financials = data if isinstance(data, dict) else {}
    income_stmts = (financials.get("Income_Statement", {}) or {}).get("yearly", {}) or {}
    balance_sheets = (financials.get("Balance_Sheet", {}) or {}).get("yearly", {}) or {}

    cutoff_year = date.today().year - lookback_years
    rows = []

    for period_str, inc in income_stmts.items():
        try:
            period_end = datetime.strptime(period_str[:10], "%Y-%m-%d").date()
            fy = period_end.year
        except ValueError:
            continue
        if fy < cutoff_year:
            continue

        bal = balance_sheets.get(period_str, {}) or {}

        def g(d: dict, *keys: str) -> Optional[float]:
            """Get first non-None value from dict using multiple key options."""
            for k in keys:
                v = d.get(k)
                if v is not None and v != "None":
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        pass
            return None

        row = {
            "ticker":         ticker,
            "fiscal_year":    fy,
            "report_date":    period_end,
            "currency":       currency,
            "accounting_std": accounting_std,
            "source":         "eodhd",
            "revenue":        g(inc, "totalRevenue", "revenue"),
            "gross_profit":   g(inc, "grossProfit"),
            "ebit":           g(inc, "ebit", "operatingIncome"),
            "net_income":     g(inc, "netIncome"),
            "interest_expense": g(inc, "interestExpense"),
            "tax_expense":    g(inc, "incomeTaxExpense"),
            "total_assets":   g(bal, "totalAssets"),
            "total_equity":   g(bal, "totalStockholderEquity", "totalEquity"),
            "total_debt":     g(bal, "longTermDebt", "shortLongTermDebt"),
            "cash":           g(bal, "cash", "cashAndCashEquivalentsAtCarryingValue"),
            "goodwill":       g(bal, "goodWill", "goodwill"),
            "intangible_assets": g(bal, "intangibleAssets"),
            # IFRS 16 fields (often present in EODHD IFRS data)
            "right_of_use_assets": g(bal, "rightOfUseAssets", "leaseRightOfUseAssets"),
            "lease_liabilities":   g(bal, "leaseLiabilities", "operatingLeaseLiability"),
            "capex":          g(inc, "capitalExpenditures"),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = _ensure_annual_columns(df)
    df = _enrich_fundamentals(df)
    return df


# ---------------------------------------------------------------------------
# yfinance fallback
# ---------------------------------------------------------------------------

def fetch_fundamentals_yfinance(
    ticker: str,
    accounting_std: str,
    currency: str,
) -> pd.DataFrame:
    """
    Fallback using yfinance. Less reliable, especially for EU stocks.
    Confidence multiplier 0.6 is applied at signal scoring level.
    """
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()

    try:
        t = yf.Ticker(ticker)
        income = t.financials  # columns = fiscal year end dates
        balance = t.balance_sheet
        if income is None or income.empty:
            return pd.DataFrame()

        rows = []
        for col in income.columns:
            fy = col.year
            inc = income[col]
            bal = balance[col] if (balance is not None and col in balance.columns) else pd.Series(dtype=float)

            def g(series: pd.Series, *keys: str) -> Optional[float]:
                for k in keys:
                    if k in series.index:
                        v = series[k]
                        if not pd.isna(v):
                            return float(v)
                return None

            row = {
                "ticker":         ticker,
                "fiscal_year":    fy,
                "report_date":    col.date() if hasattr(col, "date") else None,
                "currency":       currency,
                "accounting_std": accounting_std,
                "source":         "yfinance",
                "revenue":        g(inc, "Total Revenue"),
                "gross_profit":   g(inc, "Gross Profit"),
                "ebit":           g(inc, "EBIT", "Operating Income"),
                "net_income":     g(inc, "Net Income"),
                "interest_expense": g(inc, "Interest Expense"),
                "tax_expense":    g(inc, "Income Tax Expense"),
                "total_assets":   g(bal, "Total Assets"),
                "total_equity":   g(bal, "Stockholders Equity", "Total Equity Gross Minority Interest"),
                "total_debt":     g(bal, "Long Term Debt", "Total Debt"),
                "cash":           g(bal, "Cash And Cash Equivalents"),
                "goodwill":       g(bal, "Goodwill"),
                "intangible_assets": g(bal, "Other Intangible Assets"),
                "right_of_use_assets": None,
                "lease_liabilities":   None,
                "capex": None,
            }
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = _ensure_annual_columns(df)
        df = _enrich_fundamentals(df)
        return df

    except Exception as e:
        logger.warning(f"yfinance fundamentals failed for {ticker}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def update_fundamentals(
    conn,
    universe_df: pd.DataFrame,
    params: dict,
    eodhd_api_key: Optional[str] = None,
    force_refresh: bool = False,
) -> dict[str, str]:
    """
    Update fundamentals for all universe stocks.
    Routing logic:
      US GAAP → EDGAR first, yfinance fallback
      EU IFRS → EODHD (if key), yfinance fallback
      CA IFRS → EODHD (if key), yfinance fallback (no SEDAR+ free API)
    Returns dict: ticker -> status
    """
    from src.data.db import upsert_fundamentals_annual, is_stale, log_fetch

    staleness = params["data"]["fundamental_staleness_days"]
    results = {}

    for _, stock in universe_df.iterrows():
        ticker = stock["ticker"]
        region = stock["region"]
        accounting_std = stock["accounting_std"]
        currency = stock["currency"]
        eodhd_ticker = stock.get("eodhd_ticker") or ticker

        if not force_refresh and not is_stale(conn, "fundamentals_annual", ticker, staleness):
            results[ticker] = "cached"
            continue

        df = pd.DataFrame()

        if region == "US" and accounting_std == "GAAP":
            df = fetch_fundamentals_edgar(ticker)
            if df.empty:
                df = fetch_fundamentals_yfinance(ticker, accounting_std, currency)
                if not df.empty:
                    logger.info(f"{ticker}: fell back to yfinance fundamentals")
        elif eodhd_api_key and eodhd_ticker:
            df = fetch_fundamentals_eodhd(ticker, eodhd_ticker, eodhd_api_key,
                                          accounting_std, currency)
            if df.empty:
                df = fetch_fundamentals_yfinance(ticker, accounting_std, currency)
        else:
            df = fetch_fundamentals_yfinance(ticker, accounting_std, currency)

        if df.empty:
            results[ticker] = "failed"
            log_fetch(conn, "fundamentals", ticker, "error", "no data from any source")
            continue

        rows_written = upsert_fundamentals_annual(conn, df)
        results[ticker] = "updated"
        log_fetch(conn, "fundamentals", ticker, "success", rows_written=rows_written)

        # Brief sleep to avoid hammering APIs
        time.sleep(0.1)

    return results


# ---------------------------------------------------------------------------
# Schema helper
# ---------------------------------------------------------------------------

_ANNUAL_COLS = [
    "ticker", "fiscal_year", "report_date", "currency", "accounting_std", "source",
    "revenue", "gross_profit", "ebit", "net_income", "interest_expense", "tax_expense",
    "total_assets", "total_equity", "total_debt", "cash", "goodwill", "intangible_assets",
    "right_of_use_assets", "lease_liabilities", "capex",
    "gross_margin", "invested_capital", "nopat", "roic", "effective_tax_rate",
]


def _ensure_annual_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in _ANNUAL_COLS:
        if col not in df.columns:
            df[col] = None
    return df[_ANNUAL_COLS]
