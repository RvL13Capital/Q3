"""
Macro data fetcher: FRED (US series) + ECB Statistical Data Warehouse (EU series)
with DBnomics as fallback for EU series.
Fetches 10Y govt bond yields, CPI, PPI used by the quality signal and Kelly sizing.
"""
import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FRED_BASE    = "https://api.stlouisfed.org/fred/series/observations"
ECB_BASE     = "https://data-api.ecb.europa.eu/service/data"
DBNOMICS_BASE = "https://api.db.nomics.world/v22/series"

# FRED series IDs
FRED_SERIES = {
    "US_10Y":      "DGS10",           # US 10-year Treasury yield (daily, %)
    "US_CPI_YOY":  "CPIAUCSL",        # US CPI all items (monthly index, convert to YoY)
    "US_PPI_YOY":  "PPIACO",          # US PPI all commodities (monthly index)
    "CA_CPI_YOY":  "CPALCY01CAM661N", # Canada CPI YoY (monthly, %)
    "CA_10Y":      "IRLTLT01CAM156N", # Bank of Canada 10-year bond yield (monthly, %)
    "US_PPIENG":   "PPIENG",          # US PPI energy (level index, for X_E logistic damper)
}

# ECB SDW series keys
ECB_SERIES = {
    "EU_10Y_DE":   "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y",  # German Bund 10Y
    "EU_HICP_YOY": "ICP/M.U2.N.000000.4.ANR",               # EU HICP YoY
    "EU_PPI_YOY":  "STS/M.U2.N.PROD.NS0020.4.ANR",          # EU PPI YoY (Eurozone, annual rate)
}

# DBnomics series keys (provider/dataset/series) — fallback for EU series
# Same underlying ECB data served through the DBnomics aggregator.
DBNOMICS_SERIES = {
    "EU_10Y_DE":   "ECB/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
    "EU_HICP_YOY": "ECB/ICP/M.U2.N.000000.4.ANR",
    "EU_PPI_YOY":  "ECB/STS/M.U2.N.PROD.NS0020.4.ANR",
}


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------

def fetch_fred_series(
    series_id: str,
    fred_api_key: str,
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch a FRED time series.
    Returns DataFrame with columns: date, value, source.
    """
    if not end_date:
        end_date = date.today().isoformat()

    params = {
        "series_id":        series_id,
        "api_key":          fred_api_key,
        "file_type":        "json",
        "observation_start": start_date,
        "observation_end":   end_date,
    }
    try:
        resp = requests.get(FRED_BASE, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"FRED fetch failed for {series_id}: {e}")
        return pd.DataFrame()

    obs = data.get("observations", [])
    if not obs:
        return pd.DataFrame()

    df = pd.DataFrame(obs)[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df["source"] = "fred"
    return df


def _fred_index_to_yoy(df: pd.DataFrame) -> pd.DataFrame:
    """Convert monthly index series (e.g. CPI) to YoY % change."""
    df = df.copy().sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = df["value"].pct_change(periods=12) * 100
    df = df.dropna(subset=["value"])
    df["date"] = df["date"].dt.date.astype(str)
    return df


# ---------------------------------------------------------------------------
# ECB SDW
# ---------------------------------------------------------------------------

def fetch_ecb_series(
    series_key: str,
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch from ECB Statistical Data Warehouse REST API.
    series_key format: "DATAFLOW/SERIES_KEY" e.g. "YC/B.U2.EUR..."
    Returns DataFrame with columns: date, value, source.
    """
    if not end_date:
        end_date = date.today().isoformat()

    url = f"{ECB_BASE}/{series_key}"
    params = {
        "startPeriod": start_date,
        "endPeriod":   end_date,
        "format":      "csvdata",
    }
    try:
        resp = requests.get(url, params=params, timeout=30,
                            headers={"Accept": "text/csv"})
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
    except Exception as e:
        logger.warning(f"ECB SDW fetch failed for {series_key}: {e}")
        return pd.DataFrame()

    # ECB CSV has TIME_PERIOD and OBS_VALUE columns
    if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
        logger.warning(f"ECB SDW unexpected format for {series_key}: {df.columns.tolist()}")
        return pd.DataFrame()

    result = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
    result = result.rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "value"})
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    result = result.dropna(subset=["value"])

    # Convert YYYY-MM period strings to date strings (use last day of month)
    def period_to_date(p: str) -> str:
        try:
            if len(p) == 7:  # YYYY-MM
                import calendar
                y, m = int(p[:4]), int(p[5:7])
                last_day = calendar.monthrange(y, m)[1]
                return f"{y}-{m:02d}-{last_day:02d}"
            return p
        except Exception:
            return p

    result["date"] = result["date"].apply(period_to_date)
    result["source"] = "ecb_sdw"
    return result


# ---------------------------------------------------------------------------
# DBnomics (EU series fallback)
# ---------------------------------------------------------------------------

def fetch_dbnomics_series(
    series_path: str,       # "provider/dataset/series_id" e.g. "ECB/ICP/M.U2..."
    start_date: str,
    end_date: Optional[str] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch a macro time series from DBnomics.
    series_path: "{provider}/{dataset}/{series_code}" e.g. "ECB/YC/B.U2..."
    api_key: optional Bearer token for higher rate limits.
    Returns DataFrame with columns: date, value, source.
    """
    if not end_date:
        end_date = date.today().isoformat()

    url = f"{DBNOMICS_BASE}/{series_path}"
    params = {
        "observations": 1,
        "align_periods": 1,
    }
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        logger.warning(f"DBnomics fetch failed for {series_path}: {e}")
        return pd.DataFrame()

    try:
        series_docs = payload.get("series", {}).get("docs", [])
        if not series_docs:
            return pd.DataFrame()

        doc = series_docs[0]
        periods = doc.get("period", [])
        values  = doc.get("value", [])

        if not periods or not values:
            return pd.DataFrame()

        df = pd.DataFrame({"date": periods, "value": values})
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])

        # Convert YYYY-MM period strings to last-day-of-month date strings
        def _period_to_date(p: str) -> str:
            try:
                if len(p) == 7:
                    import calendar
                    y, m = int(p[:4]), int(p[5:7])
                    last_day = calendar.monthrange(y, m)[1]
                    return f"{y}-{m:02d}-{last_day:02d}"
                return p
            except Exception:
                return p

        df["date"] = df["date"].apply(_period_to_date)

        # Filter to requested date range
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        df["source"] = "dbnomics"
        return df
    except Exception as e:
        logger.warning(f"DBnomics parse error for {series_path}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_macro_series_stale(conn, series_id: str, staleness_days: int) -> bool:
    """Check staleness for a specific macro series_id (not the whole table)."""
    from datetime import datetime, timedelta
    try:
        result = conn.execute(
            "SELECT MAX(fetched_at) FROM macro_series WHERE series_id = ?",
            [series_id],
        ).fetchone()
        if not result or result[0] is None:
            return True
        last_fetch = result[0]
        if isinstance(last_fetch, str):
            last_fetch = datetime.fromisoformat(last_fetch)
        return last_fetch < datetime.now() - timedelta(days=staleness_days)
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Main update function
# ---------------------------------------------------------------------------

def update_macro(
    conn,
    params: dict,
    fred_api_key: Optional[str] = None,
    dbnomics_api_key: Optional[str] = None,
    force_refresh: bool = False,
) -> None:
    """
    Fetch and store all macro series defined in params.
    US/CA series via FRED. EU series via ECB SDW with DBnomics as fallback.
    """
    from src.data.db import upsert_macro, is_stale, log_fetch

    staleness = params["data"]["macro_staleness_days"]
    start_date = (date.today() - timedelta(days=365 * 6)).isoformat()

    # ── US / CA series via FRED ──────────────────────────────────────────────
    if fred_api_key:
        for our_name, fred_id in FRED_SERIES.items():
            if not force_refresh and not _is_macro_series_stale(conn, our_name, staleness):
                logger.info(f"Macro {our_name} is fresh, skipping")
                continue
            df = fetch_fred_series(fred_id, fred_api_key, start_date)
            if df.empty:
                log_fetch(conn, "macro", our_name, "error", f"FRED empty: {fred_id}")
                continue
            # Convert index series to YoY for CPI/PPI
            if our_name.endswith("_YOY") and our_name not in ("CA_CPI_YOY",):
                df = _fred_index_to_yoy(df)
            rows = upsert_macro(conn, our_name, df)
            log_fetch(conn, "macro", our_name, "success", rows_written=rows)
    else:
        logger.warning("No FRED_API_KEY provided; US/CA macro data will not be fetched")

    # ── EU series via ECB SDW → DBnomics fallback ────────────────────────────
    for our_name, ecb_key in ECB_SERIES.items():
        if not force_refresh and not _is_macro_series_stale(conn, our_name, staleness):
            continue
        df = fetch_ecb_series(ecb_key, start_date)
        if df.empty:
            logger.warning(f"ECB SDW returned empty for {our_name}, trying DBnomics")
            dbnomics_key = DBNOMICS_SERIES.get(our_name)
            if dbnomics_key:
                df = fetch_dbnomics_series(dbnomics_key, start_date, api_key=dbnomics_api_key)
        if df.empty:
            log_fetch(conn, "macro", our_name, "error", f"ECB SDW + DBnomics both empty: {ecb_key}")
            continue
        rows = upsert_macro(conn, our_name, df)
        log_fetch(conn, "macro", our_name, "success", rows_written=rows)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_risk_free_rate(conn, region: str, as_of_date: str, params: dict) -> float:
    """
    Return current 10Y govt bond yield as a decimal (e.g. 0.043 for 4.3%).
    Falls back to a reasonable default if data is unavailable.
    """
    from src.data.db import get_macro_value

    series_map = {
        "EU": params["macro"]["eu_risk_free_series"],
        "US": params["macro"]["us_risk_free_series"],
        "CA": params["macro"]["ca_risk_free_series"],
    }
    series_id = series_map.get(region, "US_10Y")
    val = get_macro_value(conn, series_id, as_of_date)

    if val is None:
        # Sensible fallback: 4.0% as of 2025
        logger.warning(f"No risk-free rate data for {region}/{series_id}, using 4.0%")
        return 0.040

    # FRED yields are already in percent (e.g., 4.3 = 4.3%)
    return val / 100.0


def get_inflation_yoy(
    conn,
    region: str,
    series_type: str,  # 'cpi' | 'ppi'
    as_of_date: str,
    params: dict,
    lookback_months: int = 24,
) -> pd.Series:
    """Return trailing YoY inflation series as pd.Series (indexed by date)."""
    from src.data.db import get_macro_series

    series_map = {
        ("EU", "cpi"): params["macro"]["eu_cpi_series"],
        ("EU", "ppi"): params["macro"]["eu_ppi_series"],
        ("US", "cpi"): params["macro"]["us_cpi_series"],
        ("US", "ppi"): params["macro"]["us_ppi_series"],
        ("CA", "cpi"): "CA_CPI_YOY",
        ("CA", "ppi"): params["macro"]["us_ppi_series"],  # proxy
    }
    series_id = series_map.get((region, series_type))
    if not series_id:
        return pd.Series(dtype=float)

    from datetime import datetime
    end = as_of_date
    start = (datetime.strptime(as_of_date, "%Y-%m-%d")
             - timedelta(days=lookback_months * 31)).strftime("%Y-%m-%d")
    return get_macro_series(conn, series_id, start, end)
