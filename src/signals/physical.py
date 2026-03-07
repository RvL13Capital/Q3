"""
Signal 1: Physical Necessity Score (X_E)

Scores how deeply a company is exposed to physically-forced megatrends.
Pure-plays (1 bucket) score 1.5/3; diversified industrials (3+ buckets) score 3/3.
Rationale: pure-plays are already the most crowded megatrend names.
"""
import pandas as pd


BUCKET_SCORE_MAP = {
    0: 0.0,
    1: 1.5,
    2: 2.0,
    3: 3.0,
}


def compute_physical_score(stock: dict | pd.Series) -> dict:
    """
    Compute physical necessity score for a single stock.

    Args:
        stock: dict or Series with keys: ticker, buckets (list), primary_bucket

    Returns dict with:
        ticker, bucket_count, primary_bucket,
        physical_raw (0–3), physical_norm (0–1), physical_confidence (always 1.0)
    """
    if isinstance(stock, pd.Series):
        stock = stock.to_dict()

    buckets = stock.get("buckets") or []
    if isinstance(buckets, str):
        # Defensive: sometimes loaded as a comma-separated string
        buckets = [b.strip() for b in buckets.split(",")]

    bucket_count = len(set(buckets))
    raw = BUCKET_SCORE_MAP.get(min(bucket_count, 3), 0.0)
    norm = raw / 3.0

    return {
        "ticker":               stock.get("ticker", ""),
        "bucket_count":         bucket_count,
        "primary_bucket":       stock.get("primary_bucket"),
        "physical_raw":         raw,
        "physical_norm":        norm,
        "physical_confidence":  1.0,
    }


def batch_physical_scores(universe_df: pd.DataFrame) -> pd.DataFrame:
    """Compute physical scores for all stocks in universe_df."""
    rows = [compute_physical_score(row) for _, row in universe_df.iterrows()]
    return pd.DataFrame(rows)
