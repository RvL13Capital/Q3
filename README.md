# EARKE Quant 3.0 (Q3)

Systematic weekly pipeline for megatrend equity investing. Screens ~180 stocks across six megatrend buckets (grid, nuclear, defense, water, critical materials, AI infra), scores them on physical momentum, fundamental quality, and crowding, and constructs a Kelly-weighted portfolio.

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # add FRED_API_KEY, EODHD_API_KEY
python src/main.py
```

---

## CLI flags

```
python src/main.py                     # run for today's date
python src/main.py --date 2025-03-01   # specific as-of date
python src/main.py --force-refresh     # bypass staleness checks, re-fetch everything
python src/main.py --skip-fetch        # skip data fetch, use only what's in the DB
python src/main.py --dry-run           # compute but don't persist portfolio snapshot
python src/main.py --trends            # also fetch Google Trends (~5s per keyword)
```

Flags can be combined freely, e.g. `--dry-run --skip-fetch --date 2025-01-06`.

---

## Dashboard

```bash
streamlit run src/reporting/dashboard.py
```

---

## Project layout

```
src/
  main.py                  # pipeline orchestrator + CLI entry point
  data/
    universe.py            # load config/universe.yaml
    prices.py              # yfinance / EODHD price fetcher
    fundamentals.py        # EODHD fundamentals fetcher
    macro.py               # FRED macro series fetcher
    db.py                  # DuckDB schema + upsert helpers + bitemporal t_k queries
  signals/
    physical.py            # price momentum, breakout, volume signals
    quality.py             # ROIC/WACC spread, margin SNR, earnings convexity
    crowding.py            # ETF correlation, short interest, rel. strength, Google Trends
    composite.py           # weighted composite score + entry/exit flags
  portfolio/
    kelly.py               # Kelly fraction + sigma estimation + Almgren-Chriss slippage
    construction.py        # weight optimisation, constraints, snapshot persistence
    monitor.py             # exit checks (6 triggers), entry checks, weekly action summary
  reporting/
    weekly_report.py       # Markdown report generation
    dashboard.py           # Streamlit dashboard

config/
  universe.yaml            # ~180 tickers across 6 megatrend buckets
  params.yaml              # signal weights, Kelly params, staleness thresholds

tests/                     # 374 tests across 13 test files (pytest)
```

---

## Signal pipeline

| Step | Module | What it produces |
|------|--------|-----------------|
| Physical | `signals/physical.py` | EROEI logistic damper X_E [0–1] (eq 4) |
| Quality | `signals/quality.py` | Moat score X_P: ROIC spread × margin SNR × convexity [0–1] (eq 5) |
| Crowding | `signals/crowding.py` | CSD index X_C: autocorr + absorption ratio [+ trends] [0–1] (eq 6) |
| Composite | `signals/composite.py` | X_E × X_P × (1−X_C); entry if > 0.30 and crowding < 0.40 (eq 7) |
| Portfolio | `portfolio/kelly.py` | Robust Kelly sizing with Almgren-Chriss participation penalty (eq 12) |
| Constraints | `portfolio/construction.py` | 8% position cap, 35% bucket cap, AUM ceiling, 10% cash floor |
| Monitor | `portfolio/monitor.py` | 6 exit triggers incl. ADV liquidity breach |

---

## GitHub Actions — weekly pipeline

Runs automatically every Monday at 09:00 UTC. Can also be triggered manually from the Actions tab with any combination of these inputs:

| Input | Default | Description |
|-------|---------|-------------|
| `date` | today | As-of date (YYYY-MM-DD) |
| `force_refresh` | false | Re-fetch all data, ignore staleness |
| `dry_run` | false | Compute but don't write portfolio snapshot |
| `skip_fetch` | false | Skip data fetch, score from cached DB data only |
| `fetch_trends` | false | Also fetch Google Trends (slow) |

The generated report is uploaded as a workflow artifact (90-day retention).

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FRED_API_KEY` | Recommended | US/CA macro data (rates, CPI, PPI) |
| `EODHD_API_KEY` | Optional | Higher-quality price + fundamentals data |
| `EODHD_API_KEY_2` | Optional | Second key for rotation under rate limits |

Without `EODHD_API_KEY` the pipeline falls back to yfinance for all data.

---

## Tests

```bash
pytest tests/ -q          # 374 tests, ~10s
```
