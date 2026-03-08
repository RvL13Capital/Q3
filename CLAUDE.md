# EARKE Quant 3.0 — Developer Guide

## Quick start

```bash
pip install -r requirements.txt
pytest tests/ -q                              # must be green before any push
python src/main.py --dry-run                  # smoke-test full pipeline
streamlit run src/reporting/dashboard.py      # visual dashboard
```

## Architecture

```
config/params.yaml + config/universe.yaml
         ↓
[1] DATA  src/data/{prices,fundamentals,macro,universe,db}.py
         ↓ DuckDB (data/q3.duckdb)
[2] SIGNALS  src/signals/{physical,quality,crowding,composite}.py
         ↓ signal_scores table
[3] PORTFOLIO  src/portfolio/{kelly,construction,monitor}.py
         ↓ portfolio_snapshots table
[4] REPORTING  src/reporting/{weekly_report,dashboard,export_snapshot}.py
```

### Signal contracts

Every batch signal function returns a `pd.DataFrame` with a `ticker` column and the following per-signal columns:

| Module | Score column | Confidence column |
|---|---|---|
| `physical.py` | `physical_norm` | `physical_confidence` |
| `quality.py` | `quality_score` | `quality_confidence` |
| `crowding.py` | `crowding_score` | `crowding_confidence` |

Confidence is always `[0, 1]`. Scores are always `[0, 1]` (crowding is inverted to `1 − X_C` by `composite.py`).

### Composite formula (EARKE eq 7)

```
composite = X_E × X_P × (1 − X_C)
μ_base    = rf + θ × composite          (θ = params.return_estimation.theta_risk_premium)
```

**Data gate:** `composite_confidence = mean(p_conf, q_conf, c_conf)`. If `< 0.40` → composite is NaN → stock excluded.

### Kelly formula (EARKE eq 12)

```
f* = argmax_f [(μ−rf)·f − ½·σ²·f² − c·σ·|f−f_old|^1.5·√(W/V)]
```

`kelly_fraction()` returns `(f_adjusted, f_full)`:
- `f_full` = unconstrained optimizer output (stored as `kelly_raw` for diagnostics)
- `f_adjusted` = `fraction × f_full`, clamped to [0, 1]
- `kelly_25pct` = `f_adjusted × composite_confidence` (σ_epist epistemic discount)

### Constraint cascade in `apply_constraints()` (order is fixed)

1. Drop `weight < min_position` (2 %)
2. Bucket cap (35 % per megatrend)
3. Per-stock cap (8 %)
4. **3.5** AUM capacity ceiling (`w_max_eur / aum_eur`) — only active when `aum_eur > 0`
5. Cash floor (≤ 90 % invested)
6. Re-check min_position after scaling

## Key invariants

- **Multiplicative composite** — all three signals must be simultaneously positive. If any is zero the composite is zero and no trade is triggered.
- **Hard gate** — `ROIC ≤ WACC` → `quality_score = 0` → composite = 0.
- **Confidence gate** — `composite_confidence < 0.40` → excluded, never overridden.
- **Crowding inversion** — high `X_C` is bad; the formula uses `(1 − X_C)`.
- **Google Trends (X_C third component)** — only applied when `trends_keyword` data exists in DB. Falls back to the two-component formula silently.

## params.yaml sync requirement

`conftest.py` fixtures must mirror `config/params.yaml`. If you change a threshold, update both:
1. `config/params.yaml` (runtime)
2. `tests/conftest.py` or inline `PARAMS` dicts in test files (test-time)

Failing to do this causes integration tests to silently test stale parameters.

## EARKE equation reference

| Eq | Symbol | Module |
|---|---|---|
| 4 | X_E logistic damper | `src/signals/physical.py` |
| 5 | X_P quality/moat | `src/signals/quality.py` |
| 6 | X_C CSD crowding | `src/signals/crowding.py` |
| 7 | composite + μ_base | `src/signals/composite.py` |
| 12 | fractional Kelly + impact | `src/portfolio/kelly.py` |

## Development workflow

```bash
# 1. Branch (session-scoped name required by CI)
git checkout -b claude/<feature>-<session-id>

# 2. Make changes — keep modules focused, avoid cross-cutting changes
# 3. Run tests
pytest tests/ -q

# 4. Commit with conventional commit prefix (fix/feat/refactor/test/docs)
git commit -m "fix(signals): ..."

# 5. Push
git push -u origin <branch>
```

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `FRED_API_KEY` | Recommended | Higher FRED rate limits |
| `EODHD_API_KEY` | Optional | Fundamentals + fallback prices |

## CLI flags (`src/main.py`)

| Flag | Effect |
|---|---|
| `--date YYYY-MM-DD` | Run as-of a past date |
| `--force-refresh` | Ignore staleness, re-fetch all data |
| `--skip-fetch` | Score only, no data fetching |
| `--dry-run` | Full pipeline, no DB writes |
| `--trends` | Enable Google Trends fetch (rate-limited, slow) |
