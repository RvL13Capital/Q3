# Architecture Audit Response — EARKE Quant 3.0
## Von Linck Capital Management

**Audit Date:** 2026-03-10
**Auditor Claims:** External quantitative architecture review
**Response:** Code-verified rebuttal with line-level evidence

---

## Executive Summary

The external audit report contains several **factually incorrect claims** about the
codebase architecture, alongside some **valid observations** about known deferred modules.
This response provides line-level code evidence for each claim.

**Critical finding:** The audit appears to have been generated without reading the actual
source code. Multiple claims describe architectures (matrix inversion, raw price CSD,
no capacity constraints) that do not exist in the codebase. The "corrupted binary archive"
disclaimer in the audit preamble confirms the auditor did not have access to the source.

---

## Claim-by-Claim Analysis

### CLAIM 1: "Naive f = Σ⁻¹μ matrix inversion causing eigenvalue singularities"

**Verdict: INCORRECT**

The audit claims the Kelly allocator uses multivariate matrix inversion (`np.linalg.inv(np.cov(...))`),
which would suffer from ill-conditioning when N approaches T.

**Actual implementation:** `src/portfolio/kelly.py:142-216`

The Kelly optimizer uses **per-stock scalar optimization** via `scipy.optimize.minimize_scalar`.
There is no covariance matrix, no matrix inversion, and no multivariate portfolio optimization
anywhere in the codebase.

```python
# kelly.py:212 — actual optimizer call
result = minimize_scalar(neg_objective, bounds=(0.0, upper), method="bounded")
```

The per-stock objective function (kelly.py:200-207) optimizes:
```
max_f [ μ_eff·f − ½·σ²_total·f² − c·σ·|f×fraction − f_old|^1.5·√(W/V) ]
```

This is a **univariate bounded optimization** per stock. The Ledoit-Wolf shrinkage
recommendation is irrelevant — there is no matrix to shrink.

**Suggested fix from audit:** Ledoit-Wolf + pseudo-inverse
**Our response:** Not applicable. No matrix inversion exists.

---

### CLAIM 2: "CSD calculated on raw, non-stationary price data"

**Verdict: INCORRECT**

The audit claims `signals/physical.py` computes variance and autocorrelation on raw prices,
introducing "severe phase-shift distortion."

**Actual implementation:** `src/signals/crowding.py:41-86`

CSD metrics are computed on **log returns**, not raw prices:

```python
# crowding.py:67-68
returns = compute_log_returns(price_df[[ticker]]).dropna()
# ...
rho_current = float(current_slice.autocorr(lag=1))
```

Log returns are approximately stationary by construction. The absorption ratio
(crowding.py:118-131) also operates on the **correlation matrix of returns**:

```python
# crowding.py:118
returns = compute_log_returns(price_df).dropna(how="all")
# ...
corr = clean.corr().values
eigvals = np.linalg.eigvalsh(corr)
```

**Note:** CSD is in `crowding.py` (X_C tensor), not `physical.py` (X_E tensor).
The audit incorrectly attributes CSD to the wrong module.

---

### CLAIM 3: "Phantom Modules — no filters.py, kalman.py, hmm.py"

**Verdict: CORRECT but already documented**

The CLAUDE.md specification explicitly declares these modules as deferred:

| Module | Status | Documentation |
|---|---|---|
| I — CD-NKBF Kalman Filter | Deferred (Phase 2) | CLAUDE.md "Module I" section |
| III — UDE / Deep Ensembles | Deferred (Phase 3) | CLAUDE.md "Module III" section |
| IV — PI-SBDM / Lévy Diffusion | Deferred (Phase 4) | CLAUDE.md "Module IV" section |

The "Current Implementation vs Full Spec" table in CLAUDE.md provides an honest
accounting of implemented vs. deferred components. There is no misrepresentation.

---

### CLAIM 4: "No AUM capacity constraints — unconstrained Kelly fractions"

**Verdict: INCORRECT**

The audit claims "the strategy ignores the physics of market impact" and
"scaling past $10M AUM will result in allocations exceeding ADV."

**Actual implementation (three layers of capacity protection):**

1. **Kelly impact term** (`kelly.py:200-207`):
   ```python
   turnover = abs(f * fraction - f_old)
   impact = impact_scaling * sigma * (turnover ** 1.5) * sqrt_wv
   ```
   Where `sqrt_wv = np.sqrt(aum / daily_dollar_volume)` — the Square-Root Law.

2. **W_max capacity ceiling** (`kelly.py:223-247`):
   ```python
   W_max = V × (μ_robust / (c · σ · |Δf|^1.5))²
   ```

3. **Construction constraint** (`construction.py:67-78`):
   ```python
   # Step 3.5: AUM capacity ceiling (eq 12 w_max)
   if aum > 0 and "w_max_eur" in df.columns:
       w_max_fracs = (df.loc[valid, "w_max_eur"] / aum).clip(upper=max_pos)
   ```

**Additionally (added in this audit response):**
4. **ADV fraction diagnostic** (`kelly.py:compute_adv_fraction`) — flags positions
   consuming >10% of ADV
5. **ADV liquidity exit trigger** (`monitor.py`) — Trigger 6 flags positions >25% of ADV

---

### CLAIM 5: "Look-ahead bias via temporal data leakage"

**Verdict: PARTIALLY VALID — already documented as Phase 2 gap**

The CLAUDE.md specification states:
> "The CD-NKBF continuous filter and bitemporale t_e/t_k ontology are **deferred** —
> the current DB uses a single timestamp. Ghost States are not yet enforced.
> This is the primary Phase 2 gap."

**Current mitigations:**
- `get_macro_value()` uses `date <= as_of_date` (basic point-in-time)
- `fundamentals_annual` has `report_date` column (though not yet used for PiT filtering)
- Price queries are bounded by `as_of_date`

**Not yet implemented:**
- Bitemporal `t_e` / `t_k` timestamp pair
- DuckDB ASOF joins on `release_date`
- Ghost State revision tracking

This is the only audit finding that identifies a genuine gap, and it was already
documented as the primary Phase 2 deliverable.

---

### CLAIM 6: "Batch CSV architecture cannot scale"

**Verdict: IRRELEVANT to current scope**

The system uses DuckDB (`data/q3.duckdb`) as its primary data store, not CSV files.
CSV caches (`data/cache/`) are write-through copies for debugging/portability.
The main pipeline operates entirely through DuckDB queries.

The recommendation to migrate to Kafka/Redis is appropriate for a production HFT system
but is out of scope for a weekly-rebalancing strategy with ~15 positions.

---

## Improvements Implemented in This Response

| Change | File | Purpose |
|---|---|---|
| Numerical stability floor on σ²_total | `kelly.py:183` | Prevents division-by-zero edge case |
| `compute_adv_fraction()` utility | `kelly.py:250-268` | ADV liquidity diagnostic |
| ADV >10% warning in Kelly weights | `kelly.py:363-369` | Execution slippage early warning |
| ADV >25% exit trigger (Trigger 6) | `monitor.py:102-116` | Automatic exit for illiquid positions |

---

## Correct Architecture Summary

```
Per-Stock Signal Flow (NOT multivariate):

  X_E (physical.py)  ──┐
                        ├──→ composite = X_E × X_P × (1−X_C)  [composite.py]
  X_P (quality.py)   ──┤        │
                        │        ├──→ μ_base = rf + θ × composite
  X_C (crowding.py)  ──┘        │
                                 ↓
                          Per-stock scalar Kelly optimization  [kelly.py]
                          f* = argmax_f [μ_eff·f − ½σ²·f² − impact(f)]
                                 │
                                 ↓
                          Constraint cascade  [construction.py]
                          min_pos → bucket_cap → stock_cap → w_max → cash_floor
```

**Key architectural property:** The system is deliberately per-stock, not multivariate.
This eliminates the covariance estimation problem entirely. Portfolio-level risk control
is achieved through the constraint cascade, not through Markowitz-style optimization.

---

## Recommendations for Future Audits

1. **Require source code access** before making architectural claims
2. **Reference specific file:line locations** for each finding
3. **Distinguish between documented future work and undiscovered gaps**
4. **Verify claims computationally** before recommending fixes

---

## Appendix: v8.1 Post-Remediation Audit Reconciliation

**Audit Date:** 2026-03-10 (v8.1 follow-up)
**Status:** All seven audit claims verified as **already implemented**.

The v8.1 audit re-raised two "severe second-order vulnerabilities" that were in fact
resolved by commits `61f55d5` (bitemporal schema) and `9f50f5e` (Almgren-Chriss
participation penalty) prior to this audit being received.

### Re-Claim 1: "Endogenous Market Impact — optimizer blind to transaction drag"

**Verdict: ALREADY IMPLEMENTED (commit 9f50f5e)**

The Kelly objective now contains five terms, not three:

```python
# kelly.py:208-226 — full eq 12 objective
def neg_objective(f):
    growth       = mu_eff * f - 0.5 * sigma_sq_total * f**2
    turnover     = abs(f * fraction - f_old)
    impact       = impact_scaling * sigma * (turnover ** 1.5) * sqrt_wv
    participation = _participation_coeff * sqrt(max(f, 0.0)) * f  # ← NEW
    return -(growth - impact - participation)
```

Where `_participation_coeff = η · σ · √(fraction · AUM / ADV)` (kelly.py:204).

This is the Almgren-Chriss participation-rate penalty. The optimizer now organically
reduces allocations to illiquid names *before* hitting the hard 10%/25% ADV boundaries.
The parameter `kelly.eta_participation = 0.50` is configurable in `params.yaml`.

**Test coverage:** 7 tests in `test_kelly.py:180-273` verify:
- Penalty reduces allocation vs baseline
- Zero η reproduces pre-enhancement behavior
- Penalty scales inversely with ADV (more illiquid → more penalty)
- Higher η → lower allocation
- ADV=0 disables penalty gracefully
- Not-Aus dominates participation penalty
- Penalty coexists with non-zero f_old

### Re-Claim 2: "Single-Axis PiT Leakage — no t_k/t_e separation"

**Verdict: ALREADY IMPLEMENTED (commit 61f55d5)**

The `t_k` (knowledge time) column is now present on all four data tables:

| Table | Migration | Backfill |
|---|---|---|
| `prices` | `ALTER TABLE ADD COLUMN t_k TIMESTAMP` | `t_k = fetched_at` |
| `fundamentals_annual` | same | `t_k = COALESCE(report_date, fetched_at)` |
| `fundamentals_quarterly` | same | `t_k = COALESCE(report_date, fetched_at)` |
| `macro_series` | same | `t_k = fetched_at` |

Four query functions now support bitemporal filtering:

| Function | Parameter | SQL Filter |
|---|---|---|
| `get_latest_fundamentals` | `as_of_date=` | `COALESCE(t_k, report_date, fetched_at) <= ?` |
| `get_margin_history` | `as_of_date=` | `COALESCE(t_k, report_date, fetched_at) <= ?` |
| `get_macro_value` | `bitemporal=True` | `COALESCE(t_k, fetched_at) <= ?` |
| `get_macro_series` | `as_of_tk=` | `COALESCE(t_k, fetched_at) <= ?` |

The `quality.py` signal module now passes `as_of_date` to all fundamentals DB queries
(`compute_roic_wacc_spread`, `compute_margin_snr`, `compute_inflation_convexity`),
closing the primary look-ahead bias vector.

**Test coverage:** 10 tests in `test_db.py:388-606` verify:
- Schema column existence
- Explicit t_k storage in upserts
- Bitemporal exclusion of future-knowledge data
- Backward-compatible (non-bitemporal) query path
- Fundamentals filtering by report_date
- Margin history filtering
- NULL report_date fallback to fetched_at
- Ghost state revision semantics (documents INSERT OR REPLACE limitation)

### Re-Claim 3: "CI/CD dotenv failure"

**Verdict: ALREADY IMPLEMENTED (commit 9f36d85)**

```python
# main.py:26-30
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional
```

Additionally, the PDF report import (matplotlib/reportlab) is now guarded similarly
(main.py:215-226), preventing CI failure in minimal environments.

**Test result:** 374/374 passing (was 356/357 before fix).

### Re-Claim 5: "ADV exit threshold hardcoded at 25%"

**Verdict: ALREADY PARAMETERIZED (commit 2cd2804)**

```python
# monitor.py:106
adv_exit_threshold = float(params["kelly"].get("adv_exit_threshold", 0.25))
```

Configurable via `params.yaml:19` and synced in `conftest.py:236`.

### Phase 3 Verification Tests Status

| Test Type | Status | Location |
|---|---|---|
| Endogenous slippage penalty | 7 tests | `test_kelly.py:180-273` |
| Bitemporal revision injection | 10 tests | `test_db.py:388-606` |
| Ergodic flatline (σ² floor) | Implicit | `kelly.py:190` + `test_kelly.py:27-36` |

### Audit Epoch 1 Roadmap Status

| Epoch 1 Deliverable | Status | Commit |
|---|---|---|
| Bitemporal dual-axis data | **Complete** | `61f55d5` |
| Endogenous sizing penalties | **Complete** | `9f50f5e` |
| CI/CD hygiene | **Complete** | `9f36d85` |
| Code review + documentation | **Complete** | `2cd2804` |

**Epoch 1 is officially closed. Phase 2 scope: append-only ghost states (PK includes t_k),
Deep Ensemble σ_epist (Module III), UKF continuous filtering (Module I).**
