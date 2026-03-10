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
