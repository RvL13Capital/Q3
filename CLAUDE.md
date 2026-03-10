# EARKE Quant 3.0 — Developer Guide
## Von Linck Capital Management · Private Research Memorandum

> **Paradigm:** Minimax-Robust Control under Non-Ergodicity.
> The optimisation objective is not maximum return but **structural survival in the
> physically worst-case scenario.  P(Ruin) = 0 is a hard constraint.**

---

## Quick start

```bash
pip install -r requirements.txt
pytest tests/ -q                              # must be green before any push
python src/main.py --dry-run                  # smoke-test full pipeline
streamlit run src/reporting/dashboard.py      # visual dashboard
```

---

## Six Fundamental Design Principles (from spec)

| # | Principle | Meaning |
|---|---|---|
| I | **Gall's Law — Modularity** | Every module is independently testable and replaceable. The system emerges from composition of simple, understood units — never from monolithic complexity. |
| II | **Physical Anchoring** | Thermodynamics (EROEI), market microstructure (Square-Root Law), and complexity physics (CSD) are *fundamental invariants* — they cannot be arbitraged away. |
| III | **Anti-Fragility** | The AI learns exclusively deviations from the physical baseline μ_base. If the AI module fails, the system degrades gracefully to the robust base model — no total failure possible. |
| IV | **Epistemic Honesty** | The system always quantifies the boundary of its own knowledge. σ_epist → ∞ signals Out-of-Distribution and triggers the automatic Not-Aus before a human reacts. |
| V | **Minimax instead of Maximisation** | Goal is structural survival in the worst physically plausible scenario. P(Ruin) = 0 is a hard constraint, not a soft penalty. |
| VI | **Causal Discipline** | Invariant Risk Minimization (IRM) forces the model to abstract only invariant causal structures across regimes. Spurious correlations are algorithmically discarded. |

---

## Five-Module Architecture

The EARKE architecture consists of five mathematically decoupled modules operating in a directed
causal chain from bitemporale data ontology to capital allocation. Each module is independently
testable, replaceable, and self-validating — Gall's Law at system level.

```
Async Inputs
    ↓
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│   MODULE I       │ → │   MODULE II      │ → │   MODULE III     │ → │   MODULE IV      │ → │   MODULE V       │
│ CD-NKBF          │   │ Oek. Physik      │   │ UDE +            │   │ PI-SBDM          │   │ Kelly            │
│ Kausales         │   │ Tensoren         │   │ Deep Ensembles   │   │ Black Swans      │   │ Capacity         │
│ Gedächtnis       │   │ Xe, Xp, Xc       │   │                  │   │ Labor            │   │ Bound            │
└──────────────────┘   └──────────────────┘   └──────────────────┘   └──────────────────┘   └──────────────────┘
                                                                                                       ↓
                                                                                               Optimal f*
```

---

### Module I — Kausales Gedächtnis (Causal Memory)
*CD-NKBF · Bitemporale DB · Ghost States*

**Spec purpose:** The biggest operational risk of quantitative funds is **look-ahead bias**.
Every data point carries two timestamps:

| Timestamp | Meaning |
|---|---|
| `t_e` (Event Time) | Economic reference period of the data |
| `t_k` (Know Time) | Microsecond-exact availability |
| System clock | Runs on the `t_k` axis |
| Ghost States | Revisions overwrite only archived states — never live states |

The module implements a **Neural Continuous-Discrete Kalman-Bucy Filter (CD-NKBF)** that
maintains a continuous latent state Z_t between publications:

```
Eq 2 — Continuous Nowcasting SDE:
  dZ_t = f_θ(Z_t, t) dt + G dW_t
  dP_t = F·P_t + P_t·F^T + Q_n
  (F = Jacobian of f_θ,  Q_n = process noise)
```

At each new observation, a discrete Kalman jump occurs:

```
Eq 3 — Discrete Kalman Update:
  K_k   = P⁻_k · H^T · (H·P⁻_k·H^T + R)⁻¹
  Z⁺_k  = Z⁻_k + K · (Y_obs − H·Z⁻_k)
  P⁺_k  = (I − K·H) · P⁻_k
```

Cross-correlations propagate causally error-free into the present → look-ahead bias is
**structurally eliminated**.

**Current implementation status:** `src/data/db.py` implements DuckDB as the data store with
staleness-based upserts. The CD-NKBF continuous filter and bitemporale t_e/t_k ontology are
**deferred** — the current DB uses a single timestamp. Ghost States are not yet enforced.
This is the primary Phase 2 gap.

---

### Module II — Deterministischer Anker (Deterministic Anchor)
*Ökonomische Physik · Drei Tensoren · Basis-Drift Synthese*

**Spec purpose:** A physically anchored, deterministic base model that cannot be
arbitraged away. Three tensors that must all be simultaneously favourable.

#### Tensor X_E — Thermodynamic (EROEI)
```
Eq 4 — EROEI Logistic Damper:
  X_E,t = 1 / (1 + exp(κ · (ECS_t − ECS_crit)))
  ECS_crit ≈ 8–10% of GDP (Energy Cost Share threshold)
  κ = steepness parameter (calibrated to 10.0 on percentile scale)
```
*Proxy used in implementation:* Direct GDP Energy Cost Share is not available at weekly
frequency. `src/signals/physical.py` uses the FRED PPIENG percentile rank over a 60-month
rolling window as the ECS proxy, with `ecs_crit_percentile = 0.70` (70th percentile ≈ 8–10%
ECS regime). κ = 10.0 is recalibrated for this [0,1] percentile input scale.

#### Tensor X_P — Microeconomic / Moat (Resilienz)
```
Eq 5 — Moat + Inflation Convexity:
  X_P,i,t = max(0, E[ROIC − WACC])
            × (1 / σ(GM))
            × exp(γ · max(0, ∂GM/∂PPI))
  GM = gross margin,  PPI = producer price index
```
Hard gate: ROIC ≤ WACC → X_P = 0 (no moat if capital costs are not covered).
Three multiplicative sub-components: spread quality, margin stability (SNR), inflation anti-fragility.

#### Tensor X_C — Systemic / CSD (Emergenz)
```
Eq 6 — Critical Slowing Down Index:
  X_C,t = clamp(ω₁·Δρ₁(t) + ω₂·Δ(λ_max/Σλ) [+ ω₃·trends_score])
  Δρ₁ = lag-1 autocorrelation delta (rising = CSD warning)
  Δ(λ_max/Σλ) = absorption ratio delta (rising = systemic co-movement)
  Optional: ω₃·trends_score from Google Trends (only applied when data present)
```

#### Basis-Drift Synthese (eq 7)
```
Eq 7 — Basis-Drift Synthese:
  μ_base(i, X_t) = r_f + (θ · X_P,i,t) × X_E,t × (1 − X_C,t)
```
Multiplicative structure: all three tensors must be simultaneously favourable.
θ = risk-premium scalar (params: `return_estimation.theta_risk_premium`, default 0.30).

**Current implementation:** Fully implemented in `src/signals/`. See signal contracts below.

---

### Module III — Hybrider Motor (Hybrid Engine)
*Universal Differential Equations · Deep Ensembles · Epistemic Uncertainty*

**Spec purpose:** The AI acts as a **Residual Learner** (Gall's Law) — it learns exclusively
the non-linear deviations from the physical baseline μ_base. The base model always remains
dominant. If the AI fails, the system degrades softly.

```
Eq 8 — Universal Differential Equation (UDE):
  dS_t = [μ_base(X_t) + tanh(ω) · N_θ(Z_t)] dt + σ_alea(X_t) dW_t
  tanh(ω): hard bounds AI influence to (−1, +1) — Fail-Safe
```

The **tanh(ω) Fail-Safe**: the most extreme AI hallucination of N_θ cannot dominate μ_base.
Anti-fragility through residual design.

Deep Ensembles of M independent networks solve the **Epistemic Uncertainty Hole** of
classical Bayesian NNs:

```
Eq 9 — Variance Decomposition (Deep Ensemble):
  μ_NN     = (1/M) · Σ_m μ̂^(m)           (ensemble mean correction)
  σ²_epist = Var_m[μ̂^(m)]                 (epistemic: disagreement between models)
  σ²_alea  = (1/M) · Σ_m σ̂²^(m)          (aleatoric: irreducible market noise)
  Normal regime: σ²_epist ≈ 0
  Black Swan:    σ²_epist → ∞  ⟹  automatic Not-Aus
```

**Current implementation status:** Module III is **deferred**. The current implementation
uses `composite_confidence` as a lightweight σ_epist proxy (epistemic discount on Kelly fraction).
True Deep Ensembles (M independent neural networks, eq 8/9) are Phase 3 scope.
The μ_NN correction term in the full Kelly eq 12 is therefore currently 0.

---

### Module IV — Kontrafaktisches Labor (Counterfactual Laboratory)
*PI-SBDM · Lévy-Diffusion · Minimax-Kalibrierung · Physik-Guardrails*

**Spec purpose:** Generate synthetic out-of-distribution (OOD) scenarios for minimax
calibration. Prevents overfitting to historical paths.

| Step | Method |
|---|---|
| **Forward** | Lévy-α-stable noise destroys historical signal (Fat Tails + Jumps) |
| **Backward** | Temporal-Fusion-Transformer learns score function `∇_x log p(x_t)` — Anderson 1982 |
| **Guidance** | Classifier-Free Guidance conditioned on ECS-Explosion, X_C = 1 |
| **Guardrails** | Physics penalty −γ·∇E(X_i): negative prices and margins > 100% are impossible |

```
Eq 10 — Minimax Calibration (10,000 OOD scenarios):
  λ* = argmax_λ  min_{S ∈ S_OOD}  TerminalWealth(EARKE(λ, S))
  s.t. P(Ruin) = 0
  S_OOD = synthetic Black Swan scenarios
```

Optimisation target is not maximum average return but **Ruin-Probability = 0** in the
physically worst-case scenario. Adversarial principle.

**Current implementation status:** Module IV is **deferred**. No Lévy diffusion, no
PI-SBDM, no synthetic OOD generation. Parameters are calibrated to historical data only.
This is Phase 4 scope.

---

### Module V — Hard-Capacity Robust Kelly
*Square-Root Impact Law · AUM-Asymptote · Not-Aus · Goodhart-Immunität*

#### Square-Root Market Impact Law
```
Eq 11 — Universal Square-Root Law (Bouchaud, Sato & Kanazawa):
  ΔP/P = Y · σ · √(Q/V)
  Y ≈ 1.0  (universal, empirically confirmed)
  Q = order volume,  V = market volume,  σ = daily volatility
```

#### Full Kelly Objective (complete spec)
```
Eq 12 — Robust Kelly — Complete Objective:
  f* = argmax_f [
      (μ_base + μ_NN) · f              ← total expected return (μ_NN = 0 until Module III)
      − λ · σ_epist · f                ← epistemic risk penalty (Not-Aus trigger)
      − ½ · (σ²_alea + σ²_epist) · f² ← total variance (aleatoric + epistemic)
      − c · σ_alea · |f − f_old|^1.5 · √(W/V)  ← market impact cost
  ]
  c = impact scaling ≈ 1.0,  W = AUM,  V = daily dollar volume
```

**Currently implemented terms:**
- `(μ_base) · f` — μ_NN = 0 pending Module III
- `− ½ · σ²_alea · f²` — σ²_epist term: lightweight proxy via `composite_confidence`
- `− c · σ_alea · |f − f_old|^1.5 · √(W/V)` — fully implemented

**Impact term convention:** In the optimizer, `f` is in full-Kelly space and `f_old` is the
previous fraction-scaled portfolio weight (`kelly_25pct`). Turnover is computed as
`|f × fraction − f_old|` to compare in the same weight space. This ensures zero impact
penalty when the position is unchanged.

**Not yet implemented:** λ·σ_epist·f linear penalty term (requires true σ_epist from Deep Ensembles).

#### Four Emergent System Properties

| Property | Mechanism | Spec trigger |
|---|---|---|
| **Not-Aus** | f* → 0 in milliseconds — instant position liquidation | σ_epist → ∞ |
| **AUM-Asymptote W_max** | Algorithmic capacity stop — AUM in impact denominator | W in √(W/V) |
| **Crisis Resilience** | Portfolio frozen at liquidity collapse | V ↓ → W_max ↓ |
| **Goodhart Immunity** | Mutation to long-term investor — Turnover → 0 | AUM ↑ |

`kelly_fraction()` returns `(f_adjusted, f_full)`:
- `f_full` = unconstrained optimizer output stored as `kelly_raw` (diagnostic)
- `f_adjusted` = `fraction × f_full`, clamped to [0, 1]; stored as `kelly_25pct`
  (σ_epist penalty is folded into the objective, not applied post-hoc)

---

## Systemic Risk Guarantees (from spec, Fig. 6)

| Risk | Mitigation | Mechanism | Module |
|---|---|---|---|
| Look-ahead Bias | **Eliminated** | Bitemporale DB · t_k system clock | I |
| Overfitting History | **Eliminated** | 10,000 synthetic OOD scenarios | IV |
| AI Hallucination | **Bounded** | tanh Fail-Safe · physical guardrails | III |
| Reflexive Self-Destruction | **Bounded** | Square-Root Law · AUM-Asymptote W_max | V |
| Black Swan | **Detected** | σ_epist → ∞ ⟹ automatic Not-Aus | III + V |
| Goodhart's Law | **Immunised** | Turnover → 0 as AUM rises | V |
| EROEI Cliff | **Built-in** | Logistic X_E damper | II |
| System Collapse (CSD) | **Detected** | Absorption Ratio + Autocorrelation X_C | II |

---

## Zentauren-Modell — Human-in-the-Loop

The algorithm and the portfolio manager are partners, not competitors.

| Algorithm does | Human does |
|---|---|
| High-dimensional data processing | Context and meaning |
| Detect hidden correlations | Premise re-calibration at Black Swans |
| Compute and monitor σ_epist | Define causal hypotheses |
| Emotionlessly execute fractional Kelly | Strategic decisions when σ_epist > threshold |
| Eliminate FOMO/panic algorithmically | Governance & sabotage prevention |
| 24/7 autonomous risk monitoring | Investor communication in crisis |

---

## Current Implementation vs Full Spec

| Module | Spec | Implementation Status |
|---|---|---|
| I — Kausales Gedächtnis | CD-NKBF, bitemporale t_e/t_k, Ghost States | **Partial** — DuckDB store, no Kalman filter, single timestamp |
| II — Deterministischer Anker | X_E, X_P, X_C, μ_base (eq 4–7) | **Complete** |
| III — Hybrider Motor | UDE, Deep Ensembles, μ_NN, σ²_epist (eq 8–9) | **Deferred** — composite_confidence proxy only |
| IV — Kontrafaktisches Labor | PI-SBDM, Lévy, minimax λ* (eq 10) | **Deferred** |
| V — Hard-Capacity Kelly | Full eq 12 with all terms, Not-Aus | **Partial** — μ_NN=0, σ_epist proxy, w_max enforced, Not-Aus not wired |

---

## Code Architecture

```
config/params.yaml + config/universe.yaml
         ↓
[DATA]      src/data/{prices,fundamentals,macro,universe,db}.py
            → DuckDB: prices, fundamentals_*, macro_series, google_trends
         ↓
[MODULE II] src/signals/{physical,quality,crowding,composite}.py
            → DuckDB: signal_scores
         ↓
[MODULE V]  src/portfolio/{kelly,construction,monitor}.py
            → DuckDB: portfolio_snapshots
         ↓
[REPORTING] src/reporting/{weekly_report,dashboard,export_snapshot}.py
```

### Signal contracts

Every batch signal function returns a `pd.DataFrame` with a `ticker` column:

| File | Score column | Confidence column |
|---|---|---|
| `signals/physical.py` | `physical_norm` | `physical_confidence` |
| `signals/quality.py` | `quality_score` | `quality_confidence` |
| `signals/crowding.py` | `crowding_score` | `crowding_confidence` |

Confidence is always `[0, 1]`. Scores are always `[0, 1]`.
Crowding is inverted by `composite.py`: the formula uses `(1 − X_C)`.

### Constraint cascade in `apply_constraints()` (order is fixed)

1. Drop `weight < min_position` (2%)
2. Bucket cap (35% per megatrend)
3. Per-stock cap (8%)
4. **Step 3.5** — AUM capacity ceiling (`w_max_eur / aum_eur`, eq 12 W_max) — active when `aum_eur > 0`
5. Cash floor (≤ 90% invested)
6. Re-check min_position after scaling

---

## Key Invariants

- **Multiplicative composite** — X_E × X_P × (1−X_C). All three tensors must be simultaneously favourable. Any zero → composite = 0 → no trade.
- **Hard gate** — ROIC ≤ WACC → X_P = 0. No moat if capital costs not covered.
- **Confidence gate** — `composite_confidence < 0.40` → NaN → excluded. Never overridden.
- **Crowding inversion** — high X_C is bad; composite uses `(1 − X_C)`.
- **tanh Fail-Safe** (spec principle, pending Module III) — AI correction bounded to (−1, +1), μ_base always dominant.
- **Google Trends (X_C third component)** — only applied when data present in DB. Falls back to two-component formula silently.
- **σ_epist Not-Aus** (spec principle, pending Module III+V wiring) — σ_epist → ∞ must trigger f* → 0.

---

## All Equations Reference

| Eq | Name | Module | File | Status |
|---|---|---|---|---|
| 1 | Ergodicity gap: ⟨g⟩_t ≠ ⟨g⟩_E | Foundation | — | Conceptual basis |
| 2 | Continuous Nowcasting SDE (CD-NKBF) | I | — | Deferred |
| 3 | Discrete Kalman Update | I | — | Deferred |
| 4 | X_E EROEI logistic damper | II | `signals/physical.py` | Implemented (PPIENG proxy) |
| 5 | X_P quality/moat (ROIC−WACC × SNR × convexity) | II | `signals/quality.py` | Implemented |
| 6 | X_C CSD (autocorr + absorption ratio [+ trends]) | II | `signals/crowding.py` | Implemented |
| 7 | μ_base Basis-Drift Synthese | II | `signals/composite.py` | Implemented |
| 8 | UDE: dS_t = [μ_base + tanh(ω)·N_θ] dt + σ_alea dW_t | III | — | Deferred |
| 9 | Deep Ensemble variance decomposition (σ²_epist, σ²_alea) | III | — | Deferred (proxy) |
| 10 | Minimax calibration λ* (10,000 OOD scenarios) | IV | — | Deferred |
| 11 | Square-Root Market Impact Law | V | `portfolio/kelly.py` | Implemented |
| 12 | Robust Kelly full objective | V | `portfolio/kelly.py` | Partial (μ_NN=0, σ_epist proxy) |

---

## params.yaml Sync Requirement

`conftest.py` fixtures must mirror `config/params.yaml`. If you change a threshold, update both:
1. `config/params.yaml` (runtime)
2. Inline `PARAMS` dicts in test files (e.g., `tests/test_kelly.py`, `tests/test_portfolio.py`)

Failing to do this causes tests to silently validate stale parameters.

---

## Development Workflow

```bash
# 1. Branch (session-scoped name required by CI)
git checkout -b claude/<feature>-<session-id>

# 2. Make changes — keep modules focused, avoid cross-cutting changes
# 3. Run tests — must be green
pytest tests/ -q

# 4. Commit with conventional prefix (fix/feat/refactor/test/docs)
git commit -m "feat(signals): ..."

# 5. Push
git push -u origin <branch>
```

**Commit scope tags:** `signals`, `portfolio`, `data`, `dashboard`, `tests`, `config`

---

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `FRED_API_KEY` | Recommended | Higher FRED rate limits (macro series) |
| `EODHD_API_KEY` | Optional | Fundamentals + fallback prices for failing yfinance tickers |

## CLI Flags (`src/main.py`)

| Flag | Effect |
|---|---|
| `--date YYYY-MM-DD` | Run pipeline as-of a historical date |
| `--force-refresh` | Ignore staleness checks, re-fetch all data |
| `--skip-fetch` | Score only — no data fetching |
| `--dry-run` | Full pipeline including scoring, no DB writes |
| `--trends` | Enable Google Trends fetch (rate-limited, ~5 s/batch) |
