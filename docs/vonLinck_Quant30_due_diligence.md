# Technical Due Diligence — von Linck Capital Management: Quant 3.0 / EARKE
**Date:** 2026-03-07
**Documents Reviewed:** `vonLinck_Quant30_v8.pdf` · `gemverkauf Theorie .docx`
**Classification:** Internal Research — Restricted

---

## Executive Summary

Both documents describe the **Epistemic-Aware Residual Kelly Engine (EARKE)** — a five-module quantitative investment architecture. The intellectual framework is sophisticated and correctly grounded in current academic literature. However, **no executable system exists**. All Julia code is illustrative pseudo-code with undefined placeholder functions. The gap between the specification and a production system is substantial.

**Verdict:** High-quality research memorandum. The mathematics is sound. The code is a blueprint, not an implementation. Treat as an architectural specification, not a deliverable.

---

## Document 1: `vonLinck_Quant30_v8.pdf` — Architecture Specification

### What It Claims
A "mathematically complete, physically anchored investment architecture" comprising five modules:

| Module | Name | Core Claim |
|--------|------|------------|
| I | Causal Memory | Neural Continuous-Discrete Kalman-Bucy Filter; bitemporal DB eliminates look-ahead bias |
| II | Deterministic Anchor | EROEI thermodynamic tensor × moat quality × CSD criticality → μ_base |
| III | Hybrid Engine | Universal Differential Equation: dS = [μ_base + tanh(ω)·N_θ] dt + σ_alea dW |
| IV | Counterfactual Lab | Physics-guided Score-Based Diffusion; 10,000 Lévy-stable Black Swan scenarios |
| V | Hard-Capacity Kelly | Square-root market impact; automatic kill-switch when σ_epist → ∞ |

### Mathematical Validity
The mathematics cited is internally consistent and references legitimate frameworks:
- **Ergodicity Economics** (Peters 2019) — correctly applied
- **Maximum Caliber** (Jaynes/Ghosh) — valid generalization of MaxEnt to trajectories
- **IRM** (Arjovsky et al.) — appropriate for regime-invariant causal learning
- **Square-Root Impact Law** (Bouchaud/Sato/Kanazawa) — Y ≈ 1.0 is empirically well-established
- **UDEs** (Rackauckas et al.) — correct framework for residual physics-ML hybrid models
- **Mean-Field Games** (Lions/Lasry) — appropriate framing for reflexive market dynamics

### What Is Missing
- **No actual code** of any kind in this document
- No backtesting results or performance attribution
- No data source specifications beyond generic mentions
- No details on how X_E (EROEI tensor) is computed from real data
- No ensemble architecture specification (number of networks, training regime, regularisation)
- σ_epist kill-switch threshold is unspecified

---

## Document 2: `gemverkauf Theorie .docx` — Technical Elaboration

### Julia Code — Verbatim Analysis

The document contains one substantive Julia snippet presented as a blueprint for Module I (Neural CD-KBF):

```julia
# Julia SciML Blueprint — Module I: Neural Continuous-Discrete Kalman-Bucy Filter
using DifferentialEquations, LinearAlgebra

function continuous_economy_drift!(dZ, Z, p, t)
    state     = @view Z[1:N]
    P_cov     = reshape(@view Z[N+1:end], N, N)
    dZ[1:N]  .= neural_drift_model(state, p.θ)          # ← UNDEFINED
    F_jacobian = compute_jacobian(state, p.θ)            # ← UNDEFINED
    dP = F_jacobian * P_cov + P_cov * F_jacobian' + p.Q_noise
    dZ[N+1:end] .= vec(dP)
end

function new_data_arrived(u, t, integrator)
    return t ∈ database.knowledge_timestamps            # ← UNDEFINED: `database`
end

function kalman_update_shock!(integrator)
    Y_observed, R_noise, H_matrix = get_pit_observation(database, t_now)  # ← UNDEFINED
    # ... Kalman gain computation ...
    integrator.u[1:N]    .= Z_post
    integrator.u[N+1:end] .= vec(P_post)
end

cb_data = DiscreteCallback(new_data_arrived, kalman_update_shock!)
prob    = ODEProblem(continuous_economy_drift!, Z_initial, (0.0, T_end), parameters)
sol     = solve(prob, Tsit5(), callback=cb_data, tstops=database.knowledge_timestamps)
```

A second snippet covers the Kelly allocator (also from the docx):

```julia
# Julia SciML Blueprint: Epistemic Kelly Allocator
function allocate_epistemic_kelly(μ_base, ensemble_preds, σ_alea_sq, λ_penalty)
    μ_ML        = mean(ensemble_preds)
    σ_epist_sq  = var(ensemble_preds)
    μ_total     = μ_base + μ_ML
    σ_total_sq  = σ_alea_sq + σ_epist_sq
    # ... robust Kelly optimization ...
end
```

### Code Assessment

| Item | Finding |
|------|---------|
| **Language** | Julia — correct choice for DifferentialEquations.jl / SciML ecosystem |
| **DifferentialEquations.jl usage** | Syntactically correct; `ODEProblem` + `DiscreteCallback` is the right pattern |
| **`neural_drift_model()`** | Undefined. Core of Module III. No architecture, no training procedure. |
| **`compute_jacobian()`** | Undefined. Required for the Riccati update — cannot be hand-computed for a neural net without `Zygote.jl` auto-diff. |
| **`get_pit_observation()`** | Undefined. Encapsulates the entire bitemporal data pipeline (Module I). |
| **`database`** | Undefined global. The entire PiT data store. |
| **`Z_initial`, `parameters`** | Undefined. No initialisation logic. |
| **`N` (state dimension)** | Undefined. Size of the latent state vector. |
| **Ensemble (Module III)** | Referenced (`ensemble_preds`) but no ensemble definition, training loop, or architecture. |
| **Lévy diffusion (Module IV)** | Described in text; no code present. |

**Summary:** The author demonstrates fluency with the correct modern tech stack (Julia, SciML, DifferentialEquations.jl, Flux.jl, Zygote.jl) and knows exactly which APIs to use. The structural skeleton is correct. The placeholders, however, represent the majority of the actual engineering work.

---

## Gap Analysis: Pseudo-Code → Production

To realise this architecture from this blueprint requires:

### 1. Data Pipeline (Module I) — High Effort
- Bitemporal database (DuckDB + transaction-time columns, or TimescaleDB)
- PiT data ingestion for fundamentals, macro series, price data with `t_know` precision
- `get_pit_observation()` implementation linking DB queries to the ODE callback

### 2. Neural Drift Model (Module III) — High Effort
- Network architecture selection (MLP, NeuralODE, Temporal Fusion Transformer)
- Training regime: what labels? What loss? IRM requires multiple environment partitions.
- Zygote-compatible implementation (no Python-side auto-diff)
- `compute_jacobian()` via `ForwardDiff.jl` or `Zygote.jl`

### 3. Deep Ensemble (Module III) — Medium Effort
- M independent networks with different initialisations
- Shared training infrastructure
- σ_epist threshold calibration

### 4. Physics-Guided Diffusion (Module IV) — Very High Effort
- Score-based diffusion model conditioned on EROEI/CSD regime
- Lévy-stable noise process integration (not natively in DifferentialEquations.jl)
- Classifier-free guidance conditioning
- Physical guardrails (price non-negativity, margin bounds)

### 5. Tensor Computation (Module II) — Medium Effort
- X_E: EROEI ratio from energy sector fundamental data — data sourcing is non-trivial
- X_P: ROIC−WACC moat score — partially implemented in the Q3 Python codebase
- X_C: Absorption Ratio from correlation matrix + lag-1 autocorrelation — tractable

### 6. Kelly Optimizer (Module V) — Medium Effort
- Differentiable objective with square-root impact term
- AUM-parameterised capacity constraint
- σ_epist kill-switch (threshold must be empirically calibrated)

---

## Relationship to Existing Q3 Python Codebase

The current Q3 repository implements a **simplified, deployable Python version** of the EARKE concept. Mapping:

| EARKE Module | Q3 Python Status |
|---|---|
| I — Bitemporal DB | Partial: DuckDB with staleness checks; no `t_know` Ghost States |
| II — Deterministic Anchor | Partial: ROIC, WACC, composite score; no EROEI X_E, no CSD X_C |
| III — UDE + Ensemble | Not implemented: yfinance/EDGAR data only; no neural component |
| IV — Diffusion Lab | Not implemented |
| V — Kelly Allocator | Not implemented: position sizing not yet active |

The Python codebase is a working data pipeline and screening tool. The full EARKE spec is a multi-year research and engineering programme.

---

## Conclusion

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Theoretical Framework | ★★★★★ | Rigorous, well-cited, state-of-the-art references |
| Mathematical Consistency | ★★★★☆ | Internally consistent; some parameters unspecified |
| Code Quality | ★★★☆☆ | Correct structural skeleton; all critical functions undefined |
| Completeness | ★★☆☆☆ | Blueprint only; no data pipeline, no training, no backtesting |
| Production Readiness | ★☆☆☆☆ | Not executable in current form |

**The author understands the problem deeply and has specified the correct solution architecture.** The code demonstrates knowledge of the right tools (Julia SciML, DifferentialEquations.jl, Flux.jl, Zygote.jl). However, `neural_drift_model()`, `get_pit_observation()`, and `load_pretrained_temporal_transformer()` are not implementation details — they *are* the system. Converting this blueprint to a production system is a substantial multi-year engineering programme, not an integration task.

---
*Technical review completed 2026-03-07. All code cited verbatim from source documents.*
