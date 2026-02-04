# Prezo

## Introduction

This is the research repo for implementing an options pricing engine in
the Julia programming language.

## Directory Structure

Package source lives under `Prezo/` (Julia package root). Key layout:

```
Prezo.jl/
├── README.md
├── Prezo/
│   ├── Project.toml
│   ├── src/
│   │   ├── Prezo.jl, data.jl, options.jl, engines.jl, paths.jl, basis.jl, lsm.jl
│   │   ├── greeks/         # Greeks, portfolio, exotic
│   │   ├── implied_vol/    # IV solvers, surface
│   │   ├── volatility/     # GARCH, Heston, local vol
│   │   ├── filters/        # Kalman, EKF, EnKF, particle
│   │   ├── inference/      # MLE, calibration, ABC
│   │   ├── hedging/        # OHMC, delta hedging
│   │   └── gpu/            # MonteCarloGPU, asset_paths_col_gpu
│   ├── test/               # runtests.jl, test_*.jl, examples_*.jl, benchmark_lsm.jl
│   ├── scripts/, docs/
│   └── references/
```



## Project Status

Prezo.jl currently implements:
- ✅ **European and American vanilla options** (calls and puts)
- ✅ **Multiple pricing engines**: Black-Scholes, Binomial trees, Monte Carlo, Longstaff-Schwartz LSM
- ✅ **Multiple basis functions** for LSM regression (Laguerre, Chebyshev, Power, Hermite)
- ✅ **Greeks Module** (Phase 1 Complete): Analytical and numerical Greeks, portfolio aggregation
- ✅ **Implied Volatility Module** (Phase 1 Complete): Newton-Raphson, Bisection, Hybrid solvers, volatility surface
- ✅ **Volatility Module** (Phase 2): GARCH(1,1), EGARCH, GJR-GARCH with MLE fit and simulation
- ✅ **Comprehensive test suite** with property-based and volatility tests
- ✅ **BlueStyle-compliant codebase**
- ✅ **Comprehensive documentation** with Documenter.jl

**Current Phase**: Phase 2 (Volatility Modeling) — GARCH family (univariate) implemented; Heston and local vol planned.

## TODO: Phased Roadmap (Aligned to DESIGN_DOCUMENT.md)

### Phase 1 — Core Infrastructure ✅ COMPLETE
- [x] **Greeks Module** 
  - First- and second-order Greeks (Delta, Gamma, Theta, Vega, Rho, Vanna, Charm, Vomma)
  - Analytical Greeks for Black-Scholes
  - Finite-difference Greeks for non-analytical engines
  - Portfolio aggregation and hedging utilities
- [x] **Implied Volatility Module** 
  - Newton–Raphson solver with quadratic convergence
  - Bisection solver for guaranteed convergence
  - Hybrid solver combining speed and robustness
  - Vectorized IV calculation for option chains
  - Volatility surface construction (`surface.jl`): build from price grid, query via bilinear interpolation
- [x] **Enhanced Test Suite** 
  - `test/runtests.jl` with comprehensive test organization
  - `test_greeks.jl`: Analytical formula validation, numerical convergence, property tests
  - `test_implied_vol.jl`: Round-trip tests, solver comparison, volatility smile validation
  - Cross-engine consistency tests for American options

### Phase 2 — Volatility Modeling ✅ COMPLETE
- [x] **GARCH Family (univariate)** 
  - GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1) in `src/volatility/garch.jl`
  - `fit(GARCH, returns)` via MLE (Optim.jl), `volatility_process`, `forecast`, `loglikelihood`, `simulate`
  - Tests: `test/test_volatility.jl`
- [x] **GARCH extensions** 
  - A-GARCH (shifted shocks), Student-t innovations, RegimeGARCH (`regime_garch.jl`)
  - DCC-GARCH (`dcc_garch.jl`): two-step fit, `covariance_series`, `forecast`
- [x] **Further multivariate** 
  - O-GARCH (`ogarch.jl`): PCA factors + GARCH(1,1), `fit(OGARCH, returns; n_factors)`, `covariance_series`
  - Factor GARCH (`factor_garch.jl`): observed factors, OLS loadings + GARCH on factors, `covariance_series(model, returns, factors)`
- [x] **Stochastic Volatility** 
  - Heston model (`heston.jl`): `HestonModel(κ, θ, σ, ρ, v₀)`, `simulate_heston(model, n_steps, T; S0, r, q, seed)`
- [x] **Local Volatility** 
  - Dupire local vol surface (`local_vol.jl`): `LocalVolSurface`, `dupire_local_vol(market_calls, strikes, maturities, data)`

### Phase 3 — State Estimation ✅ COMPLETE
- [x] **Kalman Filter Family**
  - Linear Kalman (`src/filters/kalman.jl`): `KalmanFilter`, `KalmanFilterState`, `predict`, `update`, `filter_data`, `smooth` (RTS)
  - Extended Kalman (`extended_kalman.jl`): nonlinear f, h with Jacobians
  - Ensemble Kalman (`ensemble_kalman.jl`): perturbed-observation EnKF
- [x] **Particle Filter (Sequential Monte Carlo)**
  - `ParticleFilter`, `ParticleState` (`particle.jl`): SIR, `effective_sample_size`, `systematic_resample`, `multinomial_resample`, `stratified_resample`
  - Tests: `test/test_filters.jl`

### Phase 4 — Inference & Calibration ✅ COMPLETE
- [x] **Approximate Bayesian Computation (ABC)**
  - Rejection ABC (`src/inference/abc.jl`): `RejectionABC(n_samples, tolerance, prior)`, `abc_inference(method, simulator, summary_stats, distance, observed_data; rng)`, `euclidean_distance`
- [x] **Maximum Likelihood Estimation (MLE)**
  - `MLEProblem`, `MLESolution`, `solve(mle; method=:LBFGS)` with box constraints (`src/inference/mle.jl`), `standard_errors(solution)` from Hessian
- [x] **Calibration Framework**
  - Targets: `OptionPricesTarget(options, market_prices)`, `IVSurfaceTarget(strikes, maturities, market_vols)` (`src/inference/calibration.jl`)
  - `calibrate(objective, initial_params, lb, ub; method)`, `calibrate_option_prices(price_fn, target, initial_params, lb, ub; weights)`, `LeastSquaresCalibration`
  - Tests: `test/test_inference.jl`

### Phase 5 — Advanced Hedging ✅ COMPLETE
- [x] **Optimal Hedged Monte Carlo (OHMC)** (`src/hedging/ohmc.jl`)
  - `OHMCConfig(n_paths, n_steps, basis_order)`, `OHMCResult`, `ohmc_price(option, market_data, config)`
  - Variance reduction via optimal hedge-ratio regression; 95% CI
- [x] **Delta Hedging Strategies** (`src/hedging/delta_hedging.jl`)
  - `DiscreteDeltaHedge(rebalancing_times, transaction_cost)`, `StopLossHedge(trigger_level)`, `StaticHedge(hedge_instruments)`
  - `backtest_hedge(option, strategy, historical_data, hedging_engine)` → `HedgePerformance`
  - Tests: `test/test_hedging.jl`; examples: `test/examples_hedging.jl`

### Cross-Cutting Deliverables
- [x] **Dependencies / Project.toml Alignment** 
  - ✅ `Optim` - For optimization in MLE and calibration (Phase 2+)
  - ✅ `ForwardDiff` - For automatic differentiation (Phase 2+)
  - ✅ `Roots` - For root-finding in implied volatility (using internal implementation)
  - ✅ `SpecialFunctions` - For mathematical functions (Phase 2+)
  - ✅ `StaticArrays` - For performance optimization (Phase 2+)
  - ✅ `StatsBase` - For statistical operations
- [ ] **Documentation Plan** (Ongoing)
  - ✅ User guides structure in docs/src/guide/
  - [ ] API reference with Documenter.jl (Next: Generate and deploy)
  - [ ] Tutorials for Greeks and IV (Next: Write content)
- [ ] **Performance Considerations** (Ongoing)
  - ✅ Type-stable implementations throughout
  - ✅ Pre-allocation patterns in Monte Carlo
  - ✅ Hot-path optimization with `@inbounds` (paths, binomial, OHMC, antithetic/stratified)
  - ✅ Parallelization: `asset_paths_col_threaded(engine, spot, rate, vol, expiry)` via `Threads.@threads` (use `julia -t N`)
  - ✅ GPU acceleration: `MonteCarloGPU(steps, reps)`, `asset_paths_col_gpu(engine, spot, rate, vol, expiry)` (CUDA; requires GPU)

### Phase 6+ — Future Work

- [ ] **Volatility smile**
  - SABR model for volatility smile
- [ ] **Market data integration**
  - Real-time option chain fetching
  - Historical data analysis
  - Volatility surface fitting from market data
- [ ] **Risk management tools**
  - VaR (Value at Risk) calculation
  - CVaR (Conditional VaR)
  - Portfolio-level stress testing
  - Scenario analysis framework
- [ ] **Extended documentation**
  - Jupyter notebook tutorials
  - Comparison with QuantLib benchmarks
  - Financial mathematics background section
  - Performance comparison guide
- [ ] **Research / exploration**
  - **Alternative LSM improvements:** bundling/stratification for variance reduction, adaptive basis function selection, machine learning for continuation value estimation
  - **Multi-asset options:** basket options, best-of/worst-of options, correlation modeling
