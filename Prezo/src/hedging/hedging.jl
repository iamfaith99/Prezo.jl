"""
    Hedging module: OHMC and delta hedging strategies

Phase 5 — Advanced Hedging. Provides:
- OHMC: Optimal Hedged Monte Carlo for variance-reduced option pricing
- CVaR-OHMC: Risk-aware OHMC with Lagrangian CVaR (tail-risk optimization)
- Delta hedging strategies: discrete, stop-loss, static
- Backtest framework for hedge performance
"""

include("ohmc.jl")
include("cvar_ohmc.jl")
include("delta_hedging.jl")
