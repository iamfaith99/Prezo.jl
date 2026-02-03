"""
    Hedging module: OHMC and delta hedging strategies

Phase 5 — Advanced Hedging. Provides:
- OHMC: Optimal Hedged Monte Carlo for variance-reduced option pricing
- Delta hedging strategies: discrete, stop-loss, static
- Backtest framework for hedge performance
"""

include("ohmc.jl")
include("delta_hedging.jl")
