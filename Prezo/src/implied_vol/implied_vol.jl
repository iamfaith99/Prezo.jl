"""
    ImpliedVol

Implied volatility calculation module for option pricing.

This module provides robust methods for computing implied volatility from
option market prices. The implied volatility is the volatility parameter
that makes the Black-Scholes model price equal to the observed market price.

# Key Features
- **Newton-Raphson**: Fast convergence (quadratic) when near solution
- **Bisection**: Guaranteed convergence, robust but slower
- **Hybrid Solver**: Combines both for robustness and speed
- **Vectorized**: Efficient computation for option chains

# Mathematical Background

For European options under Black-Scholes, implied volatility σ* satisfies:

    V_BS(S, K, T, r, q, σ*) = V_market

where V_BS is the Black-Scholes formula and V_market is the observed price.

# Usage

## Single Option
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
market_price = 10.5

# Using hybrid solver (recommended)
iv = implied_vol(call, market_price, data)  # Uses HybridSolver by default

# Using specific solver
iv_nr = implied_vol(call, market_price, data, NewtonRaphson())
iv_bi = implied_vol(call, market_price, data, Bisection())
```

## Option Chain
```julia
strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
market_prices = [15.2, 12.1, 10.5, 8.9, 7.3]

ivs = implied_vol_chain(strikes, market_prices, data)
```

# Solver Comparison

| Solver | Speed | Robustness | Iterations | Best For |
|--------|-------|-----------|------------|----------|
| NewtonRaphson | Fast | Medium | 3-5 | Good initial guess |
| Bisection | Slow | High | 20-50 | No initial guess |
| HybridSolver | Fast | Very High | 3-10 | General use |

See also: [`NewtonRaphson`](@ref), [`Bisection`](@ref), [`HybridSolver`](@ref),
[`implied_vol`](@ref), [`implied_vol_chain`](@ref)
"""
module ImpliedVol

using ..Prezo: EuropeanOption, EuropeanCall, EuropeanPut, MarketData, BlackScholes
using ..Prezo: price, greek, Vega
using Distributions

export IVSolver, NewtonRaphson, Bisection, HybridSolver
export implied_vol, implied_vol_chain
export iv_objective, vega_for_iv
export is_valid_price, price_bounds
export implied_vol_stats
export ImpliedVolSurface, build_implied_vol_surface, surface_iv, surface_stats

"""
    IVSolver

Abstract base type for implied volatility solvers.

Concrete subtypes:
- [`NewtonRaphson`](@ref): Fast quadratic convergence, may diverge
- [`Bisection`](@ref): Guaranteed convergence, slower
- [`HybridSolver`](@ref): Combines both for robustness (recommended)

See also: [`implied_vol`](@ref)
"""
abstract type IVSolver end

# Include solver implementations
include("newton_raphson.jl")
include("bisection.jl")
include("hybrid.jl")
include("vectorized.jl")
include("surface.jl")

end # module ImpliedVol
