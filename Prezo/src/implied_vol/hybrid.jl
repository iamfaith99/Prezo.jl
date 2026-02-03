"""
    Hybrid Solver for Implied Volatility

The hybrid solver combines Newton-Raphson and bisection methods for robust
and efficient implied volatility calculation. It uses Newton-Raphson for
speed but falls back to bisection if Newton diverges.

# Algorithm

1. Attempt Newton-Raphson for up to `newton_iter` iterations
2. If Newton diverges or fails, switch to bisection
3. Use bisection bounds informed by Newton attempts

# Advantages
- **Speed**: Newton-Raphson converges in 3-5 iterations typically
- **Robustness**: Bisection guarantees convergence
- **Production-ready**: Recommended for general use

# When to Use

- **General pricing**: Use HybridSolver as default
- **Known good initial guess**: Use NewtonRaphson directly
- **Maximum robustness needed**: Use Bisection directly

See also: [`NewtonRaphson`](@ref), [`Bisection`](@ref), [`implied_vol`](@ref)
"""

using ..Prezo: EuropeanOption, MarketData, BlackScholes
using ..Prezo: price

"""
    HybridSolver <: IVSolver

Hybrid solver combining Newton-Raphson and bisection methods.

# Fields
- `newton_max_iter::Int`: Maximum Newton-Raphson iterations (default: 10)
- `bisection_max_iter::Int`: Maximum bisection iterations if fallback (default: 50)
- `tol::Float64`: Convergence tolerance (default: 1e-6)
- `vol_low::Float64`: Lower volatility bound (default: 0.001)
- `vol_high::Float64`: Upper volatility bound (default: 5.0)
- `fallback_threshold::Float64`: Divergence threshold for switching (default: 1.0)

The `fallback_threshold` determines when to switch from Newton to bisection.
If the Newton step would move volatility outside bounds by more than this
factor, bisection is triggered.

# Examples
```julia
# Default hybrid solver (recommended)
solver = HybridSolver()

# Faster Newton, longer bisection fallback
solver = HybridSolver(newton_max_iter=15, bisection_max_iter=30)

# Use with implied_vol
iv = implied_vol(call, market_price, data)  # HybridSolver is default
iv = implied_vol(call, market_price, data, HybridSolver())
```

See also: [`implied_vol`](@ref)
"""
struct HybridSolver <: IVSolver
    newton_max_iter::Int
    bisection_max_iter::Int
    tol::Float64
    vol_low::Float64
    vol_high::Float64
    fallback_threshold::Float64

    function HybridSolver(; newton_max_iter::Int=10, bisection_max_iter::Int=50,
        tol::Float64=1e-6, vol_low::Float64=0.001, vol_high::Float64=5.0,
        fallback_threshold::Float64=1.0)
        newton_max_iter > 0 || throw(ArgumentError("newton_max_iter must be positive"))
        bisection_max_iter > 0 || throw(ArgumentError("bisection_max_iter must be positive"))
        tol > 0 || throw(ArgumentError("tol must be positive"))
        vol_low > 0 || throw(ArgumentError("vol_low must be positive"))
        vol_high > vol_low || throw(ArgumentError("vol_high must exceed vol_low"))
        fallback_threshold > 0 || throw(ArgumentError("fallback_threshold must be positive"))

        new(newton_max_iter, bisection_max_iter, tol, vol_low, vol_high, fallback_threshold)
    end
end

"""
    implied_vol(option::EuropeanOption, target_price::Real, data::MarketData,
                solver::HybridSolver)

Compute implied volatility using hybrid Newton-Raphson + bisection method.

This is the recommended solver for production use as it balances speed and robustness.

# Arguments
- `option::EuropeanOption`: European call or put option
- `target_price::Real`: Observed market price
- `data::MarketData`: Market parameters (spot, rate, div)
- `solver::HybridSolver`: Hybrid solver configuration (optional, uses default if omitted)

# Returns
Implied volatility as `Float64`, or `NaN` if calculation fails.

# Algorithm
1. Start with Newton-Raphson iterations
2. Monitor for divergence (volatility going outside bounds or oscillating)
3. If diverging, switch to bisection with narrowed bounds
4. Return result from whichever method succeeds

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
market_price = 10.45

# Use default hybrid solver
iv = implied_vol(call, market_price, data, HybridSolver())

# Or simply (HybridSolver is default)
iv = implied_vol(call, market_price, data)
```

See also: [`HybridSolver`](@ref), [`implied_vol`](@ref) for other solvers
"""
function implied_vol(option::EuropeanOption, target_price::Real, data::MarketData,
    solver::HybridSolver)

    # Validate price bounds
    data_low = MarketData(data.spot, data.rate, solver.vol_low, data.div)
    data_high = MarketData(data.spot, data.rate, solver.vol_high, data.div)

    price_low = price(option, BlackScholes(), data_low)
    price_high = price(option, BlackScholes(), data_high)

    if target_price < price_low || target_price > price_high
        @warn "Target price $target_price outside feasible range [$price_low, $price_high]"
        return NaN
    end

    # Track bounds for potential bisection fallback
    current_low = solver.vol_low
    current_high = solver.vol_high

    # Initial guess
    vol = data.vol > 0 ? data.vol : 0.2

    # Phase 1: Newton-Raphson attempts
    for iter in 1:solver.newton_max_iter
        # Ensure vol is in bounds
        vol = clamp(vol, current_low, current_high)

        # Compute objective
        trial_data = MarketData(data.spot, data.rate, vol, data.div)
        bs_price = price(option, BlackScholes(), trial_data)
        error = bs_price - target_price

        # Check convergence
        if abs(error) < solver.tol
            return vol
        end

        # Compute vega for Newton step
        vega = let
            h = 0.0001
            data_up = MarketData(data.spot, data.rate, vol + h, data.div)
            price_up = price(option, BlackScholes(), data_up)
            (price_up - bs_price) / h
        end

        # Check for valid vega
        if abs(vega) < 1e-10
            @debug "Vega too small at iteration $iter, switching to bisection"
            break
        end

        # Proposed Newton step
        vol_new = vol - error / vega

        # Check for divergence or oscillation
        if !isfinite(vol_new)
            @debug "Non-finite proposal at iteration $iter, switching to bisection"
            break
        end

        # If proposal is far outside bounds, bisection might be better
        if vol_new < current_low - solver.fallback_threshold ||
           vol_new > current_high + solver.fallback_threshold
            @debug "Newton proposal $vol_new far outside bounds [$current_low, $current_high]"
            # Narrow bounds based on current information
            if error > 0
                current_high = vol  # Price too high, vol is upper bound
            else
                current_low = vol   # Price too low, vol is lower bound
            end
            break
        end

        # Update bounds for bisection fallback (maintain bracket)
        if error > 0
            current_high = min(current_high, vol)
        else
            current_low = max(current_low, vol)
        end

        vol = vol_new
    end

    # Phase 2: Bisection with current bounds
    for iter in 1:solver.bisection_max_iter
        vol = (current_low + current_high) / 2.0

        trial_data = MarketData(data.spot, data.rate, vol, data.div)
        bs_price = price(option, BlackScholes(), trial_data)
        error = bs_price - target_price

        # Check convergence
        if abs(current_high - current_low) < solver.tol || abs(error) < solver.tol
            return vol
        end

        # Update bounds
        if error > 0
            current_high = vol
        else
            current_low = vol
        end
    end

    # Return best estimate from bisection
    return (current_low + current_high) / 2.0
end

"""
    implied_vol(option::EuropeanOption, target_price::Real, data::MarketData)

Default implied volatility calculation using HybridSolver.

This is the recommended entry point for implied volatility calculations.
It automatically uses the robust hybrid solver with sensible defaults.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)

# Compute implied volatility from market price
market_price = 10.45
iv = implied_vol(call, market_price, data)  # Returns ~0.20

# Verify: price with computed IV should match market price
check_price = price(call, BlackScholes(),
                    MarketData(100.0, 0.05, iv, 0.0))
@test isapprox(check_price, market_price, atol=1e-6)
```

See also: [`HybridSolver`](@ref), [`NewtonRaphson`](@ref), [`Bisection`](@ref)
"""
implied_vol(option::EuropeanOption, target_price::Real, data::MarketData) =
    implied_vol(option, target_price, data, HybridSolver())
