"""
    Bisection Method for Implied Volatility

The bisection method provides guaranteed convergence for finding implied
volatility. It is robust but slower than Newton-Raphson, making it ideal
as a fallback method.

# Algorithm

Given price bounds at volatilities [σ_low, σ_high]:

1. Compute price at midpoint: σ_mid = (σ_low + σ_high) / 2
2. If price(σ_mid) > target, set σ_high = σ_mid
   Else, set σ_low = σ_mid
3. Repeat until |σ_high - σ_low| < tolerance

# Convergence Properties
- **Linear convergence**: Interval halves each iteration
- **Guaranteed convergence**: If target price is within bounds
- **Typical iterations**: 20-50 for tolerance 1e-6 (starting from [0.001, 5.0])

# Advantages
- No derivative required
- Cannot diverge
- Works for any continuous pricing function

See also: [`NewtonRaphson`](@ref), [`HybridSolver`](@ref), [`implied_vol`](@ref)
"""

using ..Prezo: EuropeanOption, MarketData, BlackScholes
using ..Prezo: price

"""
    Bisection

Bisection method solver configuration for implied volatility.

# Fields
- `max_iter::Int`: Maximum iterations (default: 100)
- `tol::Float64`: Convergence tolerance for volatility (default: 1e-6)
- `vol_low::Float64`: Lower volatility bound (default: 0.001 = 0.1%)
- `vol_high::Float64`: Upper volatility bound (default: 5.0 = 500%)

# Examples
```julia
# Default settings
solver = Bisection()

# Tighter bounds for faster convergence
solver = Bisection(vol_low=0.1, vol_high=0.5, tol=1e-8)

# Use with implied_vol
iv = implied_vol(call, market_price, data, solver)
```

See also: [`implied_vol`](@ref)
"""
struct Bisection <: IVSolver
    max_iter::Int
    tol::Float64
    vol_low::Float64
    vol_high::Float64

    function Bisection(; max_iter::Int=100, tol::Float64=1e-6,
        vol_low::Float64=0.001, vol_high::Float64=5.0)
        max_iter > 0 || throw(ArgumentError("max_iter must be positive"))
        tol > 0 || throw(ArgumentError("tol must be positive"))
        vol_low > 0 || throw(ArgumentError("vol_low must be positive"))
        vol_high > vol_low || throw(ArgumentError("vol_high must exceed vol_low"))

        new(max_iter, tol, vol_low, vol_high)
    end
end

"""
    implied_vol(option::EuropeanOption, target_price::Real, data::MarketData,
                solver::Bisection)

Compute implied volatility using the bisection method.

# Arguments
- `option::EuropeanOption`: European call or put option
- `target_price::Real`: Observed market price
- `data::MarketData`: Market parameters (spot, rate, div)
- `solver::Bisection`: Bisection configuration

# Returns
Implied volatility as `Float64`, or `NaN` if calculation fails.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
market_price = 10.45

# Standard bisection
iv = implied_vol(call, market_price, data, Bisection())

# With tighter bounds for faster convergence
iv_fast = implied_vol(call, market_price, data,
                      Bisection(vol_low=0.1, vol_high=0.5))
```

See also: [`Bisection`](@ref), [`implied_vol`](@ref) for other solvers
"""
function implied_vol(option::EuropeanOption, target_price::Real, data::MarketData,
    solver::Bisection)

    # Check bounds are valid for the target price
    data_low = MarketData(data.spot, data.rate, solver.vol_low, data.div)
    data_high = MarketData(data.spot, data.rate, solver.vol_high, data.div)

    price_low = price(option, BlackScholes(), data_low)
    price_high = price(option, BlackScholes(), data_high)

    # Ensure target is within bounds
    if target_price < price_low || target_price > price_high
        @warn "Target price $target_price outside price bounds [$price_low, $price_high]"
        return NaN
    end

    # Initialize bounds
    vol_low = solver.vol_low
    vol_high = solver.vol_high

    for iter in 1:solver.max_iter
        # Midpoint
        vol_mid = (vol_low + vol_high) / 2.0

        # Price at midpoint
        data_mid = MarketData(data.spot, data.rate, vol_mid, data.div)
        price_mid = price(option, BlackScholes(), data_mid)

        # Check convergence
        if abs(vol_high - vol_low) < solver.tol
            return vol_mid
        end

        # Check if we hit the target exactly
        if abs(price_mid - target_price) < solver.tol
            return vol_mid
        end

        # Update bounds
        if price_mid > target_price
            # Price too high → decrease volatility
            vol_high = vol_mid
        else
            # Price too low → increase volatility
            vol_low = vol_mid
        end
    end

    # Return best estimate
    return (vol_low + vol_high) / 2.0
end
