"""
    Newton-Raphson Solver for Implied Volatility

The Newton-Raphson method provides fast quadratic convergence for finding
implied volatility. It uses the vega (price sensitivity to volatility) to
iteratively improve the estimate.

# Algorithm

Given current volatility estimate σₙ:

    σₙ₊₁ = σₙ - (V_BS(σₙ) - V_market) / Vega(σₙ)

where Vega(σₙ) = ∂V_BS/∂σ evaluated at σₙ.

# Convergence Properties
- **Quadratic convergence**: Error squares each iteration near solution
- **Typical iterations**: 3-5 for tolerance 1e-6
- **Requirements**: Requires vega > 0 (always true for vanilla options)

# Limitations
- May diverge if initial guess is poor
- Sensitive to extreme moneyness or very short maturity
- Use HybridSolver for production code

See also: [`Bisection`](@ref), [`HybridSolver`](@ref), [`implied_vol`](@ref)
"""

using ..Prezo: EuropeanOption, MarketData, BlackScholes
using ..Prezo: price, greek, Vega

"""
    NewtonRaphson

Newton-Raphson solver configuration for implied volatility.

# Fields
- `max_iter::Int`: Maximum iterations (default: 100)
- `tol::Float64`: Convergence tolerance (default: 1e-6)
- `min_vol::Float64`: Minimum volatility bound (default: 0.001 = 0.1%)
- `max_vol::Float64`: Maximum volatility bound (default: 5.0 = 500%)

# Examples
```julia
# Default settings
solver = NewtonRaphson()

# Custom settings
solver = NewtonRaphson(max_iter=50, tol=1e-8)

# Use with implied_vol
iv = implied_vol(call, market_price, data, solver)
```

See also: [`implied_vol`](@ref)
"""
struct NewtonRaphson <: IVSolver
    max_iter::Int
    tol::Float64
    min_vol::Float64
    max_vol::Float64

    function NewtonRaphson(; max_iter::Int=100, tol::Float64=1e-6,
        min_vol::Float64=0.001, max_vol::Float64=5.0)
        max_iter > 0 || throw(ArgumentError("max_iter must be positive"))
        tol > 0 || throw(ArgumentError("tol must be positive"))
        min_vol > 0 || throw(ArgumentError("min_vol must be positive"))
        max_vol > min_vol || throw(ArgumentError("max_vol must exceed min_vol"))

        new(max_iter, tol, min_vol, max_vol)
    end
end

"""
    iv_objective(option::EuropeanOption, target_price::Real, data::MarketData, vol::Real)

Compute the objective function for implied volatility: difference between
Black-Scholes price at volatility `vol` and target market price.

Returns: (error, bs_price) tuple where error = bs_price - target_price
"""
function iv_objective(option::EuropeanOption, target_price::Real, data::MarketData, vol::Real)
    # Create market data with trial volatility
    trial_data = MarketData(data.spot, data.rate, vol, data.div)

    # Compute Black-Scholes price
    bs_price = price(option, BlackScholes(), trial_data)

    # Return error and price
    error = bs_price - target_price
    return error, bs_price
end

"""
    vega_for_iv(option::EuropeanOption, data::MarketData, vol::Real)

Compute vega for implied volatility calculation.

Returns vega per unit volatility. The greek() function already returns
vega as ∂V/∂σ (price change per unit volatility change).
"""
function vega_for_iv(option::EuropeanOption, data::MarketData, vol::Real)
    trial_data = MarketData(data.spot, data.rate, vol, data.div)

    # greek() returns vega per unit volatility (∂V/∂σ)
    return greek(option, Vega(), trial_data)
end

"""
    is_valid_price(option::EuropeanOption, target_price::Real, data::MarketData)

Check if target price is within valid bounds for implied volatility calculation.

Theoretical bounds:
- Lower bound: Intrinsic value (immediate exercise value)
- Upper bound: Spot price for calls, strike for puts

Returns: (is_valid, lower_bound, upper_bound)
"""
function is_valid_price(option::EuropeanOption, target_price::Real, data::MarketData)
    S = data.spot
    K = option.strike
    r = data.rate
    q = data.div
    T = option.expiry

    # Discount factors
    disc_r = exp(-r * T)
    disc_q = exp(-q * T)

    # Forward price
    F = S * disc_q / disc_r

    if option isa EuropeanCall
        # Call bounds
        intrinsic = max(0.0, S * disc_q - K * disc_r)
        upper_bound = S * disc_q  # Can't be worth more than discounted spot
    else
        # Put bounds
        intrinsic = max(0.0, K * disc_r - S * disc_q)
        upper_bound = K * disc_r  # Can't be worth more than discounted strike
    end

    is_valid = intrinsic <= target_price <= upper_bound

    return is_valid, intrinsic, upper_bound
end

"""
    price_bounds(option::EuropeanOption, data::MarketData)

Compute price bounds as volatility varies from 0 to ∞.

As σ → 0: Price → intrinsic value
As σ → ∞: Price → S (calls) or K*exp(-rT) (puts) for ATM
"""
function price_bounds(option::EuropeanOption, data::MarketData)
    # Very low volatility (approaches intrinsic)
    data_low = MarketData(data.spot, data.rate, 0.001, data.div)
    price_low = price(option, BlackScholes(), data_low)

    # Very high volatility
    data_high = MarketData(data.spot, data.rate, 5.0, data.div)
    price_high = price(option, BlackScholes(), data_high)

    return price_low, price_high
end

"""
    implied_vol(option::EuropeanOption, target_price::Real, data::MarketData,
                solver::NewtonRaphson)

Compute implied volatility using Newton-Raphson method.

# Arguments
- `option::EuropeanOption`: European call or put option
- `target_price::Real`: Observed market price
- `data::MarketData`: Market parameters (spot, rate, div)
- `solver::NewtonRaphson`: Newton-Raphson configuration

# Returns
Implied volatility as `Float64`, or `NaN` if calculation fails.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
market_price = 10.45

# Standard Newton-Raphson
iv = implied_vol(call, market_price, data, NewtonRaphson())

# With custom tolerance
iv_precise = implied_vol(call, market_price, data, NewtonRaphson(tol=1e-8))
```

See also: [`NewtonRaphson`](@ref), [`implied_vol`](@ref) for other solvers
"""
function implied_vol(option::EuropeanOption, target_price::Real, data::MarketData,
    solver::NewtonRaphson)

    # Validate price is in feasible range
    is_valid, intrinsic, upper = is_valid_price(option, target_price, data)
    if !is_valid
        @warn "Price $target_price outside valid bounds [$intrinsic, $upper]"
        return NaN
    end

    # Smart initial guess using Brenner-Subrahmanyam approximation
    vol = compute_initial_vol_guess(option, target_price, data, solver)

    for iter in 1:solver.max_iter
        # Clamp volatility to bounds
        vol = clamp(vol, solver.min_vol, solver.max_vol)

        # Compute objective and derivative
        error, _ = iv_objective(option, target_price, data, vol)
        vega = vega_for_iv(option, data, vol)

        # Check convergence
        if abs(error) < solver.tol
            return vol
        end

        # Check for zero vega (shouldn't happen for valid options)
        if abs(vega) < 1e-10
            @warn "Vega too small at iteration $iter"
            return NaN
        end

        # Newton-Raphson update with step limiting
        step = error / vega

        # Limit step size to prevent overshooting (max 50% change per iteration)
        max_step = 0.5 * vol
        if abs(step) > max_step
            step = sign(step) * max_step
        end

        vol_new = vol - step

        # If we would go negative or too high, use bisection-style fallback
        if vol_new < solver.min_vol
            vol_new = (vol + solver.min_vol) / 2
        elseif vol_new > solver.max_vol
            vol_new = (vol + solver.max_vol) / 2
        end

        # Check for divergence
        if !isfinite(vol_new)
            @warn "Newton-Raphson diverged at iteration $iter"
            return NaN
        end

        vol = vol_new
    end

    @warn "Newton-Raphson did not converge in $(solver.max_iter) iterations"
    return NaN
end

"""
    compute_initial_vol_guess(option, target_price, data, solver)

Compute a smart initial volatility guess using Brenner-Subrahmanyam approximation.

For ATM options: σ ≈ price × √(2π/T) / S
This provides a much better starting point than a fixed 20% guess.
"""
function compute_initial_vol_guess(option::EuropeanOption, target_price::Real,
    data::MarketData, solver::NewtonRaphson)
    S = data.spot
    K = option.strike
    T = option.expiry
    r = data.rate
    q = data.div

    # Forward price
    F = S * exp((r - q) * T)

    # Moneyness
    moneyness = F / K

    # Brenner-Subrahmanyam approximation for ATM: σ ≈ price × √(2π/T) / S
    # Adjusted for moneyness
    if 0.8 < moneyness < 1.2
        # Near ATM - use BS approximation
        bs_approx = target_price * sqrt(2 * π / T) / S
        if solver.min_vol < bs_approx < solver.max_vol
            return bs_approx
        end
    end

    # For ITM/OTM or if BS approx is out of range, try to bracket the solution
    # by testing a few volatilities
    test_vols = [0.1, 0.2, 0.3, 0.5, 0.8]
    best_vol = 0.2
    best_error = Inf

    for test_vol in test_vols
        err, _ = iv_objective(option, target_price, data, test_vol)
        if abs(err) < abs(best_error)
            best_error = err
            best_vol = test_vol
        end
    end

    return best_vol
end
