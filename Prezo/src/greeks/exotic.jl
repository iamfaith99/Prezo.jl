"""
    exotic.jl

Numerical Greek calculations for exotic (path-dependent) options.

Exotic options (Asian, Barrier, Lookback) are path-dependent and generally
do not have closed-form Greek formulas. This module provides numerical
approximations via finite differences using Monte Carlo pricing.

Key considerations for exotic option Greeks:
- Higher variance due to Monte Carlo noise (use more paths for stability)
- Some Greeks may require special handling (e.g., barrier proximity)
- Theta calculation requires constructing a new option with modified expiry

Supported option types:
- Asian options: ArithmeticAsianCall/Put, GeometricAsianCall/Put
- Barrier options: KnockOutCall/Put, KnockInCall/Put
- Lookback options: FixedStrikeLookbackCall/Put, FloatingStrikeLookbackCall/Put
"""

using ..Prezo
using ..Prezo: ExoticOption, AsianOption, BarrierOption, LookbackOption
using ..Prezo: ArithmeticAsianCall, ArithmeticAsianPut, GeometricAsianCall, GeometricAsianPut
using ..Prezo: KnockOutCall, KnockOutPut, KnockInCall, KnockInPut
using ..Prezo: FixedStrikeLookbackCall, FixedStrikeLookbackPut
using ..Prezo: FloatingStrikeLookbackCall, FloatingStrikeLookbackPut

# ============================================================================
# Helper Functions for Option Reconstruction
# ============================================================================

"""
    get_expiry(option::ExoticOption)

Extract expiry from any exotic option type.
"""
get_expiry(option::ArithmeticAsianCall) = option.expiry
get_expiry(option::ArithmeticAsianPut) = option.expiry
get_expiry(option::GeometricAsianCall) = option.expiry
get_expiry(option::GeometricAsianPut) = option.expiry
get_expiry(option::KnockOutCall) = option.expiry
get_expiry(option::KnockOutPut) = option.expiry
get_expiry(option::KnockInCall) = option.expiry
get_expiry(option::KnockInPut) = option.expiry
get_expiry(option::FixedStrikeLookbackCall) = option.expiry
get_expiry(option::FixedStrikeLookbackPut) = option.expiry
get_expiry(option::FloatingStrikeLookbackCall) = option.expiry
get_expiry(option::FloatingStrikeLookbackPut) = option.expiry

"""
    with_shorter_expiry(option::ExoticOption, new_expiry::Real)

Create a copy of the exotic option with a shorter expiry time.
For Asian options, also scales the averaging times proportionally.
"""
function with_shorter_expiry(option::ArithmeticAsianCall, new_expiry::Real)
    ratio = new_expiry / option.expiry
    new_times = option.averaging_times .* ratio
    return ArithmeticAsianCall(option.strike, new_expiry, new_times)
end

function with_shorter_expiry(option::ArithmeticAsianPut, new_expiry::Real)
    ratio = new_expiry / option.expiry
    new_times = option.averaging_times .* ratio
    return ArithmeticAsianPut(option.strike, new_expiry, new_times)
end

function with_shorter_expiry(option::GeometricAsianCall, new_expiry::Real)
    ratio = new_expiry / option.expiry
    new_times = option.averaging_times .* ratio
    return GeometricAsianCall(option.strike, new_expiry, new_times)
end

function with_shorter_expiry(option::GeometricAsianPut, new_expiry::Real)
    ratio = new_expiry / option.expiry
    new_times = option.averaging_times .* ratio
    return GeometricAsianPut(option.strike, new_expiry, new_times)
end

function with_shorter_expiry(option::KnockOutCall, new_expiry::Real)
    return KnockOutCall(option.strike, new_expiry, option.barrier, option.barrier_type)
end

function with_shorter_expiry(option::KnockOutPut, new_expiry::Real)
    return KnockOutPut(option.strike, new_expiry, option.barrier, option.barrier_type)
end

function with_shorter_expiry(option::KnockInCall, new_expiry::Real)
    return KnockInCall(option.strike, new_expiry, option.barrier, option.barrier_type)
end

function with_shorter_expiry(option::KnockInPut, new_expiry::Real)
    return KnockInPut(option.strike, new_expiry, option.barrier, option.barrier_type)
end

function with_shorter_expiry(option::FixedStrikeLookbackCall, new_expiry::Real)
    return FixedStrikeLookbackCall(option.strike, new_expiry)
end

function with_shorter_expiry(option::FixedStrikeLookbackPut, new_expiry::Real)
    return FixedStrikeLookbackPut(option.strike, new_expiry)
end

function with_shorter_expiry(option::FloatingStrikeLookbackCall, new_expiry::Real)
    return FloatingStrikeLookbackCall(new_expiry)
end

function with_shorter_expiry(option::FloatingStrikeLookbackPut, new_expiry::Real)
    return FloatingStrikeLookbackPut(new_expiry)
end

# ============================================================================
# Numerical Greeks for Exotic Options
# ============================================================================

# ============================================================================
# Step Size Selection for Monte Carlo Greeks
# ============================================================================

"""
    mc_optimal_step_spot(spot::Real)

Compute optimal step size for spot-based Greeks with Monte Carlo pricing.

For Monte Carlo, we need larger steps than analytical methods due to noise.
Uses 1% of spot price as default, which balances accuracy and stability.
"""
mc_optimal_step_spot(spot::Real) = 0.01 * spot

"""
    mc_optimal_step_vol()

Optimal step size for volatility-based Greeks with Monte Carlo.
Uses 1% (absolute) volatility shift.
"""
mc_optimal_step_vol() = 0.01

"""
    mc_optimal_step_rate()

Optimal step size for rate-based Greeks with Monte Carlo.
Uses 1% (absolute) rate shift.
"""
mc_optimal_step_rate() = 0.01

"""
    numerical_greek(option::ExoticOption, ::Delta, engine::PricingEngine,
                     data::MarketData; h::Real=mc_optimal_step_spot(data.spot))

Numerical delta for exotic options via finite differences.

Formula: Δ ≈ (V(S+h) - V(S-h)) / (2h)

# Arguments
- `option::ExoticOption`: The exotic option contract
- `engine::PricingEngine`: Pricing engine (typically MonteCarlo)
- `data::MarketData`: Market parameters
- `h::Real`: Spot price step size (default: 1% of spot)

# Notes
For barrier options near the barrier, delta may exhibit discontinuities.
Consider using a larger step size or checking barrier proximity.

Monte Carlo pricing has inherent noise, so we use larger step sizes (1% of spot)
compared to analytical methods to ensure stable Greek estimates.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
times = collect(1:12) ./ 12
asian = ArithmeticAsianCall(100.0, 1.0, times)
engine = MonteCarlo(100, 50000)

delta = numerical_greek(asian, Delta(), engine, data)
```
"""
function numerical_greek(option::ExoticOption, ::Delta, engine::PricingEngine,
                         data::MarketData; h::Real=mc_optimal_step_spot(data.spot))

    price_func = S -> begin
        new_data = MarketData(S, data.rate, data.vol, data.div)
        return price(option, engine, new_data)
    end

    return finite_difference(price_func, data.spot, h, method=:central)
end

"""
    numerical_greek(option::ExoticOption, ::Gamma, engine::PricingEngine,
                     data::MarketData; h::Real=mc_optimal_step_spot(data.spot))

Numerical gamma for exotic options via finite differences.

Formula: Γ ≈ (V(S+h) - 2V(S) + V(S-h)) / h²

# Notes
Gamma for barrier options may be very large near the barrier level.
Consider this when interpreting results for knock-out options.

For Monte Carlo pricing, gamma estimates can be noisy due to the h² denominator.
Consider using more simulation paths for stable gamma estimates.
"""
function numerical_greek(option::ExoticOption, ::Gamma, engine::PricingEngine,
                         data::MarketData; h::Real=mc_optimal_step_spot(data.spot))

    S = data.spot

    price_up = price(option, engine, MarketData(S + h, data.rate, data.vol, data.div))
    price_mid = price(option, engine, MarketData(S, data.rate, data.vol, data.div))
    price_down = price(option, engine, MarketData(S - h, data.rate, data.vol, data.div))

    return (price_up - 2*price_mid + price_down) / (h^2)
end

"""
    numerical_greek(option::ExoticOption, ::Theta, engine::PricingEngine,
                     data::MarketData; h::Real=1/365)

Numerical theta for exotic options via finite differences.

Formula: Θ ≈ (V(T-h) - V(T)) / h

Returns theta per year (divide by 365 for daily theta).

# Notes
For Asian options, the averaging times are scaled proportionally when
computing theta. For options near expiry, theta may not be computable.
"""
function numerical_greek(option::ExoticOption, ::Theta, engine::PricingEngine,
                         data::MarketData; h::Real=1/365)

    T = get_expiry(option)

    if T <= h
        # Option is expiring soon, can't reduce time further
        return 0.0
    end

    # Price at current expiry
    price_now = price(option, engine, data)

    # Create option with shorter maturity
    option_later = with_shorter_expiry(option, T - h)
    price_later = price(option_later, engine, data)

    # Forward difference, scaled to per year
    return (price_later - price_now) / h
end

"""
    numerical_greek(option::ExoticOption, ::Vega, engine::PricingEngine,
                     data::MarketData; h::Real=mc_optimal_step_vol())

Numerical vega for exotic options via finite differences.

Formula: ν ≈ (V(σ+h) - V(σ-h)) / (2h)

# Notes
Vega for barrier options depends significantly on how close spot is to the barrier.
For lookback options, vega is generally higher than for vanilla options due to
the path-dependency capturing extreme moves.

Returns vega per unit volatility change. Multiply by 0.01 for conventional
vega (per 1% vol change).
"""
function numerical_greek(option::ExoticOption, ::Vega, engine::PricingEngine,
                         data::MarketData; h::Real=mc_optimal_step_vol())

    σ = data.vol

    # Ensure we don't go negative with volatility
    h_actual = min(h, σ * 0.9)

    price_func = vol -> begin
        new_data = MarketData(data.spot, data.rate, vol, data.div)
        return price(option, engine, new_data)
    end

    return finite_difference(price_func, σ, h_actual, method=:central)
end

"""
    numerical_greek(option::ExoticOption, ::Rho, engine::PricingEngine,
                     data::MarketData; h::Real=mc_optimal_step_rate())

Numerical rho for exotic options via finite differences.

Formula: ρ ≈ (V(r+h) - V(r-h)) / (2h)

Returns rho per unit rate change. Multiply by 0.01 for conventional
rho (per 1% rate change).
"""
function numerical_greek(option::ExoticOption, ::Rho, engine::PricingEngine,
                         data::MarketData; h::Real=mc_optimal_step_rate())

    r = data.rate

    price_func = rate -> begin
        new_data = MarketData(data.spot, rate, data.vol, data.div)
        return price(option, engine, new_data)
    end

    return finite_difference(price_func, r, h, method=:central)
end

"""
    numerical_greek(option::ExoticOption, ::Phi, engine::PricingEngine,
                     data::MarketData; h::Real=mc_optimal_step_rate())

Numerical phi (dividend sensitivity) for exotic options via finite differences.

Formula: φ ≈ (V(q+h) - V(q-h)) / (2h)

Returns phi per unit dividend yield change. Multiply by 0.01 for conventional
phi (per 1% dividend change).
"""
function numerical_greek(option::ExoticOption, ::Phi, engine::PricingEngine,
                         data::MarketData; h::Real=mc_optimal_step_rate())

    q = data.div

    price_func = div_yield -> begin
        new_data = MarketData(data.spot, data.rate, data.vol, div_yield)
        return price(option, engine, new_data)
    end

    return finite_difference(price_func, q, h, method=:central)
end

# ============================================================================
# Second Order Greeks for Exotic Options
# ============================================================================

"""
    numerical_greek(option::ExoticOption, ::Vanna, engine::PricingEngine,
                     data::MarketData; h_S::Real=mc_optimal_step_spot(data.spot), h_σ::Real=mc_optimal_step_vol())

Numerical vanna (∂Δ/∂σ = ∂ν/∂S) for exotic options.

Formula: Vanna ≈ [V(S+h,σ+h) - V(S+h,σ-h) - V(S-h,σ+h) + V(S-h,σ-h)] / (4*h_S*h_σ)

Cross-derivative measuring how delta changes with volatility.
"""
function numerical_greek(option::ExoticOption, ::Vanna, engine::PricingEngine,
                         data::MarketData; h_S::Real=mc_optimal_step_spot(data.spot), h_σ::Real=mc_optimal_step_vol())

    S, σ = data.spot, data.vol

    # Ensure we don't go negative with volatility
    h_σ_actual = min(h_σ, σ * 0.9)

    V_up_up = price(option, engine, MarketData(S + h_S, data.rate, σ + h_σ_actual, data.div))
    V_up_dn = price(option, engine, MarketData(S + h_S, data.rate, σ - h_σ_actual, data.div))
    V_dn_up = price(option, engine, MarketData(S - h_S, data.rate, σ + h_σ_actual, data.div))
    V_dn_dn = price(option, engine, MarketData(S - h_S, data.rate, σ - h_σ_actual, data.div))

    return (V_up_up - V_up_dn - V_dn_up + V_dn_dn) / (4 * h_S * h_σ_actual)
end

"""
    numerical_greek(option::ExoticOption, ::Vomma, engine::PricingEngine,
                     data::MarketData; h::Real=mc_optimal_step_vol())

Numerical vomma (∂²V/∂σ²) for exotic options.

Formula: Vomma ≈ (V(σ+h) - 2V(σ) + V(σ-h)) / h²

Second derivative of price with respect to volatility.
Measures convexity of vega.
"""
function numerical_greek(option::ExoticOption, ::Vomma, engine::PricingEngine,
                         data::MarketData; h::Real=mc_optimal_step_vol())

    σ = data.vol

    # Ensure we don't go negative with volatility
    h_actual = min(h, σ * 0.9)

    V_up = price(option, engine, MarketData(data.spot, data.rate, σ + h_actual, data.div))
    V_mid = price(option, engine, MarketData(data.spot, data.rate, σ, data.div))
    V_dn = price(option, engine, MarketData(data.spot, data.rate, σ - h_actual, data.div))

    return (V_up - 2*V_mid + V_dn) / (h_actual^2)
end

# ============================================================================
# Batch Greeks Computation
# ============================================================================

"""
    greeks(option::ExoticOption, engine::PricingEngine, data::MarketData,
           greek_types::Vector{<:Greek}; kwargs...)

Compute multiple Greeks for an exotic option.

# Arguments
- `option::ExoticOption`: The exotic option contract
- `engine::PricingEngine`: Pricing engine (typically MonteCarlo with many paths)
- `data::MarketData`: Market parameters
- `greek_types::Vector{<:Greek}`: List of Greeks to compute
- `kwargs...`: Additional arguments passed to numerical methods

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
barrier = KnockOutCall(100.0, 1.0, 120.0, :up_and_out)
engine = MonteCarlo(100, 100000)

# Compute Greeks for barrier option
barrier_greeks = greeks(barrier, engine, data, [Delta(), Gamma(), Vega()])
```
"""
function greeks(option::ExoticOption, engine::PricingEngine,
                data::MarketData, greek_types::Vector{T}; kwargs...) where T <: Greek

    result = Dict{Greek, Float64}()
    for greek_type in greek_types
        result[greek_type] = numerical_greek(option, greek_type, engine, data; kwargs...)
    end

    return result
end

"""
    all_greeks(option::ExoticOption, engine::PricingEngine, data::MarketData; kwargs...)

Compute all major first-order Greeks for an exotic option.

Returns dictionary with: Delta, Gamma, Theta, Vega, Rho, and Phi.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
times = collect(1:12) ./ 12
asian = ArithmeticAsianCall(100.0, 1.0, times)
engine = MonteCarlo(100, 50000)

greek_dict = all_greeks(asian, engine, data)
```

# Performance Notes
Computing all Greeks requires multiple pricings. For Monte Carlo engines,
consider using more paths (50,000+) to reduce noise in Greek estimates.
"""
function all_greeks(option::ExoticOption, engine::PricingEngine, data::MarketData; kwargs...)
    return greeks(option, engine, data, [Delta(), Gamma(), Theta(), Vega(), Rho(), Phi()]; kwargs...)
end

# ============================================================================
# Barrier-Specific Utilities
# ============================================================================

"""
    barrier_proximity(option::BarrierOption, data::MarketData)

Compute the proximity of spot to the barrier level.

Returns a value between 0 and 1, where:
- 0 means spot is at the barrier
- 1 means spot is far from the barrier

This is useful for understanding Greek behavior near barriers.
"""
function barrier_proximity(option::BarrierOption, data::MarketData)
    return abs(data.spot - option.barrier) / data.spot
end

"""
    is_near_barrier(option::BarrierOption, data::MarketData; threshold::Real=0.02)

Check if spot is within `threshold` (as fraction of spot) of the barrier.

Near the barrier, Greeks may be unreliable due to discontinuities.
"""
function is_near_barrier(option::BarrierOption, data::MarketData; threshold::Real=0.02)
    return barrier_proximity(option, data) < threshold
end

# ============================================================================
# Unified Interface Extensions
# ============================================================================

"""
    greek(option::ExoticOption, greek_type::Greek,
          engine::PricingEngine, data::MarketData; kwargs...)

Unified interface for computing Greeks for exotic options.

Always uses numerical methods since exotic options generally lack
closed-form Greek formulas.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
lookback = FloatingStrikeLookbackCall(1.0)
engine = MonteCarlo(252, 50000)

delta = greek(lookback, Delta(), engine, data)
vega = greek(lookback, Vega(), engine, data, h=0.005)
```

See also: [`numerical_greek`](@ref), [`all_greeks`](@ref)
"""
function greek(option::ExoticOption, greek_type::Greek,
               engine::PricingEngine, data::MarketData; kwargs...)
    return numerical_greek(option, greek_type, engine, data; kwargs...)
end
