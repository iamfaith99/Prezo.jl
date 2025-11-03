# improved_lsm_engines.jl
# This file contains improved LSM implementations using Julian patterns

using Distributions
using LinearAlgebra
using Statistics

# Import basis function types from our demo
include("julian_basis_functions.jl")

# Enhanced LSM engine that can be added to engines.jl
struct JulianLSM{B <: BasisFunction, T <: AbstractFloat}
    basis::B
    steps::Int
    reps::Int
    min_regression_paths::Int

    # Inner constructor ensures type consistency
    function JulianLSM{B, T}(basis::B, steps::Int, reps::Int, min_regression_paths::Int=50) where {B <: BasisFunction, T <: AbstractFloat}
        steps > 0 || throw(ArgumentError("steps must be positive"))
        reps > 0 || throw(ArgumentError("reps must be positive"))
        min_regression_paths > 0 || throw(ArgumentError("min_regression_paths must be positive"))
        new{B, T}(basis, steps, reps, min_regression_paths)
    end
end

# Outer constructor with type inference
function JulianLSM(basis::BasisFunction, steps::Int, reps::Int, min_regression_paths::Int=50)
    return JulianLSM{typeof(basis), Float64}(basis, steps, reps, min_regression_paths)
end

# Convenience constructors for different basis types
function LaguerreLSM(order::Int, steps::Int, reps::Int; normalization=100.0)
    basis = LaguerreBasis(order, normalization)
    return JulianLSM(basis, steps, reps)
end

function ChebyshevLSM(order::Int, steps::Int, reps::Int; domain=(25.0, 55.0))
    basis = ChebyshevBasis(order, domain[1], domain[2])
    return JulianLSM(basis, steps, reps)
end

function PowerLSM(order::Int, steps::Int, reps::Int; normalization=40.0)
    basis = PowerBasis(order, normalization)
    return JulianLSM(basis, steps, reps)
end

function HermiteLSM(order::Int, steps::Int, reps::Int; mean=40.0, std=10.0)
    basis = HermiteBasis(order, mean, std)
    return JulianLSM(basis, steps, reps)
end

# Improved pricing function with better numerical stability
function price(option::AmericanOption, engine::JulianLSM{B, T}, data::MarketData) where {B, T}
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data
    (; basis, steps, reps, min_regression_paths) = engine

    dt = T(expiry / steps)
    disc = exp(-T(rate) * dt)

    # Generate asset price paths with correct type
    paths = asset_paths_col(MonteCarlo(steps, reps), T(spot), T(rate - div), T(vol), T(expiry))

    # Initialize cash flow matrix with correct type
    cash_flows = Matrix{T}(undef, steps + 1, reps)

    # Terminal payoffs using broadcasting
    @. cash_flows[end, :] = max(zero(T), payoff_value(option, paths[end, :]))

    # Backward induction with improved regression
    for t in steps:-1:2
        # Current asset prices
        S_t = @view paths[t, :]

        # Find in-the-money paths using broadcasting
        immediate_payoffs = payoff_value.(option, S_t)
        itm_mask = immediate_payoffs .> zero(T)
        itm_count = sum(itm_mask)

        if itm_count < min_regression_paths
            # Not enough ITM paths for stable regression
            @. cash_flows[t, :] = cash_flows[t + 1, :] * disc
            continue
        end

        # Extract ITM data
        S_itm = S_t[itm_mask]
        continuation_values = cash_flows[t + 1, itm_mask] .* disc
        immediate_itm = immediate_payoffs[itm_mask]

        # Create basis matrix using multiple dispatch
        X = basis(S_itm)

        # Fit continuation value using trait-based dispatch
        try
            β = fit_continuation_value(basis, X, continuation_values)
            predicted_continuation = X * β

            # Exercise decision using broadcasting
            exercise_mask = immediate_itm .> predicted_continuation

            # Update cash flows
            @. cash_flows[t, :] = cash_flows[t + 1, :] * disc

            # Apply exercise decisions
            itm_indices = findall(itm_mask)
            exercise_indices = itm_indices[exercise_mask]

            for idx in exercise_indices
                cash_flows[t, idx] = immediate_payoffs[idx]
                @. cash_flows[t + 1:end, idx] = zero(T)  # Zero out future cash flows
            end

        catch e
            # Fallback if regression fails
            @warn "Regression failed at time step $t, using continuation value" exception=e
            @. cash_flows[t, :] = cash_flows[t + 1, :] * disc
        end
    end

    # Final exercise decision at t=1
    immediate_payoffs_t1 = payoff_value.(option, paths[1, :])
    continuation_values_t1 = cash_flows[2, :] .* disc

    final_cash_flows = max.(immediate_payoffs_t1, continuation_values_t1)

    # Return discounted expected payoff
    return T(disc) * mean(final_cash_flows)
end

# Helper function to get payoff value (more type-stable)
payoff_value(option::AmericanPut, spot::T) where T = max(zero(T), T(option.strike) - spot)
payoff_value(option::AmericanCall, spot::T) where T = max(zero(T), spot - T(option.strike))

# Add theoretical validation constraints
function validate_american_option_price(american_price::T, european_price::T,
                                       option_type::Type{<:AmericanOption},
                                       div::T, tolerance::T=T(1e-6)) where T

    if american_price < european_price - tolerance
        @warn "American option price ($(american_price)) < European price ($(european_price))"
        return false
    end

    # For calls with zero dividends, American ≈ European
    if option_type <: AmericanCall && abs(div) < tolerance
        if american_price > european_price + tolerance
            @warn "American call with zero dividends shows significant premium: $(american_price - european_price)"
            return false
        end
    end

    return true
end

# Enhanced pricing with validation
function validated_price(option::AmericanOption, engine::JulianLSM, data::MarketData)
    american_price = price(option, engine, data)

    # Compare with European price for validation
    european_option = if option isa AmericanPut
        EuropeanPut(option.strike, option.expiry)
    else
        EuropeanCall(option.strike, option.expiry)
    end

    european_price = price(european_option, Prezo.BlackScholes(), data)

    is_valid = validate_american_option_price(american_price, european_price,
                                            typeof(option), data.div)

    return (price=american_price, european_ref=european_price, valid=is_valid)
end

println("Julian LSM Implementation Features:")
println("✓ Abstract type hierarchy for basis functions")
println("✓ Multiple dispatch for different basis types")
println("✓ Type-stable implementation with parametric types")
println("✓ Broadcasting for vectorized operations")
println("✓ Trait-based regression method selection")
println("✓ Error handling for numerical stability")
println("✓ Theoretical validation constraints")
println("✓ Memory-efficient views and in-place operations")
println("✓ Functor pattern for callable basis functions")
println("✓ Zero-cost abstractions through type system")