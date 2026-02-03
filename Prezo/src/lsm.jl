using LinearAlgebra
using Statistics

"""
    LongstaffSchwartz(steps, reps, basis_order=3; antithetic=false)

Least Squares Monte Carlo engine for American option pricing.

Implements the Longstaff–Schwartz algorithm which uses regression to estimate
continuation values at each time step, enabling optimal early exercise decisions.

# Fields
- `steps::Int`: Number of time discretization steps
- `reps::Int`: Number of simulation paths (if antithetic=true, actual paths = 2×reps)
- `basis_order::Int`: Order of Laguerre polynomial basis (default: 3)
- `antithetic::Bool`: Use antithetic variates for variance reduction (default: false)

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
option = AmericanPut(100.0, 1.0)

# Standard configuration
engine = LongstaffSchwartz(50, 10000)
price(option, engine, data)

# With variance reduction (2× paths for same price)
engine = LongstaffSchwartz(50, 5000, 3; antithetic=true)
price(option, engine, data)  # Uses 10000 paths total
```

See also: [`LaguerreLSM`](@ref), [`EnhancedLongstaffSchwartz`](@ref)
"""
struct LongstaffSchwartz <: PricingEngine
    steps::Int
    reps::Int
    basis_order::Int
    antithetic::Bool
end

LongstaffSchwartz(steps::Int, reps::Int) = LongstaffSchwartz(steps, reps, 3, false)
LongstaffSchwartz(steps::Int, reps::Int, basis_order::Int; antithetic::Bool=false) =
    LongstaffSchwartz(steps, reps, basis_order, antithetic)

function price(option::AmericanOption, engine::LongstaffSchwartz, data::MarketData; paths::Union{AbstractMatrix{Float64}, Nothing}=nothing)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data
    (; steps, reps, basis_order, antithetic) = engine

    dt = expiry / steps
    disc = exp(-rate * dt)

    if paths === nothing
        paths = if antithetic
            asset_paths_antithetic(steps, reps, spot, rate - div, vol, expiry)
        else
            asset_paths_col(MonteCarlo(steps, reps), spot, rate - div, vol, expiry)
        end
    end
    # paths must be (steps+1) × n_paths, column-major
    actual_reps = size(paths, 2)

    continuation_value = Vector{Float64}(undef, actual_reps)
    next_continuation = Vector{Float64}(undef, actual_reps)

    @inbounds for i in 1:actual_reps
        continuation_value[i] = payoff(option, paths[end, i])
    end

    exercised = falses(actual_reps)

    max_itm = actual_reps
    workspace_S = Vector{Float64}(undef, max_itm)
    workspace_payoff = Vector{Float64}(undef, max_itm)
    workspace_cont = Vector{Float64}(undef, max_itm)
    workspace_X = Matrix{Float64}(undef, max_itm, basis_order + 1)

    for t in steps:-1:2
        S_t = @view paths[t, :]

        itm_count = 0
        @inbounds for i in 1:actual_reps
            if !exercised[i]
                immediate_payoff = payoff(option, S_t[i])
                if immediate_payoff > 0.0
                    itm_count += 1
                    workspace_S[itm_count] = S_t[i]
                    workspace_payoff[itm_count] = immediate_payoff
                    workspace_cont[itm_count] = continuation_value[i]
                end
            end
        end

        if itm_count == 0
            @inbounds for i in 1:actual_reps
                next_continuation[i] = continuation_value[i] * disc
            end
            continuation_value, next_continuation = next_continuation, continuation_value
            continue
        end

        S_itm = @view workspace_S[1:itm_count]
        immediate_itm = @view workspace_payoff[1:itm_count]
        cont_itm = @view workspace_cont[1:itm_count]

        @inbounds for i in 1:itm_count
            cont_itm[i] *= disc
        end

        X = create_basis_functions_inplace!(
            @view(workspace_X[1:itm_count, :]), S_itm, basis_order
        )

        β = X \ cont_itm
        predicted_continuation = X * β

        @inbounds for i in 1:actual_reps
            next_continuation[i] = continuation_value[i] * disc
        end

        itm_idx = 0
        @inbounds for i in 1:actual_reps
            if !exercised[i]
                immediate_payoff = payoff(option, S_t[i])
                if immediate_payoff > 0.0
                    itm_idx += 1
                    if immediate_itm[itm_idx] > predicted_continuation[itm_idx]
                        next_continuation[i] = immediate_itm[itm_idx]
                        exercised[i] = true
                    end
                end
            end
        end

        continuation_value, next_continuation = next_continuation, continuation_value
    end

    @inbounds for i in 1:actual_reps
        if !exercised[i]
            immediate_payoff = payoff(option, paths[1, i])
            cont_discounted = continuation_value[i] * disc
            continuation_value[i] = max(immediate_payoff, cont_discounted)
        else
            continuation_value[i] *= disc
        end
    end

    return mean(continuation_value) * disc
end

# In-place Laguerre basis for classic LSM (performance-oriented)
function create_basis_functions_inplace!(
    X::AbstractMatrix{Float64},
    S::AbstractVector{Float64},
    order::Int
)
    n = length(S)

    @inbounds for i in 1:n
        X[i, 1] = 1.0
    end

    S_norm = S ./ 100.0

    for j in 1:order
        if j == 1
            @inbounds for i in 1:n
                X[i, j+1] = 1.0 - S_norm[i]
            end
        elseif j == 2
            @inbounds for i in 1:n
                s = S_norm[i]
                X[i, j+1] = 1.0 - 2.0 * s + (s * s) / 2.0
            end
        elseif j == 3
            @inbounds for i in 1:n
                s = S_norm[i]
                s2 = s * s
                X[i, j+1] = 1.0 - 3.0 * s + 1.5 * s2 - (s2 * s) / 6.0
            end
        else
            @inbounds for i in 1:n
                X[i, j+1] = ((2 * j - 1 - S_norm[i]) * X[i, j] - (j - 1) * X[i, j-1]) / j
            end
        end
    end

    return X
end

"""
    EnhancedLongstaffSchwartz{B, T}

Enhanced LSM engine with flexible basis functions and improved numerical stability.

Use the convenience constructors [`LaguerreLSM`](@ref), [`ChebyshevLSM`](@ref),
[`PowerLSM`](@ref), or [`HermiteLSM`](@ref) instead of constructing directly.
"""
struct EnhancedLongstaffSchwartz{B<:BasisFunction,T<:AbstractFloat} <: PricingEngine
    basis::B
    steps::Int
    reps::Int
    min_regression_paths::Int

    function EnhancedLongstaffSchwartz{B,T}(
        basis::B,
        steps::Int,
        reps::Int,
        min_regression_paths::Int=50
    ) where {B<:BasisFunction,T<:AbstractFloat}
        steps > 0 || throw(ArgumentError("steps must be positive"))
        reps > 0 || throw(ArgumentError("reps must be positive"))
        min_regression_paths > 0 || throw(ArgumentError("min_regression_paths must be positive"))
        new{B,T}(basis, steps, reps, min_regression_paths)
    end
end

function EnhancedLongstaffSchwartz(
    basis::BasisFunction,
    steps::Int,
    reps::Int,
    min_regression_paths::Int=50
)
    return EnhancedLongstaffSchwartz{typeof(basis),Float64}(
        basis, steps, reps, min_regression_paths
    )
end

function LaguerreLSM(order::Int, steps::Int, reps::Int; normalization=100.0)
    basis = LaguerreBasis(order, normalization)
    return EnhancedLongstaffSchwartz(basis, steps, reps)
end

function ChebyshevLSM(order::Int, steps::Int, reps::Int; domain=(30.0, 50.0))
    basis = ChebyshevBasis(order, domain[1], domain[2])
    return EnhancedLongstaffSchwartz(basis, steps, reps)
end

function PowerLSM(order::Int, steps::Int, reps::Int; normalization=40.0)
    basis = PowerBasis(order, normalization)
    return EnhancedLongstaffSchwartz(basis, steps, reps)
end

function HermiteLSM(order::Int, steps::Int, reps::Int; mean=40.0, std=10.0)
    basis = HermiteBasis(order, mean, std)
    return EnhancedLongstaffSchwartz(basis, steps, reps)
end

function price(
    option::AmericanOption,
    engine::EnhancedLongstaffSchwartz{B,T},
    data::MarketData
) where {B,T}
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data
    (; basis, steps, reps, min_regression_paths) = engine

    dt = T(expiry / steps)
    disc = exp(-T(rate) * dt)

    paths = asset_paths_col(
        MonteCarlo(steps, reps),
        T(spot),
        T(rate - div),
        T(vol),
        T(expiry)
    )

    cash_flows = Matrix{T}(undef, steps + 1, reps)

    @. cash_flows[end, :] = max(zero(T), payoff_value(option, paths[end, :]))

    for t in steps:-1:2
        S_t = @view paths[t, :]

        immediate_payoffs = payoff_value.(option, S_t)
        itm_mask = immediate_payoffs .> zero(T)
        itm_count = sum(itm_mask)

        if itm_count < min_regression_paths
            @. cash_flows[t, :] = cash_flows[t+1, :] * disc
            continue
        end

        S_itm = S_t[itm_mask]
        continuation_values = cash_flows[t+1, itm_mask] .* disc
        immediate_itm = immediate_payoffs[itm_mask]

        X = basis(S_itm)

        try
            β = fit_continuation_value(basis, X, continuation_values)
            predicted_continuation = X * β

            exercise_mask = immediate_itm .> predicted_continuation

            @. cash_flows[t, :] = cash_flows[t+1, :] * disc

            itm_indices = findall(itm_mask)
            exercise_indices = itm_indices[exercise_mask]

            for idx in exercise_indices
                cash_flows[t, idx] = immediate_payoffs[idx]
                @. cash_flows[t+1:end, idx] = zero(T)
            end
        catch e
            @warn "Regression failed at time step $t, using continuation value" exception = e
            @. cash_flows[t, :] = cash_flows[t+1, :] * disc
        end
    end

    immediate_payoffs_t1 = payoff_value.(option, paths[1, :])
    continuation_values_t1 = cash_flows[2, :] .* disc

    final_cash_flows = max.(immediate_payoffs_t1, continuation_values_t1)

    return T(disc) * mean(final_cash_flows)
end

# Helper function for type-stable payoffs
payoff_value(option::AmericanPut, spot::T) where {T} = max(zero(T), T(option.strike) - spot)
payoff_value(option::AmericanCall, spot::T) where {T} = max(zero(T), spot - T(option.strike))

"""
    validate_american_option_price(american_price, european_price, option_type, div; tolerance=1e-6)

Validate that American option price satisfies basic pricing relationships.

Checks that:
1. American price ≥ European price (early exercise premium)
2. For American calls with zero dividends, price ≈ European price
"""
function validate_american_option_price(
    american_price::T,
    european_price::T,
    option_type::Type{<:AmericanOption},
    div::T,
    tolerance::T=T(1e-6)
) where {T}
    if american_price < european_price - tolerance
        @warn "American option price ($american_price) < European price ($european_price)"
        return false
    end

    if option_type <: AmericanCall && abs(div) < tolerance
        if american_price > european_price + tolerance
            premium = american_price - european_price
            @warn "American call with zero dividends shows significant premium: $premium"
            return false
        end
    end

    return true
end

"""
    EuropeanLongstaffSchwartz{B, T}

LSM-based regression engine for European option pricing.

Uses basis function regression on terminal asset prices to estimate option values.
This is primarily for educational purposes as Black-Scholes is more efficient.
"""
struct EuropeanLongstaffSchwartz{B<:BasisFunction,T<:AbstractFloat} <: PricingEngine
    basis::B
    reps::Int
    min_regression_paths::Int

    function EuropeanLongstaffSchwartz{B,T}(
        basis::B,
        reps::Int,
        min_regression_paths::Int=100
    ) where {B<:BasisFunction,T<:AbstractFloat}
        reps > 0 || throw(ArgumentError("reps must be positive"))
        min_regression_paths > 0 || throw(ArgumentError("min_regression_paths must be positive"))
        new{B,T}(basis, reps, min_regression_paths)
    end
end

function EuropeanLongstaffSchwartz(
    basis::BasisFunction,
    reps::Int,
    min_regression_paths::Int=100
)
    return EuropeanLongstaffSchwartz{typeof(basis),Float64}(
        basis, reps, min_regression_paths
    )
end

function EuropeanLaguerreLSM(order::Int, reps::Int; normalization=100.0)
    basis = LaguerreBasis(order, normalization)
    return EuropeanLongstaffSchwartz(basis, reps)
end

function EuropeanChebyshevLSM(order::Int, reps::Int; domain=(30.0, 50.0))
    basis = ChebyshevBasis(order, domain[1], domain[2])
    return EuropeanLongstaffSchwartz(basis, reps)
end

function EuropeanPowerLSM(order::Int, reps::Int; normalization=40.0)
    basis = PowerBasis(order, normalization)
    return EuropeanLongstaffSchwartz(basis, reps)
end

function EuropeanHermiteLSM(order::Int, reps::Int; mean=40.0, std=10.0)
    basis = HermiteBasis(order, mean, std)
    return EuropeanLongstaffSchwartz(basis, reps)
end

function price(
    option::EuropeanOption,
    engine::EuropeanLongstaffSchwartz{B,T},
    data::MarketData
) where {B,T}
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data
    (; basis, reps, min_regression_paths) = engine

    paths = zeros(T, reps)
    dt = T(expiry)
    nudt = (T(rate - div) - T(0.5) * T(vol)^2) * dt
    sidt = T(vol) * sqrt(dt)

    for i in 1:reps
        z = randn(T)
        paths[i] = T(spot) * exp(nudt + sidt * z)
    end

    terminal_payoffs = payoff_value.(option, paths)

    itm_mask = terminal_payoffs .> zero(T)
    itm_count = sum(itm_mask)

    if itm_count < min_regression_paths
        @warn "Not enough ITM paths ($itm_count < $min_regression_paths), using simple MC"
        return exp(-T(rate) * T(expiry)) * mean(terminal_payoffs)
    end

    S_itm = paths[itm_mask]
    payoffs_itm = terminal_payoffs[itm_mask]

    X = basis(S_itm)

    try
        β = fit_continuation_value(basis, X, payoffs_itm)

        X_all = basis(paths)
        predicted_values = X_all * β
        predicted_values = max.(predicted_values, zero(T))

        return exp(-T(rate) * T(expiry)) * mean(predicted_values)
    catch e
        @warn "European LSM regression failed, falling back to simple MC" exception = e
        return exp(-T(rate) * T(expiry)) * mean(terminal_payoffs)
    end
end

payoff_value(option::EuropeanPut, spot::T) where {T} = max(zero(T), T(option.strike) - spot)
payoff_value(option::EuropeanCall, spot::T) where {T} = max(zero(T), spot - T(option.strike))
