"""
    analytical.jl

Analytical Greek formulas for Black-Scholes model.

These are closed-form solutions that provide exact Greeks in O(1) time.
All formulas follow standard Black-Scholes notation with d1, d2.

Mathematical Reference:
- d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
- d2 = d1 - σ√T
- n(x) = PDF of standard normal at x
- N(x) = CDF of standard normal at x

All Greeks are continuous functions of the underlying parameters.
"""

using ..Prezo
using Distributions

# Normal distribution helpers
norm_pdf(x::Real) = pdf(Normal(0.0, 1.0), x)
norm_cdf(x::Real) = cdf(Normal(0.0, 1.0), x)

# Compute d1 and d2 for Black-Scholes
function d1_d2(S::T, K::T, r::T, q::T, σ::T, T_exp::T) where {T<:Real}
    σ_sqrt_T = σ * sqrt(T_exp)
    d1 = (log(S / K) + (r - q + 0.5 * σ^2) * T_exp) / σ_sqrt_T
    d2 = d1 - σ_sqrt_T
    return d1, d2
end

## European Call Greeks (Analytical)

"""
    greek(call::EuropeanCall, ::Delta, data::MarketData)

Analytical delta for European call options under Black-Scholes.

Formula: Δ = exp(-qT) * N(d1)
"""
function greek(call::EuropeanCall, ::Delta, data::MarketData)
    S = data.spot
    K = call.strike
    T = call.expiry
    r = data.rate
    q = data.div
    σ = data.vol

    d1, _ = d1_d2(S, K, r, q, σ, T)
    delta = exp(-q * T) * norm_cdf(d1)

    return delta
end

"""
    greek(call::EuropeanCall, ::Gamma, data::MarketData)

Analytical gamma for European call options under Black-Scholes.

Formula: Γ = exp(-qT) * n(d1) / (S * σ * √T)

Note: Same for both calls and puts (second derivative is independent of payoff).
"""
function greek(call::EuropeanCall, ::Gamma, data::MarketData)
    S = data.spot
    K = call.strike
    T = call.expiry
    r = data.rate
    q = data.div
    σ = data.vol

    d1, _ = d1_d2(S, K, r, q, σ, T)
    gamma = exp(-q * T) * norm_pdf(d1) / (S * σ * sqrt(T))

    return gamma
end

"""
    greek(call::EuropeanCall, ::Theta, data::MarketData)

Analytical theta for European call options under Black-Scholes.

Formula: Θ = -S*exp(-qT)*n(d1)*σ/(2√T) - r*K*exp(-rT)*N(d2) + q*S*exp(-qT)*N(d1)

Returns theta per year (divide by 365 for daily theta).
"""
function greek(call::EuropeanCall, ::Theta, data::MarketData)
    S = data.spot
    K = call.strike
    T = call.expiry
    r = data.rate
    q = data.div
    σ = data.vol

    d1, d2 = d1_d2(S, K, r, q, σ, T)

    term1 = -S * exp(-q * T) * norm_pdf(d1) * σ / (2 * sqrt(T))
    term2 = -r * K * exp(-r * T) * norm_cdf(d2)
    term3 = q * S * exp(-q * T) * norm_cdf(d1)

    theta = term1 + term2 + term3

    return theta
end

"""
    greek(call::EuropeanCall, ::Vega, data::MarketData)

Analytical vega for European call options under Black-Scholes.

Formula: ν = S * exp(-qT) * n(d1) * √T

Note: Same for both calls and puts.
Returns vega per unit vol (multiply by 0.01 for per 1% vol).
"""
function greek(call::EuropeanCall, ::Vega, data::MarketData)
    S = data.spot
    K = call.strike
    T = call.expiry
    r = data.rate
    q = data.div
    σ = data.vol

    d1, _ = d1_d2(S, K, r, q, σ, T)
    vega = S * exp(-q * T) * norm_pdf(d1) * sqrt(T)

    return vega
end

"""
    greek(call::EuropeanCall, ::Rho, data::MarketData)

Analytical rho for European call options under Black-Scholes.

Formula: ρ = K * T * exp(-rT) * N(d2)

Returns rho per unit rate (multiply by 0.01 for per 1% rate).
"""
function greek(call::EuropeanCall, ::Rho, data::MarketData)
    S = data.spot
    K = call.strike
    T = call.expiry
    r = data.rate
    q = data.div
    σ = data.vol

    _, d2 = d1_d2(S, K, r, q, σ, T)
    rho = K * T * exp(-r * T) * norm_cdf(d2)

    return rho
end

"""
    greek(call::EuropeanCall, ::Phi, data::MarketData)

Analytical phi (dividend sensitivity) for European call options.

Formula: φ = -S * T * exp(-qT) * N(d1)

Returns phi per unit dividend yield (multiply by 0.01 for per 1%).
"""
function greek(call::EuropeanCall, ::Phi, data::MarketData)
    S = data.spot
    K = call.strike
    T = call.expiry
    r = data.rate
    q = data.div
    σ = data.vol

    d1, _ = d1_d2(S, K, r, q, σ, T)
    phi = -S * T * exp(-q * T) * norm_cdf(d1)

    return phi
end

## European Put Greeks (Analytical)

"""
    greek(put::EuropeanPut, ::Delta, data::MarketData)

Analytical delta for European put options under Black-Scholes.

Formula: Δ = exp(-qT) * (N(d1) - 1)

Note: Put delta is call delta minus exp(-qT).
"""
function greek(put::EuropeanPut, ::Delta, data::MarketData)
    S = data.spot
    K = put.strike
    T = put.expiry
    r = data.rate
    q = data.div
    σ = data.vol

    d1, _ = d1_d2(S, K, r, q, σ, T)
    delta = exp(-q * T) * (norm_cdf(d1) - 1.0)

    return delta
end

"""
    greek(put::EuropeanPut, ::Gamma, data::MarketData)

Analytical gamma for European put options under Black-Scholes.

Formula: Same as call gamma

Γ = exp(-qT) * n(d1) / (S * σ * √T)
"""
function greek(put::EuropeanPut, ::Gamma, data::MarketData)
    # Gamma is the same for calls and puts
    S = data.spot
    K = put.strike
    T = put.expiry
    r = data.rate
    q = data.div
    σ = data.vol

    d1, _ = d1_d2(S, K, r, q, σ, T)
    gamma = exp(-q * T) * norm_pdf(d1) / (S * σ * sqrt(T))

    return gamma
end

"""
    greek(put::EuropeanPut, ::Theta, data::MarketData)

Analytical theta for European put options under Black-Scholes.

Formula: Θ = -S*exp(-qT)*n(d1)*σ/(2√T) + r*K*exp(-rT)*N(-d2) - q*S*exp(-qT)*N(-d1)
"""
function greek(put::EuropeanPut, ::Theta, data::MarketData)
    S = data.spot
    K = put.strike
    T = put.expiry
    r = data.rate
    q = data.div
    σ = data.vol

    d1, d2 = d1_d2(S, K, r, q, σ, T)

    term1 = -S * exp(-q * T) * norm_pdf(d1) * σ / (2 * sqrt(T))
    term2 = r * K * exp(-r * T) * norm_cdf(-d2)
    term3 = -q * S * exp(-q * T) * norm_cdf(-d1)

    theta = term1 + term2 + term3

    return theta
end

"""
    greek(put::EuropeanPut, ::Vega, data::MarketData)

Analytical vega for European put options under Black-Scholes.

Formula: Same as call vega

ν = S * exp(-qT) * n(d1) * √T
"""
function greek(put::EuropeanPut, ::Vega, data::MarketData)
    # Vega is the same for calls and puts
    S = data.spot
    K = put.strike
    T = put.expiry
    r = data.rate
    q = data.div
    σ = data.vol

    d1, _ = d1_d2(S, K, r, q, σ, T)
    vega = S * exp(-q * T) * norm_pdf(d1) * sqrt(T)

    return vega
end

"""
    greek(put::EuropeanPut, ::Rho, data::MarketData)

Analytical rho for European put options under Black-Scholes.

Formula: ρ = -K * T * exp(-rT) * N(-d2)

Note: Put rho is negative (puts lose value as rates rise).
"""
function greek(put::EuropeanPut, ::Rho, data::MarketData)
    S = data.spot
    K = put.strike
    T = put.expiry
    r = data.rate
    q = data.div
    σ = data.vol

    _, d2 = d1_d2(S, K, r, q, σ, T)
    rho = -K * T * exp(-r * T) * norm_cdf(-d2)

    return rho
end

"""
    greek(put::EuropeanPut, ::Phi, data::MarketData)

Analytical phi (dividend sensitivity) for European put options.

Formula: φ = S * T * exp(-qT) * N(-d1)

Note: Put phi is positive (puts benefit from dividends).
"""
function greek(put::EuropeanPut, ::Phi, data::MarketData)
    S = data.spot
    K = put.strike
    T = put.expiry
    r = data.rate
    q = data.div
    σ = data.vol

    d1, _ = d1_d2(S, K, r, q, σ, T)
    phi = S * T * exp(-q * T) * norm_cdf(-d1)

    return phi
end

## Second Order Greeks

"""
    greek(option::EuropeanOption, ::Vanna, data::MarketData)

Analytical vanna for European options under Black-Scholes.

Formula: Vanna = -exp(-qT) * n(d1) * d2 / σ

Vanna measures how delta changes with volatility (and vice versa).
"""
function greek(option::EuropeanOption, ::Vanna, data::MarketData)
    S = data.spot
    K = option.strike
    T = option.expiry
    r = data.rate
    q = data.div
    σ = data.vol

    d1, d2 = d1_d2(S, K, r, q, σ, T)
    vanna = -exp(-q * T) * norm_pdf(d1) * d2 / σ

    return vanna
end

"""
    greek(option::EuropeanOption, ::Vomma, data::MarketData)

Analytical vomma for European options under Black-Scholes.

Formula: Vomma = Vega * d1 * d2 / σ

Vomma measures convexity of vega.
"""
function greek(option::EuropeanOption, ::Vomma, data::MarketData)
    # First compute vega
    vega_val = greek(option, Vega(), data)

    S = data.spot
    K = option.strike
    T = option.expiry
    r = data.rate
    q = data.div
    σ = data.vol

    d1, d2 = d1_d2(S, K, r, q, σ, T)
    vomma = vega_val * d1 * d2 / σ

    return vomma
end

"""
    greek(option::EuropeanOption, ::Charm, data::MarketData)

Analytical charm (delta decay) for European options under Black-Scholes.

Formula for calls: Charm = -q*exp(-qT)*N(d1) + exp(-qT)*n(d1)*(2*(r-q)*T - d2*σ*√T)/(2*T*σ*√T)

Formula for puts: Charm = q*exp(-qT)*N(-d1) - exp(-qT)*n(d1)*(2*(r-q)*T - d2*σ*√T)/(2*T*σ*√T)
"""
function greek(option::EuropeanCall, ::Charm, data::MarketData)
    S = data.spot
    K = option.strike
    T = option.expiry
    r = data.rate
    q = data.div
    σ = data.vol

    d1, d2 = d1_d2(S, K, r, q, σ, T)

    term1 = -q * exp(-q * T) * norm_cdf(d1)
    term2 = exp(-q * T) * norm_pdf(d1) * (2 * (r - q) * T - d2 * σ * sqrt(T)) / (2 * T * σ * sqrt(T))

    charm = term1 + term2
    return charm
end

function greek(option::EuropeanPut, ::Charm, data::MarketData)
    S = data.spot
    K = option.strike
    T = option.expiry
    r = data.rate
    q = data.div
    σ = data.vol

    d1, d2 = d1_d2(S, K, r, q, σ, T)

    term1 = q * exp(-q * T) * norm_cdf(-d1)
    term2 = -exp(-q * T) * norm_pdf(d1) * (2 * (r - q) * T - d2 * σ * sqrt(T)) / (2 * T * σ * sqrt(T))

    charm = term1 + term2
    return charm
end

"""
    greek(option::EuropeanOption, ::Veta, data::MarketData)

Analytical veta (vega decay) for European options under Black-Scholes.

Formula: Veta = Vega * (q + [(r - q) * d1 - d2 / (2T)] / (σ√T))

Veta measures how vega changes as time passes.
"""
function greek(option::EuropeanOption, ::Veta, data::MarketData)
    S = data.spot
    K = option.strike
    T = option.expiry
    r = data.rate
    q = data.div
    σ = data.vol

    d1, d2 = d1_d2(S, K, r, q, σ, T)

    vega_val = greek(option, Vega(), data)

    # Veta = Vega * (q + [(r - q) * d1 - d2 / (2T)] / (σ√T))
    term = q + ((r - q) * d1 - d2 / (2 * T)) / (σ * sqrt(T))
    veta = vega_val * term

    return veta
end

## Batch Greek Computation

"""
    greeks(option::EuropeanOption, data::MarketData, greek_types::Vector{<:Greek})

Compute multiple Greeks at once for efficiency.

# Arguments
- `option::EuropeanOption`: The option contract
- `data::MarketData`: Market parameters
- `greek_types::Vector{<:Greek}`: List of Greeks to compute

# Returns
Dictionary mapping Greek types to values.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)

# Compute major Greeks
major_greeks = greeks(call, data, [Delta(), Gamma(), Theta(), Vega(), Rho()])
```
"""
function greeks(option::EuropeanOption, data::MarketData, greek_types::Vector{G}) where {G<:Greek}
    result = Dict{Greek,Float64}()

    # Pre-compute d1, d2 once for efficiency
    S = data.spot
    K = option.strike
    T = option.expiry
    r = data.rate
    q = data.div
    σ = data.vol

    d1, d2 = d1_d2(S, K, r, q, σ, T)

    # Cache common values
    exp_qT = exp(-q * T)
    exp_rT = exp(-r * T)
    Nd1 = norm_cdf(d1)
    Nd2 = norm_cdf(d2)
    nd1 = norm_pdf(d1)
    sqrtT = sqrt(T)

    for greek_type in greek_types
        # Compute based on type using cached values
        if greek_type isa Delta
            if option isa EuropeanCall
                result[greek_type] = exp_qT * Nd1
            else
                result[greek_type] = exp_qT * (Nd1 - 1.0)
            end
        elseif greek_type isa Gamma
            result[greek_type] = exp_qT * nd1 / (S * σ * sqrtT)
        elseif greek_type isa Vega
            result[greek_type] = S * exp_qT * nd1 * sqrtT
        elseif greek_type isa Theta
            if option isa EuropeanCall
                term1 = -S * exp_qT * nd1 * σ / (2 * sqrtT)
                term2 = -r * K * exp_rT * Nd2
                term3 = q * S * exp_qT * Nd1
                result[greek_type] = term1 + term2 + term3
            else
                term1 = -S * exp_qT * nd1 * σ / (2 * sqrtT)
                term2 = r * K * exp_rT * norm_cdf(-d2)
                term3 = -q * S * exp_qT * norm_cdf(-d1)
                result[greek_type] = term1 + term2 + term3
            end
        elseif greek_type isa Rho
            if option isa EuropeanCall
                result[greek_type] = K * T * exp_rT * Nd2
            else
                result[greek_type] = -K * T * exp_rT * norm_cdf(-d2)
            end
        elseif greek_type isa Phi
            if option isa EuropeanCall
                result[greek_type] = -S * T * exp_qT * Nd1
            else
                result[greek_type] = S * T * exp_qT * norm_cdf(-d1)
            end
        else
            # Fallback to individual greek function
            result[greek_type] = greek(option, greek_type, data)
        end
    end

    return result
end

"""
    all_greeks(option::EuropeanOption, data::MarketData)

Compute all major first-order Greeks at once.

Returns dictionary with: Delta, Gamma, Theta, Vega, Rho, and Phi.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
greek_dict = all_greeks(call, data)

# Access individual Greeks
delta = greek_dict[Delta()]
vega = greek_dict[Vega()]
```
"""
function all_greeks(option::EuropeanOption, data::MarketData)
    return greeks(option, data, [Delta(), Gamma(), Theta(), Vega(), Rho(), Phi()])
end
