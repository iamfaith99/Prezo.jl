"""
    GARCH family (univariate)

Implements GARCH(1,1), EGARCH(1,1), and GJR-GARCH(1,1) from scratch.
Shared filter: one variance recursion per model type; MLE via Optim.

# Models
- `GARCH`: h_t = ω + α*r²_{t-1} + β*h_{t-1}
- `EGARCH`: log(h_t) = ω + α*(|z_{t-1}| - E|z|) + θ*z_{t-1} + β*log(h_{t-1})
- `GJRGARCH`: h_t = ω + α*r²_{t-1} + γ*r²_{t-1}*I(r_{t-1}<0) + β*h_{t-1}
- `AGARCH`: h_t = ω + α*(z_{t-1} - γ)² + β*h_{t-1} with z = r/√h (shifted standardized shock)
- Student-t innovations: optional `dist=TDist(ν)` in `loglikelihood` and `simulate`

Regime-switching and multivariate GARCH live in separate files: `regime_garch.jl`, `garch_multivariate.jl`.

# Usage
```julia
returns = randn(1000) .* 0.01
model = fit(GARCH, returns)
h = volatility_process(model, returns)
f = forecast(model, returns, 5)
returns_sim, h_sim = simulate(model, 500; seed=42)
```
"""

using Distributions
using Optim
using Statistics
using Random

# -----------------------------------------------------------------------------
# Type hierarchy
# -----------------------------------------------------------------------------

"""
    VolatilityModel

Abstract type for volatility models (GARCH family, Heston, etc.).
"""
abstract type VolatilityModel end

"""
    GARCHModel <: VolatilityModel

Abstract type for univariate GARCH-family models.
"""
abstract type GARCHModel <: VolatilityModel end

"""
    GARCH(ω, α, β)

GARCH(1,1): h_t = ω + α*r²_{t-1} + β*h_{t-1}.
Stationarity: ω > 0, α ≥ 0, β ≥ 0, α + β < 1.
"""
struct GARCH{T<:Real} <: GARCHModel
    ω::T
    α::T
    β::T

    function GARCH(ω::T, α::T, β::T) where {T<:Real}
        ω > 0 || throw(ArgumentError("ω must be positive"))
        α ≥ 0 || throw(ArgumentError("α must be non-negative"))
        β ≥ 0 || throw(ArgumentError("β must be non-negative"))
        α + β < 1 || throw(ArgumentError("α + β must be < 1 for stationarity"))
        new{T}(ω, α, β)
    end
end

"""
    EGARCH(ω, α, θ, β)

Exponential GARCH(1,1): log(h_t) = ω + α*(|z_{t-1}| - E|z|) + θ*z_{t-1} + β*log(h_{t-1}).
Leverage effect via θ (negative θ => negative returns increase volatility).
"""
struct EGARCH{T<:Real} <: GARCHModel
    ω::T
    α::T
    θ::T
    β::T

    function EGARCH(ω::T, α::T, θ::T, β::T) where {T<:Real}
        # Persistence |β| < 1 typically required for stationarity
        abs(β) < 1 || throw(ArgumentError("|β| must be < 1 for stationarity"))
        new{T}(ω, α, θ, β)
    end
end

"""
    GJRGARCH(ω, α, γ, β)

GJR-GARCH(1,1): h_t = ω + α*r²_{t-1} + γ*r²_{t-1}*I(r_{t-1}<0) + β*h_{t-1}.
Asymmetry: γ > 0 means negative returns increase volatility more.
Stationarity: ω > 0, α ≥ 0, α+γ ≥ 0, β ≥ 0, α + γ/2 + β < 1.
"""
struct GJRGARCH{T<:Real} <: GARCHModel
    ω::T
    α::T
    γ::T
    β::T

    function GJRGARCH(ω::T, α::T, γ::T, β::T) where {T<:Real}
        ω > 0 || throw(ArgumentError("ω must be positive"))
        α ≥ 0 || throw(ArgumentError("α must be non-negative"))
        β ≥ 0 || throw(ArgumentError("β must be non-negative"))
        α + γ / 2 + β < 1 || throw(ArgumentError("α + γ/2 + β must be < 1 for stationarity"))
        new{T}(ω, α, γ, β)
    end
end

"""
    AGARCH(ω, α, γ, β)

A-GARCH(1,1) with shifted standardized shock: h_t = ω + α*(z_{t-1} - γ)² + β*h_{t-1}
where z_t = r_t/√h_t. γ > 0 gives stronger effect of negative shocks (leverage).
Stationarity: ω > 0, α ≥ 0, β ≥ 0, α + β < 1.
"""
struct AGARCH{T<:Real} <: GARCHModel
    ω::T
    α::T
    γ::T
    β::T

    function AGARCH(ω::T, α::T, γ::T, β::T) where {T<:Real}
        ω > 0 || throw(ArgumentError("ω must be positive"))
        α ≥ 0 || throw(ArgumentError("α must be non-negative"))
        β ≥ 0 || throw(ArgumentError("β must be non-negative"))
        α + β < 1 || throw(ArgumentError("α + β must be < 1 for stationarity"))
        new{T}(ω, α, γ, β)
    end
end

# -----------------------------------------------------------------------------
# Variance update (one step) — internal
# -----------------------------------------------------------------------------

function _variance_update(model::GARCH, h_prev::T, r_prev::T) where {T<:Real}
    return model.ω + model.α * r_prev^2 + model.β * h_prev
end

# E|z| for z ~ N(0,1)
const _EZ_ABS = sqrt(2 / π)

function _variance_update(model::EGARCH, h_prev::T, r_prev::T) where {T<:Real}
    z = r_prev / sqrt(max(h_prev, 1e-12))
    log_h_prev = log(max(h_prev, 1e-12))
    log_h = model.ω + model.α * (abs(z) - _EZ_ABS) + model.θ * z + model.β * log_h_prev
    return exp(log_h)
end

function _variance_update(model::GJRGARCH, h_prev::T, r_prev::T) where {T<:Real}
    shock = r_prev^2
    asym = (r_prev < 0) ? model.γ * shock : zero(T)
    return model.ω + model.α * shock + asym + model.β * h_prev
end

function _variance_update(model::AGARCH, h_prev::T, r_prev::T) where {T<:Real}
    z = r_prev / sqrt(max(h_prev, 1e-12))
    return model.ω + model.α * (z - model.γ)^2 + model.β * h_prev
end

# -----------------------------------------------------------------------------
# Unconditional variance (for initial h_0)
# -----------------------------------------------------------------------------

"""Unconditional variance E[h] for stationarity."""
function _unconditional_variance(model::GARCH)
    return model.ω / (1 - model.α - model.β)
end

function _unconditional_variance(model::EGARCH)
    return exp((model.ω - model.α * _EZ_ABS) / (1 - model.β))
end

function _unconditional_variance(model::GJRGARCH)
    return model.ω / (1 - model.α - model.γ / 2 - model.β)
end

# A-GARCH: E[(z-γ)²] = 1 + γ² for z~N(0,1), so E[h] = ω + α(1+γ²) + β*E[h] => E[h] = (ω+α(1+γ²))/(1-β)
function _unconditional_variance(model::AGARCH)
    return (model.ω + model.α * (1 + model.γ^2)) / (1 - model.β)
end

# -----------------------------------------------------------------------------
# Volatility process: returns -> conditional variances h[1:n]
# -----------------------------------------------------------------------------

"""
    volatility_process(model::GARCHModel, returns::Vector{<:Real}) -> Vector{Float64}

Compute the conditional variance series h_t for the given returns.
First value h_1 uses unconditional variance as initial condition.
"""
function volatility_process(model::GARCHModel, returns::Vector{<:Real})
    n = length(returns)
    n >= 1 || return Float64[]
    T = float(eltype(returns))
    h = Vector{T}(undef, n)
    h[1] = _unconditional_variance(model)
    for t in 2:n
        @inbounds h[t] = _variance_update(model, h[t-1], returns[t-1])
    end
    return h
end

# -----------------------------------------------------------------------------
# Log-likelihood (Gaussian or Student-t innovations)
# -----------------------------------------------------------------------------

"""
    loglikelihood(model::GARCHModel, returns::Vector{<:Real}; dist=Normal(0, 1)) -> Float64

Log-likelihood of returns under the model. `dist` is the distribution of the
standardized innovation z_t = r_t/√h_t (e.g. `Normal(0, 1)` or `TDist(ν)` with ν > 2).
"""
function loglikelihood(model::GARCHModel, returns::Vector{<:Real}; dist=Normal(0, 1))
    h = volatility_process(model, returns)
    n = length(returns)
    ll = 0.0
    for t in 1:n
        ht = max(h[t], 1e-12)
        z = returns[t] / sqrt(ht)
        ll += logpdf(dist, z) - 0.5 * log(ht)
    end
    return ll
end

# -----------------------------------------------------------------------------
# Forecast: h_{n+1}, h_{n+2}, ... (horizon steps)
# -----------------------------------------------------------------------------

"""
    forecast(model::GARCHModel, returns::Vector{<:Real}, horizon::Int) -> Vector{Float64}

Forecast conditional variance for the next `horizon` steps.
Uses last observed return and last h from volatility_process; then recurses.
"""
function forecast(model::GARCHModel, returns::Vector{<:Real}, horizon::Int)
    horizon >= 1 || return Float64[]
    h = volatility_process(model, returns)
    n = length(returns)
    T = float(eltype(returns))
    out = Vector{T}(undef, horizon)
    h_cur = n >= 1 ? h[n] : _unconditional_variance(model)
    r_last = n >= 1 ? returns[n] : zero(T)
    for i in 1:horizon
        h_cur = _variance_update(model, h_cur, r_last)
        @inbounds out[i] = h_cur
        r_last = 0.0  # E[r]=0 for forecast
    end
    return out
end

# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------

"""
    simulate(model::GARCHModel, n::Int; seed=nothing, dist=Normal(0, 1)) -> Tuple{Vector{Float64}, Vector{Float64}}

Simulate (returns, conditional variances) for n steps. `dist` is the innovation distribution
(e.g. `Normal(0, 1)` or `TDist(ν)`). For TDist, use ν > 2.
"""
function simulate(model::GARCHModel, n::Int; seed=nothing, dist=Normal(0, 1))
    seed === nothing || Random.seed!(seed)
    T = Float64
    h = Vector{T}(undef, n)
    r = Vector{T}(undef, n)
    h[1] = _unconditional_variance(model)
    z1 = rand(dist)
    r[1] = z1 * sqrt(h[1])
    for t in 2:n
        @inbounds h[t] = _variance_update(model, h[t-1], r[t-1])
        @inbounds r[t] = rand(dist) * sqrt(max(h[t], 1e-12))
    end
    return r, h
end

# -----------------------------------------------------------------------------
# MLE fit: negative log-likelihood and box constraints
# -----------------------------------------------------------------------------

function _neg_loglik_garch(params::Vector{T}, returns::Vector{<:Real}) where {T<:Real}
    ω, α, β = params[1], params[2], params[3]
    if ω <= 0 || α < 0 || β < 0 || α + β >= 1
        return T(Inf)
    end
    model = GARCH(ω, α, β)
    return -loglikelihood(model, returns)
end

function _neg_loglik_egarch(params::Vector{T}, returns::Vector{<:Real}) where {T<:Real}
    ω, α, θ, β = params[1], params[2], params[3], params[4]
    if abs(β) >= 1
        return T(Inf)
    end
    model = EGARCH(ω, α, θ, β)
    return -loglikelihood(model, returns)
end

function _neg_loglik_gjr(params::Vector{T}, returns::Vector{<:Real}) where {T<:Real}
    ω, α, γ, β = params[1], params[2], params[3], params[4]
    if ω <= 0 || α < 0 || β < 0 || α + γ / 2 + β >= 1
        return T(Inf)
    end
    model = GJRGARCH(ω, α, γ, β)
    return -loglikelihood(model, returns)
end

function _neg_loglik_agarch(params::Vector{T}, returns::Vector{<:Real}) where {T<:Real}
    ω, α, γ, β = params[1], params[2], params[3], params[4]
    if ω <= 0 || α < 0 || β < 0 || α + β >= 1
        return T(Inf)
    end
    model = AGARCH(ω, α, γ, β)
    return -loglikelihood(model, returns)
end

"""
    fit(::Type{GARCH}, returns::Vector{<:Real}; method=:MLE) -> GARCH{Float64}

Estimate GARCH(1,1) by maximum likelihood (Optim LBFGS).
"""
function fit(::Type{GARCH}, returns::Vector{<:Real}; method=:MLE)
    method == :MLE || throw(ArgumentError("Only :MLE supported"))
    n = length(returns)
    n >= 10 || throw(ArgumentError("Need at least 10 observations"))
    var_r = var(returns)
    mean_r = mean(returns)
    # Centered returns for estimation
    r = Float64.(returns .- mean_r)
    # Initial: ω = 0.1*var, α = 0.1, β = 0.85
    ω0 = 0.1 * var_r
    α0 = 0.1
    β0 = 0.85
    lb = [1e-8, 0.0, 0.0]
    ub = [Inf, Inf, 1.0]
    x0 = [ω0, α0, β0]
    obj = x -> _neg_loglik_garch(x, r)
    result = optimize(obj, lb, ub, x0, Fminbox(LBFGS()); inplace=false)
    ω, α, β = Optim.minimizer(result)
    return GARCH(ω, α, β)
end

"""
    fit(::Type{EGARCH}, returns::Vector{<:Real}; method=:MLE) -> EGARCH{Float64}
"""
function fit(::Type{EGARCH}, returns::Vector{<:Real}; method=:MLE)
    method == :MLE || throw(ArgumentError("Only :MLE supported"))
    n = length(returns)
    n >= 10 || throw(ArgumentError("Need at least 10 observations"))
    r = Float64.(returns .- mean(returns))
    ω0, α0, θ0, β0 = 0.0, 0.1, -0.05, 0.9
    lb = [-Inf, 0.0, -Inf, -1.0]
    ub = [Inf, Inf, Inf, 1.0]
    x0 = [ω0, α0, θ0, β0]
    obj = x -> _neg_loglik_egarch(x, r)
    result = optimize(obj, lb, ub, x0, Fminbox(LBFGS()); inplace=false)
    ω, α, θ, β = Optim.minimizer(result)
    return EGARCH(ω, α, θ, β)
end

"""
    fit(::Type{GJRGARCH}, returns::Vector{<:Real}; method=:MLE) -> GJRGARCH{Float64}
"""
function fit(::Type{GJRGARCH}, returns::Vector{<:Real}; method=:MLE)
    method == :MLE || throw(ArgumentError("Only :MLE supported"))
    n = length(returns)
    n >= 10 || throw(ArgumentError("Need at least 10 observations"))
    r = Float64.(returns .- mean(returns))
    var_r = var(r)
    ω0 = 0.1 * var_r
    α0 = 0.05
    γ0 = 0.1
    β0 = 0.85
    lb = [1e-8, 0.0, -Inf, 0.0]
    ub = [Inf, Inf, Inf, 1.0]
    x0 = [ω0, α0, γ0, β0]
    obj = x -> _neg_loglik_gjr(x, r)
    result = optimize(obj, lb, ub, x0, Fminbox(LBFGS()); inplace=false)
    ω, α, γ, β = Optim.minimizer(result)
    return GJRGARCH(ω, α, γ, β)
end

"""
    fit(::Type{AGARCH}, returns::Vector{<:Real}; method=:MLE) -> AGARCH{Float64}
"""
function fit(::Type{AGARCH}, returns::Vector{<:Real}; method=:MLE)
    method == :MLE || throw(ArgumentError("Only :MLE supported"))
    n = length(returns)
    n >= 10 || throw(ArgumentError("Need at least 10 observations"))
    r = Float64.(returns .- mean(returns))
    var_r = var(r)
    ω0 = 0.1 * var_r
    α0 = 0.08
    γ0 = 0.5
    β0 = 0.85
    lb = [1e-8, 0.0, -10.0, 0.0]
    ub = [Inf, Inf, 10.0, 1.0]
    x0 = [ω0, α0, γ0, β0]
    obj = x -> _neg_loglik_agarch(x, r)
    result = optimize(obj, lb, ub, x0, Fminbox(LBFGS()); inplace=false)
    ω, α, γ, β = Optim.minimizer(result)
    return AGARCH(ω, α, γ, β)
end
