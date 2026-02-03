"""
    Regime-switching GARCH (two regimes)

Two-regime Markov-switching GARCH(1,1). Separate file from univariate GARCH family
per design: different structure (regimes, transitions, per-regime GARCH).
Depends on GARCH and VolatilityModel from garch.jl.
"""

using Random

# GARCH(1,1) helpers (avoid coupling to garch.jl internals)
_unconditional_variance_garch(g::GARCH) = g.ω / (1 - g.α - g.β)
_variance_update_garch(g::GARCH, h_prev::T, r_prev::T) where {T<:Real} =
    g.ω + g.α * r_prev^2 + g.β * h_prev

"""
    RegimeGARCH(regime1::GARCH, regime2::GARCH, p11::Real, p22::Real)

Two-regime Markov-switching GARCH(1,1). p11 = P(s_t=1 | s_{t-1}=1), p22 = P(s_t=2 | s_{t-1}=2).
"""
struct RegimeGARCH{T<:Real} <: VolatilityModel
    regime1::GARCH{T}
    regime2::GARCH{T}
    p11::T
    p22::T

    function RegimeGARCH(regime1::GARCH{T}, regime2::GARCH{T}, p11::T, p22::T) where {T<:Real}
        0 < p11 < 1 || throw(ArgumentError("p11 must be in (0,1)"))
        0 < p22 < 1 || throw(ArgumentError("p22 must be in (0,1)"))
        new{T}(regime1, regime2, p11, p22)
    end
end

"""
    volatility_process(model::RegimeGARCH, returns::Vector{<:Real}) -> Tuple{Vector{Float64}, Vector{Float64}}

Returns (h, regime_probs) where h is conditional variance and regime_probs[t] is P(s_t=1).
Uses Hamilton-style filter with regime-specific variances.
"""
function volatility_process(model::RegimeGARCH, returns::Vector{<:Real})
    n = length(returns)
    n >= 1 || return (Float64[], Float64[])
    T = float(eltype(returns))
    h = Vector{T}(undef, n)
    pr1 = Vector{T}(undef, n)
    h1_cur = _unconditional_variance_garch(model.regime1)
    h2_cur = _unconditional_variance_garch(model.regime2)
    π2_ss = (1 - model.p11) / (2 - model.p11 - model.p22)
    π1_ss = 1 - π2_ss
    h[1] = π1_ss * h1_cur + π2_ss * h2_cur
    pr1[1] = π1_ss
    for t in 2:n
        r_prev = returns[t-1]
        σ1 = sqrt(max(h1_cur, 1e-12))
        σ2 = sqrt(max(h2_cur, 1e-12))
        L1 = exp(-0.5 * (r_prev / σ1)^2) / σ1
        L2 = exp(-0.5 * (r_prev / σ2)^2) / σ2
        p1 = pr1[t-1]
        p2 = 1 - p1
        η1 = model.p11 * L1 * p1 + (1 - model.p22) * L2 * p2
        η2 = (1 - model.p11) * L1 * p1 + model.p22 * L2 * p2
        norm = η1 + η2
        pr1[t] = η1 / norm
        h1_cur = _variance_update_garch(model.regime1, h1_cur, r_prev)
        h2_cur = _variance_update_garch(model.regime2, h2_cur, r_prev)
        h[t] = pr1[t] * h1_cur + (1 - pr1[t]) * h2_cur
    end
    return h, pr1
end

"""
    forecast(model::RegimeGARCH, returns::Vector{<:Real}, horizon::Int) -> Vector{Float64}

Forecast conditional variance using last filtered regime probability; each step blends
the two regime GARCH updates with E[r]=0.
"""
function forecast(model::RegimeGARCH, returns::Vector{<:Real}, horizon::Int)
    horizon >= 1 || return Float64[]
    h, pr1 = volatility_process(model, returns)
    n = length(returns)
    T = float(eltype(returns))
    out = Vector{T}(undef, horizon)
    h_cur = n >= 1 ? h[n] : 0.5 * (_unconditional_variance_garch(model.regime1) + _unconditional_variance_garch(model.regime2))
    p1 = n >= 1 ? pr1[n] : 0.5
    for i in 1:horizon
        h1 = _variance_update_garch(model.regime1, h_cur, zero(T))
        h2 = _variance_update_garch(model.regime2, h_cur, zero(T))
        h_cur = p1 * h1 + (1 - p1) * h2
        @inbounds out[i] = h_cur
    end
    return out
end

"""
    simulate(model::RegimeGARCH, n::Int; seed=nothing) -> Tuple{Vector{Float64}, Vector{Float64}, Vector{Int}}

Returns (returns, h, regimes) where regimes[t] ∈ {1, 2}.
"""
function simulate(model::RegimeGARCH, n::Int; seed=nothing)
    seed === nothing || Random.seed!(seed)
    T = Float64
    h = Vector{T}(undef, n)
    r = Vector{T}(undef, n)
    s = Vector{Int}(undef, n)
    π2_ss = (1 - model.p11) / (2 - model.p11 - model.p22)
    s[1] = rand() < (1 - π2_ss) ? 1 : 2
    g = s[1] == 1 ? model.regime1 : model.regime2
    h[1] = _unconditional_variance_garch(g)
    r[1] = randn() * sqrt(h[1])
    for t in 2:n
        s[t] = (s[t-1] == 1) ? (rand() < model.p11 ? 1 : 2) : (rand() < model.p22 ? 2 : 1)
        g = s[t] == 1 ? model.regime1 : model.regime2
        @inbounds h[t] = _variance_update_garch(g, h[t-1], r[t-1])
        @inbounds r[t] = randn() * sqrt(max(h[t], 1e-12))
    end
    return r, h, s
end
