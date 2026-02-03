"""
    Heston stochastic volatility model

dS_t = (r - q) S_t dt + √v_t S_t dW₁
dv_t = κ(θ - v_t) dt + σ √v_t dW₂,  E[dW₁ dW₂] = ρ dt

# Usage
```julia
m = HestonModel(2.0, 0.04, 0.3, -0.5, 0.04)
spot, vol, returns = simulate_heston(m, 500, 1.0; S0=100.0, r=0.05, q=0.0, seed=42)
```
"""

using Random
using Statistics

"""
    HestonModel(κ, θ, σ, ρ, v₀)

Heston model parameters.
- κ: mean reversion speed (variance)
- θ: long-term variance
- σ: vol-of-vol
- ρ: correlation between spot and variance shocks, ∈ [-1, 1]
- v₀: initial variance

Feller condition 2κθ > σ² ensures variance stays positive (not enforced; user responsibility).
"""
struct HestonModel{T<:Real} <: VolatilityModel
    κ::T
    θ::T
    σ::T
    ρ::T
    v₀::T

    function HestonModel(κ::T, θ::T, σ::T, ρ::T, v₀::T) where {T<:Real}
        θ > 0 || throw(ArgumentError("θ must be positive"))
        σ > 0 || throw(ArgumentError("σ must be positive"))
        v₀ > 0 || throw(ArgumentError("v₀ must be positive"))
        -1 ≤ ρ ≤ 1 || throw(ArgumentError("ρ must be in [-1, 1]"))
        κ > 0 || throw(ArgumentError("κ must be positive"))
        new{T}(κ, θ, σ, ρ, v₀)
    end
end

"""
    simulate_heston(
        model::HestonModel,
        n_steps::Int,
        T::Real;
        S0::Real=1.0,
        r::Real=0.0,
        q::Real=0.0,
        seed=nothing,
    ) -> (spot::Vector{Float64}, variance::Vector{Float64}, returns::Vector{Float64})

Euler-Maruyama simulation. Returns spot path (length n_steps+1), variance path (n_steps+1),
and log-returns (length n_steps). Reflection at v=0 for variance to avoid negative values.
"""
function simulate_heston(
    model::HestonModel,
    n_steps::Int,
    T::Real;
    S0::Real=1.0,
    r::Real=0.0,
    q::Real=0.0,
    seed=nothing,
)
    n_steps >= 1 || throw(ArgumentError("n_steps must be ≥ 1"))
    T > 0 || throw(ArgumentError("T must be positive"))
    rng = seed === nothing ? Random.GLOBAL_RNG : MersenneTwister(seed)
    dt = T / n_steps
    sqrt_dt = sqrt(dt)
    (; κ, θ, σ, ρ, v₀) = model
    # Correlated Brownian: W2 = ρ*W1 + sqrt(1-ρ²)*W⊥
    sqrt_1mrho2 = sqrt(max(1 - ρ^2, 0.0))

    spot = Vector{Float64}(undef, n_steps + 1)
    variance = Vector{Float64}(undef, n_steps + 1)
    spot[1] = Float64(S0)
    variance[1] = Float64(v₀)

    for t in 1:n_steps
        dW1 = randn(rng) * sqrt_dt
        dW2 = (ρ * dW1 + sqrt_1mrho2 * randn(rng) * sqrt_dt)
        v = max(variance[t], 1e-14)
        sqrt_v = sqrt(v)
        # Euler-Maruyama with reflection: v_new = max(v + dv, 0)
        dv = κ * (θ - v) * dt + σ * sqrt_v * dW2
        variance[t + 1] = max(v + dv, 1e-14)
        dS = (r - q) * spot[t] * dt + spot[t] * sqrt_v * dW1
        spot[t + 1] = max(spot[t] + dS, 1e-14)
    end

    returns = Vector{Float64}(undef, n_steps)
    for t in 1:n_steps
        returns[t] = log(spot[t + 1] / spot[t])
    end
    return spot, variance, returns
end
