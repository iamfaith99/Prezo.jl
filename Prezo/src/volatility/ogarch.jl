"""
    O-GARCH (Orthogonal / PCA-based GARCH)

Multivariate GARCH via PCA: decompose returns into orthogonal factors (PCs),
fit univariate GARCH(1,1) to each factor, reconstruct conditional covariance
as H_t = V * D_t * V' where D_t = diag(h_1,t, ..., h_k,t).

# Usage
```julia
returns = randn(300, 4) .* 0.01  # T x n
model = fit(OGARCH, returns; n_factors=4)
covs = covariance_series(model, returns)
```
"""

using LinearAlgebra
using Statistics

"""
    OGARCH(loadings::Matrix{Float64}, garch_models::Vector{GARCH{Float64}})

O-GARCH model. `loadings` is the n×k matrix V (eigenvectors); `garch_models[j]` is
GARCH(1,1) for the j-th principal component. Conditional cov = V * diag(h_t) * V'.
"""
struct OGARCH <: VolatilityModel
    loadings::Matrix{Float64}   # n x k (assets x factors)
    garch_models::Vector{GARCH{Float64}}
end

"""
    fit(::Type{OGARCH}, returns::Matrix{<:Real}; n_factors::Int=size(returns,2)) -> OGARCH

Fit O-GARCH: PCA on returns, then GARCH(1,1) on each of the first `n_factors` PCs.
"""
function fit(::Type{OGARCH}, returns::Matrix{<:Real}; n_factors::Int=size(returns, 2))
    T, n = size(returns)
    T >= 30 || throw(ArgumentError("Need at least 30 observations"))
    n >= 2 || throw(ArgumentError("Need at least 2 assets"))
    k = min(n_factors, n)
    r = returns .- mean(returns, dims=1)
    Σ = cov(r)
    F = eigen(Symmetric(Σ))
    # Order by eigenvalue descending
    perm = sortperm(F.values; rev=true)
    λ = F.values[perm]
    V = F.vectors[:, perm]   # n x n
    Vk = V[:, 1:k]           # n x k
    # Factor returns: r_t * Vk gives k-vector at time t
    factors = r * Vk         # T x k
    models = GARCH{Float64}[]
    for j in 1:k
        push!(models, fit(GARCH, factors[:, j]))
    end
    return OGARCH(Vk, models)
end

"""
    covariance_series(model::OGARCH, returns::Matrix{<:Real}) -> Vector{Matrix{Float64}}

Conditional covariance sequence. Uses loadings and GARCH variance per factor;
returns are only used to get length T (same as size(returns,1)).
"""
function covariance_series(model::OGARCH, returns::Matrix{<:Real})
    T, n = size(returns)
    T >= 1 || return Matrix{Float64}[]
    k = length(model.garch_models)
    V = model.loadings
    # Factor returns for this sample (needed for GARCH recursion)
    r = returns .- mean(returns, dims=1)
    factors = r * V
    h_mat = Matrix{Float64}(undef, T, k)
    for j in 1:k
        h_mat[:, j] = volatility_process(model.garch_models[j], factors[:, j])
    end
    out = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        D = Diagonal(h_mat[t, :])
        H = V * D * V'
        @inbounds out[t] = Matrix(Symmetric(H))
    end
    return out
end
