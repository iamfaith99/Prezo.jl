"""
    Factor GARCH

Multivariate GARCH with observed factors: returns r_t = B * f_t + ε_t (or factor
decomposition). Fit GARCH(1,1) to each factor; conditional covariance
H_t = B * diag(h_f,t) * B' + D where D is diagonal residual variance (optional).
Simplified: H_t = B * diag(h_f,t) * B' (no residual, so factors explain all).

# Usage
```julia
returns = randn(300, 3)
factors = randn(300, 2)  # T x k
model = fit(FactorGARCH, returns, factors)
covs = covariance_series(model, returns, factors)
```
"""

using LinearAlgebra
using Statistics

"""
    FactorGARCH(loadings::Matrix{Float64}, garch_models::Vector{GARCH{Float64}}, resid_var::Vector{Float64})

Factor GARCH. `loadings` is n×k (B); `garch_models` are GARCH(1,1) for each factor;
`resid_var` is n-vector of residual variances (can be zeros). H_t = B*diag(h_f)*B' + diag(resid_var).
"""
struct FactorGARCH <: VolatilityModel
    loadings::Matrix{Float64}   # n x k
    garch_models::Vector{GARCH{Float64}}
    resid_var::Vector{Float64}
end

"""
    fit(::Type{FactorGARCH}, returns::Matrix{<:Real}, factors::Matrix{<:Real}) -> FactorGARCH

Estimate B by OLS (returns on factors), fit GARCH(1,1) to each factor column,
estimate residual variances from OLS residuals. factors is T×k.
"""
function fit(::Type{FactorGARCH}, returns::Matrix{<:Real}, factors::Matrix{<:Real})
    T, n = size(returns)
    T2, k = size(factors)
    T == T2 || throw(ArgumentError("returns and factors must have same number of rows"))
    T >= 30 || throw(ArgumentError("Need at least 30 observations"))
    k >= 1 || throw(ArgumentError("Need at least 1 factor"))
    # OLS: returns = factors * B' + resid  => B' = (F'F)^{-1} F' R  => B = R' F (F'F)^{-1}
    F = factors
    R = returns
    Bt = (F' * F) \ (F' * R)
    B = Bt'
    resid = R - F * Bt
    resid_var = max.(var(resid, dims=1)[:], 1e-12)
    # Fit GARCH to each factor
    models = GARCH{Float64}[]
    for j in 1:k
        push!(models, fit(GARCH, F[:, j]))
    end
    return FactorGARCH(B, models, resid_var)
end

"""
    covariance_series(model::FactorGARCH, returns::Matrix{<:Real}, factors::Matrix{<:Real}) -> Vector{Matrix{Float64}}
"""
function covariance_series(model::FactorGARCH, returns::Matrix{<:Real}, factors::Matrix{<:Real})
    T, n = size(returns)
    k = length(model.garch_models)
    B = model.loadings
    D_resid = Diagonal(model.resid_var)
    h_mat = Matrix{Float64}(undef, T, k)
    for j in 1:k
        h_mat[:, j] = volatility_process(model.garch_models[j], factors[:, j])
    end
    out = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        D_f = Diagonal(h_mat[t, :])
        H = B * D_f * B' + D_resid
        @inbounds out[t] = Matrix(Symmetric(H))
    end
    return out
end
