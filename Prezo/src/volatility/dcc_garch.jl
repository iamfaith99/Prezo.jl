"""
    DCC-GARCH (Dynamic Conditional Correlation)

Multivariate GARCH with univariate GARCH(1,1) per asset and a scalar
GARCH(1,1)-style recursion for the conditional correlation matrix.
Two-step estimation: (1) fit univariate GARCH to each series, (2) estimate
DCC parameters (a, b) on standardized residuals.

# Model
- Per-asset variances: h_{i,t} from GARCH(1,1)
- Standardized residuals: z_{i,t} = r_{i,t} / √h_{i,t}
- Q_t = (1-a-b)*Q̄ + a*z_{t-1}*z_{t-1}' + b*Q_{t-1}; R_t = corr(Q_t)
- Conditional covariance: H_t = D_t R_t D_t, D_t = diag(√h_{1,t}, ..., √h_{n,t})

# Usage
```julia
returns = randn(500, 3) .* 0.01  # T x n
model = fit(DCCGARCH, returns)
covs = covariance_series(model, returns)
f = forecast(model, returns, 5)
```
"""

using LinearAlgebra
using Optim
using Statistics

"""
    DCCGARCH(univariate_models::Vector{GARCH{Float64}}, a::Float64, b::Float64, Q_bar::Matrix{Float64})

DCC-GARCH model. `univariate_models[j]` is GARCH(1,1) for asset j; `a`, `b` are DCC parameters;
`Q_bar` is the unconditional correlation of standardized residuals (n x n).
"""
struct DCCGARCH <: VolatilityModel
    univariate_models::Vector{GARCH{Float64}}
    a::Float64
    b::Float64
    Q_bar::Matrix{Float64}

    function DCCGARCH(
        univariate_models::Vector{GARCH{Float64}},
        a::Float64,
        b::Float64,
        Q_bar::Matrix{Float64}
    )
        n = length(univariate_models)
        size(Q_bar) == (n, n) || throw(ArgumentError("Q_bar must be n×n"))
        (a >= 0 && b >= 0 && a + b < 1) ||
            throw(ArgumentError("DCC: a,b >= 0 and a+b < 1 for stationarity"))
        new(univariate_models, a, b, Q_bar)
    end
end

function _standardized_residuals(returns::Matrix{<:Real}, garch_models::Vector{GARCH{Float64}})
    T, n = size(returns)
    z = Matrix{Float64}(undef, T, n)
    for j in 1:n
        h = volatility_process(garch_models[j], returns[:, j])
        for t in 1:T
            z[t, j] = returns[t, j] / sqrt(max(h[t], 1e-12))
        end
    end
    return z
end

"""Q_t from DCC recursion; then normalize to correlation R_t."""
function _q_to_correlation(Q::Matrix{Float64})
    n = size(Q, 1)
    d = sqrt.(max.(diag(Q), 1e-12))
    R = Q ./ (d .* d')
    for i in 1:n
        R[i, i] = 1.0
    end
    return R
end

"""
    covariance_series(model::DCCGARCH, returns::Matrix{<:Real}) -> Vector{Matrix{Float64}}

Sequence of conditional covariance matrices H_t (one per time step).
`returns` is T×n (time × assets).
"""
function covariance_series(model::DCCGARCH, returns::Matrix{<:Real})
    T, n = size(returns)
    T >= 1 || return Matrix{Float64}[]
    # Per-asset variances
    h_mat = Matrix{Float64}(undef, T, n)
    for j in 1:n
        h_mat[:, j] = volatility_process(model.univariate_models[j], returns[:, j])
    end
    z = _standardized_residuals(returns, model.univariate_models)
    # Q_bar as initial Q; then recursion
    Q = copy(model.Q_bar)
    out = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        # Q_t = (1-a-b)*Q_bar + a*z_{t-1}z_{t-1}' + b*Q_{t-1}
        if t == 1
            Q = copy(model.Q_bar)
        else
            z_prev = z[t-1, :]
            Q = (1 - model.a - model.b) * model.Q_bar +
                model.a * (z_prev * z_prev') + model.b * Q
        end
        R = _q_to_correlation(Q)
        D = Diagonal(sqrt.(max.(h_mat[t, :], 1e-12)))
        H = D * R * D
        @inbounds out[t] = Matrix(H)
    end
    return out
end

"""
    forecast(model::DCCGARCH, returns::Matrix{<:Real}, horizon::Int) -> Vector{Matrix{Float64}}

Forecast conditional covariance matrices for the next `horizon` steps.
Uses last observed Q and last per-asset h; E[z]=0 for future.
"""
function forecast(model::DCCGARCH, returns::Matrix{<:Real}, horizon::Int)
    horizon >= 1 || return Matrix{Float64}[]
    T, n = size(returns)
    h_forecasts = Vector{Vector{Float64}}(undef, n)
    for j in 1:n
        h_forecasts[j] = forecast(model.univariate_models[j], returns[:, j], horizon)
    end
    z = _standardized_residuals(returns, model.univariate_models)
    Q = copy(model.Q_bar)
    if T >= 1
        for t in 1:T
            if t > 1
                z_prev = z[t-1, :]
                Q = (1 - model.a - model.b) * model.Q_bar +
                    model.a * (z_prev * z_prev') + model.b * Q
            end
        end
    end
    out = Vector{Matrix{Float64}}(undef, horizon)
    for i in 1:horizon
        # Future: E[z]=0 => Q = (1-a-b)*Q_bar + b*Q
        Q = (1 - model.a - model.b) * model.Q_bar + model.b * Q
        R = _q_to_correlation(Q)
        d = Float64[sqrt(max(h_forecasts[j][i], 1e-12)) for j in 1:n]
        D = Diagonal(d)
        out[i] = Matrix(D * R * D)
    end
    return out
end

# Negative log-likelihood for DCC part (correlation): sum_t (log|R_t| + z_t' R_t^{-1} z_t)
function _dcc_neg_loglik(params::Vector{Float64}, z::Matrix{Float64}, Q_bar::Matrix{Float64})
    a, b = params[1], params[2]
    if a < 0 || b < 0 || a + b >= 1
        return Inf
    end
    T, n = size(z)
    Q = copy(Q_bar)
    nll = 0.0
    for t in 1:T
        if t > 1
            z_prev = z[t-1, :]
            Q = (1 - a - b) * Q_bar + a * (z_prev * z_prev') + b * Q
        end
        R = _q_to_correlation(Q)
        R = (R + R') / 2
        try
            Rinv = inv(R)
            nll += log(det(R)) + dot(z[t, :], Rinv, z[t, :])
        catch
            return Inf
        end
    end
    return 0.5 * nll
end

"""
    fit(::Type{DCCGARCH}, returns::Matrix{<:Real}; method=:MLE) -> DCCGARCH

Two-step estimation. Step 1: fit GARCH(1,1) to each column. Step 2: estimate (a, b) on
standardized residuals; Q_bar is sample correlation of z.
"""
function fit(::Type{DCCGARCH}, returns::Matrix{<:Real}; method=:MLE)
    method == :MLE || throw(ArgumentError("Only :MLE supported"))
    T, n = size(returns)
    T >= 30 || throw(ArgumentError("Need at least 30 observations"))
    n >= 2 || throw(ArgumentError("Need at least 2 assets"))
    # Demean per asset
    r = returns .- mean(returns, dims=1)
    # Step 1: univariate GARCH per column
    models = GARCH{Float64}[]
    for j in 1:n
        push!(models, fit(GARCH, r[:, j]))
    end
    # Standardized residuals
    z = _standardized_residuals(r, models)
    # Q_bar = sample correlation of z (unconditional)
    Q_bar = cov(z)
    scale = sqrt.(diag(Q_bar))
    for i in 1:n, j in 1:n
        Q_bar[i, j] /= max(scale[i] * scale[j], 1e-12)
    end
    for i in 1:n
        Q_bar[i, i] = 1.0
    end
    Q_bar = (Q_bar + Q_bar') / 2
    # Step 2: estimate a, b
    a0, b0 = 0.05, 0.90
    lb = [1e-8, 1e-8]
    ub = [1.0, 1.0]
    obj = p -> _dcc_neg_loglik(p, z, Q_bar)
    result = optimize(obj, lb, ub, [a0, b0], Fminbox(LBFGS()); inplace=false)
    a, b = Optim.minimizer(result)
    return DCCGARCH(models, a, b, Q_bar)
end
