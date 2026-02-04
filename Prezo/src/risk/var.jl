"""
    Value at Risk (VaR) and Conditional VaR (CVaR / Expected Shortfall)

Three VaR methods:
- Historical: empirical quantile of returns
- Parametric: assumes normal distribution
- Monte Carlo: simulation-based

CVaR (Expected Shortfall) is the expected loss given that loss exceeds VaR.
"""

using Statistics
using Distributions

# ============================================================================
# VaR Methods
# ============================================================================

"""
    VaRMethod

Abstract type for VaR calculation methods.
"""
abstract type VaRMethod end

"""
    HistoricalVaR()

Historical simulation VaR: uses empirical quantile of returns.
"""
struct HistoricalVaR <: VaRMethod end

"""
    ParametricVaR()

Parametric VaR: assumes returns are normally distributed.
"""
struct ParametricVaR <: VaRMethod end

"""
    MonteCarloVaR(n_simulations)

Monte Carlo VaR: simulates future returns and computes quantile.

# Fields
- `n_simulations::Int`: Number of Monte Carlo simulations
"""
struct MonteCarloVaR <: VaRMethod
    n_simulations::Int
end
MonteCarloVaR() = MonteCarloVaR(10_000)

# ============================================================================
# VaR Calculation
# ============================================================================

"""
    value_at_risk(returns::Vector{<:Real}, confidence::Real; method=HistoricalVaR()) -> Float64

Calculate Value at Risk (VaR) for a portfolio.

VaR is the maximum expected loss at a given confidence level over a time horizon.
A 95% VaR of 0.05 means there's a 5% chance of losing more than 5% of portfolio value.

# Arguments
- `returns`: Vector of historical returns (e.g., daily log returns)
- `confidence`: Confidence level (e.g., 0.95 for 95% VaR)
- `method`: VaR calculation method (default: HistoricalVaR())

# Returns
VaR as a positive number (loss magnitude).

# Examples
```julia
returns = randn(252) * 0.02  # ~2% daily vol
var_95 = value_at_risk(returns, 0.95)
var_99 = value_at_risk(returns, 0.99; method=ParametricVaR())
```
"""
function value_at_risk(returns::Vector{<:Real}, confidence::Real; method::VaRMethod=HistoricalVaR())
    0.0 < confidence < 1.0 || error("Confidence must be in (0, 1)")
    return _var(returns, confidence, method)
end

function _var(returns::Vector{<:Real}, confidence::Real, ::HistoricalVaR)
    α = 1.0 - confidence
    return -quantile(returns, α)
end

function _var(returns::Vector{<:Real}, confidence::Real, ::ParametricVaR)
    μ = mean(returns)
    σ = std(returns)
    α = 1.0 - confidence
    z = quantile(Normal(), α)
    return -(μ + σ * z)
end

function _var(returns::Vector{<:Real}, confidence::Real, method::MonteCarloVaR)
    μ = mean(returns)
    σ = std(returns)
    simulated = μ .+ σ .* randn(method.n_simulations)
    α = 1.0 - confidence
    return -quantile(simulated, α)
end

"""
    value_at_risk(returns::Vector{<:Real}, confidence::Real, horizon::Int; 
                  method=HistoricalVaR()) -> Float64

Calculate VaR scaled to a multi-period horizon using square-root-of-time rule.

# Arguments
- `returns`: Vector of single-period returns
- `confidence`: Confidence level
- `horizon`: Number of periods to scale to
- `method`: VaR calculation method

# Examples
```julia
daily_returns = randn(252) * 0.02
var_10day = value_at_risk(daily_returns, 0.95, 10)  # 10-day VaR
```
"""
function value_at_risk(returns::Vector{<:Real}, confidence::Real, horizon::Int; 
                       method::VaRMethod=HistoricalVaR())
    horizon > 0 || error("Horizon must be positive")
    var_1 = value_at_risk(returns, confidence; method=method)
    return var_1 * sqrt(horizon)
end

# ============================================================================
# CVaR (Expected Shortfall)
# ============================================================================

"""
    conditional_var(returns::Vector{<:Real}, confidence::Real; method=HistoricalVaR()) -> Float64

Calculate Conditional VaR (CVaR), also known as Expected Shortfall (ES).

CVaR is the expected loss given that the loss exceeds VaR. It's a coherent
risk measure (unlike VaR) and captures tail risk better.

# Arguments
- `returns`: Vector of historical returns
- `confidence`: Confidence level (e.g., 0.95)
- `method`: Calculation method

# Returns
CVaR as a positive number (expected loss in tail).

# Examples
```julia
returns = randn(252) * 0.02
cvar_95 = conditional_var(returns, 0.95)
# CVaR >= VaR always
@assert cvar_95 >= value_at_risk(returns, 0.95)
```
"""
function conditional_var(returns::Vector{<:Real}, confidence::Real; method::VaRMethod=HistoricalVaR())
    0.0 < confidence < 1.0 || error("Confidence must be in (0, 1)")
    return _cvar(returns, confidence, method)
end

function _cvar(returns::Vector{<:Real}, confidence::Real, ::HistoricalVaR)
    α = 1.0 - confidence
    threshold = quantile(returns, α)
    tail_returns = filter(r -> r <= threshold, returns)
    isempty(tail_returns) && return -threshold
    return -mean(tail_returns)
end

function _cvar(returns::Vector{<:Real}, confidence::Real, ::ParametricVaR)
    μ = mean(returns)
    σ = std(returns)
    α = 1.0 - confidence
    z_α = quantile(Normal(), α)
    # ES for normal: μ - σ * φ(z_α) / α
    es = μ - σ * pdf(Normal(), z_α) / α
    return -es
end

function _cvar(returns::Vector{<:Real}, confidence::Real, method::MonteCarloVaR)
    μ = mean(returns)
    σ = std(returns)
    simulated = μ .+ σ .* randn(method.n_simulations)
    α = 1.0 - confidence
    threshold = quantile(simulated, α)
    tail = filter(r -> r <= threshold, simulated)
    isempty(tail) && return -threshold
    return -mean(tail)
end

# ============================================================================
# Portfolio VaR
# ============================================================================

"""
    PortfolioVaR

Result of portfolio VaR calculation.

# Fields
- `var::Float64`: Portfolio VaR
- `cvar::Float64`: Portfolio CVaR (Expected Shortfall)
- `confidence::Float64`: Confidence level used
- `method::VaRMethod`: Method used for calculation
- `component_var::Vector{Float64}`: Component VaR for each asset (optional)
"""
struct PortfolioVaR
    var::Float64
    cvar::Float64
    confidence::Float64
    method::VaRMethod
    component_var::Vector{Float64}
end

"""
    portfolio_var(returns::Matrix{<:Real}, weights::Vector{<:Real}, confidence::Real;
                  method=HistoricalVaR()) -> PortfolioVaR

Calculate VaR and CVaR for a portfolio of assets.

# Arguments
- `returns`: Matrix of asset returns (T × N, rows = time, cols = assets)
- `weights`: Portfolio weights (should sum to 1)
- `confidence`: Confidence level
- `method`: VaR calculation method

# Returns
`PortfolioVaR` with portfolio VaR, CVaR, and component VaR.

# Examples
```julia
# 252 days, 5 assets
returns = randn(252, 5) * 0.02
weights = [0.2, 0.2, 0.2, 0.2, 0.2]
result = portfolio_var(returns, weights, 0.95)
println("Portfolio VaR: ", result.var)
println("Portfolio CVaR: ", result.cvar)
```
"""
function portfolio_var(returns::Matrix{<:Real}, weights::Vector{<:Real}, confidence::Real;
                       method::VaRMethod=HistoricalVaR())
    size(returns, 2) == length(weights) || error("Weights must match number of assets")
    abs(sum(weights) - 1.0) < 0.01 || @warn "Weights do not sum to 1"
    
    # Portfolio returns
    portfolio_returns = returns * weights
    
    # Portfolio VaR and CVaR
    var = value_at_risk(portfolio_returns, confidence; method=method)
    cvar = conditional_var(portfolio_returns, confidence; method=method)
    
    # Component VaR (marginal contribution)
    n_assets = length(weights)
    component = zeros(n_assets)
    ε = 1e-4
    for i in 1:n_assets
        w_up = copy(weights)
        w_up[i] += ε
        w_up ./= sum(w_up)
        var_up = value_at_risk(returns * w_up, confidence; method=method)
        component[i] = (var_up - var) / ε * weights[i]
    end
    
    return PortfolioVaR(var, cvar, confidence, method, component)
end
