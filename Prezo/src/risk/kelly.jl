"""
    Kelly Criterion for Position Sizing

Implements the Kelly criterion and its variants for optimal position sizing:
- Classic Kelly (binary outcomes)
- Continuous Kelly (Gaussian returns)
- Fractional Kelly (reduced risk)
- Multi-asset Kelly (portfolio optimization)
"""

using Statistics
using LinearAlgebra

# ============================================================================
# Classic Kelly (Binary Outcomes)
# ============================================================================

"""
    kelly_fraction(win_prob::Real, win_loss_ratio::Real) -> Float64

Calculate the Kelly fraction for binary outcomes (win/lose).

The Kelly criterion maximizes the expected logarithm of wealth, giving
the optimal fraction of capital to bet.

Formula: f* = p - q/b = p - (1-p)/b

# Arguments
- `win_prob`: Probability of winning (0 < p < 1)
- `win_loss_ratio`: Ratio of win amount to loss amount (b > 0)

# Returns
Optimal fraction of capital to bet. Can be negative (don't bet) or > 1 (use leverage).

# Examples
```julia
# Coin flip with 2:1 payout
f = kelly_fraction(0.5, 2.0)  # = 0.25

# Edge in sports betting
f = kelly_fraction(0.55, 1.0)  # = 0.10
```

See also: [`fractional_kelly`](@ref), [`kelly_continuous`](@ref)
"""
function kelly_fraction(win_prob::Real, win_loss_ratio::Real)
    0.0 < win_prob < 1.0 || error("win_prob must be in (0, 1)")
    win_loss_ratio > 0 || error("win_loss_ratio must be positive")
    
    p = win_prob
    q = 1.0 - p
    b = win_loss_ratio
    
    return p - q / b
end

"""
    fractional_kelly(win_prob::Real, win_loss_ratio::Real, fraction::Real=0.5) -> Float64

Calculate a fractional Kelly bet for reduced volatility.

Full Kelly is aggressive and can lead to large drawdowns. Fractional Kelly
(e.g., half-Kelly) reduces risk while still growing wealth.

# Arguments
- `win_prob`: Probability of winning
- `win_loss_ratio`: Ratio of win to loss
- `fraction`: Kelly fraction to use (default 0.5 = half-Kelly)

# Returns
Position size as fraction of capital.

# Examples
```julia
full_kelly = kelly_fraction(0.6, 1.5)
half_kelly = fractional_kelly(0.6, 1.5, 0.5)  # More conservative
```
"""
function fractional_kelly(win_prob::Real, win_loss_ratio::Real, fraction::Real=0.5)
    0.0 < fraction <= 1.0 || error("fraction must be in (0, 1]")
    return fraction * kelly_fraction(win_prob, win_loss_ratio)
end

# ============================================================================
# Continuous Kelly (Gaussian Returns)
# ============================================================================

"""
    kelly_continuous(expected_return::Real, variance::Real, risk_free_rate::Real=0.0) -> Float64

Calculate Kelly fraction for continuous returns (Gaussian model).

For assets with normally distributed returns, the Kelly criterion gives:
f* = (μ - r) / σ²

where μ is expected return, r is risk-free rate, σ² is variance.

# Arguments
- `expected_return`: Expected return of the asset (μ)
- `variance`: Variance of returns (σ²)
- `risk_free_rate`: Risk-free rate (default 0)

# Returns
Optimal leverage/position size.

# Examples
```julia
# Stock with 10% expected return, 20% volatility
f = kelly_continuous(0.10, 0.04, 0.02)  # (0.10 - 0.02) / 0.04 = 2.0
# Kelly says 2x leverage, but fractional Kelly is safer
```

See also: [`kelly_fraction`](@ref), [`kelly_portfolio`](@ref)
"""
function kelly_continuous(expected_return::Real, variance::Real, risk_free_rate::Real=0.0)
    variance > 0 || error("variance must be positive")
    return (expected_return - risk_free_rate) / variance
end

"""
    kelly_from_sharpe(sharpe_ratio::Real, volatility::Real) -> Float64

Calculate Kelly leverage from Sharpe ratio and volatility.

# Arguments
- `sharpe_ratio`: Annualized Sharpe ratio
- `volatility`: Annualized volatility (standard deviation)

# Returns
Optimal leverage.

# Examples
```julia
# Strategy with Sharpe 1.5, 15% vol
leverage = kelly_from_sharpe(1.5, 0.15)  # = 10x leverage
```
"""
function kelly_from_sharpe(sharpe_ratio::Real, volatility::Real)
    volatility > 0 || error("volatility must be positive")
    return sharpe_ratio / volatility
end

# ============================================================================
# Multi-Asset Kelly (Portfolio)
# ============================================================================

"""
    KellyPortfolio

Result of multi-asset Kelly optimization.

# Fields
- `weights::Vector{Float64}`: Optimal portfolio weights
- `expected_return::Float64`: Portfolio expected return
- `volatility::Float64`: Portfolio volatility
- `sharpe_ratio::Float64`: Portfolio Sharpe ratio
- `kelly_leverage::Float64`: Total Kelly leverage (sum of absolute weights)
"""
struct KellyPortfolio
    weights::Vector{Float64}
    expected_return::Float64
    volatility::Float64
    sharpe_ratio::Float64
    kelly_leverage::Float64
end

"""
    kelly_portfolio(expected_returns::Vector{<:Real}, covariance::Matrix{<:Real};
                    risk_free_rate::Real=0.0, max_leverage::Real=Inf) -> KellyPortfolio

Calculate optimal Kelly portfolio weights for multiple assets.

For N assets with excess returns μ and covariance Σ, Kelly weights are:
f* = Σ⁻¹ μ

# Arguments
- `expected_returns`: Vector of expected returns for each asset
- `covariance`: Covariance matrix of returns
- `risk_free_rate`: Risk-free rate (default 0)
- `max_leverage`: Maximum total leverage (optional constraint)

# Returns
`KellyPortfolio` with optimal weights and metrics.

# Examples
```julia
# Two assets
μ = [0.10, 0.15]  # Expected returns
Σ = [0.04 0.01; 0.01 0.09]  # Covariance
result = kelly_portfolio(μ, Σ; risk_free_rate=0.02)
println("Weights: ", result.weights)
println("Sharpe: ", result.sharpe_ratio)
```
"""
function kelly_portfolio(expected_returns::Vector{<:Real}, covariance::Matrix{<:Real};
                         risk_free_rate::Real=0.0, max_leverage::Real=Inf)
    n = length(expected_returns)
    size(covariance) == (n, n) || error("Covariance matrix size mismatch")
    
    # Excess returns
    μ = expected_returns .- risk_free_rate
    
    # Kelly weights: Σ⁻¹ μ
    Σ_inv = inv(covariance + 1e-8 * I)
    weights = Σ_inv * μ
    
    # Apply leverage constraint
    total_leverage = sum(abs.(weights))
    if total_leverage > max_leverage
        weights .*= max_leverage / total_leverage
        total_leverage = max_leverage
    end
    
    # Portfolio metrics
    port_return = dot(weights, expected_returns)
    port_var = dot(weights, covariance * weights)
    port_vol = sqrt(max(port_var, 0.0))
    sharpe = port_vol > 0 ? (port_return - risk_free_rate) / port_vol : 0.0
    
    return KellyPortfolio(weights, port_return, port_vol, sharpe, total_leverage)
end

"""
    fractional_kelly_portfolio(expected_returns::Vector{<:Real}, covariance::Matrix{<:Real};
                               risk_free_rate::Real=0.0, fraction::Real=0.5) -> KellyPortfolio

Calculate fractional Kelly portfolio (e.g., half-Kelly).

# Arguments
- `expected_returns`: Expected returns vector
- `covariance`: Covariance matrix
- `risk_free_rate`: Risk-free rate
- `fraction`: Kelly fraction (default 0.5)

# Examples
```julia
result = fractional_kelly_portfolio(μ, Σ; fraction=0.25)  # Quarter-Kelly
```
"""
function fractional_kelly_portfolio(expected_returns::Vector{<:Real}, covariance::Matrix{<:Real};
                                    risk_free_rate::Real=0.0, fraction::Real=0.5)
    0.0 < fraction <= 1.0 || error("fraction must be in (0, 1]")
    
    full_kelly = kelly_portfolio(expected_returns, covariance; risk_free_rate=risk_free_rate)
    scaled_weights = full_kelly.weights .* fraction
    
    port_return = dot(scaled_weights, expected_returns)
    port_var = dot(scaled_weights, covariance * scaled_weights)
    port_vol = sqrt(max(port_var, 0.0))
    sharpe = port_vol > 0 ? (port_return - risk_free_rate) / port_vol : 0.0
    leverage = sum(abs.(scaled_weights))
    
    return KellyPortfolio(scaled_weights, port_return, port_vol, sharpe, leverage)
end

# ============================================================================
# Kelly Metrics and Utilities
# ============================================================================

"""
    kelly_growth_rate(fraction::Real, win_prob::Real, win_loss_ratio::Real) -> Float64

Calculate expected log growth rate for a given betting fraction.

G(f) = p * log(1 + bf) + q * log(1 - f)

# Arguments
- `fraction`: Betting fraction (f)
- `win_prob`: Probability of winning (p)
- `win_loss_ratio`: Win/loss ratio (b)

# Returns
Expected log growth rate.
"""
function kelly_growth_rate(fraction::Real, win_prob::Real, win_loss_ratio::Real)
    0.0 <= fraction <= 1.0 || return -Inf
    p = win_prob
    q = 1.0 - p
    b = win_loss_ratio
    
    win_term = p * log(1.0 + b * fraction)
    lose_term = q * log(max(1.0 - fraction, 1e-10))
    
    return win_term + lose_term
end

"""
    kelly_drawdown_probability(kelly_fraction::Real, drawdown::Real, n_bets::Int) -> Float64

Estimate probability of experiencing a drawdown of given magnitude.

Uses the approximation for drawdown probability in Kelly betting.

# Arguments
- `kelly_fraction`: Betting fraction used
- `drawdown`: Drawdown level (e.g., 0.5 for 50% drawdown)
- `n_bets`: Number of bets

# Returns
Approximate probability of experiencing the drawdown.
"""
function kelly_drawdown_probability(kelly_fraction::Real, drawdown::Real, n_bets::Int)
    0.0 < drawdown < 1.0 || error("drawdown must be in (0, 1)")
    kelly_fraction > 0 || return 0.0
    
    # Approximation: P(drawdown > d) ≈ d^(2/f - 1) for small f
    # More conservative estimate
    exponent = 2.0 / kelly_fraction
    return min(1.0, drawdown^exponent * n_bets / 100)
end

"""
    optimal_bet_size(bankroll::Real, kelly_frac::Real; 
                     min_bet::Real=0.0, max_bet::Real=Inf) -> Float64

Calculate the actual bet size given bankroll and Kelly fraction.

# Arguments
- `bankroll`: Current bankroll/capital
- `kelly_frac`: Kelly fraction (from `kelly_fraction` or similar)
- `min_bet`: Minimum bet size
- `max_bet`: Maximum bet size

# Returns
Recommended bet size.
"""
function optimal_bet_size(bankroll::Real, kelly_frac::Real; 
                          min_bet::Real=0.0, max_bet::Real=Inf)
    bankroll > 0 || return 0.0
    raw_bet = bankroll * max(kelly_frac, 0.0)
    return clamp(raw_bet, min_bet, max_bet)
end
