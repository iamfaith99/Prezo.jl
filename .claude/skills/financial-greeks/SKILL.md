# Financial Greeks & Risk Metrics

Expertise in computing option Greeks (sensitivities) and risk metrics for portfolio management and hedging.

## The Greeks Overview

### First-Order Greeks

**Delta (Δ)**: Sensitivity to spot price
- ∂V/∂S
- Call: 0 < Δ < 1
- Put: -1 < Δ < 0
- **Interpretation**: For $1 move in stock, option moves $Δ

**Vega (ν)**: Sensitivity to volatility
- ∂V/∂σ
- Always positive for long options
- **Interpretation**: For 1% change in vol, option moves $ν

**Theta (Θ)**: Time decay
- ∂V/∂t or -∂V/∂T
- Usually negative (options lose value over time)
- **Interpretation**: Daily P&L from passage of time

**Rho (ρ)**: Sensitivity to interest rate
- ∂V/∂r
- Call: positive, Put: negative
- **Interpretation**: For 1% change in rate, option moves $ρ

### Second-Order Greeks

**Gamma (Γ)**: Rate of change of Delta
- ∂²V/∂S² = ∂Δ/∂S
- Always positive for long options
- Highest for ATM options near expiry
- **Interpretation**: Convexity; how much delta changes

**Vanna**: Cross-sensitivity
- ∂²V/∂S∂σ = ∂Δ/∂σ = ∂ν/∂S
- **Interpretation**: How delta changes with vol

**Volga (Vomma)**: Vol convexity
- ∂²V/∂σ²
- **Interpretation**: How vega changes with vol

## Analytical Greeks (Black-Scholes)

### Implementation

```julia
struct GreekResults{T<:AbstractFloat}
    delta::T
    gamma::T
    vega::T
    theta::T
    rho::T
end

function greeks(option::EuropeanCall, engine::BlackScholes, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data

    # Common terms
    sqrt_T = sqrt(expiry)
    d1 = (log(spot / strike) + (rate - div + 0.5 * vol^2) * expiry) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T

    Φ_d1 = norm_cdf(d1)
    Φ_d2 = norm_cdf(d2)
    φ_d1 = norm_pdf(d1)  # Standard normal PDF

    disc_spot = spot * exp(-div * expiry)
    disc_strike = strike * exp(-rate * expiry)

    # Delta
    delta = exp(-div * expiry) * Φ_d1

    # Gamma
    gamma = exp(-div * expiry) * φ_d1 / (spot * vol * sqrt_T)

    # Vega (per 1% change in vol)
    vega = disc_spot * φ_d1 * sqrt_T / 100

    # Theta (per day)
    theta_year = (-disc_spot * φ_d1 * vol / (2 * sqrt_T)
                  - rate * disc_strike * Φ_d2
                  + div * disc_spot * Φ_d1)
    theta = theta_year / 365

    # Rho (per 1% change in rate)
    rho = disc_strike * expiry * Φ_d2 / 100

    return GreekResults(delta, gamma, vega, theta, rho)
end

function greeks(option::EuropeanPut, engine::BlackScholes, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data

    sqrt_T = sqrt(expiry)
    d1 = (log(spot / strike) + (rate - div + 0.5 * vol^2) * expiry) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T

    Φ_md1 = norm_cdf(-d1)
    Φ_md2 = norm_cdf(-d2)
    φ_d1 = norm_pdf(d1)

    disc_spot = spot * exp(-div * expiry)
    disc_strike = strike * exp(-rate * expiry)

    # Delta
    delta = -exp(-div * expiry) * Φ_md1

    # Gamma (same for call and put)
    gamma = exp(-div * expiry) * φ_d1 / (spot * vol * sqrt_T)

    # Vega (same for call and put)
    vega = disc_spot * φ_d1 * sqrt_T / 100

    # Theta
    theta_year = (-disc_spot * φ_d1 * vol / (2 * sqrt_T)
                  + rate * disc_strike * Φ_md2
                  - div * disc_spot * Φ_md1)
    theta = theta_year / 365

    # Rho
    rho = -disc_strike * expiry * Φ_md2 / 100

    return GreekResults(delta, gamma, vega, theta, rho)
end

# Helper functions
function norm_cdf(x)
    return 0.5 * (1 + erf(x / sqrt(2)))
end

function norm_pdf(x)
    return exp(-0.5 * x^2) / sqrt(2π)
end
```

## Numerical Greeks (Finite Difference)

### Central Difference (Most Accurate)

```julia
function delta_finite_difference(
    option::VanillaOption,
    engine::PricingEngine,
    data::MarketData;
    h::Float64=0.01  # Bump size
)
    # Central difference: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
    data_up = MarketData(data.spot + h, data.rate, data.vol, data.div)
    data_down = MarketData(data.spot - h, data.rate, data.vol, data.div)

    price_up = price(option, engine, data_up)
    price_down = price(option, engine, data_down)

    return (price_up - price_down) / (2h)
end

function gamma_finite_difference(
    option::VanillaOption,
    engine::PricingEngine,
    data::MarketData;
    h::Float64=0.01
)
    # Second derivative: f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
    data_up = MarketData(data.spot + h, data.rate, data.vol, data.div)
    data_down = MarketData(data.spot - h, data.rate, data.vol, data.div)

    price_up = price(option, engine, data_up)
    price_mid = price(option, engine, data)
    price_down = price(option, engine, data_down)

    return (price_up - 2 * price_mid + price_down) / (h^2)
end

function vega_finite_difference(
    option::VanillaOption,
    engine::PricingEngine,
    data::MarketData;
    h::Float64=0.01  # 1% vol bump
)
    data_up = MarketData(data.spot, data.rate, data.vol + h, data.div)
    data_down = MarketData(data.spot, data.rate, data.vol - h, data.div)

    price_up = price(option, engine, data_up)
    price_down = price(option, engine, data_down)

    return (price_up - price_down) / (2h)
end

function theta_finite_difference(
    option::VanillaOption,
    engine::PricingEngine,
    data::MarketData;
    h::Float64=1/365  # 1 day
)
    # Theta = -∂V/∂T = ∂V/∂t (time to expiry decreases)
    if option.expiry <= h
        return 0.0  # Can't bump past expiry
    end

    option_future = typeof(option)(option.strike, option.expiry)
    option_near = typeof(option)(option.strike, option.expiry - h)

    price_future = price(option_future, engine, data)
    price_near = price(option_near, engine, data)

    return (price_near - price_future) / h  # Note: this is -∂V/∂T
end

function rho_finite_difference(
    option::VanillaOption,
    engine::PricingEngine,
    data::MarketData;
    h::Float64=0.01  # 1% rate bump
)
    data_up = MarketData(data.spot, data.rate + h, data.vol, data.div)
    data_down = MarketData(data.spot, data.rate - h, data.vol, data.div)

    price_up = price(option, engine, data_up)
    price_down = price(option, engine, data_down)

    return (price_up - price_down) / (2h)
end
```

### Choosing Bump Size (h)

**Trade-off:**
- Too large: truncation error (formula approximation)
- Too small: roundoff error (finite precision)

**Optimal h** (for central difference):
```julia
# Rule of thumb
h_optimal = cbrt(eps(Float64))  # ≈ 6e-6

# Practical values:
h_spot = 0.01  # $1 or 1% of spot
h_vol = 0.01   # 1% absolute (e.g., 20% → 21%)
h_rate = 0.0001  # 1 bp (0.01%)
```

## Pathwise Greeks (Monte Carlo)

For Monte Carlo pricing, pathwise derivatives are more efficient than finite difference.

### Pathwise Delta

```julia
function delta_pathwise_mc(
    option::EuropeanCall,
    data::MarketData,
    n_paths::Int
)
    (; spot, rate, vol, div, expiry) = data
    (; strike) = option

    drift = (rate - div - 0.5 * vol^2) * expiry
    vol_sqrt_T = vol * sqrt(expiry)
    disc = exp(-rate * expiry)

    sum_delta = 0.0

    for i in 1:n_paths
        Z = randn()

        # Terminal spot
        S_T = spot * exp(drift + vol_sqrt_T * Z)

        # Pathwise derivative: ∂S_T/∂S_0 = S_T/S_0
        ∂S_∂S0 = S_T / spot

        # Payoff derivative: ∂payoff/∂S_T
        ∂payoff_∂ST = S_T > strike ? 1.0 : 0.0

        # Chain rule
        sum_delta += ∂payoff_∂ST * ∂S_∂S0
    end

    delta = disc * sum_delta / n_paths
    return delta
end
```

**Advantages:**
- Same paths used for price and Greeks
- Lower variance than finite difference
- No choice of bump size

**Limitations:**
- Requires smooth payoffs (not digital options)
- More complex implementation

### Pathwise Vega

```julia
function vega_pathwise_mc(
    option::EuropeanCall,
    data::MarketData,
    n_paths::Int
)
    (; spot, rate, vol, div, expiry) = data
    (; strike) = option

    drift = (rate - div - 0.5 * vol^2) * expiry
    sqrt_T = sqrt(expiry)
    disc = exp(-rate * expiry)

    sum_vega = 0.0

    for i in 1:n_paths
        Z = randn()

        # Terminal spot
        S_T = spot * exp(drift + vol * sqrt_T * Z)

        # ∂S_T/∂σ
        ∂S_∂σ = S_T * (Z * sqrt_T - vol * expiry)

        # Payoff derivative
        ∂payoff_∂ST = S_T > strike ? 1.0 : 0.0

        sum_vega += ∂payoff_∂ST * ∂S_∂σ
    end

    vega = disc * sum_vega / n_paths / 100  # Per 1% vol change
    return vega
end
```

## Likelihood Ratio Method (Monte Carlo)

Alternative to pathwise, works for discontinuous payoffs.

### LR Delta

```julia
function delta_likelihood_ratio_mc(
    option::VanillaOption,
    data::MarketData,
    n_paths::Int
)
    (; spot, rate, vol, div, expiry) = data

    drift = (rate - div - 0.5 * vol^2) * expiry
    vol_sqrt_T = vol * sqrt(expiry)
    disc = exp(-rate * expiry)

    sum_weighted_payoff = 0.0

    for i in 1:n_paths
        Z = randn()

        # Terminal spot
        S_T = spot * exp(drift + vol_sqrt_T * Z)

        # Payoff
        poff = payoff(option, S_T)

        # Score function (derivative of log-likelihood)
        score = Z / (spot * vol_sqrt_T)

        sum_weighted_payoff += poff * score
    end

    delta = disc * sum_weighted_payoff / n_paths
    return delta
end
```

**Advantages:**
- Works for discontinuous payoffs
- Works for American options

**Disadvantages:**
- Higher variance than pathwise
- Can be numerically unstable

## Greeks for American Options

### Finite Difference (Most Common)

```julia
function greeks_american_fd(
    option::AmericanOption,
    engine::PricingEngine,
    data::MarketData
)
    h_spot = 0.01
    h_vol = 0.01
    h_rate = 0.0001

    # Delta
    data_up = MarketData(data.spot + h_spot, data.rate, data.vol, data.div)
    data_down = MarketData(data.spot - h_spot, data.rate, data.vol, data.div)
    delta = (price(option, engine, data_up) - price(option, engine, data_down)) / (2 * h_spot)

    # Gamma
    price_mid = price(option, engine, data)
    gamma = (price(option, engine, data_up) - 2 * price_mid + price(option, engine, data_down)) / (h_spot^2)

    # Vega
    data_vol_up = MarketData(data.spot, data.rate, data.vol + h_vol, data.div)
    data_vol_down = MarketData(data.spot, data.rate, data.vol - h_vol, data.div)
    vega = (price(option, engine, data_vol_up) - price(option, engine, data_vol_down)) / (2 * h_vol)

    # Theta (1 day)
    h_time = 1/365
    if option.expiry > h_time
        option_near = typeof(option)(option.strike, option.expiry - h_time)
        theta = (price(option_near, engine, data) - price_mid) / h_time
    else
        theta = NaN
    end

    # Rho
    data_rate_up = MarketData(data.spot, data.rate + h_rate, data.vol, data.div)
    data_rate_down = MarketData(data.spot, data.rate - h_rate, data.vol, data.div)
    rho = (price(option, engine, data_rate_up) - price(option, engine, data_rate_down)) / (2 * h_rate)

    return GreekResults(delta, gamma, vega, theta, rho)
end
```

### Parallel Computation

```julia
using Base.Threads

function greeks_american_parallel(
    option::AmericanOption,
    engine::PricingEngine,
    data::MarketData
)
    # Pre-allocate results
    prices = zeros(7)  # [down_spot, mid, up_spot, down_vol, up_vol, down_rate, up_rate]

    # Data scenarios
    h_spot, h_vol, h_rate = 0.01, 0.01, 0.0001

    scenarios = [
        MarketData(data.spot - h_spot, data.rate, data.vol, data.div),
        data,
        MarketData(data.spot + h_spot, data.rate, data.vol, data.div),
        MarketData(data.spot, data.rate, data.vol - h_vol, data.div),
        MarketData(data.spot, data.rate, data.vol + h_vol, data.div),
        MarketData(data.spot, data.rate - h_rate, data.vol, data.div),
        MarketData(data.spot, data.rate + h_rate, data.vol, data.div),
    ]

    @threads for i in 1:7
        prices[i] = price(option, engine, scenarios[i])
    end

    # Compute Greeks
    delta = (prices[3] - prices[1]) / (2 * h_spot)
    gamma = (prices[3] - 2 * prices[2] + prices[1]) / (h_spot^2)
    vega = (prices[5] - prices[4]) / (2 * h_vol)
    rho = (prices[7] - prices[6]) / (2 * h_rate)

    # Theta
    h_time = 1/365
    if option.expiry > h_time
        option_near = typeof(option)(option.strike, option.expiry - h_time)
        theta = (price(option_near, engine, data) - prices[2]) / h_time
    else
        theta = NaN
    end

    return GreekResults(delta, gamma, vega, theta, rho)
end
```

## Portfolio Greeks

Aggregate Greeks across positions.

```julia
struct Position
    option::VanillaOption
    quantity::Int  # Positive = long, negative = short
end

function portfolio_greeks(
    positions::Vector{Position},
    engine::PricingEngine,
    data::MarketData
)
    total_delta = 0.0
    total_gamma = 0.0
    total_vega = 0.0
    total_theta = 0.0
    total_rho = 0.0

    for pos in positions
        gks = greeks(pos.option, engine, data)

        total_delta += pos.quantity * gks.delta
        total_gamma += pos.quantity * gks.gamma
        total_vega += pos.quantity * gks.vega
        total_theta += pos.quantity * gks.theta
        total_rho += pos.quantity * gks.rho
    end

    return GreekResults(total_delta, total_gamma, total_vega, total_theta, total_rho)
end
```

## Delta Hedging

### Delta-Neutral Portfolio

```julia
function delta_hedge_quantity(
    option_delta::Float64,
    option_quantity::Int,
    stock_delta::Float64=1.0
)
    # How many shares to hold for delta-neutral position?
    # Total delta = option_quantity * option_delta + hedge_quantity * stock_delta = 0

    hedge_quantity = -(option_quantity * option_delta) / stock_delta

    return hedge_quantity
end

# Example: Long 100 calls with delta = 0.6
# Hedge: Short 60 shares (delta = -60)
```

### Dynamic Delta Hedging

```julia
function delta_hedge_rebalance(
    current_hedge::Float64,
    new_option_greeks::GreekResults,
    option_quantity::Int
)
    target_hedge = -option_quantity * new_option_greeks.delta
    rebalance_quantity = target_hedge - current_hedge

    return (target_hedge=target_hedge, rebalance=rebalance_quantity)
end
```

## Gamma Scalping

P&L from gamma when delta hedged.

```julia
function gamma_pnl(
    gamma::Float64,
    spot_change::Float64,
    option_quantity::Int
)
    # Gamma P&L ≈ 0.5 * Γ * (ΔS)²
    pnl = 0.5 * option_quantity * gamma * spot_change^2
    return pnl
end
```

## Risk Metrics

### Value at Risk (VaR)

```julia
function portfolio_var(
    greeks::GreekResults,
    spot::Float64,
    vol::Float64;
    confidence::Float64=0.95,
    horizon_days::Int=1
)
    # Simple delta-gamma approximation
    # ΔV ≈ Δ * ΔS + 0.5 * Γ * (ΔS)²

    # Daily volatility
    daily_vol = vol * sqrt(horizon_days / 365)

    # Worst-case spot move (e.g., 95% confidence = 1.645σ)
    z = quantile(Normal(), confidence)
    worst_spot_move = spot * daily_vol * z

    # Portfolio change
    delta_change = greeks.delta * worst_spot_move
    gamma_change = 0.5 * greeks.gamma * worst_spot_move^2

    var = -(delta_change + gamma_change)  # Negative of loss

    return var
end
```

### Greeks Ladder

Display Greeks at different spot levels.

```julia
function greeks_ladder(
    option::VanillaOption,
    engine::PricingEngine,
    data::MarketData;
    spot_range::Float64=0.2  # ±20%
)
    spot_levels = range(
        data.spot * (1 - spot_range),
        data.spot * (1 + spot_range),
        length=11
    )

    results = []

    for S in spot_levels
        data_level = MarketData(S, data.rate, data.vol, data.div)
        prc = price(option, engine, data_level)
        gks = greeks(option, engine, data_level)

        push!(results, (spot=S, price=prc, greeks=gks))
    end

    return results
end
```

## Advanced: Greeks with Automatic Differentiation

Use ForwardDiff.jl for exact derivatives.

```julia
using ForwardDiff

function delta_autodiff(
    option::VanillaOption,
    engine::PricingEngine,
    data::MarketData
)
    # Define function: spot → price
    f(S) = price(option, engine, MarketData(S, data.rate, data.vol, data.div))

    # Automatic derivative
    delta = ForwardDiff.derivative(f, data.spot)
    return delta
end

function greeks_autodiff(
    option::VanillaOption,
    engine::PricingEngine,
    data::MarketData
)
    # Delta: ∂V/∂S
    f_S(S) = price(option, engine, MarketData(S, data.rate, data.vol, data.div))
    delta = ForwardDiff.derivative(f_S, data.spot)
    gamma = ForwardDiff.derivative(S -> ForwardDiff.derivative(f_S, S), data.spot)

    # Vega: ∂V/∂σ
    f_σ(σ) = price(option, engine, MarketData(data.spot, data.rate, σ, data.div))
    vega = ForwardDiff.derivative(f_σ, data.vol) / 100

    # Rho: ∂V/∂r
    f_r(r) = price(option, engine, MarketData(data.spot, r, data.vol, data.div))
    rho = ForwardDiff.derivative(f_r, data.rate) / 100

    # Theta: approximate with finite difference
    h = 1/365
    if option.expiry > h
        option_near = typeof(option)(option.strike, option.expiry - h)
        theta = (price(option_near, engine, data) - price(option, engine, data)) / h
    else
        theta = 0.0
    end

    return GreekResults(delta, gamma, vega, theta, rho)
end
```

**Advantages:**
- Exact derivatives (no finite difference error)
- No need to choose bump size

**Limitations:**
- Requires code to be ForwardDiff-compatible
- Can be slower for complex functions

## Testing Greeks

```julia
@testset "Greeks Validation" begin
    # Test delta bounds
    call = EuropeanCall(100.0, 1.0)
    data = MarketData(100.0, 0.05, 0.2, 0.0)

    gks = greeks(call, BlackScholes(), data)

    @test 0 < gks.delta < 1  # Call delta in (0,1)
    @test gks.gamma > 0      # Always positive
    @test gks.vega > 0       # Always positive
    @test gks.theta < 0      # Time decay

    # Test delta-gamma relationship (spot bump)
    h = 1.0
    data_up = MarketData(100.0 + h, 0.05, 0.2, 0.0)
    gks_up = greeks(call, BlackScholes(), data_up)

    delta_change = gks_up.delta - gks.delta
    gamma_estimate = delta_change / h

    @test isapprox(gamma_estimate, gks.gamma, rtol=0.01)

    # Compare analytical vs finite difference
    delta_fd = delta_finite_difference(call, BlackScholes(), data)
    @test isapprox(gks.delta, delta_fd, rtol=0.001)
end
```

## Further Reading

- Hull: "Options, Futures, and Other Derivatives" (Greeks chapter)
- Taleb: "Dynamic Hedging" (Practical Greek management)
- Wilmott: "Paul Wilmott on Quantitative Finance" (Greek derivations)
- Glasserman: "Monte Carlo Methods" (Pathwise and LR methods)
