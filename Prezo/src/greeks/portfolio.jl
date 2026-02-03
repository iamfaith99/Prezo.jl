"""
    portfolio.jl

Portfolio-level Greek aggregation and risk management.

This module provides tools for:
- Aggregating Greeks across multiple positions
- Computing portfolio-level risk metrics
- Delta-neutral and gamma-neutral hedging
- P&L attribution

A position is defined as an option contract with a quantity (positive for long,
negative for short). Portfolio Greeks are weighted sums of individual position Greeks.
"""

using ..Prezo

"""
    Position(option::VanillaOption, quantity::Real)

Represents a position in an option contract.

# Fields
- `option::VanillaOption`: The option contract
- `quantity::Real`: Number of contracts (positive=long, negative=short)

# Examples
```julia
# Long 10 ATM calls
pos1 = Position(EuropeanCall(100.0, 1.0), 10)

# Short 5 OTM puts
pos2 = Position(EuropeanPut(90.0, 1.0), -5)

# Portfolio
portfolio = [pos1, pos2]
```

See also: [`portfolio_greeks`](@ref)
"""
struct Position{T<:Real}
    option::VanillaOption
    quantity::T

    function Position(option::VanillaOption, quantity::T) where {T<:Real}
        new{T}(option, quantity)
    end
end

Base.broadcastable(p::Position) = Ref(p)

"""
    portfolio_greeks(positions::Vector{Position}, data::MarketData)

Compute aggregated Greeks for a portfolio of positions using Black-Scholes.

Portfolio Greeks are computed as:
Portfolio Greek = Σ (position_quantity × individual_greek)

# Arguments
- `positions::Vector{Position}`: Portfolio positions
- `data::MarketData`: Market parameters

# Returns
Dictionary mapping Greek types to portfolio-level values.

# Examples
```julia
# Create a delta-neutral straddle
positions = [
    Position(EuropeanCall(100.0, 1.0), 100),   # Long 100 calls
    Position(EuropeanPut(100.0, 1.0), 100),  # Long 100 puts
]

data = MarketData(100.0, 0.05, 0.2, 0.0)
portfolio_greeks(positions, data)

# Output will show near-zero delta (delta-neutral straddle)
```

See also: [`Position`](@ref), [`hedge_ratio`](@ref)
"""
function portfolio_greeks(positions::Vector{<:Position}, data::MarketData)
    # Initialize accumulators for each Greek
    greek_types = [Delta(), Gamma(), Theta(), Vega(), Rho(), Phi()]
    portfolio_values = Dict{Greek,Float64}(g => 0.0 for g in greek_types)

    for position in positions
        # Compute individual Greeks
        individual_greeks = all_greeks(position.option, data)

        # Accumulate weighted by position size
        for (greek_type, value) in individual_greeks
            portfolio_values[greek_type] += position.quantity * value
        end
    end

    return portfolio_values
end

"""
    portfolio_greeks(positions::Vector{Position}, engine::BlackScholes, data::MarketData; kwargs...)

Compute aggregated Greeks for a portfolio using Black-Scholes (analytical).

For European options with Black-Scholes engine, uses fast analytical Greeks.

# Arguments
- `positions::Vector{Position}`: Portfolio positions (must be European options)
- `engine::BlackScholes`: Black-Scholes pricing engine
- `data::MarketData`: Market parameters
- `kwargs...`: Additional arguments (ignored for analytical Greeks)

# Examples
```julia
positions = [
    Position(EuropeanCall(100.0, 1.0), 10),
    Position(EuropeanPut(100.0, 1.0), -5),
]

data = MarketData(100.0, 0.05, 0.2, 0.0)
portfolio_greeks(positions, BlackScholes(), data)
```
"""
function portfolio_greeks(positions::Vector{<:Position}, engine::BlackScholes, data::MarketData; kwargs...)
    # For BlackScholes, delegate to analytical method
    return portfolio_greeks(positions, data)
end

"""
    portfolio_greeks(positions::Vector{Position}, engine::PricingEngine, data::MarketData; kwargs...)

Compute aggregated Greeks for a portfolio using any pricing engine.

This version supports American options and other engines that don't have
analytical Greek formulas.

# Arguments
- `positions::Vector{Position}`: Portfolio positions
- `engine::PricingEngine`: Pricing engine for numerical Greeks
- `data::MarketData`: Market parameters
- `kwargs...`: Additional arguments passed to numerical methods

# Examples
```julia
# Delta-neutral portfolio with American options
positions = [
    Position(AmericanCall(100.0, 1.0), 50),
    Position(AmericanPut(100.0, 1.0), 50),
]

engine = Binomial(500)
data = MarketData(100.0, 0.05, 0.2, 0.0)
portfolio_greeks(positions, engine, data)
```
"""
function portfolio_greeks(positions::Vector{<:Position}, engine::PricingEngine, data::MarketData; kwargs...)
    # Otherwise, use numerical methods
    greek_types = [Delta(), Gamma(), Theta(), Vega(), Rho(), Phi()]
    portfolio_values = Dict{Greek,Float64}(g => 0.0 for g in greek_types)

    for position in positions
        # Compute individual Greeks numerically
        individual_greeks = all_greeks(position.option, engine, data; kwargs...)

        # Accumulate weighted by position size
        for (greek_type, value) in individual_greeks
            portfolio_values[greek_type] += position.quantity * value
        end
    end

    return portfolio_values
end

"""
    hedge_ratio(positions::Vector{Position}, greek_type::Greek, data::MarketData)

Compute the number of underlying shares needed to hedge a specific Greek.

# Arguments
- `positions::Vector{Position}`: Portfolio to hedge
- `greek_type::Greek`: Greek to hedge (typically Delta or Gamma)
- `data::MarketData`: Market parameters

# Returns
Number of underlying shares to trade:
- Positive = buy shares
- Negative = sell shares

# Examples
```julia
# Delta hedge a short option position
positions = [Position(EuropeanCall(100.0, 1.0), -100)]  # Short 100 calls
data = MarketData(100.0, 0.05, 0.2, 0.0)

# Shares needed to delta-neutralize
shares = hedge_ratio(positions, Delta(), data)

# Execute: buy `shares` shares of underlying
```

# Hedging Strategies

## Delta Hedging
Neutralizes directional exposure:
```julia
shares = hedge_ratio(portfolio, Delta(), data)
```

## Gamma Hedging (with options)
Requires trading other options, not just underlying:
```julia
# Target: Make portfolio gamma-neutral
# Trade additional options to offset gamma
```

## Vega Hedging
Trade options with different expirations:
```julia
# Portfolio has positive vega (long options)
# Sell other options to reduce vega exposure
```

See also: [`portfolio_greeks`](@ref), [`is_delta_neutral`](@ref)
"""
function hedge_ratio(positions::Vector{<:Position}, greek_type::Greek, data::MarketData)
    # Compute portfolio Greek
    pgreeks = portfolio_greeks(positions, data)
    portfolio_greek_value = pgreeks[greek_type]

    # For delta hedging with underlying:
    # Underlying has delta = 1, gamma = 0, vega = 0, etc.
    if greek_type isa Delta
        # To hedge: trade -portfolio_delta shares
        return -portfolio_greek_value
    elseif greek_type isa Gamma
        # Underlying has zero gamma, can't gamma-hedge with stock alone
        @warn "Cannot gamma-hedge using underlying alone (underlying has gamma=0). Use options for gamma hedging."
        return 0.0
    elseif greek_type isa Vega
        # Underlying has zero vega
        @warn "Cannot vega-hedge using underlying alone (underlying has vega=0). Use options for vega hedging."
        return 0.0
    else
        # For other Greeks, underlying has zero sensitivity
        @warn "Underlying has zero $(typeof(greek_type)). Cannot hedge using stock alone."
        return 0.0
    end
end

"""
    hedge_ratio(positions::Vector{Position}, greek_type::Greek,
                hedge_option::VanillaOption, data::MarketData)

Compute number of hedge_option contracts needed to neutralize a specific Greek.

This version allows hedging with other options (for gamma, vega hedging).

# Arguments
- `positions::Vector{Position}`: Portfolio to hedge
- `greek_type::Greek`: Greek to hedge
- `hedge_option::VanillaOption`: Option to use as hedge instrument
- `data::MarketData`: Market parameters

# Returns
Number of hedge_option contracts to trade:
- Positive = buy options
- Negative = sell options

# Examples
```julia
# Gamma hedge using ATM options
portfolio = [Position(EuropeanCall(100.0, 1.0), 100)]
hedge_option = EuropeanPut(100.0, 1.0)  # Use puts to hedge gamma

contracts = hedge_ratio(portfolio, Gamma(), hedge_option, data)
```
"""
function hedge_ratio(positions::Vector{<:Position}, greek_type::Greek,
    hedge_option::VanillaOption, data::MarketData)
    # Compute portfolio Greek
    pgreeks = portfolio_greeks(positions, data)
    portfolio_greek_value = pgreeks[greek_type]

    # Compute hedge option Greek
    hedge_greek = greek(hedge_option, greek_type, data)

    if abs(hedge_greek) < 1e-10
        error("Hedge option has near-zero $(typeof(greek_type)). Cannot use for hedging.")
    end

    # Number of contracts needed
    # portfolio_greek + n * hedge_greek = 0
    # n = -portfolio_greek / hedge_greek
    contracts = -portfolio_greek_value / hedge_greek

    return contracts
end

"""
    is_delta_neutral(positions::Vector{Position}, data::MarketData; tolerance=1e-6)

Check if a portfolio is delta-neutral (within tolerance).

# Arguments
- `positions::Vector{Position}`: Portfolio to check
- `data::MarketData`: Market parameters
- `tolerance::Real`: Maximum allowed delta for neutrality (default: 1e-6)

# Returns
`true` if |portfolio_delta| ≤ tolerance, `false` otherwise.

# Examples
```julia
portfolio = [
    Position(EuropeanCall(100.0, 1.0), 100),
    Position(EuropeanPut(100.0, 1.0), 100),
]

is_delta_neutral(portfolio, data)  # true for straddle
```
"""
function is_delta_neutral(positions::Vector{<:Position}, data::MarketData; tolerance::Real=1e-6)
    pgreeks = portfolio_greeks(positions, data)
    portfolio_delta = pgreeks[Delta()]
    return abs(portfolio_delta) <= tolerance
end

"""
    is_gamma_neutral(positions::Vector{Position}, data::MarketData; tolerance=1e-6)

Check if a portfolio is gamma-neutral (within tolerance).

Gamma neutrality requires balanced convexity, typically achieved with
offsetting long and short option positions.
"""
function is_gamma_neutral(positions::Vector{<:Position}, data::MarketData; tolerance::Real=1e-6)
    pgreeks = portfolio_greeks(positions, data)
    portfolio_gamma = pgreeks[Gamma()]
    return abs(portfolio_gamma) <= tolerance
end

"""
    pnl_attribution(positions::Vector{Position},
                    data_before::MarketData, data_after::MarketData,
                    greeks_before::Dict{Greek, Float64})

Attribute P&L changes to Greek sensitivities (Greek decomposition).

Uses Taylor expansion to approximate P&L:
ΔP&L ≈ Δ×ΔS + ½×Γ×(ΔS)² + ν×Δσ + Θ×Δt + ρ×Δr + φ×Δq

# Arguments
- `positions::Vector{Position}`: Portfolio positions
- `data_before::MarketData`: Market state before move
- `data_after::MarketData`: Market state after move
- `greeks_before::Dict{Greek, Float64}`: Portfolio Greeks at start

# Returns
Dictionary with P&L attribution:
- `delta_pnl`: P&L from spot move
- `gamma_pnl`: P&L from convexity (second-order)
- `vega_pnl`: P&L from vol change
- `theta_pnl`: P&L from time decay
- `rho_pnl`: P&L from rate change
- `phi_pnl`: P&L from dividend change
- `total_approx`: Sum of all Greek contributions
- `total_actual`: Actual P&L from full re-pricing

# Examples
```julia
# Initial state
data_before = MarketData(100.0, 0.05, 0.2, 0.0)
positions = [Position(EuropeanCall(100.0, 1.0), 100)]

# After market move
data_after = MarketData(105.0, 0.05, 0.22, 0.0)

# Attribution
pgreeks = portfolio_greeks(positions, data_before)
attribution = pnl_attribution(positions, data_before, data_after, pgreeks)

# Show breakdown
println("Delta P&L: ", attribution[:delta_pnl])
println("Gamma P&L: ", attribution[:gamma_pnl])
println("Vega P&L: ", attribution[:vega_pnl])
```

See also: [`portfolio_greeks`](@ref)
"""
function pnl_attribution(positions::Vector{<:Position},
    data_before::MarketData, data_after::MarketData,
    greeks_before::Dict{Greek,Float64})

    # Compute market moves
    dS = data_after.spot - data_before.spot
    dvol = data_after.vol - data_before.vol
    dr = data_after.rate - data_before.rate
    dq = data_after.div - data_before.div
    dt = 0.0  # Assume same time point (instantaneous attribution)

    # Get portfolio Greeks
    delta = greeks_before[Delta()]
    gamma = greeks_before[Gamma()]
    vega = greeks_before[Vega()]
    theta = greeks_before[Theta()]
    rho = greeks_before[Rho()]
    phi = greeks_before[Phi()]

    # Attribution using Taylor expansion
    delta_pnl = delta * dS
    gamma_pnl = 0.5 * gamma * dS^2
    vega_pnl = vega * dvol
    theta_pnl = theta * dt
    rho_pnl = rho * dr
    phi_pnl = phi * dq

    total_approx = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + rho_pnl + phi_pnl

    # Compute actual P&L by re-pricing
    value_before = sum(p -> p.quantity * price(p.option, BlackScholes(), data_before), positions)
    value_after = sum(p -> p.quantity * price(p.option, BlackScholes(), data_after), positions)
    total_actual = value_after - value_before

    return Dict(
        :delta_pnl => delta_pnl,
        :gamma_pnl => gamma_pnl,
        :vega_pnl => vega_pnl,
        :theta_pnl => theta_pnl,
        :rho_pnl => rho_pnl,
        :phi_pnl => phi_pnl,
        :total_approx => total_approx,
        :total_actual => total_actual,
        :unexplained => total_actual - total_approx
    )
end

"""
    scenario_analysis(positions::Vector{Position}, data::MarketData,
                     spot_changes::Vector{<:Real}, vol_changes::Vector{<:Real})

Compute portfolio value under different market scenarios.

# Arguments
- `positions::Vector{Position}`: Portfolio positions
- `data::MarketData`: Base market parameters
- `spot_changes::Vector{<:Real}`: Spot price scenarios (absolute changes)
- `vol_changes::Vector{<:Real}`: Volatility scenarios (absolute changes)

# Returns
Matrix of portfolio values for each (spot_change, vol_change) combination.

# Examples
```julia
positions = [Position(EuropeanCall(100.0, 1.0), 100)]
data = MarketData(100.0, 0.05, 0.2, 0.0)

# Scenario grid
spot_changes = [-10.0, -5.0, 0.0, 5.0, 10.0]  # -10 to +10
vol_changes = [-0.05, 0.0, 0.05]              # -5% to +5%

# Compute scenario matrix
scenarios = scenario_analysis(positions, data, spot_changes, vol_changes)
# scenarios[i, j] = value when spot_changes[i] and vol_changes[j]
```

See also: [`pnl_attribution`](@ref)
"""
function scenario_analysis(positions::Vector{<:Position}, data::MarketData,
    spot_changes::Vector{T}, vol_changes::Vector{U}) where {T<:Real,U<:Real}

    n_spot = length(spot_changes)
    n_vol = length(vol_changes)

    # Compute base value
    base_value = sum(p -> p.quantity * price(p.option, BlackScholes(), data), positions)

    # Scenario matrix
    scenarios = zeros(n_spot, n_vol)

    for i in 1:n_spot
        for j in 1:n_vol
            # New market data for this scenario
            new_spot = data.spot + spot_changes[i]
            new_vol = data.vol + vol_changes[j]
            new_data = MarketData(new_spot, data.rate, new_vol, data.div)

            # Portfolio value in this scenario
            scenario_value = sum(p -> p.quantity * price(p.option, BlackScholes(), new_data), positions)

            # Store P&L (change from base)
            scenarios[i, j] = scenario_value - base_value
        end
    end

    return scenarios
end
