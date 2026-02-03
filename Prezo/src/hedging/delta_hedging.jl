"""
    Delta hedging strategies and backtesting

Implements discrete delta hedge, stop-loss hedge, static hedge, and a backtest
framework that tracks P&L, transaction costs, and hedge slippage.
"""

using Statistics

"""
    HedgingStrategy

Abstract type for delta hedging strategies.

Subtypes: [`DiscreteDeltaHedge`](@ref), [`StopLossHedge`](@ref), [`StaticHedge`](@ref).
"""
abstract type HedgingStrategy end

"""
    DiscreteDeltaHedge(rebalancing_times, transaction_cost)

Rebalance delta at fixed times; pay a cost per trade.

# Fields
- `rebalancing_times::Vector{Float64}`: Times at which to rebalance (e.g. [0.25, 0.5, 0.75])
- `transaction_cost::Float64`: Cost per unit of underlying traded (absolute)

# Examples
```julia
strategy = DiscreteDeltaHedge([0.25, 0.5, 0.75, 1.0], 0.001)
```
"""
struct DiscreteDeltaHedge <: HedgingStrategy
    rebalancing_times::Vector{Float64}
    transaction_cost::Float64
end

"""
    StopLossHedge(trigger_level)

Rebalance when delta deviation from target exceeds a threshold.

# Fields
- `trigger_level::Float64`: Rebalance when |delta - target_delta| > trigger_level
  (e.g. 0.1 for 10% of a contract)

# Examples
```julia
strategy = StopLossHedge(0.15)
```
"""
struct StopLossHedge <: HedgingStrategy
    trigger_level::Float64
end

"""
    StaticHedge(hedge_instruments::Vector{EuropeanOption})

Hold a fixed portfolio of options (no dynamic rebalancing).

# Fields
- `hedge_instruments::Vector{EuropeanOption}`: Options used to hedge (e.g. replicating portfolio)

# Examples
```julia
strategy = StaticHedge([EuropeanCall(100.0, 1.0), EuropeanPut(100.0, 1.0)])
```
"""
struct StaticHedge <: HedgingStrategy
    hedge_instruments::Vector{EuropeanOption}
end

"""
    HedgePerformance(total_pnl, total_cost, variance, max_drawdown, n_trades)

Result of a hedge backtest.

# Fields
- `total_pnl::Float64`: Total P&L (option payoff - hedging cost - transaction costs)
- `total_cost::Float64`: Sum of transaction costs
- `variance::Float64`: Variance of period-wise P&L (or path-wise if simulated)
- `max_drawdown::Float64`: Maximum cumulative drawdown during the period
- `n_trades::Int`: Number of rebalancing trades
"""
struct HedgePerformance
    total_pnl::Float64
    total_cost::Float64
    variance::Float64
    max_drawdown::Float64
    n_trades::Int
end

"""
    backtest_hedge(option::EuropeanOption, strategy::HedgingStrategy,
                  historical_data::Matrix{Float64}, hedging_engine::PricingEngine)
                  -> HedgePerformance

Backtest a hedging strategy along a single path of market data.

# Arguments
- `option::EuropeanOption`: Option being hedged (short position: we sold the option)
- `strategy::HedgingStrategy`: Hedging strategy to use
- `historical_data`: Matrix of size (n_obs × 4): each row = [time, spot, rate, vol]
- `hedging_engine::PricingEngine`: Engine for pricing and delta (e.g. BlackScholes())

# Returns
`HedgePerformance` with total P&L, transaction costs, variance, max drawdown, trade count.

# Notes
- Assumes short option: we receive premium at t=0 and must replicate payoff at expiry.
- Delta at each time uses remaining time to expiry (option with expiry T - t).
"""
function backtest_hedge(
    option::EuropeanOption,
    strategy::HedgingStrategy,
    historical_data::Matrix{Float64},
    hedging_engine::PricingEngine,
)
    n_obs = size(historical_data, 1)
    n_obs >= 2 || return HedgePerformance(0.0, 0.0, 0.0, 0.0, 0)

    times = historical_data[:, 1]
    spots = historical_data[:, 2]
    rates = historical_data[:, 3]
    vols = historical_data[:, 4]
    T = times[end]
    S0 = spots[1]
    r0 = rates[1]
    vol0 = vols[1]
    q0 = 0.0

    # Premium received at t=0 (we sold the option)
    data0 = MarketData(S0, r0, vol0, q0)
    premium = price(option, hedging_engine, data0)

    # Option payoff at expiry (last observation)
    payoff_at_expiry = payoff(option, spots[end])

    if strategy isa StaticHedge
        # Static: no rebalancing; P&L = premium - payoff (no dynamic hedge)
        return HedgePerformance(
            premium - payoff_at_expiry,
            0.0,
            0.0,
            max(0.0, payoff_at_expiry - premium),
            0,
        )
    end

    # Dynamic strategies: track shares, cash, costs
    shares = 0.0
    cash = premium  # Start with premium
    cost_total = 0.0
    n_trades = 0
    target_delta = 0.0
    pnls = Float64[]
    cum = 0.0
    max_dd = 0.0

    # Rebalancing: at which observation indices we rebalance
    rebalance_mask = falses(n_obs)
    rebalance_mask[1] = true
    rebalance_mask[end] = true
    if strategy isa DiscreteDeltaHedge
        for tr in strategy.rebalancing_times
            idx = argmin(abs.(times .- tr))
            rebalance_mask[idx] = true
        end
    end

    # Option with remaining expiry (for delta at time t)
    option_at_t(t_rem) = typeof(option)(option.strike, t_rem)

    for i in 1:n_obs
        t = times[i]
        tau = max(T - t, 1e-10)  # time to expiry
        S = spots[i]
        r = rates[i]
        vol = vols[i]
        data = MarketData(S, r, vol, q0)
        opt_t = option_at_t(tau)

        # Target delta (short option => hedge with +delta of underlying)
        target_delta = -greek(opt_t, Delta(), data)

        if strategy isa StopLossHedge
            if i > 1 && abs(shares - target_delta) > strategy.trigger_level
                rebalance_mask[i] = true
            end
        end

        if rebalance_mask[i]
            delta_trade = target_delta - shares
            shares = target_delta
            cost_trade = strategy isa DiscreteDeltaHedge ?
                strategy.transaction_cost * abs(delta_trade) * S : 0.01 * abs(delta_trade) * S
            cost_total += cost_trade
            cash -= delta_trade * S + cost_trade
            n_trades += 1
        end

        if i < n_obs
            opt_value = price(opt_t, hedging_engine, data)
            mtm = -opt_value + shares * S - (premium - cash)
            push!(pnls, mtm)
            cum = mtm
            max_dd = max(max_dd, premium - cum)
        end
    end

    final_cash = cash + shares * spots[end]
    total_pnl = final_cash - payoff_at_expiry
    variance_pnl = length(pnls) > 1 ? var(pnls) : 0.0

    HedgePerformance(total_pnl, cost_total, variance_pnl, max_dd, n_trades)
end
