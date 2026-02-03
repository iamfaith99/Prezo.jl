using Distributions: Normal, cdf
using Statistics

norm_cdf(x) = cdf(Normal(0.0, 1.0), x)

"""
    PricingEngine

Abstract base type for all option pricing engines.

Concrete subtypes include:
- [`BlackScholes`](@ref): Analytical Black-Scholes-Merton formula
- [`Binomial`](@ref): Binomial tree method
- [`MonteCarlo`](@ref): Monte Carlo simulation
- [`MonteCarloAntithetic`](@ref): Monte Carlo with antithetic variates
- [`MonteCarloStratified`](@ref): Monte Carlo with stratified sampling

See also: [`price`](@ref)
"""
abstract type PricingEngine end

"""
    Binomial(steps)

Binomial tree pricing engine for European and American options.

The binomial model discretizes time into `steps` intervals and builds a recombining tree
of possible asset prices. Converges to Black-Scholes for European options as steps → ∞.

# Fields
- `steps::Int`: Number of time steps in the tree
"""
struct Binomial <: PricingEngine
    steps::Int
end

"""
    price(option, engine, data)

Price an option using the specified pricing engine and market data.

This is the main pricing interface using Julia's multiple dispatch. The appropriate
pricing method is selected based on the option type and engine type.

# Arguments
- `option::VanillaOption`: The option contract to price
- `engine`: The pricing engine (BlackScholes, Binomial, MonteCarlo, etc.)
- `data::MarketData`: Market parameters

# Returns
The option price as a Float64.
"""
function price(option::EuropeanOption, engine::Binomial, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data
    steps = engine.steps

    dt = expiry / steps
    u = exp((rate - div) * dt + vol * sqrt(dt))
    d = exp((rate - div) * dt - vol * sqrt(dt))
    pu = (exp((rate - div) * dt) - d) / (u - d)
    pd = 1 - pu
    disc = exp(-rate * dt)

    s = zeros(steps + 1)
    x = zeros(steps + 1)

    @inbounds for i in 1:steps+1
        s[i] = spot * u^(steps + 1 - i) * d^(i - 1)
        x[i] = payoff(option, s[i])
    end

    for j in steps:-1:1
        @inbounds for i in 1:j
            x[i] = disc * (pu * x[i] + pd * x[i+1])
        end
    end

    return x[1]
end

function price(option::AmericanOption, engine::Binomial, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data
    steps = engine.steps

    dt = expiry / steps
    u = exp((rate - div) * dt + vol * sqrt(dt))
    d = exp((rate - div) * dt - vol * sqrt(dt))
    pu = (exp((rate - div) * dt) - d) / (u - d)
    pd = 1 - pu
    disc = exp(-rate * dt)

    s = zeros(steps + 1, steps + 1)
    x = zeros(steps + 1, steps + 1)

    for i in 0:steps
        for j in 0:i
            s[j+1, i+1] = spot * u^(i - j) * d^j
        end
    end

    for j in 1:steps+1
        x[j, steps+1] = payoff(option, s[j, steps+1])
    end

    for i in steps:-1:1
        for j in 1:i
            continuation = disc * (pu * x[j, i+1] + pd * x[j+1, i+1])
            exercise = payoff(option, s[j, i])
            x[j, i] = max(continuation, exercise)
        end
    end

    return x[1, 1]
end

"""
    BlackScholes()

Black-Scholes-Merton analytical pricing engine for European options.
"""
struct BlackScholes <: PricingEngine end

function price(option::EuropeanCall, engine::BlackScholes, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data

    d1 = (log(spot / strike) + (rate - div + 0.5 * vol^2) * expiry) / (vol * sqrt(expiry))
    d2 = d1 - vol * sqrt(expiry)

    return spot * exp(-div * expiry) * norm_cdf(d1) - strike * exp(-rate * expiry) * norm_cdf(d2)
end

function price(option::EuropeanPut, engine::BlackScholes, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data

    d1 = (log(spot / strike) + (rate - div + 0.5 * vol^2) * expiry) / (vol * sqrt(expiry))
    d2 = d1 - vol * sqrt(expiry)

    return strike * exp(-rate * expiry) * norm_cdf(-d2) - spot * exp(-div * expiry) * norm_cdf(-d1)
end

"""
    MonteCarlo(steps, reps)

Monte Carlo simulation engine for European option pricing.

# Fields
- `steps::Int`: Number of time steps per simulation path
- `reps::Int`: Number of simulation paths (replications)
"""
struct MonteCarlo <: PricingEngine
    steps::Int
    reps::Int
end

function price(option::EuropeanOption, engine::MonteCarlo, data::MarketData)
    (; expiry) = option
    (; spot, rate, vol) = data

    paths = asset_paths(engine, spot, rate, vol, expiry)
    payoffs = payoff.(option, paths[:, end])

    return exp(-rate * expiry) * mean(payoffs)
end

struct MonteCarloAntithetic <: PricingEngine
    steps::Int
    reps::Int
end

function price(option::EuropeanOption, engine::MonteCarloAntithetic, data::MarketData)
    (; expiry) = option
    (; spot, rate, vol) = data
    (; steps, reps) = engine

    paths = asset_paths_antithetic(steps, reps, spot, rate, vol, expiry)
    payoffs = payoff.(option, paths[end, :])

    return exp(-rate * expiry) * mean(payoffs)
end

struct MonteCarloStratified <: PricingEngine
    steps::Int
    reps::Int
end

function price(option::EuropeanOption, engine::MonteCarloStratified, data::MarketData)
    (; expiry) = option
    (; spot, rate, vol) = data
    (; steps, reps) = engine

    paths = asset_paths_stratified(steps, reps, spot, rate, vol, expiry)
    payoffs = payoff.(option, paths[end, :])

    return exp(-rate * expiry) * mean(payoffs)
end

# ============================================================================
# Exotic Option Pricing (Monte Carlo)
# ============================================================================

# ----------------------------------------------------------------------------
# Asian Options
# ----------------------------------------------------------------------------

"""
    price(option::AsianOption, engine::MonteCarlo, data::MarketData)

Price an Asian option using Monte Carlo simulation.

Generates paths and computes the average price at the specified averaging times.
"""
function price(option::AsianOption, engine::MonteCarlo, data::MarketData)
    (; expiry, averaging_times) = option
    (; spot, rate, vol, div) = data
    (; steps, reps) = engine

    # Generate paths with enough granularity
    dt = expiry / steps
    time_grid = collect(0:dt:expiry)

    # Find indices closest to averaging times
    avg_indices = [argmin(abs.(time_grid .- t)) for t in averaging_times]

    # Generate paths (column-major: steps+1 x reps)
    paths = asset_paths_col(engine, spot, rate - div, vol, expiry)

    # Compute payoffs for each path
    payoffs = zeros(reps)
    for j in 1:reps
        # Extract prices at averaging times
        avg_prices = [paths[i, j] for i in avg_indices]
        payoffs[j] = payoff(option, avg_prices)
    end

    return exp(-rate * expiry) * mean(payoffs)
end

"""
    price(option::GeometricAsianCall, engine::BlackScholes, data::MarketData)

Analytical pricing for geometric Asian call option.

The geometric average of log-normal variables is log-normal, enabling closed-form pricing.
"""
function price(option::GeometricAsianCall, engine::BlackScholes, data::MarketData)
    (; strike, expiry, averaging_times) = option
    (; spot, rate, vol, div) = data

    n = length(averaging_times)

    # Adjusted parameters for geometric average
    # For continuous averaging approximation
    σ_adj = vol * sqrt((2 * n + 1) / (6 * (n + 1)))
    r_adj = 0.5 * (rate - div - vol^2 / 2 + σ_adj^2)

    d1 = (log(spot / strike) + (r_adj + 0.5 * σ_adj^2) * expiry) / (σ_adj * sqrt(expiry))
    d2 = d1 - σ_adj * sqrt(expiry)

    return exp(-rate * expiry) * (spot * exp(r_adj * expiry) * norm_cdf(d1) - strike * norm_cdf(d2))
end

"""
    price(option::GeometricAsianPut, engine::BlackScholes, data::MarketData)

Analytical pricing for geometric Asian put option.
"""
function price(option::GeometricAsianPut, engine::BlackScholes, data::MarketData)
    (; strike, expiry, averaging_times) = option
    (; spot, rate, vol, div) = data

    n = length(averaging_times)

    σ_adj = vol * sqrt((2 * n + 1) / (6 * (n + 1)))
    r_adj = 0.5 * (rate - div - vol^2 / 2 + σ_adj^2)

    d1 = (log(spot / strike) + (r_adj + 0.5 * σ_adj^2) * expiry) / (σ_adj * sqrt(expiry))
    d2 = d1 - σ_adj * sqrt(expiry)

    return exp(-rate * expiry) * (strike * norm_cdf(-d2) - spot * exp(r_adj * expiry) * norm_cdf(-d1))
end

# ----------------------------------------------------------------------------
# Barrier Options
# ----------------------------------------------------------------------------

"""
    price(option::BarrierOption, engine::MonteCarlo, data::MarketData)

Price a barrier option using Monte Carlo simulation.

Generates paths and checks for barrier breaches along each path.
"""
function price(option::BarrierOption, engine::MonteCarlo, data::MarketData)
    (; expiry) = option
    (; spot, rate, vol, div) = data
    (; steps, reps) = engine

    # Generate paths (column-major: steps+1 x reps)
    paths = asset_paths_col(engine, spot, rate - div, vol, expiry)

    # Compute payoffs for each path
    payoffs = zeros(reps)
    for j in 1:reps
        path = paths[:, j]
        payoffs[j] = payoff(option, path)
    end

    return exp(-rate * expiry) * mean(payoffs)
end

"""
    price(option::KnockOutCall, engine::BlackScholes, data::MarketData)

Analytical pricing for knock-out call option (continuous monitoring).

Uses the reflection principle for barrier option valuation.
"""
function price(option::KnockOutCall, engine::BlackScholes, data::MarketData)
    (; strike, expiry, barrier, barrier_type) = option
    (; spot, rate, vol, div) = data

    # Standard European call price
    vanilla_call = price(EuropeanCall(strike, expiry), engine, data)

    if barrier_type == :up_and_out
        if spot >= barrier
            return 0.0  # Already knocked out
        end

        # Rebate = 0 assumed
        λ = (rate - div + vol^2 / 2) / vol^2
        y = log(barrier^2 / (spot * strike)) / (vol * sqrt(expiry)) + λ * vol * sqrt(expiry)
        x1 = log(spot / barrier) / (vol * sqrt(expiry)) + λ * vol * sqrt(expiry)
        y1 = log(barrier / spot) / (vol * sqrt(expiry)) + λ * vol * sqrt(expiry)

        knock_in = spot * exp(-div * expiry) * (barrier / spot)^(2 * λ) * norm_cdf(y) -
                   strike * exp(-rate * expiry) * (barrier / spot)^(2 * λ - 2) * norm_cdf(y - vol * sqrt(expiry))

        return max(0.0, vanilla_call - knock_in)

    else  # :down_and_out
        if spot <= barrier
            return 0.0  # Already knocked out
        end

        λ = (rate - div + vol^2 / 2) / vol^2
        y1 = log(barrier / spot) / (vol * sqrt(expiry)) + λ * vol * sqrt(expiry)

        knock_in = spot * exp(-div * expiry) * (barrier / spot)^(2 * λ) * norm_cdf(y1) -
                   strike * exp(-rate * expiry) * (barrier / spot)^(2 * λ - 2) * norm_cdf(y1 - vol * sqrt(expiry))

        return max(0.0, vanilla_call - knock_in)
    end
end

"""
    price(option::KnockOutPut, engine::BlackScholes, data::MarketData)

Analytical pricing for knock-out put option (continuous monitoring).
"""
function price(option::KnockOutPut, engine::BlackScholes, data::MarketData)
    (; strike, expiry, barrier, barrier_type) = option
    (; spot, rate, vol, div) = data

    vanilla_put = price(EuropeanPut(strike, expiry), engine, data)

    if barrier_type == :down_and_out
        if spot <= barrier
            return 0.0
        end

        λ = (rate - div + vol^2 / 2) / vol^2
        y1 = log(barrier / spot) / (vol * sqrt(expiry)) + λ * vol * sqrt(expiry)

        knock_in = -spot * exp(-div * expiry) * (barrier / spot)^(2 * λ) * norm_cdf(-y1) +
                   strike * exp(-rate * expiry) * (barrier / spot)^(2 * λ - 2) * norm_cdf(-y1 + vol * sqrt(expiry))

        return max(0.0, vanilla_put - knock_in)

    else  # :up_and_out
        if spot >= barrier
            return 0.0
        end

        λ = (rate - div + vol^2 / 2) / vol^2
        y = log(barrier^2 / (spot * strike)) / (vol * sqrt(expiry)) + λ * vol * sqrt(expiry)

        knock_in = -spot * exp(-div * expiry) * (barrier / spot)^(2 * λ) * norm_cdf(-y) +
                   strike * exp(-rate * expiry) * (barrier / spot)^(2 * λ - 2) * norm_cdf(-y + vol * sqrt(expiry))

        return max(0.0, vanilla_put - knock_in)
    end
end

"""
    price(option::KnockInCall, engine::BlackScholes, data::MarketData)

Analytical pricing for knock-in call option using in-out parity.

Knock-in + Knock-out = Vanilla
"""
function price(option::KnockInCall, engine::BlackScholes, data::MarketData)
    (; strike, expiry, barrier, barrier_type) = option

    vanilla = price(EuropeanCall(strike, expiry), engine, data)

    # Map knock-in type to knock-out type
    ko_type = barrier_type == :up_and_in ? :up_and_out : :down_and_out
    knock_out = price(KnockOutCall(strike, expiry, barrier, ko_type), engine, data)

    return vanilla - knock_out
end

"""
    price(option::KnockInPut, engine::BlackScholes, data::MarketData)

Analytical pricing for knock-in put option using in-out parity.
"""
function price(option::KnockInPut, engine::BlackScholes, data::MarketData)
    (; strike, expiry, barrier, barrier_type) = option

    vanilla = price(EuropeanPut(strike, expiry), engine, data)

    ko_type = barrier_type == :up_and_in ? :up_and_out : :down_and_out
    knock_out = price(KnockOutPut(strike, expiry, barrier, ko_type), engine, data)

    return vanilla - knock_out
end

# ----------------------------------------------------------------------------
# Lookback Options
# ----------------------------------------------------------------------------

"""
    price(option::LookbackOption, engine::MonteCarlo, data::MarketData)

Price a lookback option using Monte Carlo simulation.

Generates paths and tracks the maximum/minimum price along each path.
"""
function price(option::LookbackOption, engine::MonteCarlo, data::MarketData)
    (; expiry) = option
    (; spot, rate, vol, div) = data
    (; steps, reps) = engine

    # Generate paths (column-major: steps+1 x reps)
    paths = asset_paths_col(engine, spot, rate - div, vol, expiry)

    # Compute payoffs for each path
    payoffs = zeros(reps)
    for j in 1:reps
        path = paths[:, j]
        payoffs[j] = payoff(option, path)
    end

    return exp(-rate * expiry) * mean(payoffs)
end

"""
    price(option::FloatingStrikeLookbackCall, engine::BlackScholes, data::MarketData)

Price floating-strike lookback call using Monte Carlo fallback.

Payoff: S_T - S_min

# TODO: Implement Goldman-Sosin-Gatto analytical formula
"""
function price(option::FloatingStrikeLookbackCall, ::BlackScholes, data::MarketData)
    # Analytical lookback formulas are complex; use MC with high granularity
    return price(option, MonteCarlo(252, 10000), data)
end

"""
    price(option::FloatingStrikeLookbackPut, engine::BlackScholes, data::MarketData)

Price floating-strike lookback put using Monte Carlo fallback.

Payoff: S_max - S_T

# TODO: Implement Goldman-Sosin-Gatto analytical formula
"""
function price(option::FloatingStrikeLookbackPut, ::BlackScholes, data::MarketData)
    return price(option, MonteCarlo(252, 10000), data)
end

"""
    price(option::FixedStrikeLookbackCall, engine::BlackScholes, data::MarketData)

Price fixed-strike lookback call using Monte Carlo fallback.

Payoff: max(0, S_max - K)

# TODO: Implement Conze-Viswanathan analytical formula
"""
function price(option::FixedStrikeLookbackCall, ::BlackScholes, data::MarketData)
    return price(option, MonteCarlo(252, 10000), data)
end

"""
    price(option::FixedStrikeLookbackPut, engine::BlackScholes, data::MarketData)

Price fixed-strike lookback put using Monte Carlo fallback.

Payoff: max(0, K - S_min)

# TODO: Implement Conze-Viswanathan analytical formula
"""
function price(option::FixedStrikeLookbackPut, ::BlackScholes, data::MarketData)
    return price(option, MonteCarlo(252, 10000), data)
end
