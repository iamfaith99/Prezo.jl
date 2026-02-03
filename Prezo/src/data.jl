"""
    MarketData(spot, rate, vol, div)

Market parameters for option pricing.

# Fields
- `spot::Float64`: Current spot price of the underlying asset
- `rate::Float64`: Risk-free interest rate (annualized, as decimal, e.g., 0.05 for 5%)
- `vol::Float64`: Volatility (annualized standard deviation, as decimal, e.g., 0.2 for 20%)
- `div::Float64`: Dividend yield (continuous, annualized, as decimal)

# Examples
```julia
# Non-dividend paying stock
data = MarketData(100.0, 0.05, 0.2, 0.0)

# Stock with 2% dividend yield
data = MarketData(100.0, 0.05, 0.2, 0.02)
```
"""
struct MarketData
    spot::Float64
    rate::Float64
    vol::Float64
    div::Float64
end

"""
    DiscreteDividend(time, amount)

A single discrete dividend payment.

# Fields
- `time::Float64`: Time of dividend payment (in years from now)
- `amount::Float64`: Dividend amount (absolute, not yield)

# Examples
```julia
# Dividend of \$2 in 3 months
div1 = DiscreteDividend(0.25, 2.0)

# Dividend of \$2.50 in 9 months
div2 = DiscreteDividend(0.75, 2.5)
```
"""
struct DiscreteDividend
    time::Float64
    amount::Float64
end

"""
    MarketDataExt(spot, rate, vol, div, discrete_dividends)

Extended market parameters with support for discrete dividends.

# Fields
- `spot::Float64`: Current spot price of the underlying asset
- `rate::Float64`: Risk-free interest rate (annualized, as decimal)
- `vol::Float64`: Volatility (annualized standard deviation, as decimal)
- `div::Float64`: Continuous dividend yield (annualized, as decimal)
- `discrete_dividends::Vector{DiscreteDividend}`: Vector of discrete dividend payments

# Examples
```julia
# Stock with continuous yield + discrete dividends
divs = [DiscreteDividend(0.25, 2.0), DiscreteDividend(0.75, 2.5)]
data = MarketDataExt(100.0, 0.05, 0.2, 0.01, divs)

# No discrete dividends (equivalent to MarketData)
data = MarketDataExt(100.0, 0.05, 0.2, 0.02, DiscreteDividend[])
```

See also: [`MarketData`](@ref), [`DiscreteDividend`](@ref)
"""
struct MarketDataExt
    spot::Float64
    rate::Float64
    vol::Float64
    div::Float64
    discrete_dividends::Vector{DiscreteDividend}
end

# Convenience constructor with no discrete dividends
MarketDataExt(spot, rate, vol, div) = MarketDataExt(spot, rate, vol, div, DiscreteDividend[])

# Convert MarketData to MarketDataExt
MarketDataExt(data::MarketData) = MarketDataExt(data.spot, data.rate, data.vol, data.div, DiscreteDividend[])

"""
    present_value_dividends(data::MarketDataExt, T::Float64)

Calculate the present value of discrete dividends up to time T.

# Arguments
- `data::MarketDataExt`: Market data with discrete dividends
- `T::Float64`: Time horizon (in years)

# Returns
Present value of all discrete dividends occurring before time T.
"""
function present_value_dividends(data::MarketDataExt, T::Float64)
    pv = 0.0
    for d in data.discrete_dividends
        if d.time <= T
            pv += d.amount * exp(-data.rate * d.time)
        end
    end
    return pv
end

"""
    adjusted_spot(data::MarketDataExt, T::Float64)

Calculate spot price adjusted for discrete dividends (for pricing purposes).

# Arguments
- `data::MarketDataExt`: Market data with discrete dividends
- `T::Float64`: Time horizon (in years)

# Returns
Spot price minus present value of dividends.
"""
function adjusted_spot(data::MarketDataExt, T::Float64)
    return data.spot - present_value_dividends(data, T)
end
