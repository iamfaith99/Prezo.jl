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