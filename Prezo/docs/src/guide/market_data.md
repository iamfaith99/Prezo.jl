# Market Data

The `MarketData` struct encapsulates all market parameters needed for option pricing.

## Structure

```julia
struct MarketData
    spot::Float64    # Current spot price of the underlying asset
    rate::Float64    # Risk-free interest rate (annualized)
    vol::Float64     # Volatility (annualized standard deviation)
    div::Float64     # Dividend yield (continuous, annualized)
end
```

## Creating Market Data

```julia
using Prezo

# Basic example: no dividends
data = MarketData(100.0, 0.05, 0.2, 0.0)

# With dividends
data_div = MarketData(100.0, 0.05, 0.2, 0.02)
```

## Parameters

### Spot Price
The current market price of the underlying asset.

```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
println(data.spot)  # 100.0
```

### Risk-Free Rate
The annualized risk-free interest rate, typically based on government bonds. Should be expressed as a decimal (e.g., 0.05 for 5%).

```julia
rate = 0.05  # 5% per annum
```

### Volatility
The annualized standard deviation of the asset's returns. This is a key input to option pricing models.

```julia
vol = 0.2  # 20% annualized volatility
```

### Dividend Yield
The continuous dividend yield of the underlying asset. For non-dividend paying stocks, set to 0.0.

```julia
div = 0.0    # No dividends
div = 0.02   # 2% dividend yield
```

## Impact on Pricing

### Dividends and American Options
Dividends have an important impact on American option pricing:

- **American Calls**: Early exercise becomes optimal when dividends are high enough
- **American Puts**: Early exercise premium increases with dividends

For non-dividend paying stocks (`div = 0.0`), an American call has the same value as a European call, since early exercise is never optimal.

### Volatility
Higher volatility increases option values for both calls and puts, as it increases the probability of large price movements.

### Interest Rates
Interest rates affect:
- The present value of future payoffs
- The forward price of the underlying asset
- Generally increase call values and decrease put values
