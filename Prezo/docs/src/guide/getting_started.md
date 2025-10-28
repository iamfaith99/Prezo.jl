# Getting Started

This guide will help you get started with Prezo.jl for pricing options.

## Installation

Add Prezo.jl to your Julia environment:

```julia
using Pkg
Pkg.add("Prezo")
```

## Basic Workflow

The basic workflow for pricing an option involves three steps:

1. **Define market data**
2. **Create an option contract**
3. **Select a pricing engine and compute the price**

### Example: Pricing a European Call

```julia
using Prezo

# Step 1: Define market data
# MarketData(spot, rate, volatility, dividend)
data = MarketData(100.0, 0.05, 0.2, 0.0)

# Step 2: Create option
# EuropeanCall(strike, expiry)
option = EuropeanCall(100.0, 1.0)

# Step 3: Price using different engines
bs_price = price(option, BlackScholes(), data)
binomial_price = price(option, Binomial(100), data)
mc_price = price(option, MonteCarlo(10000, 100), data)

println("Black-Scholes: $bs_price")
println("Binomial: $binomial_price")
println("Monte Carlo: $mc_price")
```

### Example: Pricing an American Put

American options require simulation-based methods that account for early exercise:

```julia
using Prezo

# Market data
data = MarketData(100.0, 0.05, 0.2, 0.0)

# American put option
option = AmericanPut(100.0, 1.0)

# Price using Longstaff-Schwartz method
# LongstaffSchwartz(time_steps, num_paths, basis_order)
lsm = LongstaffSchwartz(50, 10000, 3)
american_price = price(option, lsm, data)

println("American Put Price: $american_price")
```

## Multiple Dispatch

Prezo.jl uses Julia's multiple dispatch to provide a clean interface:

```julia
price(option::OptionType, engine::EngineType, data::MarketData)
```

This design allows you to easily switch between different option types and pricing engines while keeping the same calling convention.
