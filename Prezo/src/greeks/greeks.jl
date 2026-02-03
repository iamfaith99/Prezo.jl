"""
    greek(option, greek_type, data_or_engine)

Calculate option Greeks (sensitivities) using analytical or numerical methods.

This module provides comprehensive Greek calculations for vanilla options,
supporting both analytical formulas (Black-Scholes) and numerical methods
(finite differences) for all pricing engines.

# Supported Greeks

## First Order (Price Sensitivities)
- `Delta`: Price sensitivity to underlying spot price
- `Theta`: Price sensitivity to time decay (per year)
- `Vega`: Price sensitivity to volatility (per 1% vol change)
- `Rho`: Price sensitivity to interest rate (per 1% rate change)

## Second Order (Convexity)
- `Gamma`: Delta sensitivity to spot price (second derivative)
- `Vanna`: Delta sensitivity to volatility (cross derivative)
- `Vomma`: Vega sensitivity to volatility (second derivative)
- `Charm`: Delta sensitivity to time (cross derivative)
- `Veta`: Vega sensitivity to time (cross derivative)
- `Speed`: Gamma sensitivity to spot (third derivative)
- `Color`: Gamma sensitivity to time (cross derivative)
- `Ultima`: Vomma sensitivity to vol (third derivative)

## Dividend Greeks
- `Phi` or `RhoDiv`: Sensitivity to dividend yield

# API

## Analytical Greeks (Black-Scholes only)
```julia
greek(option::EuropeanOption, ::Delta, data::MarketData) -> Float64
greek(option::EuropeanOption, ::Gamma, data::MarketData) -> Float64
greek(option::EuropeanOption, ::Vega, data::MarketData) -> Float64
```

## Numerical Greeks (any engine)
```julia
greek(option::VanillaOption, ::Delta, engine::PricingEngine, data::MarketData; h=1e-4) -> Float64
greek(option::VanillaOption, ::Gamma, engine::PricingEngine, data::MarketData; h=1e-4) -> Float64
greek(option::VanillaOption, ::Vega, engine::PricingEngine, data::MarketData; h=1e-4) -> Float64
```

# Examples

## Analytical Greeks
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)

# Calculate delta
delta_call = greek(call, Delta(), data)

# Calculate all major Greeks at once
all_greeks = greeks(call, data)  # Returns Dict{Greek, Float64}
```

## Numerical Greeks for American Options
```julia
put = AmericanPut(100.0, 1.0)
engine = Binomial(500)

# Delta via finite differences
delta_am = greek(put, Delta(), engine, data)

# Vega with custom step size
vega_am = greek(put, Vega(), engine, data, h=0.01)
```

## Portfolio Greeks
```julia
positions = [
    Position(EuropeanCall(100.0, 1.0), 10),   # Long 10 calls
    Position(EuropeanPut(100.0, 1.0), -5),    # Short 5 puts
]
portfolio = portfolio_greeks(positions, BlackScholes(), data)
# Returns: Dict with aggregated Delta, Gamma, Vega, etc.
```

See also: [`Delta`](@ref), [`Gamma`](@ref), [`Vega`](@ref), [`Theta`](@ref),
[`Rho`](@ref), [`greeks`](@ref), [`portfolio_greeks`](@ref)
"""
module Greeks

# Import types from parent module
using ..Prezo: VanillaOption, EuropeanOption, EuropeanCall, EuropeanPut
using ..Prezo: AmericanOption, AmericanCall, AmericanPut
using ..Prezo: MarketData, BlackScholes, PricingEngine, MonteCarlo
using ..Prezo: price, payoff

# Import exotic option types
using ..Prezo: ExoticOption, AsianOption, BarrierOption, LookbackOption
using ..Prezo: ArithmeticAsianCall, ArithmeticAsianPut, GeometricAsianCall, GeometricAsianPut
using ..Prezo: KnockOutCall, KnockOutPut, KnockInCall, KnockInPut
using ..Prezo: FixedStrikeLookbackCall, FixedStrikeLookbackPut
using ..Prezo: FloatingStrikeLookbackCall, FloatingStrikeLookbackPut

using Distributions
using LinearAlgebra

export Greek, FirstOrderGreek, SecondOrderGreek, ThirdOrderGreek
export Delta, Gamma, Theta, Vega, Rho, Phi, RhoDiv
export Vanna, Charm, Vomma, Veta, Speed, Color, Ultima
export greek, greeks, all_greeks
export portfolio_greeks, Position
export numerical_greek, finite_difference
export hedge_ratio, is_delta_neutral, is_gamma_neutral
export pnl_attribution, scenario_analysis

# Exotic option Greeks exports
export get_expiry, with_shorter_expiry
export barrier_proximity, is_near_barrier
export mc_optimal_step_spot, mc_optimal_step_vol, mc_optimal_step_rate

# Include type definitions first
include("types.jl")

# Declare the main greek function (will be defined in submodules)
function greek end
function greeks end
function all_greeks end
function numerical_greek end
function portfolio_greeks end
function hedge_ratio end
function is_delta_neutral end
function is_gamma_neutral end
function pnl_attribution end
function scenario_analysis end

# Exotic option helper functions
function get_expiry end
function with_shorter_expiry end
function barrier_proximity end
function is_near_barrier end

# Include submodules
include("analytical.jl")
include("numerical.jl")
include("portfolio.jl")
include("exotic.jl")

end # module Greeks
