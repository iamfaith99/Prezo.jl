# Prezo.jl

A Julia package for pricing financial derivatives using multiple numerical methods.

## Overview

Prezo.jl provides a flexible framework for pricing European and American options using various computational approaches:

- **Analytical Methods**: Black-Scholes-Merton formula
- **Discrete-Time Methods**: Binomial tree models
- **Simulation Methods**: Monte Carlo and Least Squares Monte Carlo (Longstaff-Schwartz)

## Features

- Multiple pricing engines with consistent interfaces
- Support for both European and American options
- Flexible basis function framework for LSM methods
- Market data structure with spot, rate, volatility, and dividend yield
- Path visualization for Monte Carlo simulations

## Installation

```julia
using Pkg
Pkg.add("Prezo")
```

## Quick Start

```julia
using Prezo

# Define market data
data = MarketData(100.0, 0.05, 0.2, 0.0)

# Create an option
option = EuropeanCall(100.0, 1.0)

# Price using Black-Scholes
bs_engine = BlackScholes()
price(option, bs_engine, data)
```

## Package Structure

- `MarketData`: Market parameters (spot price, interest rate, volatility, dividend yield)
- `Options`: European and American calls and puts
- `Pricing Engines`: BlackScholes, Binomial, MonteCarlo, LongstaffSchwartz
- `Basis Functions`: Laguerre, Chebyshev, Power, and Hermite polynomials for LSM methods

## Index

```@index
```
