# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Prezo.jl is a Julia package implementing an options pricing engine for financial derivatives. It provides multiple pricing engines (Black-Scholes-Merton, Binomial, and Monte Carlo) for European and American options.

## Commands for Development

### Basic Julia Package Commands
- **Activate environment**: `julia --project=Prezo`
- **Install dependencies**: In Julia REPL: `using Pkg; Pkg.instantiate()`
- **Run examples**: `julia --project=Prezo examples.jl` (from Prezo directory)
- **Start REPL with package**: `julia --project=Prezo -e "using Prezo"`

### Testing
- No formal test suite exists (no `test/runtests.jl` file)
- Use `examples.jl` for basic European option validation
- Use `spot_sensitivity_test.jl` for spot price sensitivity analysis
- Use `american_options_test.jl` for American option validation with Longstaff-Schwartz

### Documentation
- **Build documentation**: `cd Prezo/docs && julia --project=.. make.jl`
- **View documentation**: Open `Prezo/docs/build/index.html` in browser
- **Documentation framework**: Uses Documenter.jl v1.15.0
- All public functions and types have comprehensive docstrings
- Documentation includes user guides, API reference, and examples

## Code Architecture

### Package Structure
```
Prezo/
├── src/
│   ├── Prezo.jl         # Main module, exports, includes
│   ├── data.jl          # Market data structures
│   ├── options.jl       # Option type definitions and payoff functions
│   └── engines.jl       # Pricing engines and algorithms
├── docs/
│   ├── make.jl          # Documentation build script
│   └── src/             # Documentation source files (markdown)
│       ├── index.md
│       ├── api.md
│       └── guide/
├── examples.jl          # Usage examples and validation
└── Project.toml         # Package dependencies
```

### Core Components

**Market Data (`data.jl`)**
- `MarketData` struct with fields: `spot`, `rate`, `vol`, `div`

**Option Types (`options.jl`)**
- Abstract type hierarchy: `VanillaOption` → `EuropeanOption`/`AmericanOption`
- Concrete types: `EuropeanCall`, `EuropeanPut`, `AmericanCall`, `AmericanPut`
- Each option has `strike` and `expiry` fields
- `payoff()` functions compute option payoffs at expiration
- Broadcasting enabled via `Base.broadcastable()` implementations

**Pricing Engines (`engines.jl`)**
- `BlackScholes`: Analytical formula for European options
- `Binomial`: Discrete-time tree method with configurable steps
- `MonteCarlo`: Simulation-based pricing with path generation
- `LongstaffSchwartz`: Least Squares Monte Carlo for American options

### Key Patterns

**Multiple Dispatch Architecture**
The pricing system uses Julia's multiple dispatch with signature:
```julia
price(option::OptionType, engine::EngineType, data::MarketData)
```

**Asset Path Generation**
Three Monte Carlo path generation methods:
- `asset_paths()`: Returns `(reps, steps+1)` matrix
- `asset_paths_col()`: Returns `(steps+1, reps)` matrix
- `asset_paths_ax()`: Uses `axes()` for iteration (most Julian)

### Dependencies
- **Distributions.jl**: Normal distribution for Monte Carlo and Black-Scholes
- **Plots.jl**: Path visualization and plotting functionality
- **Random.jl**: Random number generation for simulations
- **LinearAlgebra.jl**: Matrix operations for LSM regression
- **Statistics.jl**: Statistical functions for Monte Carlo
- **Documenter.jl**: Documentation generation (dev dependency)

### American Option Pricing

**Longstaff-Schwartz Algorithm**
- Least Squares Monte Carlo method for American options
- Backward induction with regression-based continuation value estimation
- Uses Laguerre polynomial basis functions for regression (default)
- Constructor: `LongstaffSchwartz(steps, paths, basis_order)` or `LongstaffSchwartz(steps, paths)` (default basis_order=3)
- Works with both `AmericanPut` and `AmericanCall` options
- Enhanced versions available with multiple basis functions:
  - `LaguerreLSM`: Laguerre polynomials (recommended)
  - `ChebyshevLSM`: Chebyshev polynomials
  - `PowerLSM`: Simple power basis
  - `HermiteLSM`: Hermite polynomials

## Documentation Guidelines

When modifying or extending the codebase:

1. **Add docstrings** to all public functions and types using Julia docstring format
2. **Include examples** in docstrings showing typical usage
3. **Rebuild documentation** after changes: `cd Prezo/docs && julia --project=.. make.jl`
4. **Update user guides** in `docs/src/guide/` for new features
5. **Cross-reference** related functions using `` [`FunctionName`](@ref) ``

### Docstring Format
```julia
"""
    function_name(arg1, arg2; kwarg=default)

Brief description of what the function does.

# Arguments
- `arg1::Type`: Description of argument 1
- `arg2::Type`: Description of argument 2

# Keyword Arguments
- `kwarg::Type`: Description (default: `default`)

# Returns
Description of return value.

# Examples
```julia
# Example usage
result = function_name(value1, value2)
```

See also: [`RelatedFunction`](@ref)
"""
```

## Important Notes

- Package uses `AbstractFloat` for option parameters to allow flexibility
- All pricing engines implement different numerical methods for validation
- Monte Carlo engine includes path plotting capabilities for visualization
- American options implemented via Longstaff-Schwartz Least Squares Monte Carlo
- Dividend handling available in market data and used in American option pricing
- American call options ≈ European call options when dividends are zero
- American put options show significant early exercise premiums
- All public API is fully documented with docstrings and examples