# Julia Documentation with Documenter.jl

Expertise in creating comprehensive documentation for Julia packages using Documenter.jl.

## Docstring Best Practices

### Standard Format

```julia
"""
    function_name(arg1, arg2; kwarg=default)

Brief one-line description.

Detailed explanation of what the function does, including any important
algorithms, assumptions, or mathematical background.

# Arguments
- `arg1::Type`: Description of argument 1
- `arg2::Type`: Description of argument 2

# Keyword Arguments
- `kwarg::Type`: Description (default: `default`)

# Returns
Description of return value and its type.

# Examples
```julia
# Basic usage
result = function_name(value1, value2)

# With keyword argument
result = function_name(value1, value2; kwarg=custom_value)
```

# Notes
Any important caveats, performance considerations, or related information.

See also: [`related_function`](@ref), [`OtherType`](@ref)
"""
function function_name(arg1, arg2; kwarg=default)
    # implementation
end
```

### Math in Docstrings

Use LaTeX math notation:

```julia
"""
Black-Scholes formula for European call options:

```math
C = S e^{-qT} \\Phi(d_1) - K e^{-rT} \\Phi(d_2)
```

where:
```math
\\begin{aligned}
d_1 &= \\frac{\\ln(S/K) + (r - q + \\sigma^2/2)T}{\\sigma\\sqrt{T}} \\\\
d_2 &= d_1 - \\sigma\\sqrt{T}
\\end{aligned}
```
"""
```

### Code Examples in Docstrings

Always include runnable examples:

```julia
"""
# Examples
```jldoctest
julia> data = MarketData(100.0, 0.05, 0.2, 0.0)
MarketData(100.0, 0.05, 0.2, 0.0)

julia> option = EuropeanCall(100.0, 1.0)
EuropeanCall(100.0, 1.0)

julia> price(option, BlackScholes(), data)
10.45
```
"""
```

Use `jldoctest` for automatic testing of examples.

## Documentation Structure

### Typical Layout

```
docs/
├── make.jl              # Build script
├── Project.toml         # Doc dependencies
└── src/
    ├── index.md         # Home page
    ├── api.md           # API Reference
    ├── guide/
    │   ├── getting_started.md
    │   ├── basic_usage.md
    │   ├── advanced.md
    │   └── examples.md
    └── assets/
        ├── logo.png
        └── custom.css
```

### make.jl Template

```julia
using Documenter
using Prezo

makedocs(;
    modules=[Prezo],
    authors="Your Name <email@example.com>",
    repo="https://github.com/username/Prezo.jl/blob/{commit}{path}#{line}",
    sitename="Prezo.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://username.github.io/Prezo.jl",
        assets=String[],
        mathengine=Documenter.MathJax3(),
    ),
    pages=[
        "Home" => "index.md",
        "User Guide" => [
            "Getting Started" => "guide/getting_started.md",
            "Basic Usage" => "guide/basic_usage.md",
            "Advanced Topics" => "guide/advanced.md",
            "Examples" => "guide/examples.md",
        ],
        "API Reference" => "api.md",
    ],
    warnonly=[:missing_docs],
)

deploydocs(;
    repo="github.com/username/Prezo.jl",
    devbranch="main",
)
```

### index.md Template

```markdown
# Prezo.jl

Documentation for Prezo.jl - A Julia package for pricing financial derivatives.

## Overview

Prezo.jl provides fast, accurate pricing for European and American options using:
- Black-Scholes analytical formulas
- Binomial trees
- Monte Carlo simulation
- Longstaff-Schwartz algorithm for American options

## Quick Start

```julia
using Prezo

# Market data
data = MarketData(100.0, 0.05, 0.2, 0.0)

# Option
option = EuropeanCall(100.0, 1.0)

# Price
price(option, BlackScholes(), data)
```

## Features

- **Multiple pricing engines**: Black-Scholes, Binomial, Monte Carlo, LSM
- **American options**: Binomial and Longstaff-Schwartz
- **Type-stable**: Optimized for performance
- **Well-tested**: Comprehensive test suite

## Installation

```julia
using Pkg
Pkg.add("Prezo")
```

## Contents

```@contents
Pages = [
    "guide/getting_started.md",
    "guide/basic_usage.md",
    "api.md"
]
Depth = 2
```
```

### api.md Template

```markdown
# API Reference

## Market Data

```@docs
MarketData
```

## Options

```@docs
VanillaOption
EuropeanOption
AmericanOption
EuropeanCall
EuropeanPut
AmericanCall
AmericanPut
payoff
```

## Pricing Engines

```@docs
BlackScholes
Binomial
MonteCarlo
LongstaffSchwartz
price
```

## Index

```@index
```
```

## Cross-References

Link to other documented items:

```julia
"""
See also: [`BlackScholes`](@ref), [`price`](@ref)
"""
```

Link to sections:

```markdown
See the [User Guide](@ref) for examples.
```

Link to external URLs:

```markdown
Based on [Black-Scholes model](https://en.wikipedia.org/wiki/Black–Scholes_model).
```

## Building Documentation

### Local Build

```bash
cd docs
julia --project=.. make.jl
```

### View Locally

```bash
# Open in browser
open build/index.html
```

### Serve with LiveServer.jl

```julia
using LiveServer
serve(dir="docs/build")
```

## Deploying Documentation

### GitHub Actions (Recommended)

`.github/workflows/docs.yml`:

```yaml
name: Documentation

on:
  push:
    branches:
      - main
    tags: '*'
  pull_request:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: julia-actions/setup-julia@latest
        with:
          version: '1'
      - name: Install dependencies
        run: julia --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
      - name: Build and deploy
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          DOCUMENTER_KEY: ${{ secrets.DOCUMENTER_KEY }}
        run: julia --project=docs/ docs/make.jl
```

### Setting up Documenter Key

```julia
using DocumenterTools
DocumenterTools.genkeys(user="username", repo="Prezo.jl")
```

Follow instructions to add SSH key to GitHub.

## Advanced Features

### Custom Themes

`docs/src/assets/custom.css`:

```css
/* Custom colors */
:root {
    --documenter-primary: #4285f4;
}

/* Code blocks */
pre {
    background-color: #f5f5f5;
    border-radius: 4px;
}
```

### LaTeX Macros

```julia
makedocs(;
    ...
    format=Documenter.HTML(;
        mathengine=Documenter.MathJax3(Dict(
            :TeX => Dict(
                :Macros => Dict(
                    "call" => "C",
                    "put" => "P",
                    "spot" => "S",
                ),
            ),
        )),
    ),
)
```

### Hiding Implementation Details

```julia
"""
Public function.
"""
function public_function()
    _internal_helper()
end

# No docstring = won't appear in docs
function _internal_helper()
    # implementation
end
```

### Organizing Large APIs

Group related functions:

```markdown
## Pricing

```@docs
price
BlackScholes
Binomial
```

## Greeks

```@docs
greeks
delta
gamma
```
```

## Documentation Checklist

For a well-documented package:

1. ✓ Docstring for every exported function/type
2. ✓ Examples in every docstring
3. ✓ Getting Started guide
4. ✓ API reference with all exports
5. ✓ Mathematical formulas where relevant
6. ✓ Cross-references between related items
7. ✓ Build documentation locally to check
8. ✓ Set up automatic deployment
9. ✓ Add badges to README
10. ✓ Keep examples up-to-date with code

## Badges for README.md

```markdown
[![Docs Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://username.github.io/Prezo.jl/stable/)
[![Docs Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://username.github.io/Prezo.jl/dev/)
```

## Documenting Examples

Create example notebooks in `docs/src/examples/`:

```julia
# # Pricing European Options
#
# This example demonstrates basic option pricing.

using Prezo

# Define market data
data = MarketData(100.0, 0.05, 0.2, 0.0)

# Create option
option = EuropeanCall(100.0, 1.0)

# Price with different engines
bs_price = price(option, BlackScholes(), data)
binom_price = price(option, Binomial(100), data)

println("Black-Scholes: $bs_price")
println("Binomial: $binom_price")
```

Convert with Literate.jl:

```julia
using Literate
Literate.markdown("examples/pricing.jl", "docs/src/examples/")
```

## Further Reading

- Documenter.jl docs: https://juliadocs.github.io/Documenter.jl/
- Julia docstring style guide
- DocumenterTools.jl for utilities
