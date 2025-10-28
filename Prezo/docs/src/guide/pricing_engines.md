# Pricing Engines

Prezo.jl provides multiple pricing engines, each with different strengths and use cases.

## Overview

| Engine | Method | European | American | Speed | Accuracy |
|--------|--------|----------|----------|-------|----------|
| BlackScholes | Analytical | ✓ | ✗ | Very Fast | Exact |
| Binomial | Tree | ✓ | ✓ | Fast | Convergent |
| MonteCarlo | Simulation | ✓ | ✗ | Moderate | Convergent |
| LongstaffSchwartz | LSM | ✗ | ✓ | Slow | Convergent |

## Black-Scholes

The Black-Scholes-Merton model provides exact closed-form solutions for European options.

```julia
engine = BlackScholes()
price(option, engine, data)
```

**Advantages:**
- Exact analytical solution
- Very fast computation
- No convergence issues

**Limitations:**
- European options only
- Assumes constant volatility and interest rates
- Log-normal price distribution

## Binomial Tree

The binomial model discretizes time and builds a tree of possible price paths.

```julia
# Binomial(num_steps)
engine = Binomial(100)
price(option, engine, data)
```

**Parameters:**
- `num_steps`: Number of time steps (more steps = higher accuracy)

**Advantages:**
- Handles both European and American options
- Intuitive and easy to understand
- Converges to Black-Scholes for European options

**Typical values:**
- Quick estimate: 50-100 steps
- Production: 500-1000 steps

## Monte Carlo

Monte Carlo simulation generates random price paths and averages discounted payoffs.

```julia
# MonteCarlo(num_paths, num_steps)
engine = MonteCarlo(10000, 100)
price(option, engine, data)
```

**Parameters:**
- `num_paths`: Number of simulated paths
- `num_steps`: Time steps per path

**Advantages:**
- Natural for path-dependent options
- Easily extensible to complex payoffs
- Can visualize sample paths

**Path Visualization:**
```julia
plot_paths(option, MonteCarlo(100, 100), data, num_paths=10)
```

**Typical values:**
- Quick estimate: 1,000-10,000 paths
- Production: 100,000+ paths

## Longstaff-Schwartz (LSM)

The Least Squares Monte Carlo method prices American options by estimating continuation values using regression.

```julia
# LongstaffSchwartz(num_steps, num_paths, basis_order)
engine = LongstaffSchwartz(50, 10000, 3)
price(option, engine, data)
```

**Parameters:**
- `num_steps`: Time discretization steps
- `num_paths`: Number of simulated paths
- `basis_order`: Polynomial order for regression (default: 3)

**Advantages:**
- Handles American early exercise
- Flexible regression framework
- Works for complex payoffs

**Typical values:**
- Time steps: 50-100
- Paths: 10,000-100,000
- Basis order: 2-4

### Basis Functions

Prezo.jl supports multiple polynomial basis families:

```julia
# Laguerre polynomials (default, recommended)
lsm = LaguerreLSM(50, 10000, 3)

# Chebyshev polynomials
lsm = ChebyshevLSM(50, 10000, 3)

# Simple power basis
lsm = PowerLSM(50, 10000, 3)

# Hermite polynomials
lsm = HermiteLSM(50, 10000, 3)
```

## Choosing an Engine

### For European Options:
1. **Black-Scholes**: Use this when possible (fastest and exact)
2. **Binomial**: When you need to verify BS or modify the model
3. **Monte Carlo**: For path-dependent payoffs or visualizations

### For American Options:
1. **Binomial**: Good balance of speed and accuracy
2. **Longstaff-Schwartz**: More flexible, better for complex payoffs

### Convergence Testing

Always test convergence when using numerical methods:

```julia
using Prezo

data = MarketData(100.0, 0.05, 0.2, 0.0)
option = AmericanPut(100.0, 1.0)

# Test different numbers of paths
for n in [1000, 5000, 10000, 50000]
    engine = LongstaffSchwartz(50, n, 3)
    p = price(option, engine, data)
    println("Paths: $n, Price: $p")
end
```
