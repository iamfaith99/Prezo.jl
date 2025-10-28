using Distributions
using Plots
using Random
using LinearAlgebra
using Statistics

normcdf(x) = cdf(Normal(0.0, 1.0), x)


"""
    Binomial(steps)

Binomial tree pricing engine for European and American options.

The binomial model discretizes time into `steps` intervals and builds a recombining tree
of possible asset prices. Converges to Black-Scholes for European options as steps → ∞.

# Fields
- `steps::Int`: Number of time steps in the tree

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
option = EuropeanCall(100.0, 1.0)

# Quick estimate with 50 steps
engine = Binomial(50)
price(option, engine, data)

# More accurate with 500 steps
engine = Binomial(500)
price(option, engine, data)
```

See also: [`price`](@ref), [`BlackScholes`](@ref)
"""
struct Binomial
    steps::Int
end

"""
    price(option, engine, data)

Price an option using the specified pricing engine and market data.

This is the main pricing interface using Julia's multiple dispatch. The appropriate
pricing method is selected based on the option type and engine type.

# Arguments
- `option::VanillaOption`: The option contract to price
- `engine`: The pricing engine (BlackScholes, Binomial, MonteCarlo, LongstaffSchwartz, etc.)
- `data::MarketData`: Market parameters

# Returns
The option price as a Float64.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)

# European call with different engines
call = EuropeanCall(100.0, 1.0)
price(call, BlackScholes(), data)
price(call, Binomial(100), data)
price(call, MonteCarlo(100, 10000), data)

# American put
put = AmericanPut(100.0, 1.0)
price(put, Binomial(100), data)
price(put, LongstaffSchwartz(50, 10000), data)
```

See also: [`BlackScholes`](@ref), [`Binomial`](@ref), [`MonteCarlo`](@ref), [`LongstaffSchwartz`](@ref)
"""
function price(option::EuropeanOption, engine::Binomial, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data
    steps = engine.steps

    dt = expiry / steps
    u = exp(rate * dt + vol * sqrt(dt))
    d = exp(rate * dt - vol * sqrt(dt))
    pu = (exp(rate * dt) - d) / (u - d)
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
            x[i] = disc * (pu * x[i] + pd * x[i + 1])
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

    # Asset prices at each node
    s = zeros(steps + 1, steps + 1)
    # Option values at each node
    x = zeros(steps + 1, steps + 1)

    # Initialize asset prices
    for i in 0:steps
        for j in 0:i
            s[j + 1, i + 1] = spot * u^(i - j) * d^j
        end
    end

    # Terminal payoffs
    for j in 1:steps+1
        x[j, steps + 1] = payoff(option, s[j, steps + 1])
    end

    # Backward induction with early exercise
    for i in steps:-1:1
        for j in 1:i
            # Continuation value
            continuation = disc * (pu * x[j, i + 1] + pd * x[j + 1, i + 1])
            # Immediate exercise value
            exercise = payoff(option, s[j, i])
            # Take maximum for American option
            x[j, i] = max(continuation, exercise)
        end
    end

    return x[1, 1]
end


"""
    BlackScholes()

Black-Scholes-Merton analytical pricing engine for European options.

Provides exact closed-form solutions using the Black-Scholes formula. This is the
fastest and most accurate method for European options under the Black-Scholes assumptions
(constant volatility, log-normal price distribution, no dividends).

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
put = EuropeanPut(100.0, 1.0)

engine = BlackScholes()
call_price = price(call, engine, data)
put_price = price(put, engine, data)
```

See also: [`price`](@ref), [`Binomial`](@ref)
"""
struct BlackScholes end

function price(option::EuropeanCall, engine::BlackScholes, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data

    d1 = (log(spot / strike) + (rate + 0.5 * vol^2) * expiry) / (vol * sqrt(expiry))
    d2 = d1 - vol * sqrt(expiry)

    price = spot * normcdf(d1) - strike * exp(-rate * expiry) * normcdf(d2)

    return price 
end

function price(option::EuropeanPut, engine::BlackScholes, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol) = data

    d1 = (log(spot / strike) + (rate + 0.5 * vol^2) * expiry) / (vol * sqrt(expiry))
    d2 = d1 - vol * sqrt(expiry)

    price = strike * exp(-rate * expiry) * normcdf(-d2) - spot * normcdf(-d1)

    return price 
end


"""
    MonteCarlo(steps, reps)

Monte Carlo simulation engine for European option pricing.

Generates random price paths and computes the discounted expected payoff.
Useful for path-dependent options and provides visualization capabilities.

# Fields
- `steps::Int`: Number of time steps per simulation path
- `reps::Int`: Number of simulation paths (replications)

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
option = EuropeanCall(100.0, 1.0)

# Quick estimate with 1,000 paths
engine = MonteCarlo(100, 1000)
price(option, engine, data)

# Production quality with 100,000 paths
engine = MonteCarlo(100, 100000)
price(option, engine, data)

# Visualize sample paths
paths = asset_paths(engine, 100.0, 0.05, 0.2, 1.0)
```

See also: [`asset_paths`](@ref), [`asset_paths_col`](@ref), [`plot_paths`](@ref)
"""
struct MonteCarlo
    steps::Int
    reps::Int
end

"""
    asset_paths(engine::MonteCarlo, spot, rate, vol, expiry)

Generate asset price paths using geometric Brownian motion.

Returns a matrix of simulated asset prices with dimensions `(reps, steps+1)`.

# Arguments
- `engine::MonteCarlo`: Monte Carlo engine with steps and reps
- `spot`: Initial spot price
- `rate`: Risk-free interest rate
- `vol`: Volatility
- `expiry`: Time to expiration

# Returns
Matrix of size `(reps, steps+1)` where each row is a simulated path.

# Examples
```julia
engine = MonteCarlo(100, 1000)
paths = asset_paths(engine, 100.0, 0.05, 0.2, 1.0)
size(paths)  # (1000, 101)
```

See also: [`asset_paths_col`](@ref), [`asset_paths_ax`](@ref)
"""
function asset_paths(engine::MonteCarlo, spot, rate, vol, expiry)
    (; steps, reps) = engine

    dt = expiry / steps
    nudt = (rate - 0.5 * vol^2) * dt
    sidt = vol * sqrt(dt)
    paths = zeros(reps, steps + 1)
    paths[:, 1] .= spot

    @inbounds for i in 1:reps
        @inbounds for j in 2:steps + 1
            z = rand(Normal(0.0, 1.0))
            paths[i, j] = paths[i, j - 1] * exp(nudt + sidt * z)
        end
    end

    return paths
end


"""
    asset_paths_col(engine::MonteCarlo, spot, rate, vol, expiry)

Generate asset price paths with column-major layout.

Returns a matrix of simulated asset prices with dimensions `(steps+1, reps)`.
This layout is preferred for time-based operations.

# Returns
Matrix of size `(steps+1, reps)` where each column is a simulated path.

See also: [`asset_paths`](@ref), [`asset_paths_ax`](@ref)
"""
function asset_paths_col(engine::MonteCarlo, spot, rate, vol, expiry)
    (; steps, reps) = engine

    dt = expiry / steps
    nudt = (rate - 0.5 * vol^2) * dt
    sidt = vol * sqrt(dt)
    paths = zeros(steps + 1, reps)
    paths[1, :] .= spot

    @inbounds for i in 1:reps
        @inbounds for j in 2:steps +1
            z = rand(Normal(0.0, 1.0))
            paths[j, i] = paths[j - 1, i] * exp(nudt + sidt * z)
        end
    end

    return paths
end


"""
    asset_paths_ax(engine::MonteCarlo, spot, rate, vol, expiry)

Generate asset price paths using Julian axis-based iteration.

More idiomatic Julia implementation using `axes()` for iteration.

See also: [`asset_paths`](@ref), [`asset_paths_col`](@ref)
"""
function asset_paths_ax(engine::MonteCarlo, spot, rate, vol, expiry)
    (; steps, reps) = engine

    dt = expiry / steps
    nudt = (rate - 0.5 * vol^2) * dt
    sidt = vol * sqrt(dt)
    paths = zeros(steps + 1, reps)
    paths[1, :] .= spot

    for j in axes(paths, 2), i in 2:last(axes(paths, 1))
        z = rand(Normal(0.0, 1.0))
        paths[i, j] = paths[i - 1, j] * exp(nudt + sidt * z)
    end

    return paths

end

"""
    plot_paths(paths, num)

Visualize simulated asset price paths.

# Arguments
- `paths`: Matrix of simulated paths (from `asset_paths`)
- `num`: Number of paths to plot

# Examples
```julia
engine = MonteCarlo(100, 1000)
paths = asset_paths(engine, 100.0, 0.05, 0.2, 1.0)
plot_paths(paths, 10)  # Plot first 10 paths
```

See also: [`asset_paths`](@ref), [`MonteCarlo`](@ref)
"""
function plot_paths(paths, num)
    # The second dimension is steps+1 columns, so:
    steps = size(paths, 2) - 1

    # Plot the first path
    plot(0:steps, paths[1, :], label="", legend=false)

    # Add the remaining paths (2 through 10) to the same plot
    for i in 2:num
        plot!(0:steps, paths[i, :], label="", legend=false)
    end

    xaxis!("Time step")
    yaxis!("Asset price")
    title!("First 10 Simulated Paths")
end

function price(option::EuropeanOption, engine::MonteCarlo, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol) = data
    (; steps, reps) = engine

    paths = asset_paths(engine, spot, rate, vol, expiry)
    payoffs = payoff.(option, paths[:, end])

    return exp(-rate * expiry) * mean(payoffs)
end


"""
    LongstaffSchwartz(steps, reps, basis_order=3)

Least Squares Monte Carlo engine for American option pricing.

Implements the Longstaff-Schwartz algorithm which uses regression to estimate
continuation values at each time step, enabling optimal early exercise decisions.

# Fields
- `steps::Int`: Number of time discretization steps
- `reps::Int`: Number of simulation paths
- `basis_order::Int`: Order of Laguerre polynomial basis (default: 3)

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
option = AmericanPut(100.0, 1.0)

# Standard configuration
engine = LongstaffSchwartz(50, 10000)
price(option, engine, data)

# Custom basis order
engine = LongstaffSchwartz(50, 10000, 4)
price(option, engine, data)
```

# References
Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation:
a simple least-squares approach. The Review of Financial Studies, 14(1), 113-147.

See also: [`LaguerreLSM`](@ref), [`EnhancedLongstaffSchwartz`](@ref)
"""
struct LongstaffSchwartz
    steps::Int
    reps::Int
    basis_order::Int
end

# Default constructor with reasonable basis order
LongstaffSchwartz(steps::Int, reps::Int) = LongstaffSchwartz(steps, reps, 3)

function price(option::AmericanOption, engine::LongstaffSchwartz, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data
    (; steps, reps, basis_order) = engine

    dt = expiry / steps
    disc = exp(-rate * dt)

    # Generate all asset price paths
    paths = asset_paths_col(MonteCarlo(steps, reps), spot, rate - div, vol, expiry)

    # Initialize cash flow matrix
    # cash_flows[t, i] = discounted cash flow from path i at time t
    cash_flows = zeros(steps + 1, reps)

    # At expiration, cash flow is the payoff
    for i in 1:reps
        cash_flows[end, i] = payoff(option, paths[end, i])
    end

    # Work backwards through time
    for t in steps:-1:2
        # Current asset prices
        S_t = paths[t, :]

        # Find in-the-money paths
        itm_payoffs = payoff.(option, S_t)
        itm_indices = findall(x -> x > 0, itm_payoffs)

        if length(itm_indices) == 0
            # No paths are in-the-money, continue
            cash_flows[t, :] = cash_flows[t + 1, :] * disc
            continue
        end

        # Asset prices for in-the-money paths
        S_itm = S_t[itm_indices]

        # Continuation values (discounted future cash flows)
        continuation_values = cash_flows[t + 1, itm_indices] * disc

        # Create basis functions for regression
        # Using Laguerre polynomials as in original Longstaff-Schwartz paper
        X = create_basis_functions(S_itm, basis_order)

        # Regression: continuation_value = X * β + ε
        β = X \ continuation_values

        # Predicted continuation values
        predicted_continuation = X * β

        # Exercise decision: exercise if immediate payoff > continuation value
        immediate_payoffs = itm_payoffs[itm_indices]
        exercise_indices = immediate_payoffs .> predicted_continuation

        # Update cash flows
        cash_flows[t, :] = cash_flows[t + 1, :] * disc

        for (idx, itm_idx) in enumerate(itm_indices)
            if exercise_indices[idx]
                # Exercise at time t
                cash_flows[t, itm_idx] = immediate_payoffs[idx]
                # Zero out future cash flows from this path
                cash_flows[t + 1:end, itm_idx] .= 0.0
            end
        end
    end

    # At t=1 (first time step), decide whether to exercise immediately
    immediate_payoffs_t1 = payoff.(option, paths[1, :])
    continuation_values_t1 = cash_flows[2, :] * disc

    final_cash_flows = max.(immediate_payoffs_t1, continuation_values_t1)

    # Discount back to present value
    return mean(final_cash_flows) * disc
end

function create_basis_functions(S::Vector{Float64}, order::Int)
    n = length(S)
    X = ones(n, order + 1)  # Include constant term

    # Normalized asset prices for better numerical stability
    S_norm = S / 100.0

    for j in 1:order
        if j == 1
            # L₁(x) = 1 - x
            X[:, j + 1] = 1.0 .- S_norm
        elseif j == 2
            # L₂(x) = 1 - 2x + x²/2
            X[:, j + 1] = 1.0 .- 2.0 * S_norm .+ (S_norm .^ 2) / 2.0
        elseif j == 3
            # L₃(x) = 1 - 3x + 3x²/2 - x³/6
            X[:, j + 1] = 1.0 .- 3.0 * S_norm .+ 1.5 * (S_norm .^ 2) .- (S_norm .^ 3) / 6.0
        else
            # For higher orders, use recursive relation
            # L_{n+1}(x) = (2n+1-x)L_n(x)/n+1 - nL_{n-1}(x)/n+1
            X[:, j + 1] = ((2 * j - 1 .- S_norm) .* X[:, j] .- (j - 1) * X[:, j - 1]) / j
        end
    end

    return X
end


# Enhanced Julian LSM Implementation with Multiple Basis Functions
# ==============================================================

"""
    BasisFunction

Abstract base type for polynomial basis functions used in LSM regression.

Subtypes include:
- [`LaguerreBasis`](@ref): Laguerre polynomials (recommended)
- [`ChebyshevBasis`](@ref): Chebyshev polynomials
- [`PowerBasis`](@ref): Simple power basis
- [`HermiteBasis`](@ref): Hermite polynomials
"""
abstract type BasisFunction end

"""
    LaguerreBasis(order, normalization)

Laguerre polynomial basis functions for LSM regression.

Laguerre polynomials are orthogonal and provide good numerical stability.
This is the recommended basis for most applications.

# Fields
- `order::Int`: Maximum polynomial order
- `normalization::Float64`: Normalization factor for asset prices

See also: [`EnhancedLongstaffSchwartz`](@ref), [`LaguerreLSM`](@ref)
"""
struct LaguerreBasis <: BasisFunction
    order::Int
    normalization::Float64
end

"""
    ChebyshevBasis(order, domain_min, domain_max)

Chebyshev polynomial basis functions for LSM regression.

Chebyshev polynomials are orthogonal on [-1, 1] and provide excellent
approximation properties. Asset prices are mapped to this domain.

# Fields
- `order::Int`: Maximum polynomial order
- `domain_min::Float64`: Minimum of expected price domain
- `domain_max::Float64`: Maximum of expected price domain

See also: [`EnhancedLongstaffSchwartz`](@ref), [`ChebyshevLSM`](@ref)
"""
struct ChebyshevBasis <: BasisFunction
    order::Int
    domain_min::Float64
    domain_max::Float64
end

"""
    PowerBasis(order, normalization)

Simple power basis (1, x, x², x³, ...) for LSM regression.

Simple but can suffer from numerical instability for higher orders
due to non-orthogonality.

# Fields
- `order::Int`: Maximum polynomial order
- `normalization::Float64`: Normalization factor for asset prices

See also: [`EnhancedLongstaffSchwartz`](@ref), [`PowerLSM`](@ref)
"""
struct PowerBasis <: BasisFunction
    order::Int
    normalization::Float64
end

"""
    HermiteBasis(order, mean, std)

Hermite polynomial basis functions for LSM regression.

Hermite polynomials are orthogonal with respect to Gaussian weight functions.
Asset prices are standardized before applying the polynomials.

# Fields
- `order::Int`: Maximum polynomial order
- `mean::Float64`: Mean for standardization
- `std::Float64`: Standard deviation for standardization

See also: [`EnhancedLongstaffSchwartz`](@ref), [`HermiteLSM`](@ref)
"""
struct HermiteBasis <: BasisFunction
    order::Int
    mean::Float64
    std::Float64
end

# Multiple dispatch for different basis types
function evaluate_basis(basis::LaguerreBasis, S::AbstractVector{T}) where T
    (; order, normalization) = basis
    n = length(S)
    X = ones(T, n, order + 1)

    S_norm = S ./ normalization

    for j in 1:order
        if j == 1
            X[:, j + 1] = @. 1.0 - S_norm
        elseif j == 2
            X[:, j + 1] = @. 1.0 - 2.0 * S_norm + (S_norm^2) / 2.0
        elseif j == 3
            X[:, j + 1] = @. 1.0 - 3.0 * S_norm + 1.5 * (S_norm^2) - (S_norm^3) / 6.0
        else
            X[:, j + 1] = @. ((2 * j - 1 - S_norm) * X[:, j] - (j - 1) * X[:, j - 1]) / j
        end
    end

    return X
end

function evaluate_basis(basis::ChebyshevBasis, S::AbstractVector{T}) where T
    (; order, domain_min, domain_max) = basis
    n = length(S)
    X = ones(T, n, order + 1)

    S_mapped = @. 2.0 * (S - domain_min) / (domain_max - domain_min) - 1.0

    if order >= 1
        X[:, 2] = S_mapped
    end

    for j in 2:order
        X[:, j + 1] = @. 2.0 * S_mapped * X[:, j] - X[:, j - 1]
    end

    return X
end

function evaluate_basis(basis::PowerBasis, S::AbstractVector{T}) where T
    (; order, normalization) = basis
    n = length(S)
    X = ones(T, n, order + 1)

    S_norm = S ./ normalization

    for j in 1:order
        X[:, j + 1] = S_norm .^ j
    end

    return X
end

function evaluate_basis(basis::HermiteBasis, S::AbstractVector{T}) where T
    (; order, mean, std) = basis
    n = length(S)
    X = ones(T, n, order + 1)

    S_std = @. (S - mean) / std

    if order >= 1
        X[:, 2] = @. 2.0 * S_std
    end

    for j in 2:order
        X[:, j + 1] = @. 2.0 * S_std * X[:, j] - 2.0 * j * X[:, j - 1]
    end

    return X
end

# Trait-based design for numerical stability
abstract type BasisProperties end
struct Orthogonal <: BasisProperties end
struct NonOrthogonal <: BasisProperties end

basis_properties(::Type{LaguerreBasis}) = Orthogonal()
basis_properties(::Type{ChebyshevBasis}) = Orthogonal()
basis_properties(::Type{PowerBasis}) = NonOrthogonal()
basis_properties(::Type{HermiteBasis}) = Orthogonal()

function fit_continuation_value(X::Matrix, y::Vector, ::Orthogonal)
    F = qr(X)
    return F \ y
end

function fit_continuation_value(X::Matrix, y::Vector, ::NonOrthogonal)
    return X \ y
end

function fit_continuation_value(basis::BasisFunction, X::Matrix, y::Vector)
    return fit_continuation_value(X, y, basis_properties(typeof(basis)))
end

# Functor pattern - make basis functions callable
(basis::BasisFunction)(S::AbstractVector) = evaluate_basis(basis, S)

"""
    EnhancedLongstaffSchwartz{B, T}

Enhanced LSM engine with flexible basis functions and improved numerical stability.

This is a parametric type that allows different basis functions and precision levels.
Use the convenience constructors [`LaguerreLSM`](@ref), [`ChebyshevLSM`](@ref),
[`PowerLSM`](@ref), or [`HermiteLSM`](@ref) instead of constructing directly.

# Type Parameters
- `B <: BasisFunction`: Type of basis function
- `T <: AbstractFloat`: Floating point precision (default: Float64)

# Fields
- `basis::B`: Basis function instance
- `steps::Int`: Time discretization steps
- `reps::Int`: Number of simulation paths
- `min_regression_paths::Int`: Minimum ITM paths required for regression

See also: [`LaguerreLSM`](@ref), [`ChebyshevLSM`](@ref), [`PowerLSM`](@ref), [`HermiteLSM`](@ref)
"""
struct EnhancedLongstaffSchwartz{B <: BasisFunction, T <: AbstractFloat}
    basis::B
    steps::Int
    reps::Int
    min_regression_paths::Int

    function EnhancedLongstaffSchwartz{B, T}(basis::B, steps::Int, reps::Int, min_regression_paths::Int=50) where {B <: BasisFunction, T <: AbstractFloat}
        steps > 0 || throw(ArgumentError("steps must be positive"))
        reps > 0 || throw(ArgumentError("reps must be positive"))
        min_regression_paths > 0 || throw(ArgumentError("min_regression_paths must be positive"))
        new{B, T}(basis, steps, reps, min_regression_paths)
    end
end

function EnhancedLongstaffSchwartz(basis::BasisFunction, steps::Int, reps::Int, min_regression_paths::Int=50)
    return EnhancedLongstaffSchwartz{typeof(basis), Float64}(basis, steps, reps, min_regression_paths)
end

"""
    LaguerreLSM(order, steps, reps; normalization=100.0)

Create an enhanced LSM engine with Laguerre polynomial basis (recommended).

# Arguments
- `order::Int`: Polynomial order (typically 2-4)
- `steps::Int`: Time discretization steps
- `reps::Int`: Number of simulation paths

# Keyword Arguments
- `normalization::Float64`: Normalization factor for prices (default: 100.0)

# Examples
```julia
engine = LaguerreLSM(3, 50, 10000)
option = AmericanPut(100.0, 1.0)
data = MarketData(100.0, 0.05, 0.2, 0.0)
price(option, engine, data)
```
"""
function LaguerreLSM(order::Int, steps::Int, reps::Int; normalization=100.0)
    basis = LaguerreBasis(order, normalization)
    return EnhancedLongstaffSchwartz(basis, steps, reps)
end

"""
    ChebyshevLSM(order, steps, reps; domain=(30.0, 50.0))

Create an enhanced LSM engine with Chebyshev polynomial basis.

# Arguments
- `order::Int`: Polynomial order
- `steps::Int`: Time discretization steps
- `reps::Int`: Number of simulation paths

# Keyword Arguments
- `domain::Tuple{Float64,Float64}`: Expected price range (min, max)
"""
function ChebyshevLSM(order::Int, steps::Int, reps::Int; domain=(30.0, 50.0))
    basis = ChebyshevBasis(order, domain[1], domain[2])
    return EnhancedLongstaffSchwartz(basis, steps, reps)
end

"""
    PowerLSM(order, steps, reps; normalization=40.0)

Create an enhanced LSM engine with simple power basis.

# Arguments
- `order::Int`: Polynomial order
- `steps::Int`: Time discretization steps
- `reps::Int`: Number of simulation paths

# Keyword Arguments
- `normalization::Float64`: Normalization factor for prices
"""
function PowerLSM(order::Int, steps::Int, reps::Int; normalization=40.0)
    basis = PowerBasis(order, normalization)
    return EnhancedLongstaffSchwartz(basis, steps, reps)
end

"""
    HermiteLSM(order, steps, reps; mean=40.0, std=10.0)

Create an enhanced LSM engine with Hermite polynomial basis.

# Arguments
- `order::Int`: Polynomial order
- `steps::Int`: Time discretization steps
- `reps::Int`: Number of simulation paths

# Keyword Arguments
- `mean::Float64`: Mean for standardization
- `std::Float64`: Standard deviation for standardization
"""
function HermiteLSM(order::Int, steps::Int, reps::Int; mean=40.0, std=10.0)
    basis = HermiteBasis(order, mean, std)
    return EnhancedLongstaffSchwartz(basis, steps, reps)
end

# Enhanced pricing function with better numerical stability
function price(option::AmericanOption, engine::EnhancedLongstaffSchwartz{B, T}, data::MarketData) where {B, T}
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data
    (; basis, steps, reps, min_regression_paths) = engine

    dt = T(expiry / steps)
    disc = exp(-T(rate) * dt)

    paths = asset_paths_col(MonteCarlo(steps, reps), T(spot), T(rate - div), T(vol), T(expiry))

    cash_flows = Matrix{T}(undef, steps + 1, reps)

    # Terminal payoffs using broadcasting
    @. cash_flows[end, :] = max(zero(T), payoff_value(option, paths[end, :]))

    # Backward induction with improved regression
    for t in steps:-1:2
        S_t = @view paths[t, :]

        immediate_payoffs = payoff_value.(option, S_t)
        itm_mask = immediate_payoffs .> zero(T)
        itm_count = sum(itm_mask)

        if itm_count < min_regression_paths
            @. cash_flows[t, :] = cash_flows[t + 1, :] * disc
            continue
        end

        S_itm = S_t[itm_mask]
        continuation_values = cash_flows[t + 1, itm_mask] .* disc
        immediate_itm = immediate_payoffs[itm_mask]

        X = basis(S_itm)

        try
            β = fit_continuation_value(basis, X, continuation_values)
            predicted_continuation = X * β

            exercise_mask = immediate_itm .> predicted_continuation

            @. cash_flows[t, :] = cash_flows[t + 1, :] * disc

            itm_indices = findall(itm_mask)
            exercise_indices = itm_indices[exercise_mask]

            for idx in exercise_indices
                cash_flows[t, idx] = immediate_payoffs[idx]
                @. cash_flows[t + 1:end, idx] = zero(T)
            end

        catch e
            @warn "Regression failed at time step $t, using continuation value" exception=e
            @. cash_flows[t, :] = cash_flows[t + 1, :] * disc
        end
    end

    immediate_payoffs_t1 = payoff_value.(option, paths[1, :])
    continuation_values_t1 = cash_flows[2, :] .* disc

    final_cash_flows = max.(immediate_payoffs_t1, continuation_values_t1)

    return T(disc) * mean(final_cash_flows)
end

# Helper function for type-stable payoffs
payoff_value(option::AmericanPut, spot::T) where T = max(zero(T), T(option.strike) - spot)
payoff_value(option::AmericanCall, spot::T) where T = max(zero(T), spot - T(option.strike))

"""
    validate_american_option_price(american_price, european_price, option_type, div; tolerance=1e-6)

Validate that American option price satisfies basic pricing relationships.

Checks that:
1. American price ≥ European price (early exercise premium)
2. For American calls with zero dividends, price ≈ European price

# Arguments
- `american_price::T`: American option price
- `european_price::T`: Corresponding European option price
- `option_type::Type{<:AmericanOption}`: Type of American option
- `div::T`: Dividend yield
- `tolerance::T`: Numerical tolerance (default: 1e-6)

# Returns
`true` if validations pass, `false` otherwise (with warning messages).

# Examples
```julia
am_price = price(AmericanPut(100.0, 1.0), LaguerreLSM(3, 50, 10000), data)
eu_price = price(EuropeanPut(100.0, 1.0), BlackScholes(), data)
validate_american_option_price(am_price, eu_price, AmericanPut, 0.0)
```
"""
function validate_american_option_price(american_price::T, european_price::T,
                                      option_type::Type{<:AmericanOption},
                                      div::T, tolerance::T=T(1e-6)) where T

    if american_price < european_price - tolerance
        @warn "American option price ($american_price) < European price ($european_price)"
        return false
    end

    if option_type <: AmericanCall && abs(div) < tolerance
        if american_price > european_price + tolerance
            @warn "American call with zero dividends shows significant premium: $(american_price - european_price)"
            return false
        end
    end

    return true
end

# European LSM Implementation
# ===========================
# For European options, LSM becomes a regression method to estimate option value
# using basis functions on the terminal asset price

"""
    EuropeanLongstaffSchwartz{B, T}

LSM-based regression engine for European option pricing.

Uses basis function regression on terminal asset prices to estimate option values.
This is primarily for educational purposes as Black-Scholes is more efficient.

# Type Parameters
- `B <: BasisFunction`: Type of basis function
- `T <: AbstractFloat`: Floating point precision

See also: [`EuropeanLaguerreLSM`](@ref), [`BlackScholes`](@ref)
"""
struct EuropeanLongstaffSchwartz{B <: BasisFunction, T <: AbstractFloat}
    basis::B
    reps::Int
    min_regression_paths::Int

    function EuropeanLongstaffSchwartz{B, T}(basis::B, reps::Int, min_regression_paths::Int=100) where {B <: BasisFunction, T <: AbstractFloat}
        reps > 0 || throw(ArgumentError("reps must be positive"))
        min_regression_paths > 0 || throw(ArgumentError("min_regression_paths must be positive"))
        new{B, T}(basis, reps, min_regression_paths)
    end
end

function EuropeanLongstaffSchwartz(basis::BasisFunction, reps::Int, min_regression_paths::Int=100)
    return EuropeanLongstaffSchwartz{typeof(basis), Float64}(basis, reps, min_regression_paths)
end

"""
    EuropeanLaguerreLSM(order, reps; normalization=100.0)

Create a European LSM engine with Laguerre basis (for educational purposes).

For production use, prefer [`BlackScholes`](@ref) for European options.

See also: [`EuropeanLongstaffSchwartz`](@ref), [`BlackScholes`](@ref)
"""
function EuropeanLaguerreLSM(order::Int, reps::Int; normalization=100.0)
    basis = LaguerreBasis(order, normalization)
    return EuropeanLongstaffSchwartz(basis, reps)
end

"""
    EuropeanChebyshevLSM(order, reps; domain=(30.0, 50.0))

Create a European LSM engine with Chebyshev basis.

See also: [`EuropeanLongstaffSchwartz`](@ref)
"""
function EuropeanChebyshevLSM(order::Int, reps::Int; domain=(30.0, 50.0))
    basis = ChebyshevBasis(order, domain[1], domain[2])
    return EuropeanLongstaffSchwartz(basis, reps)
end

"""
    EuropeanPowerLSM(order, reps; normalization=40.0)

Create a European LSM engine with power basis.

See also: [`EuropeanLongstaffSchwartz`](@ref)
"""
function EuropeanPowerLSM(order::Int, reps::Int; normalization=40.0)
    basis = PowerBasis(order, normalization)
    return EuropeanLongstaffSchwartz(basis, reps)
end

"""
    EuropeanHermiteLSM(order, reps; mean=40.0, std=10.0)

Create a European LSM engine with Hermite basis.

See also: [`EuropeanLongstaffSchwartz`](@ref)
"""
function EuropeanHermiteLSM(order::Int, reps::Int; mean=40.0, std=10.0)
    basis = HermiteBasis(order, mean, std)
    return EuropeanLongstaffSchwartz(basis, reps)
end

# European LSM pricing - regression on terminal payoffs
function price(option::EuropeanOption, engine::EuropeanLongstaffSchwartz{B, T}, data::MarketData) where {B, T}
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data
    (; basis, reps, min_regression_paths) = engine

    # Generate terminal asset prices (single time step)
    paths = zeros(T, reps)
    dt = T(expiry)
    nudt = (T(rate - div) - T(0.5) * T(vol)^2) * dt
    sidt = T(vol) * sqrt(dt)

    # Generate terminal prices using geometric Brownian motion
    for i in 1:reps
        z = randn(T)
        paths[i] = T(spot) * exp(nudt + sidt * z)
    end

    # Calculate terminal payoffs
    terminal_payoffs = payoff_value.(option, paths)

    # Find in-the-money paths for regression
    itm_mask = terminal_payoffs .> zero(T)
    itm_count = sum(itm_mask)

    if itm_count < min_regression_paths
        # Fallback to simple Monte Carlo if not enough ITM paths
        @warn "Not enough ITM paths ($itm_count < $min_regression_paths), using simple MC"
        return exp(-T(rate) * T(expiry)) * mean(terminal_payoffs)
    end

    # Extract ITM data for regression
    S_itm = paths[itm_mask]
    payoffs_itm = terminal_payoffs[itm_mask]

    # Create basis matrix
    X = basis(S_itm)

    try
        # Fit payoff function using basis regression
        β = fit_continuation_value(basis, X, payoffs_itm)

        # Predict option values for all paths using the fitted model
        X_all = basis(paths)
        predicted_values = X_all * β

        # Ensure non-negative values (option prices can't be negative)
        predicted_values = max.(predicted_values, zero(T))

        # Return discounted expected value
        return exp(-T(rate) * T(expiry)) * mean(predicted_values)

    catch e
        @warn "European LSM regression failed, falling back to simple MC" exception=e
        return exp(-T(rate) * T(expiry)) * mean(terminal_payoffs)
    end
end

# Helper functions for European option payoffs (type-stable)
payoff_value(option::EuropeanPut, spot::T) where T = max(zero(T), T(option.strike) - spot)
payoff_value(option::EuropeanCall, spot::T) where T = max(zero(T), spot - T(option.strike))