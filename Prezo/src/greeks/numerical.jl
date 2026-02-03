"""
    numerical.jl

Numerical Greek calculations using finite differences.

This module provides numerical approximations for Greeks when analytical
formulas are not available (e.g., American options, complex engines).

Methods:
- Forward differences (1-point): Fast but O(h) error
- Central differences (2-point): Balanced accuracy, O(h²) error
- Richardson extrapolation (4-point): Higher accuracy, O(h⁴) error

Default step size: h = √ε * S ≈ 1e-4 for typical spot prices
where ε is machine epsilon (~1e-16 for Float64).
"""

using ..Prezo

"""
    finite_difference(f::Function, x::Real, h::Real; method=:central)

Compute numerical derivative using finite differences.

# Arguments
- `f::Function`: Function to differentiate
- `x::Real`: Point at which to evaluate derivative
- `h::Real`: Step size
- `method::Symbol`: Finite difference method (:forward, :central, :richardson)

# Returns
Approximate derivative df/dx at point x.

# Methods
- `:forward`: (f(x+h) - f(x)) / h, error O(h)
- `:central`: (f(x+h) - f(x-h)) / (2h), error O(h²) ✓ recommended
- `:richardson`: (f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)) / (12h), error O(h⁴)

# Examples
```julia
f(x) = x^2
# Analytical derivative: 2x, at x=3: 6.0

# Forward difference
deriv = finite_difference(f, 3.0, 1e-6, method=:forward)  # ≈ 6.0

# Central difference (more accurate)
deriv = finite_difference(f, 3.0, 1e-6, method=:central)  # ≈ 6.0
```
"""
function finite_difference(f::Function, x::Real, h::Real; method::Symbol=:central)
    if method == :forward
        return (f(x + h) - f(x)) / h
    elseif method == :central
        return (f(x + h) - f(x - h)) / (2 * h)
    elseif method == :richardson
        # 4-point central difference (higher accuracy)
        return (f(x - 2h) - 8*f(x - h) + 8*f(x + h) - f(x + 2h)) / (12 * h)
    else
        throw(ArgumentError("Unknown method: $method. Use :forward, :central, or :richardson"))
    end
end

"""
    optimal_step_size(x::Real, ::Type{T}=Float64) where T

Compute optimal step size for finite differences based on machine precision.

For central differences: h = √ε * |x| where ε is machine epsilon

# Examples
```julia
h = optimal_step_size(100.0)  # ≈ 1e-6 for typical spot prices
```
"""
function optimal_step_size(x::Real, ::Type{T}=Float64) where T
    eps_val = eps(T)
    return sqrt(eps_val) * max(abs(x), one(T))
end

"""
    numerical_greek(option::VanillaOption, ::Delta, engine::PricingEngine, 
                     data::MarketData; h::Real=optimal_step_size(data.spot))

Numerical delta via finite differences.

Formula: Δ ≈ (V(S+h) - V(S-h)) / (2h)

# Arguments
- `option::VanillaOption`: The option contract
- `engine::PricingEngine`: Pricing engine (Binomial, MonteCarlo, LSM, etc.)
- `data::MarketData`: Market parameters
- `h::Real`: Spot price step size (default: optimal for Float64)

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
put = AmericanPut(100.0, 1.0)
engine = Binomial(500)

# Numerical delta for American option
delta = numerical_greek(put, Delta(), engine, data)
```

See also: [`greek`](@ref) for unified interface
"""
function numerical_greek(option::VanillaOption, ::Delta, engine::PricingEngine, 
                         data::MarketData; h::Real=optimal_step_size(data.spot))
    
    # Price function
    price_func = S -> begin
        new_data = MarketData(S, data.rate, data.vol, data.div)
        return price(option, engine, new_data)
    end
    
    # Central difference
    delta = finite_difference(price_func, data.spot, h, method=:central)
    
    return delta
end

"""
    numerical_greek(option::VanillaOption, ::Gamma, engine::PricingEngine, 
                     data::MarketData; h::Real=optimal_step_size(data.spot))

Numerical gamma via finite differences.

Formula: Γ ≈ (V(S+h) - 2V(S) + V(S-h)) / h²

Uses second-order central difference for second derivative.
"""
function numerical_greek(option::VanillaOption, ::Gamma, engine::PricingEngine, 
                         data::MarketData; h::Real=optimal_step_size(data.spot))
    
    # Price at S+h, S, S-h
    S = data.spot
    
    price_up = price(option, engine, MarketData(S + h, data.rate, data.vol, data.div))
    price_mid = price(option, engine, MarketData(S, data.rate, data.vol, data.div))
    price_down = price(option, engine, MarketData(S - h, data.rate, data.vol, data.div))
    
    # Second-order central difference
    gamma = (price_up - 2*price_mid + price_down) / (h^2)
    
    return gamma
end

"""
    numerical_greek(option::VanillaOption, ::Theta, engine::PricingEngine, 
                     data::MarketData; h::Real=1/365)

Numerical theta via finite differences.

Formula: Θ ≈ (V(T-h) - V(T)) / h

Note: For theta, we use a small time step (1 day default) because:
- Time is always positive (can't use central differences easily)
- Options near expiry need fine resolution

Returns theta per year (standard convention).
"""
function numerical_greek(option::VanillaOption, ::Theta, engine::PricingEngine, 
                         data::MarketData; h::Real=1/365)
    
    T = option.expiry
    
    if T <= h
        # Option is expiring soon, can't reduce time further
        # Use very small h or return 0
        return 0.0
    end
    
    # Price at T and T-h
    price_now = price(option, engine, data)
    
    # Create option with shorter maturity
    if option isa EuropeanCall
        option_later = EuropeanCall(option.strike, T - h)
    elseif option isa EuropeanPut
        option_later = EuropeanPut(option.strike, T - h)
    elseif option isa AmericanCall
        option_later = AmericanCall(option.strike, T - h)
    elseif option isa AmericanPut
        option_later = AmericanPut(option.strike, T - h)
    else
        throw(ArgumentError("Unsupported option type: $(typeof(option))"))
    end
    
    price_later = price(option_later, engine, data)
    
    # Forward difference, scaled to per year
    theta = (price_later - price_now) / h
    
    return theta
end

"""
    numerical_greek(option::VanillaOption, ::Vega, engine::PricingEngine, 
                     data::MarketData; h::Real=0.01)

Numerical vega via finite differences.

Formula: ν ≈ (V(σ+h) - V(σ-h)) / (2h)

Default h = 0.01 (1% vol change) since vega is conventionally reported per 1%.
Returns vega per unit vol (multiply by 0.01 for conventional vega).
"""
function numerical_greek(option::VanillaOption, ::Vega, engine::PricingEngine, 
                         data::MarketData; h::Real=0.01)
    
    σ = data.vol
    
    # Price function
    price_func = vol -> begin
        new_data = MarketData(data.spot, data.rate, vol, data.div)
        return price(option, engine, new_data)
    end
    
    # Central difference
    vega = finite_difference(price_func, σ, h, method=:central)
    
    return vega
end

"""
    numerical_greek(option::VanillaOption, ::Rho, engine::PricingEngine, 
                     data::MarketData; h::Real=0.01)

Numerical rho via finite differences.

Formula: ρ ≈ (V(r+h) - V(r-h)) / (2h)

Default h = 0.01 (1% rate change).
Returns rho per unit rate (multiply by 0.01 for conventional rho).
"""
function numerical_greek(option::VanillaOption, ::Rho, engine::PricingEngine, 
                         data::MarketData; h::Real=0.01)
    
    r = data.rate
    
    # Price function
    price_func = rate -> begin
        new_data = MarketData(data.spot, rate, data.vol, data.div)
        return price(option, engine, new_data)
    end
    
    # Central difference
    rho = finite_difference(price_func, r, h, method=:central)
    
    return rho
end

"""
    numerical_greek(option::VanillaOption, ::Phi, engine::PricingEngine, 
                     data::MarketData; h::Real=0.01)

Numerical phi (dividend sensitivity) via finite differences.

Formula: φ ≈ (V(q+h) - V(q-h)) / (2h)

Default h = 0.01 (1% dividend yield change).
"""
function numerical_greek(option::VanillaOption, ::Phi, engine::PricingEngine, 
                         data::MarketData; h::Real=0.01)
    
    q = data.div
    
    # Price function
    price_func = div_yield -> begin
        new_data = MarketData(data.spot, data.rate, data.vol, div_yield)
        return price(option, engine, new_data)
    end
    
    # Central difference
    phi = finite_difference(price_func, q, h, method=:central)
    
    return phi
end

## Unified Interface

"""
    greek(option::VanillaOption, greek_type::FirstOrderGreek, 
          engine::PricingEngine, data::MarketData; kwargs...)

Unified interface for computing Greeks with any pricing engine.

This method automatically:
1. Uses analytical formulas for European options with Black-Scholes
2. Falls back to numerical methods for other engines
3. Handles American options via finite differences

# Arguments
- `option::VanillaOption`: The option contract
- `greek_type::FirstOrderGreek`: Type of Greek to compute (Delta, Gamma, etc.)
- `engine::PricingEngine`: Pricing engine
- `data::MarketData`: Market parameters
- `kwargs...`: Additional arguments passed to numerical methods

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
put = AmericanPut(100.0, 1.0)
engine = Binomial(500)

# Automatic numerical computation
delta = greek(put, Delta(), engine, data)
vega = greek(put, Vega(), engine, data, h=0.005)  # Custom step size
```

See also: [`numerical_greek`](@ref) for explicit numerical methods
"""
function greek(option::VanillaOption, greek_type::Greek, 
                             engine::PricingEngine, data::MarketData; kwargs...)
    
    # Use analytical formulas for European + Black-Scholes
    if engine isa BlackScholes && option isa EuropeanOption
        return greek(option, greek_type, data)
    end
    
    # Otherwise, use numerical methods
    return numerical_greek(option, greek_type, engine, data; kwargs...)
end

"""
    greeks(option::VanillaOption, engine::PricingEngine, data::MarketData, 
           greek_types::Vector{<:Greek}; kwargs...)

Compute multiple Greeks using a specific pricing engine.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
put = AmericanPut(100.0, 1.0)
engine = LongstaffSchwartz(50, 10000)

# Compute Greeks for American option
american_greeks = greeks(put, engine, data, [Delta(), Gamma(), Vega()])
```
"""
function greeks(option::VanillaOption, engine::PricingEngine, 
                              data::MarketData, greek_types::Vector{T}; kwargs...) where T <: Greek
    
    # For European + Black-Scholes, use analytical batch computation
    if engine isa BlackScholes && option isa EuropeanOption
        return greeks(option, data, greek_types)
    end
    
    # Otherwise, compute numerically one by one
    result = Dict{Greek, Float64}()
    for greek_type in greek_types
        result[greek_type] = numerical_greek(option, greek_type, engine, data; kwargs...)
    end
    
    return result
end

"""
    all_greeks(option::VanillaOption, engine::PricingEngine, data::MarketData; kwargs...)

Compute all major first-order Greeks using a specific pricing engine.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
put = AmericanPut(100.0, 1.0)
engine = Binomial(500)

# All Greeks for American put
greek_dict = all_greeks(put, engine, data)
```
"""
function all_greeks(option::VanillaOption, engine::PricingEngine, data::MarketData; kwargs...)
    return greeks(option, engine, data, [Delta(), Gamma(), Theta(), Vega(), Rho(), Phi()]; kwargs...)
end
