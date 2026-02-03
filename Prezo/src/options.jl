"""
    VanillaOption

Abstract base type for all vanilla option contracts.

See also: [`EuropeanOption`](@ref), [`AmericanOption`](@ref)
"""
abstract type VanillaOption end

"""
    EuropeanOption <: VanillaOption

Abstract type for European-style options that can only be exercised at expiration.

See also: [`EuropeanCall`](@ref), [`EuropeanPut`](@ref)
"""
abstract type EuropeanOption <: VanillaOption end

"""
    AmericanOption <: VanillaOption

Abstract type for American-style options that can be exercised at any time up to
and including expiration.

See also: [`AmericanCall`](@ref), [`AmericanPut`](@ref)
"""
abstract type AmericanOption <: VanillaOption end

## European Options

"""
    EuropeanCall(strike, expiry)

A European call option that gives the holder the right (but not the obligation)
to buy the underlying asset at the strike price at expiration.

# Fields
- `strike::AbstractFloat`: Strike price (exercise price)
- `expiry::AbstractFloat`: Time to expiration in years

# Payoff
The payoff at expiration is `max(0, S - K)` where `S` is the spot price and `K`
is the strike.

# Examples
```julia
# At-the-money call with 1 year to expiration
call = EuropeanCall(100.0, 1.0)

# Out-of-the-money call with 6 months to expiration
call = EuropeanCall(110.0, 0.5)
```

See also: [`EuropeanPut`](@ref), [`payoff`](@ref)
"""
struct EuropeanCall <: EuropeanOption
    strike::AbstractFloat
    expiry::AbstractFloat
end

Base.broadcastable(x::EuropeanCall) = Ref(x)

function payoff(option::EuropeanCall, spot)
    return max(0.0, spot - option.strike)
end


"""
    EuropeanPut(strike, expiry)

A European put option that gives the holder the right (but not the obligation)
to sell the underlying asset at the strike price at expiration.

# Fields
- `strike::AbstractFloat`: Strike price (exercise price)
- `expiry::AbstractFloat`: Time to expiration in years

# Payoff
The payoff at expiration is `max(0, K - S)` where `S` is the spot price and `K`
is the strike.

# Examples
```julia
# At-the-money put with 1 year to expiration
put = EuropeanPut(100.0, 1.0)

# In-the-money put with 6 months to expiration
put = EuropeanPut(110.0, 0.5)
```

See also: [`EuropeanCall`](@ref), [`payoff`](@ref)
"""
struct EuropeanPut <: EuropeanOption
    strike::AbstractFloat
    expiry::AbstractFloat
end

Base.broadcastable(x::EuropeanPut) = Ref(x)

"""
    payoff(option::VanillaOption, spot)

Calculate the intrinsic value (payoff) of an option at a given spot price.

# Arguments
- `option::VanillaOption`: The option contract
- `spot`: Spot price(s) of the underlying asset (can be scalar or vector)

# Returns
The intrinsic value of the option. For calls: `max(0, S - K)`, for puts: `max(0, K - S)`.

# Examples
```julia
call = EuropeanCall(100.0, 1.0)
payoff(call, 110.0)  # Returns 10.0

put = EuropeanPut(100.0, 1.0)
payoff(put, 90.0)    # Returns 10.0
payoff(put, 110.0)   # Returns 0.0
```
"""
function payoff(option::EuropeanPut, spot)
    return max.(0.0, option.strike - spot)
end


## American Options

"""
    AmericanCall(strike, expiry)

An American call option that can be exercised at any time up to and including expiration.

# Fields
- `strike::AbstractFloat`: Strike price (exercise price)
- `expiry::AbstractFloat`: Time to expiration in years

# Notes
For non-dividend paying stocks, American calls have the same value as European calls
since early exercise is never optimal.

# Examples
```julia
call = AmericanCall(100.0, 1.0)
```

See also: [`AmericanPut`](@ref), [`EuropeanCall`](@ref)
"""
struct AmericanCall <: AmericanOption
    strike::AbstractFloat
    expiry::AbstractFloat
end

Base.broadcastable(x::AmericanCall) = Ref(x)

function payoff(option::AmericanCall, spot)
    return max(0.0, spot - option.strike)
end


"""
    AmericanPut(strike, expiry)

An American put option that can be exercised at any time up to and including expiration.

# Fields
- `strike::AbstractFloat`: Strike price (exercise price)
- `expiry::AbstractFloat`: Time to expiration in years

# Notes
American puts always trade at a premium to European puts due to the early exercise feature.
This early exercise option is particularly valuable when interest rates are high or
the option is deep in-the-money.

# Examples
```julia
put = AmericanPut(100.0, 1.0)
```

See also: [`AmericanCall`](@ref), [`EuropeanPut`](@ref)
"""
struct AmericanPut <: AmericanOption
    strike::AbstractFloat
    expiry::AbstractFloat
end

Base.broadcastable(x::AmericanPut) = Ref(x)

function payoff(option::AmericanPut, spot)
    return max(0.0, option.strike - spot)
end


# ============================================================================
# Exotic Options
# ============================================================================

"""
    ExoticOption <: VanillaOption

Abstract base type for exotic (path-dependent) options.

Subtypes include:
- [`AsianOption`](@ref): Options on average price
- [`BarrierOption`](@ref): Options with knock-in/knock-out barriers
- [`LookbackOption`](@ref): Options on extreme prices

See also: [`VanillaOption`](@ref)
"""
abstract type ExoticOption <: VanillaOption end

# ----------------------------------------------------------------------------
# Asian Options
# ----------------------------------------------------------------------------

"""
    AsianOption <: ExoticOption

Abstract type for Asian options whose payoff depends on the average price
of the underlying over a set of observation times.

See also: [`ArithmeticAsianCall`](@ref), [`GeometricAsianCall`](@ref)
"""
abstract type AsianOption <: ExoticOption end

"""
    ArithmeticAsianCall(strike, expiry, averaging_times)

Asian call option with arithmetic average.

The payoff is `max(0, A - K)` where A is the arithmetic average of spot prices
at the specified averaging times.

# Fields
- `strike::Float64`: Strike price
- `expiry::Float64`: Time to expiration in years
- `averaging_times::Vector{Float64}`: Observation times for averaging (in years)

# Notes
No closed-form solution exists; requires Monte Carlo pricing.

# Examples
```julia
# Monthly averaging over 1 year
times = collect(1:12) ./ 12
call = ArithmeticAsianCall(100.0, 1.0, times)
```
"""
struct ArithmeticAsianCall <: AsianOption
    strike::Float64
    expiry::Float64
    averaging_times::Vector{Float64}
end

Base.broadcastable(x::ArithmeticAsianCall) = Ref(x)

"""
    ArithmeticAsianPut(strike, expiry, averaging_times)

Asian put option with arithmetic average.

The payoff is `max(0, K - A)` where A is the arithmetic average.
"""
struct ArithmeticAsianPut <: AsianOption
    strike::Float64
    expiry::Float64
    averaging_times::Vector{Float64}
end

Base.broadcastable(x::ArithmeticAsianPut) = Ref(x)

"""
    GeometricAsianCall(strike, expiry, averaging_times)

Asian call option with geometric average.

The payoff is `max(0, G - K)` where G is the geometric average of spot prices.

# Notes
Has a closed-form solution under Black-Scholes assumptions since the geometric
average of log-normal variables is log-normal.
"""
struct GeometricAsianCall <: AsianOption
    strike::Float64
    expiry::Float64
    averaging_times::Vector{Float64}
end

Base.broadcastable(x::GeometricAsianCall) = Ref(x)

"""
    GeometricAsianPut(strike, expiry, averaging_times)

Asian put option with geometric average.

The payoff is `max(0, K - G)` where G is the geometric average.
"""
struct GeometricAsianPut <: AsianOption
    strike::Float64
    expiry::Float64
    averaging_times::Vector{Float64}
end

Base.broadcastable(x::GeometricAsianPut) = Ref(x)

# Asian option payoffs (require price path)
function payoff(option::ArithmeticAsianCall, prices::Vector{<:Real})
    avg = sum(prices) / length(prices)
    return max(0.0, avg - option.strike)
end

function payoff(option::ArithmeticAsianPut, prices::Vector{<:Real})
    avg = sum(prices) / length(prices)
    return max(0.0, option.strike - avg)
end

function payoff(option::GeometricAsianCall, prices::Vector{<:Real})
    geo_avg = exp(sum(log.(prices)) / length(prices))
    return max(0.0, geo_avg - option.strike)
end

function payoff(option::GeometricAsianPut, prices::Vector{<:Real})
    geo_avg = exp(sum(log.(prices)) / length(prices))
    return max(0.0, option.strike - geo_avg)
end

# ----------------------------------------------------------------------------
# Barrier Options
# ----------------------------------------------------------------------------

"""
    BarrierOption <: ExoticOption

Abstract type for barrier options that are activated or deactivated
when the underlying crosses a specified barrier level.

Barrier types (specified as `Symbol`):
- `:up_and_out`: Knocked out if price rises above barrier
- `:down_and_out`: Knocked out if price falls below barrier
- `:up_and_in`: Activated if price rises above barrier
- `:down_and_in`: Activated if price falls below barrier

See also: [`KnockOutCall`](@ref), [`KnockInCall`](@ref)
"""
abstract type BarrierOption <: ExoticOption end

"""
    KnockOutCall(strike, expiry, barrier, barrier_type)

Barrier call option that becomes worthless if the barrier is breached.

# Fields
- `strike::Float64`: Strike price
- `expiry::Float64`: Time to expiration in years
- `barrier::Float64`: Barrier level
- `barrier_type::Symbol`: Either `:up_and_out` or `:down_and_out`

# Examples
```julia
# Up-and-out call: worthless if spot rises above 120
call = KnockOutCall(100.0, 1.0, 120.0, :up_and_out)

# Down-and-out call: worthless if spot falls below 80
call = KnockOutCall(100.0, 1.0, 80.0, :down_and_out)
```
"""
struct KnockOutCall <: BarrierOption
    strike::Float64
    expiry::Float64
    barrier::Float64
    barrier_type::Symbol  # :up_and_out or :down_and_out
end

Base.broadcastable(x::KnockOutCall) = Ref(x)

"""
    KnockOutPut(strike, expiry, barrier, barrier_type)

Barrier put option that becomes worthless if the barrier is breached.
"""
struct KnockOutPut <: BarrierOption
    strike::Float64
    expiry::Float64
    barrier::Float64
    barrier_type::Symbol
end

Base.broadcastable(x::KnockOutPut) = Ref(x)

"""
    KnockInCall(strike, expiry, barrier, barrier_type)

Barrier call option that is activated only if the barrier is breached.

# Fields
- `strike::Float64`: Strike price
- `expiry::Float64`: Time to expiration in years
- `barrier::Float64`: Barrier level
- `barrier_type::Symbol`: Either `:up_and_in` or `:down_and_in`

# Examples
```julia
# Down-and-in call: activated if spot falls below 80
call = KnockInCall(100.0, 1.0, 80.0, :down_and_in)
```
"""
struct KnockInCall <: BarrierOption
    strike::Float64
    expiry::Float64
    barrier::Float64
    barrier_type::Symbol  # :up_and_in or :down_and_in
end

Base.broadcastable(x::KnockInCall) = Ref(x)

"""
    KnockInPut(strike, expiry, barrier, barrier_type)

Barrier put option that is activated only if the barrier is breached.
"""
struct KnockInPut <: BarrierOption
    strike::Float64
    expiry::Float64
    barrier::Float64
    barrier_type::Symbol
end

Base.broadcastable(x::KnockInPut) = Ref(x)

# Check if barrier was breached
function barrier_breached(option::BarrierOption, path::Vector{<:Real})
    if option.barrier_type == :up_and_out || option.barrier_type == :up_and_in
        return any(path .>= option.barrier)
    else  # :down_and_out or :down_and_in
        return any(path .<= option.barrier)
    end
end

# Barrier option payoffs (require full price path)
function payoff(option::KnockOutCall, path::Vector{<:Real})
    if barrier_breached(option, path)
        return 0.0
    else
        return max(0.0, path[end] - option.strike)
    end
end

function payoff(option::KnockOutPut, path::Vector{<:Real})
    if barrier_breached(option, path)
        return 0.0
    else
        return max(0.0, option.strike - path[end])
    end
end

function payoff(option::KnockInCall, path::Vector{<:Real})
    if barrier_breached(option, path)
        return max(0.0, path[end] - option.strike)
    else
        return 0.0
    end
end

function payoff(option::KnockInPut, path::Vector{<:Real})
    if barrier_breached(option, path)
        return max(0.0, option.strike - path[end])
    else
        return 0.0
    end
end

# ----------------------------------------------------------------------------
# Lookback Options
# ----------------------------------------------------------------------------

"""
    LookbackOption <: ExoticOption

Abstract type for lookback options whose payoff depends on the extreme
(maximum or minimum) price reached during the option's life.

See also: [`FixedStrikeLookbackCall`](@ref), [`FloatingStrikeLookbackCall`](@ref)
"""
abstract type LookbackOption <: ExoticOption end

"""
    FixedStrikeLookbackCall(strike, expiry)

Lookback call with fixed strike.

Payoff is `max(0, S_max - K)` where S_max is the maximum spot price
observed during the option's life.

# Fields
- `strike::Float64`: Strike price
- `expiry::Float64`: Time to expiration in years
"""
struct FixedStrikeLookbackCall <: LookbackOption
    strike::Float64
    expiry::Float64
end

Base.broadcastable(x::FixedStrikeLookbackCall) = Ref(x)

"""
    FixedStrikeLookbackPut(strike, expiry)

Lookback put with fixed strike.

Payoff is `max(0, K - S_min)` where S_min is the minimum spot price
observed during the option's life.
"""
struct FixedStrikeLookbackPut <: LookbackOption
    strike::Float64
    expiry::Float64
end

Base.broadcastable(x::FixedStrikeLookbackPut) = Ref(x)

"""
    FloatingStrikeLookbackCall(expiry)

Lookback call with floating strike.

Payoff is `S_T - S_min` where S_T is the terminal price and S_min is the
minimum price observed. Always in-the-money at expiration.

# Fields
- `expiry::Float64`: Time to expiration in years
"""
struct FloatingStrikeLookbackCall <: LookbackOption
    expiry::Float64
end

Base.broadcastable(x::FloatingStrikeLookbackCall) = Ref(x)

"""
    FloatingStrikeLookbackPut(expiry)

Lookback put with floating strike.

Payoff is `S_max - S_T` where S_max is the maximum price observed.
Always in-the-money at expiration.
"""
struct FloatingStrikeLookbackPut <: LookbackOption
    expiry::Float64
end

Base.broadcastable(x::FloatingStrikeLookbackPut) = Ref(x)

# Lookback option payoffs (require full price path)
function payoff(option::FixedStrikeLookbackCall, path::Vector{<:Real})
    return max(0.0, maximum(path) - option.strike)
end

function payoff(option::FixedStrikeLookbackPut, path::Vector{<:Real})
    return max(0.0, option.strike - minimum(path))
end

function payoff(option::FloatingStrikeLookbackCall, path::Vector{<:Real})
    return path[end] - minimum(path)
end

function payoff(option::FloatingStrikeLookbackPut, path::Vector{<:Real})
    return maximum(path) - path[end]
end
