"""
    Vectorized Implied Volatility

Efficient computation of implied volatility for multiple options simultaneously.
Useful for analyzing option chains, volatility surfaces, and market data processing.
"""

using ..Prezo: EuropeanOption, EuropeanCall, EuropeanPut, MarketData, BlackScholes

"""
    implied_vol_chain(
        option_specs::Vector{<:NamedTuple},
        market_prices::Vector{<:Real},
        data::MarketData;
        solver=HybridSolver(),
        progress_fn=nothing
    )

Compute implied volatilities for an option chain.

# Arguments
- `option_specs`: Vector of NamedTuples with fields (strike, expiry, is_call)
- `market_prices`: Observed market prices (same length as specs)
- `data::MarketData`: Base market parameters
- `solver`: IV solver to use (default: HybridSolver())
- `progress_fn`: Optional callback `fn(current, total)` for progress tracking

# Returns
Vector of implied volatilities (NaN for failed calculations).
"""
function implied_vol_chain(
    option_specs::Vector{<:NamedTuple},
    market_prices::Vector{T},
    data::MarketData;
    solver=HybridSolver(),
    progress_fn=nothing
) where {T<:Real}

    n = length(option_specs)
    @assert n == length(market_prices) "Option specs and prices must have same length"

    ivs = Vector{Float64}(undef, n)

    idx = 0
    for spec in option_specs
        idx += 1
        price_val = market_prices[idx]

        # Create option object
        if spec.is_call
            option = EuropeanCall(spec.strike, spec.expiry)
        else
            option = EuropeanPut(spec.strike, spec.expiry)
        end

        # Compute implied vol
        ivs[idx] = implied_vol(option, price_val, data, solver)

        # Call progress callback if provided
        if progress_fn !== nothing
            progress_fn(idx, n)
        end
    end

    return ivs
end

"""
    implied_vol_chain(
        strikes::Vector{<:Real},
        market_prices::Vector{<:Real},
        expiry::Real,
        is_call::Bool,
        data::MarketData;
        kwargs...
    )

Convenience method for computing IVs across strikes with common expiry.
"""
function implied_vol_chain(
    strikes::Vector{<:Real},
    market_prices::Vector{<:Real},
    expiry::Real,
    is_call::Bool,
    data::MarketData;
    kwargs...
)
    # Convert strikes to option specs
    specs = [
        (strike=Float64(k), expiry=Float64(expiry), is_call=is_call)
        for k in strikes
    ]

    return implied_vol_chain(specs, market_prices, data; kwargs...)
end

"""
    implied_vol_stats(ivs::Vector{<:Real})

Compute statistics on implied volatility vector, ignoring NaN values.

# Returns
NamedTuple with fields:
- `mean`: Mean of valid IVs
- `std`: Standard deviation of valid IVs
- `minimum`: Minimum IV value
- `maximum`: Maximum IV value
- `range`: Range (max - min)
- `n_valid`: Number of valid (non-NaN) values
- `n_total`: Total number of values
"""
function implied_vol_stats(ivs::Vector{T}) where {T<:Real}
    # Filter out NaN values
    valid_ivs = filter(!isnan, ivs)

    n_valid = length(valid_ivs)
    n_total = length(ivs)

    if n_valid == 0
        return (
            mean=NaN,
            std=NaN,
            minimum=NaN,
            maximum=NaN,
            range=NaN,
            n_valid=0,
            n_total=n_total
        )
    end

    min_val = minimum(valid_ivs)
    max_val = maximum(valid_ivs)

    return (
        mean=sum(valid_ivs) / n_valid,
        std=n_valid > 1 ? sqrt(sum((v - sum(valid_ivs) / n_valid)^2 for v in valid_ivs) / (n_valid - 1)) : 0.0,
        minimum=min_val,
        maximum=max_val,
        range=max_val - min_val,
        n_valid=n_valid,
        n_total=n_total
    )
end
