"""
    Dupire local volatility

Build a local volatility surface σ(K,T) from market call prices via the Dupire equation:
  σ²(K,T) = 2 * (∂C/∂T + (r-q)K ∂C/∂K + (r-q)C) / (K² ∂²C/∂K²)

Finite-difference approximations for derivatives; interior points only (boundaries set to NaN or extrapolate).
"""

using LinearAlgebra

# MarketData lives in Prezo (data.jl), available when Volatility is loaded by Prezo
using ..Prezo: MarketData

"""
    LocalVolSurface

Local volatility surface σ(K, T).

# Fields
- `strikes::Vector{Float64}`: Strike grid (ascending).
- `maturities::Vector{Float64}`: Maturity grid in years (ascending).
- `local_vols::Matrix{Float64}`: local_vols[i,j] = σ(strikes[i], maturities[j]). May contain NaN at boundaries.
"""
struct LocalVolSurface
    strikes::Vector{Float64}
    maturities::Vector{Float64}
    local_vols::Matrix{Float64}
end

"""
    _central_diff(x::AbstractVector, y::AbstractVector, i::Int) -> Float64

Central difference dy/dx at index i; one-sided at boundaries.
"""
function _central_diff(x::AbstractVector, y::AbstractVector, i::Int)
    n = length(x)
    i < 1 || i > n && return NaN
    if i == 1
        return (y[2] - y[1]) / (x[2] - x[1])
    end
    if i == n
        return (y[n] - y[n-1]) / (x[n] - x[n-1])
    end
    return (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
end

"""
    _second_diff(x::AbstractVector, y::AbstractVector, i::Int) -> Float64

Second derivative d²y/dx² at index i (central); one-sided or NaN at boundaries.
"""
function _second_diff(x::AbstractVector, y::AbstractVector, i::Int)
    n = length(x)
    if i <= 1 || i >= n
        return NaN
    end
    dx1 = x[i] - x[i-1]
    dx2 = x[i+1] - x[i]
    # d²y/dx² ≈ 2 * (y[i-1]/(dx1*(dx1+dx2)) - y[i]/(dx1*dx2) + y[i+1]/(dx2*(dx1+dx2)))
    den = dx1 * dx2 * (dx1 + dx2)
    den ≈ 0 && return NaN
    return 2.0 * (y[i-1] * dx2 - y[i] * (dx1 + dx2) + y[i+1] * dx1) / den
end

"""
    dupire_local_vol(
        market_calls::Matrix{<:Real},
        strikes::Vector{<:Real},
        maturities::Vector{<:Real},
        data::MarketData,
    ) -> LocalVolSurface

Compute local variance σ²(K,T) from Dupire formula using finite differences.
- market_calls[i,j] = call price at strikes[i], maturities[j].
- data.rate, data.div used as r, q. data.spot/data.vol unused.
Returns LocalVolSurface with σ (sqrt of variance); invalid/negative variance set to NaN.
"""
function dupire_local_vol(
    market_calls::Matrix{<:Real},
    strikes::Vector{<:Real},
    maturities::Vector{<:Real},
    data::MarketData,
)
    nk = length(strikes)
    nt = length(maturities)
    if size(market_calls) != (nk, nt)
        throw(ArgumentError("market_calls size must be (length(strikes), length(maturities))"))
    end
    r = Float64(data.rate)
    q = Float64(data.div)
    K = Float64.(strikes)
    T = Float64.(maturities)
    C = Float64.(market_calls)

    # ∂C/∂T: differentiate along columns (maturities)
    dCdT = fill(NaN, nk, nt)
    for i in 1:nk
        for j in 1:nt
            dCdT[i, j] = _central_diff(T, C[i, :], j)
        end
    end

    # ∂C/∂K: differentiate along rows (strikes)
    dCdK = fill(NaN, nk, nt)
    for j in 1:nt
        for i in 1:nk
            dCdK[i, j] = _central_diff(K, C[:, j], i)
        end
    end

    # ∂²C/∂K²
    d2CdK2 = fill(NaN, nk, nt)
    for j in 1:nt
        for i in 1:nk
            d2CdK2[i, j] = _second_diff(K, C[:, j], i)
        end
    end

    # σ² = 2 * (∂C/∂T + (r-q)K ∂C/∂K + (r-q)C) / (K² ∂²C/∂K²)
    local_vols = fill(NaN, nk, nt)
    for j in 1:nt
        for i in 1:nk
            ki = K[i]
            ki > 0 || continue
            num = 2.0 * (dCdT[i, j] + (r - q) * ki * dCdK[i, j] + (r - q) * C[i, j])
            den = ki^2 * d2CdK2[i, j]
            if den > 1e-14 && num >= 0
                local_vols[i, j] = sqrt(num / den)
            end
        end
    end
    return LocalVolSurface(K, T, local_vols)
end
