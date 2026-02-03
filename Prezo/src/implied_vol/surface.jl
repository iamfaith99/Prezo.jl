"""
    Volatility Surface Construction

Build and query an implied volatility surface from a grid of (strike, expiry) and
market prices. The surface can be constructed from a rectangular price matrix and
queried at arbitrary (K, T) via bilinear interpolation.

# Design
- **Functional core**: Pure construction and interpolation; no hidden state.
- **Encapsulation**: Surface is an immutable struct; no mutation after build.

See also: [`ImpliedVolSurface`](@ref), [`build_implied_vol_surface`](@ref), [`surface_iv`](@ref)
"""

using ..Prezo: EuropeanCall, EuropeanPut, MarketData

"""
    ImpliedVolSurface

Holds a rectangular grid of implied volatilities: one IV per (strike, expiry) point.

# Fields
- `strikes::Vector{Float64}`: Strike prices (ascending).
- `expiries::Vector{Float64}`: Times to expiry in years (ascending).
- `iv_matrix::Matrix{Float64}`: `iv_matrix[i,j]` = IV at `strikes[i]`, `expiries[j]`. Use `NaN` for missing/failed.
- `is_call::Bool`: `true` for call surface, `false` for put surface.

# Invariants
- `size(iv_matrix) == (length(strikes), length(expiries))`.
- Strikes and expiries are sorted (ascending).
"""
struct ImpliedVolSurface
    strikes::Vector{Float64}
    expiries::Vector{Float64}
    iv_matrix::Matrix{Float64}
    is_call::Bool

    function ImpliedVolSurface(
        strikes::Vector{Float64},
        expiries::Vector{Float64},
        iv_matrix::Matrix{Float64},
        is_call::Bool
    )
        nk = length(strikes)
        nt = length(expiries)
        if size(iv_matrix) != (nk, nt)
            throw(ArgumentError("iv_matrix size $(size(iv_matrix)) must be ($nk, $nt)"))
        end
        # Allow unsorted input but store as-is; interpolation assumes ascending
        new(strikes, expiries, iv_matrix, is_call)
    end
end

"""
    build_implied_vol_surface(
        strikes::Vector{<:Real},
        expiries::Vector{<:Real},
        price_matrix::Matrix{<:Real},
        data::MarketData;
        is_call::Bool=true,
        solver=HybridSolver()
    ) -> ImpliedVolSurface

Build an implied volatility surface from a rectangular grid of market prices.

# Arguments
- `strikes`: Strike prices (length n).
- `expiries`: Times to expiry in years (length m).
- `price_matrix`: Matrix of size (n, m). `price_matrix[i,j]` = market price at `strikes[i]`, `expiries[j]`.
  Use `NaN` for missing quotes; corresponding IV will be `NaN`.
- `data`: Market data (spot, rate, div); vol field is only used as initial guess where relevant.
- `is_call`: If `true`, treat options as calls; otherwise puts.
- `solver`: IV solver to use (default `HybridSolver()`).

# Returns
`ImpliedVolSurface` with the same grid; failed or invalid prices yield `NaN` in the IV matrix.

# Example
```julia
strikes = [90.0, 100.0, 110.0]
expiries = [0.25, 0.5, 1.0]
prices = [12.5 11.2 10.1; 5.2 6.1 7.0; 1.1 2.0 3.2]  # 3×3
data = MarketData(100.0, 0.05, 0.2, 0.0)
surf = build_implied_vol_surface(strikes, expiries, prices, data)
```
"""
function build_implied_vol_surface(
    strikes::Vector{<:Real},
    expiries::Vector{<:Real},
    price_matrix::Matrix{<:Real},
    data::MarketData;
    is_call::Bool=true,
    solver=HybridSolver()
)
    nk = length(strikes)
    nt = length(expiries)
    if size(price_matrix) != (nk, nt)
        throw(ArgumentError("price_matrix size $(size(price_matrix)) must be ($nk, $nt)"))
    end

    k_vec = Float64.(strikes)
    t_vec = Float64.(expiries)
    iv_mat = Matrix{Float64}(undef, nk, nt)

    for j in 1:nt
        T = t_vec[j]
        for i in 1:nk
            K = k_vec[i]
            p = price_matrix[i, j]
            if isnan(p) || !(p > 0)
                iv_mat[i, j] = NaN
                continue
            end
            option = is_call ? EuropeanCall(K, T) : EuropeanPut(K, T)
            iv_mat[i, j] = implied_vol(option, p, data, solver)
        end
    end

    return ImpliedVolSurface(k_vec, t_vec, iv_mat, is_call)
end

"""
    surface_iv(surface::ImpliedVolSurface, strike::Real, expiry::Real) -> Float64

Return implied volatility at (strike, expiry) via bilinear interpolation.

# Arguments
- `surface`: Built implied vol surface.
- `strike`: Strike price.
- `expiry`: Time to expiry in years.

# Returns
Interpolated IV, or `NaN` if (strike, expiry) is outside the surface grid or
all surrounding nodes are NaN.
"""
function surface_iv(surface::ImpliedVolSurface, strike::Real, expiry::Real)
    k = Float64(strike)
    t = Float64(expiry)
    strikes = surface.strikes
    expiries = surface.expiries
    M = surface.iv_matrix
    nk = length(strikes)
    nt = length(expiries)

    # Outside grid → NaN
    if nk == 0 || nt == 0
        return NaN
    end
    if k < strikes[1] || k > strikes[nk]
        return NaN
    end
    if t < expiries[1] || t > expiries[nt]
        return NaN
    end

    # Find indices for bilinear interpolation
    i_lo = _index_lo(strikes, k)
    i_hi = min(i_lo + 1, nk)
    j_lo = _index_lo(expiries, t)
    j_hi = min(j_lo + 1, nt)

    # Single point in one dimension: use 1D interpolation or constant
    if i_lo == i_hi && j_lo == j_hi
        return M[i_lo, j_lo]
    end
    if i_lo == i_hi
        return _lerp_1d(expiries[j_lo], expiries[j_hi], M[i_lo, j_lo], M[i_lo, j_hi], t)
    end
    if j_lo == j_hi
        return _lerp_1d(strikes[i_lo], strikes[i_hi], M[i_lo, j_lo], M[i_hi, j_lo], k)
    end

    # Bilinear: interpolate in strike then in expiry (or vice versa)
    v_lo = _lerp_1d(strikes[i_lo], strikes[i_hi], M[i_lo, j_lo], M[i_hi, j_lo], k)
    v_hi = _lerp_1d(strikes[i_lo], strikes[i_hi], M[i_lo, j_hi], M[i_hi, j_hi], k)
    return _lerp_1d(expiries[j_lo], expiries[j_hi], v_lo, v_hi, t)
end

# Index of the lower grid point: strikes[i] <= x < strikes[i+1], or last if x == strikes[end]
function _index_lo(xs::Vector{Float64}, x::Float64)
    idx = 1
    for i in 1:(length(xs) - 1)
        if xs[i + 1] > x
            return i
        end
        idx = i + 1
    end
    return idx
end

# Linear interpolation; returns NaN if either value is NaN
function _lerp_1d(x0::Float64, x1::Float64, v0::Float64, v1::Float64, x::Float64)
    if isnan(v0) || isnan(v1)
        return NaN
    end
    if x0 == x1
        return v0
    end
    w = (x - x0) / (x1 - x0)
    return v0 + w * (v1 - v0)
end

"""
    surface_stats(surface::ImpliedVolSurface) -> NamedTuple

Return summary statistics for the surface (valid IVs only).

# Returns
NamedTuple with: `mean`, `std`, `minimum`, `maximum`, `range`, `n_valid`, `n_total`.
"""
function surface_stats(surface::ImpliedVolSurface)
    flat = vec(surface.iv_matrix)
    return implied_vol_stats(flat)
end
