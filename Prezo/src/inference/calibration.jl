"""
    Model Calibration Framework

Calibrate model parameters to match observed option prices or implied vol surface.
Least-squares: minimize sum of squared errors (option prices or IVs).
"""

using Optim
using LinearAlgebra

"""
    CalibrationTarget

Abstract type for calibration targets (option prices, IV surface, etc.).
"""
abstract type CalibrationTarget end

"""
    OptionPricesTarget(options, market_prices)

Target: vector of options and corresponding market prices.
`options[i]` is priced at `market_prices[i]`.
"""
struct OptionPricesTarget <: CalibrationTarget
    options::Vector
    market_prices::Vector{Float64}
end

"""
    IVSurfaceTarget(strikes, maturities, market_vols)

Target: grid of (strike, maturity) and observed implied volatilities.
`market_vols[i,j]` = IV at `strikes[i]`, `maturities[j]`.
"""
struct IVSurfaceTarget <: CalibrationTarget
    strikes::Vector{Float64}
    maturities::Vector{Float64}
    market_vols::Matrix{Float64}
end

"""
    CalibrationMethod

Abstract type for calibration methods (least squares, regularized, etc.).
"""
abstract type CalibrationMethod end

"""
    LeastSquaresCalibration(; weights=nothing)

Minimize sum of squared errors. Optionally weight by `weights` (same length as target).
"""
struct LeastSquaresCalibration <: CalibrationMethod
    weights::Union{Vector{Float64},Nothing}
end
LeastSquaresCalibration() = LeastSquaresCalibration(nothing)

"""
    RegularizedCalibration(regularization::Function, λ::Real)

Minimize objective(params) + λ * regularization(params). E.g. L2 on deviation from ref:
  regularization(p) = sum((p - ref).^2)
"""
struct RegularizedCalibration{T<:Real} <: CalibrationMethod
    regularization::Function  # params -> scalar penalty
    λ::T
end

"""
    CalibrationResult(params, loss, converged, iterations)

Result of calibrate().
"""
struct CalibrationResult{T<:Real}
    params::Vector{T}
    loss::T
    converged::Bool
    iterations::Int
end

"""
    calibrate(
        objective::Function,
        initial_params::Vector{<:Real},
        lower_bounds::Vector{<:Real},
        upper_bounds::Vector{<:Real};
        method::CalibrationMethod=LeastSquaresCalibration(),
        verbose::Bool=false,
    ) -> CalibrationResult

Generic calibration: minimize `objective(params)` over box-constrained params.
`objective(params)` should return a scalar loss (e.g. sum of squared errors).
"""
function calibrate(
    objective::Function,
    initial_params::Vector{<:Real},
    lower_bounds::Vector{<:Real},
    upper_bounds::Vector{<:Real};
    method::CalibrationMethod=LeastSquaresCalibration(),
    verbose::Bool=false,
)
    if method isa RegularizedCalibration
        full_objective = function (params)
            objective(params) + method.λ * method.regularization(params)
        end
        result = optimize(
            full_objective,
            lower_bounds,
            upper_bounds,
            initial_params,
            Fminbox(LBFGS());
            inplace=false,
        )
    else
        result = optimize(
            objective,
            lower_bounds,
            upper_bounds,
            initial_params,
            Fminbox(LBFGS());
            inplace=false,
        )
    end
    params = Optim.minimizer(result)
    loss = Optim.minimum(result)
    converged = Optim.converged(result)
    iterations = Optim.iterations(result)
    return CalibrationResult(params, loss, converged, iterations)
end

"""
    calibrate_option_prices(
        price_fn::Function,
        target::OptionPricesTarget,
        initial_params::Vector{<:Real},
        lower_bounds::Vector{<:Real},
        upper_bounds::Vector{<:Real};
        weights=nothing,
        verbose=false,
    ) -> CalibrationResult

Calibrate parameters so model prices match target. `price_fn(option, params) -> Float64`.
Loss = sum_i w_i * (price_fn(options[i], params) - market_prices[i])^2.
"""
function calibrate_option_prices(
    price_fn::Function,
    target::OptionPricesTarget,
    initial_params::Vector{<:Real},
    lower_bounds::Vector{<:Real},
    upper_bounds::Vector{<:Real};
    weights::Union{Vector{<:Real},Nothing}=nothing,
    method::CalibrationMethod=LeastSquaresCalibration(),
    verbose::Bool=false,
)
    (; options, market_prices) = target
    n = length(options)
    length(market_prices) == n || throw(ArgumentError("options and market_prices length mismatch"))
    w = weights === nothing ? ones(n) : Float64.(weights)
    length(w) == n || throw(ArgumentError("weights length must match options"))

    function objective(params)
        s = 0.0
        for i in 1:n
            err = price_fn(options[i], params) - market_prices[i]
            s += w[i] * err^2
        end
        return s
    end

    return calibrate(objective, initial_params, lower_bounds, upper_bounds; method=method, verbose=verbose)
end
