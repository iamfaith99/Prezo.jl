"""
    Maximum Likelihood Estimation (MLE)

Box-constrained optimization of log-likelihood via Optim.jl (LBFGS).
Standard errors from observed information matrix (inverse Hessian of negative log-likelihood).
"""

using Optim
using LinearAlgebra
using ForwardDiff

"""
    MLEProblem(loglikelihood, initial_params, lower_bounds, upper_bounds)

MLE with box constraints. Minimizes negative log-likelihood.
- `loglikelihood(params::Vector) -> Float64`: log-likelihood (will minimize -loglikelihood)
- `initial_params`: starting point
- `lower_bounds`, `upper_bounds`: vectors of bounds (length = length(initial_params))
"""
struct MLEProblem{T<:Real}
    loglikelihood::Function
    initial_params::Vector{T}
    lower_bounds::Vector{T}
    upper_bounds::Vector{T}
end

function MLEProblem(
    loglikelihood::Function,
    initial_params::Vector{T},
    constraints::Vector{Tuple{T,T}},
) where {T<:Real}
    lb = [c[1] for c in constraints]
    ub = [c[2] for c in constraints]
    return MLEProblem(loglikelihood, initial_params, lb, ub)
end

"""
    MLESolution(params, minimum, converged, iterations, hessian)

Result of MLE solve. `minimum` is the negative log-likelihood at optimum.
`hessian` is the Hessian of the negative log-likelihood at params (for standard errors).
"""
struct MLESolution{T<:Real}
    params::Vector{T}
    minimum::T
    converged::Bool
    iterations::Int
    hessian::Matrix{T}
end

"""
    solve(mle::MLEProblem; method=:LBFGS) -> MLESolution

Minimize negative log-likelihood subject to box constraints.
Uses Optim.Fminbox(LBFGS()). Hessian at solution via ForwardDiff for standard errors.
"""
function solve(mle::MLEProblem; method=:LBFGS)
    (; loglikelihood, initial_params, lower_bounds, upper_bounds) = mle
    nll(params) = -loglikelihood(params)
    # Fminbox for box constraints (Optim uses finite diff by default for box)
    result = optimize(
        nll,
        lower_bounds,
        upper_bounds,
        initial_params,
        Fminbox(LBFGS());
        inplace=false,
    )
    params = Optim.minimizer(result)
    minimum_f = Optim.minimum(result)
    converged = Optim.converged(result)
    iterations = Optim.iterations(result)
    # Hessian of nll at params (observed information)
    hessian = ForwardDiff.hessian(nll, params)
    return MLESolution(params, minimum_f, converged, iterations, hessian)
end

"""
    standard_errors(solution::MLESolution) -> Vector{Float64}

Standard errors from observed information matrix: sqrt(diag(inv(hessian))).
Returns NaN for parameters where inverse is singular.
"""
function standard_errors(solution::MLESolution)
    H = solution.hessian
    try
        invH = inv(Symmetric(H))
        return sqrt.(max.(diag(invH), 0.0))
    catch
        return fill(NaN, length(solution.params))
    end
end
