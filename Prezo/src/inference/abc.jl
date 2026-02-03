"""
    Approximate Bayesian Computation (ABC)

Likelihood-free inference: accept parameter draws when simulated data
is "close" to observed data in summary-statistic space.

Rejection ABC: sample from prior, simulate, accept if distance(summary(sim), summary(obs)) <= tolerance.
"""

using Distributions
using Random
using Statistics
using LinearAlgebra

"""
    ABCMethod

Abstract type for ABC algorithms (rejection, MCMC-ABC, SMC-ABC, etc.).
"""
abstract type ABCMethod end

"""
    RejectionABC(n_samples, tolerance, prior)

Rejection ABC. Sample `n_samples` from `prior`, simulate, accept if
distance(summary(simulated), summary(observed)) <= tolerance.
`prior` should support `rand(rng, prior)` returning a vector (e.g. Product of univariates).
"""
struct RejectionABC{T<:Real} <: ABCMethod
    n_samples::Int
    tolerance::T
    prior::Distribution
end

"""
    abc_inference(
        method::RejectionABC,
        simulator::Function,
        summary_stats::Function,
        distance::Function,
        observed_data;
        rng=GLOBAL_RNG,
    ) -> (accepted_params::Matrix{Float64}, accepted_count::Int)

Run rejection ABC.
- `simulator(params::Vector)`: returns synthetic data (e.g. vector of returns)
- `summary_stats(data)`: returns summary vector (e.g. [mean, std, ...])
- `distance(s1, s2)`: returns scalar distance between two summary vectors (e.g. euclidean)
- `observed_data`: the observed dataset (summary_stats(observed_data) used as target)

Returns matrix of accepted parameters (n_accepted × n_params) and count.
"""
function abc_inference(
    method::RejectionABC,
    simulator::Function,
    summary_stats::Function,
    distance::Function,
    observed_data;
    rng::Union{AbstractRNG,Nothing}=nothing,
)
    (; n_samples, tolerance, prior) = method
    rng = rng === nothing ? Random.GLOBAL_RNG : rng
    obs_summary = summary_stats(observed_data)
    # Infer param dimension from one prior draw
    params_draw = rand(rng, prior)
    n_params = length(params_draw)
    # Collect accepted params
    accepted = Vector{Vector{Float64}}()
    for _ in 1:n_samples
        params = Float64.(rand(rng, prior))
        data_sim = simulator(params)
        sim_summary = summary_stats(data_sim)
        d = distance(obs_summary, sim_summary)
        if d <= tolerance
            push!(accepted, params)
        end
    end
    n_acc = length(accepted)
    if n_acc == 0
        return Matrix{Float64}(undef, 0, n_params), 0
    end
    out = reduce(hcat, accepted)'
    return out, n_acc
end

"""
    euclidean_distance(a::AbstractVector, b::AbstractVector) -> Float64

Euclidean distance between two vectors (for ABC summary distance).
"""
function euclidean_distance(a::AbstractVector, b::AbstractVector)
    length(a) == length(b) || throw(ArgumentError("length mismatch"))
    return sqrt(sum((a[i] - b[i])^2 for i in eachindex(a)))
end
