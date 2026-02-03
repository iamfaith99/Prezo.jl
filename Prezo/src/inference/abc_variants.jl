"""
    ABC variants: MCMC-ABC, SMC-ABC, regression-adjusted, model choice, hierarchical

Extends base Rejection ABC with:
- MCMCABC: Metropolis-Hastings ABC chain
- ABCSMC: Sequential Monte Carlo with decreasing tolerance
- RegressionABC: post-hoc regression adjustment of accepted params on summaries
- ABCModelChoice: posterior model probabilities from multiple simulators
- HierarchicalABC: infer hyperparameters when params are drawn from a hyperprior
"""

using Distributions
using Random
using Statistics
using LinearAlgebra

# -----------------------------------------------------------------------------
# MCMC-ABC
# -----------------------------------------------------------------------------

"""
    MCMCABC(n_steps, tolerance, prior, proposal_scale)

MCMC-ABC (Marjoram et al.). Metropolis-Hastings with ABC acceptance: propose from
current state (e.g. current + proposal_scale * randn), simulate, accept if distance <= tolerance
and with MH ratio (prior ratio; proposal symmetric). `proposal_scale` is a vector of step sizes per dimension.
"""
struct MCMCABC{T<:Real} <: ABCMethod
    n_steps::Int
    tolerance::T
    prior::Distribution
    proposal_scale::Vector{T}
end

function abc_inference(
    method::MCMCABC,
    simulator::Function,
    summary_stats::Function,
    distance::Function,
    observed_data;
    rng::Union{AbstractRNG,Nothing}=nothing,
)
    (; n_steps, tolerance, prior, proposal_scale) = method
    rng = rng === nothing ? Random.GLOBAL_RNG : rng
    obs_summary = summary_stats(observed_data)
    n_params = length(proposal_scale)
    # Find initial state by rejection
    current = Float64.(rand(rng, prior))
    for _ in 1:10_000
        data_sim = simulator(current)
        if distance(obs_summary, summary_stats(data_sim)) <= tolerance
            break
        end
        current = Float64.(rand(rng, prior))
    end
    chain = Matrix{Float64}(undef, n_steps, n_params)
    chain[1, :] = current
    lp_current = logpdf(prior, current)
    for t in 2:n_steps
        proposal = current .+ proposal_scale .* randn(rng, n_params)
        # Prior: reject if logpdf is -Inf (outside support)
        try
            logpdf(prior, proposal) == -Inf && (chain[t, :] = current; continue)
        catch
            chain[t, :] = current
            continue
        end
        data_sim = simulator(proposal)
        d = distance(obs_summary, summary_stats(data_sim))
        if d > tolerance
            chain[t, :] = current
            continue
        end
        lp_prop = logpdf(prior, proposal)
        if rand(rng) < exp(lp_prop - lp_current)
            current = proposal
            lp_current = lp_prop
        end
        chain[t, :] = current
    end
    return chain, n_steps
end

# -----------------------------------------------------------------------------
# SMC-ABC (Sequential Monte Carlo)
# -----------------------------------------------------------------------------

"""
    ABCSMC(n_particles, n_generations, eps_schedule, prior)

SMC-ABC: maintain n_particles, decrease tolerance over n_generations via eps_schedule.
Generation 1: rejection ABC at eps_schedule[1]. Later: perturb particles, reweight, resample, accept at current eps.
Simplified: each generation run rejection with current eps, drawing from previous generation's accepted (perturbed).
"""
struct ABCSMC{T<:Real} <: ABCMethod
    n_particles::Int
    n_generations::Int
    eps_schedule::Vector{T}
    prior::Distribution
    perturbation_scale::Vector{T}  # scale for perturbing particles between generations
end

function abc_inference(
    method::ABCSMC,
    simulator::Function,
    summary_stats::Function,
    distance::Function,
    observed_data;
    rng::Union{AbstractRNG,Nothing}=nothing,
)
    (; n_particles, n_generations, eps_schedule, prior, perturbation_scale) = method
    rng = rng === nothing ? Random.GLOBAL_RNG : rng
    obs_summary = summary_stats(observed_data)
    n_params = length(perturbation_scale)
    # Generation 1: rejection at eps_schedule[1]
    particles = Vector{Vector{Float64}}()
    while length(particles) < n_particles
        params = Float64.(rand(rng, prior))
        if distance(obs_summary, summary_stats(simulator(params))) <= eps_schedule[1]
            push!(particles, params)
        end
    end
    # Later generations: perturb and filter
    for gen in 2:n_generations
        eps = gen <= length(eps_schedule) ? eps_schedule[gen] : eps_schedule[end]
        new_particles = Vector{Vector{Float64}}()
        while length(new_particles) < n_particles
            idx = rand(rng, 1:length(particles))
            perturbed = particles[idx] .+ perturbation_scale .* randn(rng, n_params)
            try
                logpdf(prior, perturbed) == -Inf && continue
            catch
                continue
            end
            if distance(obs_summary, summary_stats(simulator(perturbed))) <= eps
                push!(new_particles, perturbed)
            end
        end
        particles = new_particles
    end
    out = reduce(hcat, particles)'
    return out, n_particles
end

# -----------------------------------------------------------------------------
# Regression-adjusted ABC
# -----------------------------------------------------------------------------

"""
    RegressionABC(base_method::ABCMethod, regression::Symbol)

After running base ABC, adjust accepted parameters by regressing params on summary statistics
(observed - simulated) to correct tolerance bias. regression in (:linear, :local_linear).
:linear = global linear; :local_linear = weighted linear using kernel weights on distance.
"""
struct RegressionABC <: ABCMethod
    base_method::ABCMethod
    regression::Symbol  # :linear or :local_linear
end

function abc_inference(
    method::RegressionABC,
    simulator::Function,
    summary_stats::Function,
    distance::Function,
    observed_data;
    rng::Union{AbstractRNG,Nothing}=nothing,
)
    (; base_method, regression) = method
    accepted, n_acc = abc_inference(base_method, simulator, summary_stats, distance, observed_data; rng=rng)
    if n_acc < 2
        return accepted, n_acc
    end
    obs_summary = summary_stats(observed_data)
    n_params = size(accepted, 2)
    n_summary = length(obs_summary)
    # Simulated summaries for each accepted param (recompute for adjustment)
    S = Matrix{Float64}(undef, n_acc, n_summary)
    for i in 1:n_acc
        S[i, :] = summary_stats(simulator(accepted[i, :]))
    end
    # Residuals: observed - simulated summary
    diff = obs_summary' .- S
    # Linear adjustment: θ_adj = θ + B * (s_obs - s_sim); B from regression θ ~ diff
    if regression == :linear
        # θ = α + B * (s_obs - s_sim) => B = Cov(θ, diff) * inv(Var(diff))
        B = (accepted' .- mean(accepted, dims=1)') * (diff .- mean(diff, dims=1)) / n_acc *
            inv(Symmetric((diff .- mean(diff, dims=1))' * (diff .- mean(diff, dims=1)) / n_acc + 1e-10 * I))
        # Ensure vector so B * delta is (n_params,); avoid broadcast producing matrix
        delta = obs_summary .- vec(mean(S, dims=1))
        theta_obs = vec(mean(accepted, dims=1)) .+ B * delta
        adjusted = Matrix{Float64}(undef, n_acc, n_params)
        for i in 1:n_acc
            adjusted[i, :] = theta_obs
        end
    else
        # Local linear: weight by distance; single adjusted point = weighted regression at s_obs
        adjusted = copy(accepted)
        for i in 1:n_acc
            d = vec(sqrt.(sum((S .- obs_summary') .^ 2, dims=2)))
            w = max.(1 .- d ./ (maximum(d) + 1e-10), 0)
            w = w ./ sum(w)
            sw = sum(w)
            mS = sum(w .* eachrow(S))
            mT = sum(w .* eachrow(accepted))
            C = (accepted' .- mT') * (w .* (diff .- (obs_summary' .- mS)))
            V = (diff .- (obs_summary' .- mS))' * (w .* (diff .- (obs_summary' .- mS))) + 1e-10 * I
            B = C * inv(Symmetric(V))
            adjusted[i, :] = mT .+ B * (obs_summary .- mS)
        end
    end
    return adjusted, n_acc
end

# -----------------------------------------------------------------------------
# Model choice ABC
# -----------------------------------------------------------------------------

"""
    ABCModelChoice(base_method::ABCMethod, model_priors::Vector{Float64})

Multiple simulators (one per model). base_method used per model with same observed_data.
simulator(i, params) = i-th model simulator. Returns (accepted_params_per_model, model_probs).
model_probs = posterior P(model | data) ∝ prior(model) * acceptance_rate(model).
"""
struct ABCModelChoice <: ABCMethod
    base_method::ABCMethod
    model_priors::Vector{Float64}
end

function abc_model_choice(
    method::ABCModelChoice,
    simulators::Vector{Function},  # one per model
    summary_stats::Function,
    distance::Function,
    observed_data;
    rng::Union{AbstractRNG,Nothing}=nothing,
)
    (; base_method, model_priors) = method
    K = length(simulators)
    length(model_priors) == K || throw(ArgumentError("model_priors length must match simulators"))
    rng = rng === nothing ? Random.GLOBAL_RNG : rng
    accepted_per_model = Vector{Matrix{Float64}}()
    n_acc = Int[]
    for k in 1:K
        sim_k = params -> simulators[k](params)
        out, n = abc_inference(base_method, sim_k, summary_stats, distance, observed_data; rng=rng)
        push!(accepted_per_model, out)
        push!(n_acc, n)
    end
    total = sum(n_acc)
    model_probs = total > 0 ? (model_priors .* n_acc) ./ (total * sum(model_priors)) : model_priors
    model_probs = model_probs ./ sum(model_probs)
    return accepted_per_model, model_probs
end

# -----------------------------------------------------------------------------
# Hierarchical ABC
# -----------------------------------------------------------------------------

"""
    HierarchicalABC(base_method::ABCMethod, hyperprior::Distribution)

Hierarchical model: hyperparams ~ hyperprior, then params ~ f(hyperparams), then data ~ simulator(params).
simulator(hyperparams) should draw params from the conditional prior given hyperparams, then simulate data.
So simulator(hyperparams) = simulate(given(hyperparams)) -> data.
Inference returns accepted hyperparameters (and optionally per-unit params if desired).
"""
struct HierarchicalABC{T<:Real} <: ABCMethod
    base_method::ABCMethod
    hyperprior::Distribution
end
# Outer constructor so T is fixed when not inferrable from arguments
HierarchicalABC(base_method::ABCMethod, hyperprior::Distribution) = HierarchicalABC{Float64}(base_method, hyperprior)

function abc_inference(
    method::HierarchicalABC,
    simulator::Function,
    summary_stats::Function,
    distance::Function,
    observed_data;
    rng::Union{AbstractRNG,Nothing}=nothing,
)
    # simulator(hyperparams) -> synthetic data (internally draws params from hyperprior conditionals)
    abc_inference(method.base_method, simulator, summary_stats, distance, observed_data; rng=rng)
end
