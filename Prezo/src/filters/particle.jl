"""
    Particle Filter (Sequential Monte Carlo, SIR)

State-space with arbitrary process and observation models. Maintains a weighted
ensemble (particles); weights updated by observation likelihood; resample when
effective sample size falls below threshold.

  x_t ~ f(x_{t-1})   (propagate; f can include process noise)
  z_t | x_t ~ p(z_t | x_t)   (observation likelihood)

Weights: w_i ∝ p(z | x_i). Resample when ESS < threshold * n_particles.
"""

using LinearAlgebra
using Statistics
using Random
using Distributions

"""
    ParticleState(particles, weights, t)

- `particles`: N×D matrix (N particles, D state dimensions)
- `weights`: length-N normalized weights (sum = 1)
- `t`: time index
"""
struct ParticleState{T<:Real}
    particles::Matrix{T}
    weights::Vector{T}
    t::Int
end

"""
    ParticleFilter(f, h, R, n_particles, resample_threshold)

SIR particle filter with Gaussian observation likelihood.
- `f(x::Vector[, rng]) -> Vector`: state transition (optional rng for process noise)
- `h(x::Vector) -> Vector`: observation map (for likelihood: z ~ N(h(x), R))
- `R`: observation noise covariance (for weighting)
- `n_particles::Int`: number of particles
- `resample_threshold::Float64`: resample when ESS < threshold * n_particles (e.g. 0.5)
"""
struct ParticleFilter{T<:Real} <: StateSpaceFilter
    f::Function      # f(x, rng) -> x_new
    h::Function
    R::Matrix{T}
    n_particles::Int
    resample_threshold::Float64
end

"""
    effective_sample_size(weights::Vector{<:Real}) -> Float64

ESS = 1 / sum(w_i^2). Used to decide when to resample.
"""
function effective_sample_size(weights::Vector{<:Real})
    w = Float64.(weights)
    s = sum(w .^ 2)
    s <= 0 && return 0.0
    return 1.0 / s
end

"""
    systematic_resample(weights::Vector{<:Real}; rng=GLOBAL_RNG) -> Vector{Int}

Systematic resampling: returns indices into the particle array (length N).
"""
function systematic_resample(
    weights::Vector{<:Real};
    rng::Union{AbstractRNG,Nothing}=nothing,
)
    rng = rng === nothing ? Random.GLOBAL_RNG : rng
    N = length(weights)
    u0 = rand(rng)
    indices = Vector{Int}(undef, N)
    c = 0.0
    j = 1
    for i in 1:N
        c += weights[i]
        while c > (i - 1 + u0) / N && j <= N
            indices[j] = i
            j += 1
        end
    end
    # Handle numerical edge: ensure we have exactly N indices
    while j <= N
        indices[j] = N
        j += 1
    end
    return indices
end

"""
    multinomial_resample(weights::Vector{<:Real}; rng=GLOBAL_RNG) -> Vector{Int}

Multinomial resampling: draw N indices with replacement according to weights.
"""
function multinomial_resample(
    weights::Vector{<:Real};
    rng::Union{AbstractRNG,Nothing}=nothing,
)
    rng = rng === nothing ? Random.GLOBAL_RNG : rng
    N = length(weights)
    indices = rand(rng, Categorical(weights), N)
    return indices
end

"""
    stratified_resample(weights::Vector{<:Real}; rng=GLOBAL_RNG) -> Vector{Int}

Stratified resampling: one draw per stratum [ (i-1)/N, i/N ).
"""
function stratified_resample(
    weights::Vector{<:Real};
    rng::Union{AbstractRNG,Nothing}=nothing,
)
    rng = rng === nothing ? Random.GLOBAL_RNG : rng
    N = length(weights)
    cdf = cumsum(weights)
    indices = Vector{Int}(undef, N)
    for i in 1:N
        u = (i - 1 + rand(rng)) / N
        idx = searchsortedfirst(cdf, u)
        indices[i] = min(idx, N)
    end
    return indices
end

"""
    update(
        filter::ParticleFilter,
        state::ParticleState,
        z::Vector{<:Real};
        rng=GLOBAL_RNG,
        resample_fn=systematic_resample,
    ) -> ParticleState

One step: propagate particles, compute weights from p(z|x), normalize, resample if ESS < threshold.
"""
function update(
    filter::ParticleFilter,
    state::ParticleState,
    z::Vector{<:Real};
    rng::Union{AbstractRNG,Nothing}=nothing,
    resample_fn=systematic_resample,
)
    (; f, h, R, n_particles, resample_threshold) = filter
    rng = rng === nothing ? Random.GLOBAL_RNG : rng
    z = Float64.(z)
    N, D = size(state.particles)
    # Propagate (f may take (x) or (x, rng))
    particles_new = Matrix{Float64}(undef, N, D)
    for i in 1:N
        x = state.particles[i, :]
        particles_new[i, :] = applicable(filter.f, x, rng) ? filter.f(x, rng) : filter.f(x)
    end
    # Weights: log w_i = -0.5 * (z - h(x_i))' * inv(R) * (z - h(x_i))  (Gaussian)
    R_chol = cholesky(Symmetric(R))
    R_inv = inv(R_chol)
    log_w = zeros(N)
    for i in 1:N
        y = z - h(particles_new[i, :])
        log_w[i] = -0.5 * dot(y, R_inv \ y)
    end
    # Normalize (log-sum-exp for stability)
    m = maximum(log_w)
    w = exp.(log_w .- m)
    s = sum(w)
    w = s > 0 ? w ./ s : fill(1.0 / N, N)
    # Resample if ESS too low
    ess = effective_sample_size(w)
    if ess < resample_threshold * N
        idx = resample_fn(w; rng=rng)
        particles_new = particles_new[idx, :]
        w = fill(1.0 / N, N)
    end
    return ParticleState(particles_new, w, state.t + 1)
end

"""
    filter_data(
        filter::ParticleFilter,
        observations::AbstractVector{<:AbstractVector{<:Real}},
        initial_particles::Matrix{<:Real};
        rng=GLOBAL_RNG,
        resample_fn=systematic_resample,
    ) -> Vector{ParticleState}

initial_particles is (n_particles × n_state).
"""
function filter_data(
    filter::ParticleFilter,
    observations::AbstractVector{<:AbstractVector{<:Real}},
    initial_particles::Matrix{<:Real};
    rng::Union{AbstractRNG,Nothing}=nothing,
    resample_fn=systematic_resample,
)
    T = length(observations)
    out = Vector{ParticleState}(undef, T)
    N = size(initial_particles, 1)
    w0 = fill(1.0 / N, N)
    state = ParticleState(Matrix{Float64}(initial_particles), w0, 0)
    for t in 1:T
        state = update(filter, state, observations[t]; rng=rng, resample_fn=resample_fn)
        @inbounds out[t] = state
    end
    return out
end
