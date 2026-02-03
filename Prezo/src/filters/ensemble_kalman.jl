"""
    Ensemble Kalman Filter (EnKF)

Particle-based approximation: maintain an ensemble of states; mean and sample covariance
approximate the Kalman state and P. No need for Jacobians; works for nonlinear f, h.

Predict: each member x_i = f(x_i) + w_i, w_i ~ N(0,Q).
Update: perturb observations z_j = z + v_j, v_j ~ N(0,R); then Kalman update for each member
using ensemble mean/covariance (or square-root form). Standard EnKF: update each member
with same gain K from ensemble covariance.
"""

using LinearAlgebra
using Statistics
using Random

"""
    EnsembleKalmanState(ensemble::Matrix{T}, t::Int)

Ensemble of states: each column is one member (n_state × n_ensemble), or we use (n_ensemble × n_state).
Design: particles N×D (N particles, D dimensions). So ensemble is (n_ensemble × n_state).
"""
struct EnsembleKalmanState{T<:Real}
    ensemble::Matrix{T}  # n_ensemble × n_state
    t::Int
end

"""
    EnsembleKalmanFilter(f, h, n_ensemble, Q, R)

- `f(x::Vector) -> Vector`: state transition (deterministic; we add Q noise in predict)
- `h(x::Vector) -> Vector`: observation map
- `n_ensemble::Int`: ensemble size
- `Q`, `R`: process and observation noise covariances
"""
struct EnsembleKalmanFilter{T<:Real} <: StateSpaceFilter
    f::Function
    h::Function
    n_ensemble::Int
    Q::Matrix{T}
    R::Matrix{T}
end

function _ensemble_mean(ens::Matrix{<:Real})
    return vec(mean(ens, dims=1))
end

function _ensemble_cov(ens::Matrix{<:Real})
    n_ens, n_state = size(ens)
    μ = _ensemble_mean(ens)
    X = ens .- μ'
    return (X' * X) / max(n_ens - 1, 1)
end

"""
    predict(filter::EnsembleKalmanFilter, state::EnsembleKalmanState; rng=GLOBAL_RNG) -> EnsembleKalmanState

Propagate each member: x_i = f(x_i) + w_i, w_i ~ N(0, Q).
"""
function predict(
    filter::EnsembleKalmanFilter,
    state::EnsembleKalmanState;
    rng::Union{AbstractRNG,Nothing}=nothing,
)
    (; f, n_ensemble, Q) = filter
    rng = rng === nothing ? Random.GLOBAL_RNG : rng
    n_state = size(state.ensemble, 2)
    ens_new = Matrix{Float64}(undef, n_ensemble, n_state)
    L = cholesky(Symmetric(Q)).L
    for i in 1:n_ensemble
        x = state.ensemble[i, :]
        ens_new[i, :] = f(x) + L * randn(rng, n_state)
    end
    return EnsembleKalmanState(ens_new, state.t + 1)
end

"""
    update(filter::EnsembleKalmanFilter, state::EnsembleKalmanState, z::Vector{<:Real}; rng=GLOBAL_RNG) -> EnsembleKalmanState

Perturbed-observation EnKF: perturb z with R noise, then update each member using ensemble gain.
"""
function update(
    filter::EnsembleKalmanFilter,
    state::EnsembleKalmanState,
    z::Vector{<:Real};
    rng::Union{AbstractRNG,Nothing}=nothing,
)
    (; h, n_ensemble, R) = filter
    rng = rng === nothing ? Random.GLOBAL_RNG : rng
    z = Float64.(z)
    n_state = size(state.ensemble, 2)
    m_obs = length(z)
    # Ensemble of predictions in observation space: H_ens[i,:] = h(ensemble[i,:])
    H_ens = Matrix{Float64}(undef, n_ensemble, m_obs)
    for i in 1:n_ensemble
        H_ens[i, :] = h(state.ensemble[i, :])
    end
    y_mean = vec(mean(H_ens, dims=1))
    X = state.ensemble .- _ensemble_mean(state.ensemble)'
    Y = H_ens .- y_mean'
    P_xy = (X' * Y) / max(n_ensemble - 1, 1)
    P_yy = (Y' * Y) / max(n_ensemble - 1, 1) + R
    K = P_xy / Symmetric(P_yy)
    L_R = cholesky(Symmetric(R)).L
    ens_new = copy(state.ensemble)
    for i in 1:n_ensemble
        z_pert = z + L_R * randn(rng, m_obs)
        ens_new[i, :] = state.ensemble[i, :] + K * (z_pert - H_ens[i, :])
    end
    return EnsembleKalmanState(ens_new, state.t)
end

"""
    filter_step(filter::EnsembleKalmanFilter, state::EnsembleKalmanState, z::Union{Vector{<:Real}, Nothing}; rng=GLOBAL_RNG) -> EnsembleKalmanState
"""
function filter_step(
    filter::EnsembleKalmanFilter,
    state::EnsembleKalmanState,
    z::Union{Vector{<:Real}, Nothing};
    rng::Union{AbstractRNG,Nothing}=nothing,
)
    pred = predict(filter, state; rng=rng)
    z === nothing && return pred
    return update(filter, pred, z; rng=rng)
end

"""
    filter_data(
        filter::EnsembleKalmanFilter,
        observations::AbstractVector{<:AbstractVector{<:Real}},
        initial_ensemble::Matrix{<:Real};
        rng=GLOBAL_RNG,
    ) -> Vector{EnsembleKalmanState}

initial_ensemble is (n_ensemble × n_state).
"""
function filter_data(
    filter::EnsembleKalmanFilter,
    observations::AbstractVector{<:AbstractVector{<:Real}},
    initial_ensemble::Matrix{<:Real};
    rng::Union{AbstractRNG,Nothing}=nothing,
)
    T = length(observations)
    out = Vector{EnsembleKalmanState}(undef, T)
    state = EnsembleKalmanState(Matrix{Float64}(initial_ensemble), 0)
    for t in 1:T
        state = filter_step(filter, state, observations[t]; rng=rng)
        @inbounds out[t] = state
    end
    return out
end
