"""
    Linear Kalman Filter

Discrete-time linear Gaussian state-space:
  x_t = F * x_{t-1} + w_t,  w_t ~ N(0, Q)
  z_t = H * x_t + v_t,      v_t ~ N(0, R)

Predict: x_pred = F*x, P_pred = F*P*F' + Q
Update:  K = P_pred*H'/(H*P_pred*H' + R), x = x_pred + K*(z - H*x_pred), P = (I - K*H)*P_pred
"""

using LinearAlgebra

"""
    StateSpaceFilter

Abstract type for state-space filters (Kalman family, particle filter).
"""
abstract type StateSpaceFilter end

"""
    KalmanFilterState(x, P, t)

State of the Kalman filter at time step t.
- `x`: state estimate (mean)
- `P`: error covariance matrix
- `t`: time index
"""
struct KalmanFilterState{T<:Real}
    x::Vector{T}
    P::Matrix{T}
    t::Int
end

"""
    KalmanFilter(F, H, Q, R)

Linear Kalman filter.
- `F`: state transition matrix (n×n)
- `H`: observation matrix (m×n)
- `Q`: process noise covariance (n×n)
- `R`: observation noise covariance (m×m)
"""
struct KalmanFilter{T<:Real} <: StateSpaceFilter
    F::Matrix{T}
    H::Matrix{T}
    Q::Matrix{T}
    R::Matrix{T}

    function KalmanFilter(F::Matrix{T}, H::Matrix{T}, Q::Matrix{T}, R::Matrix{T}) where {T<:Real}
        n = size(F, 1)
        m = size(H, 1)
        size(F) == (n, n) || throw(ArgumentError("F must be square"))
        size(H) == (m, n) || throw(ArgumentError("H must be m×n"))
        size(Q) == (n, n) || throw(ArgumentError("Q must be n×n"))
        size(R) == (m, m) || throw(ArgumentError("R must be m×m"))
        new{T}(F, H, Q, R)
    end
end

"""
    predict(filter::KalmanFilter, state::KalmanFilterState) -> KalmanFilterState

Prediction step: x_pred = F*x, P_pred = F*P*F' + Q.
"""
function predict(filter::KalmanFilter, state::KalmanFilterState)
    (; F, Q) = filter
    x_pred = F * state.x
    P_pred = Symmetric(F * state.P * F' + Q)
    return KalmanFilterState(x_pred, Matrix(P_pred), state.t + 1)
end

"""
    update(filter::KalmanFilter, state::KalmanFilterState, z::Vector{<:Real}) -> KalmanFilterState

Update step with observation z. state is the *predicted* state (from predict).
"""
function update(filter::KalmanFilter, state::KalmanFilterState, z::Vector{<:Real})
    (; H, R) = filter
    z = Float64.(z)
    y = z - H * state.x
    S = Symmetric(H * state.P * H' + R)
    S_inv = inv(S)
    K = state.P * H' * S_inv
    x_new = state.x + K * y
    P_new = Symmetric((I - K * H) * state.P)
    return KalmanFilterState(x_new, Matrix(P_new), state.t)
end

"""
    filter_step(filter::KalmanFilter, state::KalmanFilterState, z::Union{Vector{<:Real}, Nothing}) -> KalmanFilterState

One step: predict, then update with z if provided. If z is nothing, returns predicted state only.
"""
function filter_step(
    filter::KalmanFilter,
    state::KalmanFilterState,
    z::Union{Vector{<:Real}, Nothing},
)
    pred = predict(filter, state)
    z === nothing && return pred
    return update(filter, pred, z)
end

"""
    filter_data(
        filter::KalmanFilter,
        observations::AbstractVector{<:AbstractVector{<:Real}},
        initial_state::KalmanFilterState,
    ) -> Vector{KalmanFilterState}

Run filter over observation sequence. observations[t] is the measurement at time t (1-based).
Initial state is the prior at t=0; first observation is at t=1.
Returns filtered states (posterior at each t) of length length(observations).
"""
function filter_data(
    filter::KalmanFilter,
    observations::AbstractVector{<:AbstractVector{<:Real}},
    initial_state::KalmanFilterState,
)
    T = length(observations)
    out = Vector{KalmanFilterState}(undef, T)
    state = initial_state
    for t in 1:T
        state = filter_step(filter, state, observations[t])
        @inbounds out[t] = state
    end
    return out
end

"""
    smooth(filter::KalmanFilter, filtered_states::Vector{KalmanFilterState}) -> Vector{KalmanFilterState}

Rauch–Tung–Striebel (RTS) smoother. filtered_states from filter_data.
Returns smoothed states (same length). Backward pass: smooth state at t uses filtered state at t and smoothed at t+1.
"""
function smooth(
    filter::KalmanFilter,
    filtered_states::Vector{KalmanFilterState},
)
    (; F) = filter
    T = length(filtered_states)
    T == 0 && return KalmanFilterState[]
    smoothed = Vector{KalmanFilterState}(undef, T)
    smoothed[T] = filtered_states[T]
    for t in (T - 1):-1:1
        pred = predict(filter, filtered_states[t])
        P_pred = F * filtered_states[t].P * F' + filter.Q
        J = filtered_states[t].P * F' * inv(Symmetric(P_pred))
        x_s = filtered_states[t].x + J * (smoothed[t + 1].x - pred.x)
        P_s = Matrix(Symmetric(filtered_states[t].P + J * (smoothed[t + 1].P - P_pred) * J'))
        smoothed[t] = KalmanFilterState(x_s, P_s, filtered_states[t].t)
    end
    return smoothed
end
