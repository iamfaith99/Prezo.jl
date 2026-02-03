"""
    Extended Kalman Filter (EKF)

Nonlinear state-space with Gaussian noise, linearized at current estimate:
  x_t = f(x_{t-1}) + w_t,  w_t ~ N(0, Q)
  z_t = h(x_t) + v_t,      v_t ~ N(0, R)

Predict: x_pred = f(x), F = ∂f/∂x, P_pred = F*P*F' + Q
Update:  H = ∂h/∂x at x_pred, then standard Kalman update with z - h(x_pred).
"""

using LinearAlgebra

"""
    ExtendedKalmanFilter(f, h, F_jac, H_jac, Q, R)

EKF for nonlinear f (state transition) and h (observation).
- `f(x::Vector) -> Vector`: state transition
- `h(x::Vector) -> Vector`: observation map
- `F_jac(x::Vector) -> Matrix`: Jacobian of f at x (n×n)
- `H_jac(x::Vector) -> Matrix`: Jacobian of h at x (m×n)
- `Q`, `R`: process and observation noise covariances
"""
struct ExtendedKalmanFilter{T<:Real} <: StateSpaceFilter
    f::Function
    h::Function
    F_jac::Function
    H_jac::Function
    Q::Matrix{T}
    R::Matrix{T}
end

"""
    predict(filter::ExtendedKalmanFilter, state::KalmanFilterState) -> KalmanFilterState

Prediction: x_pred = f(x), P_pred = F*P*F' + Q with F = F_jac(x).
"""
function predict(filter::ExtendedKalmanFilter, state::KalmanFilterState)
    (; f, F_jac, Q) = filter
    x_pred = f(state.x)
    F = F_jac(state.x)
    P_pred = Symmetric(F * state.P * F' + Q)
    return KalmanFilterState(x_pred, Matrix(P_pred), state.t + 1)
end

"""
    update(filter::ExtendedKalmanFilter, state::KalmanFilterState, z::Vector{<:Real}) -> KalmanFilterState

Update with observation z. state is the predicted state. Innovation: z - h(state.x).
"""
function update(filter::ExtendedKalmanFilter, state::KalmanFilterState, z::Vector{<:Real})
    (; h, H_jac, R) = filter
    z = Float64.(z)
    H = H_jac(state.x)
    y = z - h(state.x)
    S = Symmetric(H * state.P * H' + R)
    S_inv = inv(S)
    K = state.P * H' * S_inv
    x_new = state.x + K * y
    P_new = Matrix(Symmetric((I - K * H) * state.P))
    return KalmanFilterState(x_new, P_new, state.t)
end

"""
    filter_step(filter::ExtendedKalmanFilter, state::KalmanFilterState, z::Union{Vector{<:Real}, Nothing}) -> KalmanFilterState
"""
function filter_step(
    filter::ExtendedKalmanFilter,
    state::KalmanFilterState,
    z::Union{Vector{<:Real}, Nothing},
)
    pred = predict(filter, state)
    z === nothing && return pred
    return update(filter, pred, z)
end

"""
    filter_data(
        filter::ExtendedKalmanFilter,
        observations::AbstractVector{<:AbstractVector{<:Real}},
        initial_state::KalmanFilterState,
    ) -> Vector{KalmanFilterState}
"""
function filter_data(
    filter::ExtendedKalmanFilter,
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
