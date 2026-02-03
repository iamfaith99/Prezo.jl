"""
    Filters

State-space filters for Phase 3: Kalman family (linear, extended, ensemble) and
particle filter (sequential Monte Carlo).

# Kalman Filter (linear)
- [`KalmanFilter`](@ref), [`KalmanFilterState`](@ref)
- `predict`, `update`, `filter_step`, `filter_data`, `smooth`

# Extended Kalman Filter (EKF)
- [`ExtendedKalmanFilter`](@ref): nonlinear f, h with Jacobians
- Same interface: `predict`, `update`, `filter_data`

# Ensemble Kalman Filter (EnKF)
- [`EnsembleKalmanFilter`](@ref), [`EnsembleKalmanState`](@ref)
- `predict`, `update`, `filter_data` (initial_ensemble)

# Particle Filter
- [`ParticleFilter`](@ref), [`ParticleState`](@ref)
- `update`, `filter_data`, `effective_sample_size`
- Resampling: `systematic_resample`, `multinomial_resample`, `stratified_resample`
"""
module Filters

using LinearAlgebra
using Statistics
using Random
using Distributions

include("kalman.jl")
include("extended_kalman.jl")
include("ensemble_kalman.jl")
include("particle.jl")

export StateSpaceFilter
export KalmanFilter, KalmanFilterState
export predict, update, filter_step, filter_data, smooth
export ExtendedKalmanFilter
export EnsembleKalmanFilter, EnsembleKalmanState
export ParticleFilter, ParticleState
export effective_sample_size
export systematic_resample, multinomial_resample, stratified_resample

end # module Filters
