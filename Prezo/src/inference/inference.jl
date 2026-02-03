"""
    Inference

Phase 4: MLE, calibration, and Approximate Bayesian Computation (ABC).

# MLE
- [`MLEProblem`](@ref), [`MLESolution`](@ref)
- `solve(mle; method=:LBFGS)`, `standard_errors(solution)`

# Calibration
- [`CalibrationTarget`](@ref), [`OptionPricesTarget`](@ref), [`IVSurfaceTarget`](@ref)
- [`CalibrationMethod`](@ref), [`LeastSquaresCalibration`](@ref)
- [`calibrate`](@ref), [`calibrate_option_prices`](@ref), [`CalibrationResult`](@ref)

# ABC
- [`ABCMethod`](@ref), [`RejectionABC`](@ref)
- `abc_inference(method, simulator, summary_stats, distance, observed_data; rng)`
- `euclidean_distance(a, b)` for summary distance
"""
module Inference

using Optim
using LinearAlgebra
using ForwardDiff
using Distributions
using Random
using Statistics

include("mle.jl")
include("calibration.jl")
include("abc.jl")
include("abc_variants.jl")

export MLEProblem, MLESolution, solve, standard_errors
export CalibrationTarget, OptionPricesTarget, IVSurfaceTarget
export CalibrationMethod, LeastSquaresCalibration, RegularizedCalibration
export calibrate, calibrate_option_prices, CalibrationResult
export ABCMethod, RejectionABC
export MCMCABC, ABCSMC, RegressionABC, ABCModelChoice, HierarchicalABC
export abc_inference, abc_model_choice, euclidean_distance

end # module Inference
