"""
    GPU acceleration for Monte Carlo path generation and European option pricing

Requires CUDA.jl and a CUDA-capable GPU. Paths are generated on device; payoff and
discount are computed on GPU; only the final scalar price is transferred to CPU.

Use when `reps` is very large (e.g. 10^6+) to benefit from GPU parallelism.
"""

using CUDA
using LinearAlgebra
using Statistics

# -----------------------------------------------------------------------------
# MonteCarloGPU engine
# -----------------------------------------------------------------------------

"""
    MonteCarloGPU(steps, reps)

Monte Carlo pricing engine that runs path generation and payoff on GPU.

# Fields
- `steps::Int`: Number of time steps per path
- `reps::Int`: Number of simulation paths

# Requirements
- CUDA.jl and a CUDA-capable GPU
- Use `asset_paths_col_gpu` for raw GPU path matrix (CuArray)

# Examples
```julia
engine = MonteCarloGPU(100, 1_000_000)
price(EuropeanCall(100.0, 1.0), engine, MarketData(100.0, 0.05, 0.2, 0.0))
```
"""
struct MonteCarloGPU <: PricingEngine
    steps::Int
    reps::Int
end

# -----------------------------------------------------------------------------
# GPU path generation
# -----------------------------------------------------------------------------

"""
    asset_paths_col_gpu(engine::MonteCarlo, spot, rate, vol, expiry)

Generate asset price paths on GPU (column-major: (steps+1) × reps).

Returns a `CuArray{Float64,2}`. Requires a CUDA-capable GPU.

# Arguments
- `engine::MonteCarlo`: Engine with `steps` and `reps`
- `spot`, `rate`, `vol`, `expiry`: GBM parameters

# Returns
`CuArray` of size `(steps+1, reps)`.
"""
function asset_paths_col_gpu(engine::MonteCarlo, spot, rate, vol, expiry)
    (; steps, reps) = engine
    dt = expiry / steps
    nudt = Float64((rate - 0.5 * vol^2) * dt)
    sidt = Float64(vol * sqrt(dt))

    paths = CUDA.zeros(Float64, steps + 1, reps)
    paths[1, :] .= Float64(spot)

    for t in 1:steps
        z = CUDA.randn(Float64, reps)
        paths[t+1, :] .= paths[t, :] .* exp.(nudt .+ sidt .* z)
    end
    return paths
end

# -----------------------------------------------------------------------------
# GPU pricing (European option)
# -----------------------------------------------------------------------------

function price(option::EuropeanOption, engine::MonteCarloGPU, data::MarketData)
    (; steps, reps) = engine
    (; spot, rate, vol) = data
    (; expiry) = option

    paths = asset_paths_col_gpu(MonteCarlo(steps, reps), spot, rate, vol, expiry)
    terminal = paths[end, :]
    payoffs = payoff.(Ref(option), terminal)
    discounted = exp(-rate * expiry) .* payoffs
    # Reduce on GPU; copy scalar to CPU
    option_price = Float64(sum(discounted) / length(discounted))
    return option_price
end
