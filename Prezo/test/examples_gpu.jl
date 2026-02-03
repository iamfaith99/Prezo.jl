#=
    Examples: GPU acceleration (CUDA) for Monte Carlo

Run from Prezo package root:
    julia --project=. test/examples_gpu.jl

Requires CUDA.jl and a CUDA-capable GPU. If no GPU is available, the script
prints a message and exits.
=#

using Prezo
using CUDA

if !CUDA.functional()
    println("No CUDA GPU available. GPU examples skipped.")
    exit(0)
end

println("=== GPU Monte Carlo (CUDA) ===")
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)

# Raw GPU path matrix
engine = MonteCarlo(100, 50_000)
paths = asset_paths_col_gpu(engine, 100.0, 0.05, 0.2, 1.0)
println("  Paths on GPU: size $(size(paths)), type $(typeof(paths))")

# Price via MonteCarloGPU engine
engine_gpu = MonteCarloGPU(100, 50_000)
price_gpu = price(call, engine_gpu, data)
price_bs = price(call, BlackScholes(), data)
println("  MonteCarloGPU price: $(round(price_gpu, digits=4))")
println("  Black-Scholes:       $(round(price_bs, digits=4))")

println("\nDone.")