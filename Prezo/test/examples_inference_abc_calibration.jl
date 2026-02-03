#=
    Examples: ABC variants (MCMC-ABC, SMC-ABC, regression-adjusted, model choice, hierarchical)
    and regularized calibration.

Run from Prezo package root:
    julia --project=. test/examples_inference_abc_calibration.jl
=#

using Prezo
using Statistics
using Random
using Distributions
using LinearAlgebra

Random.seed!(123)

# -----------------------------------------------------------------------------
# Shared setup: Normal(μ, σ) simulator and summaries
# -----------------------------------------------------------------------------
true_μ, true_σ = 0.0, 1.0
observed = randn(200) .* true_σ .+ true_μ
simulator(params) = randn(200) .* max(params[2], 0.1) .+ params[1]
summary_stats(data) = [mean(data), std(data)]
prior = product_distribution([Normal(0, 2), Uniform(0.3, 3.0)])

println("=== 1. Rejection ABC (baseline) ===")
method_rej = RejectionABC(300, 0.2, prior)
accepted, n = abc_inference(method_rej, simulator, summary_stats, euclidean_distance, observed; rng=MersenneTwister(1))
println("  Accepted: $n")
if n >= 5
    println("  Mean(μ): $(round(first(mean(accepted[:,1])), digits=4)), true μ = $true_μ")
    println("  Mean(σ): $(round(first(mean(accepted[:,2])), digits=4)), true σ = $true_σ")
end

println("\n=== 2. MCMC-ABC ===")
method_mcmc = MCMCABC(500, 0.2, prior, [0.15, 0.15])
chain, _ = abc_inference(method_mcmc, simulator, summary_stats, euclidean_distance, observed; rng=MersenneTwister(2))
# Discard burn-in
chain_use = chain[101:end, :]
println("  Chain length: $(size(chain_use, 1))")
println("  Mean(μ): $(round(first(mean(chain_use[:,1])), digits=4))")
println("  Mean(σ): $(round(first(mean(chain_use[:,2])), digits=4))")

println("\n=== 3. SMC-ABC (sequential tolerance) ===")
method_smc = ABCSMC(80, 3, [0.3, 0.2, 0.15], prior, [0.1, 0.1])
particles, _ = abc_inference(method_smc, simulator, summary_stats, euclidean_distance, observed; rng=MersenneTwister(3))
println("  Particles: $(size(particles, 1))")
if size(particles, 1) >= 5
    println("  Mean(μ): $(round(first(mean(particles[:,1])), digits=4))")
    println("  Mean(σ): $(round(first(mean(particles[:,2])), digits=4))")
end

println("\n=== 4. Regression-adjusted ABC ===")
method_reg = RegressionABC(RejectionABC(200, 0.25, prior), :linear)
adjusted, n_reg = abc_inference(method_reg, simulator, summary_stats, euclidean_distance, observed; rng=MersenneTwister(4))
println("  Adjusted samples: $n_reg")
if n_reg >= 5
    println("  Mean adjusted(μ): $(round(first(mean(adjusted[:,1])), digits=4))")
    println("  Mean adjusted(σ): $(round(first(mean(adjusted[:,2])), digits=4))")
end

println("\n=== 5. Model choice ABC (two models: Normal vs Laplace-like) ===")
# Model 1: Normal(μ, σ)
sim1(params) = randn(200) .* max(params[2], 0.1) .+ params[1]
# Model 2: Laplace(μ, b) - different summary behavior (element-wise ±1)
sim2(params) = (1 .- 2 .* (rand(200) .< 0.5)) .* randexp(200) .* max(params[2], 0.1) .+ params[1]
simulators = [sim1, sim2]
method_mc = ABCModelChoice(RejectionABC(150, 0.3, prior), [0.5, 0.5])
accepted_per_model, model_probs = abc_model_choice(method_mc, simulators, summary_stats, euclidean_distance, observed; rng=MersenneTwister(5))
println("  Posterior P(Model 1): $(round(model_probs[1], digits=4))")
println("  Posterior P(Model 2): $(round(model_probs[2], digits=4))")
# Data were Normal, so Model 1 should get higher probability

println("\n=== 6. Hierarchical ABC (hyperprior on (μ, σ)) ===")
# Simulator: draw (μ, σ) from hyperprior, then simulate data
hyperprior = product_distribution([Normal(0, 2), Uniform(0.3, 3.0)])
simulator_hier(hyperparams) = randn(200) .* max(hyperparams[2], 0.1) .+ hyperparams[1]
method_hier = HierarchicalABC(RejectionABC(200, 0.2, hyperprior), hyperprior)
hier_accepted, n_hier = abc_inference(method_hier, simulator_hier, summary_stats, euclidean_distance, observed; rng=MersenneTwister(6))
println("  Accepted hyperparams: $n_hier")
if n_hier >= 5
    println("  Mean(μ): $(round(first(mean(hier_accepted[:,1])), digits=4))")
    println("  Mean(σ): $(round(first(mean(hier_accepted[:,2])), digits=4))")
end

println("\n=== 7. Regularized calibration (option prices + L2 penalty) ===")
data = MarketData(100.0, 0.05, 0.22, 0.0)
options = [EuropeanCall(90.0, 1.0), EuropeanCall(100.0, 1.0), EuropeanCall(110.0, 1.0)]
market_prices = [price(opt, BlackScholes(), data) for opt in options]
target = OptionPricesTarget(options, market_prices)
price_fn(opt, params) = price(opt, BlackScholes(), MarketData(100.0, 0.05, max(params[1], 0.01), 0.0))
ref_vol = 0.20
reg_L2(p) = (p[1] - ref_vol)^2
method_reg_cal = RegularizedCalibration(reg_L2, 0.1)
result_reg = calibrate_option_prices(price_fn, target, [0.21], [0.05], [0.5]; method=method_reg_cal)
println("  Calibrated vol (regularized toward $ref_vol): $(round(result_reg.params[1], digits=4))")
println("  Loss: $(round(result_reg.loss, digits=6)), converged: $(result_reg.converged)")
# Unregularized for comparison
result_ls = calibrate_option_prices(price_fn, target, [0.21], [0.05], [0.5])
println("  Calibrated vol (least squares): $(round(result_ls.params[1], digits=4))")

println("\nDone.")
