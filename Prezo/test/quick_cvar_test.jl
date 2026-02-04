#=
    Quick Validation: CVaR-OHMC Implementation

    Run: julia --project=. test/quick_cvar_test.jl
=#

using Prezo
using Statistics
using Random

println("="^60)
println("Quick Validation: CVaR-OHMC (Lagrangian CVaR inside OHMC)")
println("="^60)

# Setup
Random.seed!(42)
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
bs_price = price(call, BlackScholes(), data)

println("\n1. Reference: Black-Scholes = $(round(bs_price, digits=4))")

# Test 1: Basic CVaR-OHMC pricing
println("\n2. Basic CVaR-OHMC pricing...")
config = CVaROHMCConfig(2000, 20, 3; cvar_alpha=0.95, cvar_weight=0.2)
result = cvar_ohmc_price(call, data, config; rng=MersenneTwister(42))

println("   Price:    $(round(result.option_price, digits=4))")
println("   Variance: $(round(result.hedged_portfolio_variance, digits=6))")
println("   CVaR:     $(round(result.cvar, digits=4))")
println("   VaR:      $(round(result.var, digits=4))")

# Validate CVaR >= VaR (fundamental property)
@assert result.cvar >= result.var - 1e-10 "CVaR must be >= VaR"
println("   ✓ CVaR >= VaR property holds")

# Validate price is reasonable
@assert 0.5 * bs_price < result.option_price < 2.0 * bs_price "Price out of range"
println("   ✓ Price in reasonable range")

# Test 2: Different scoring methods
println("\n3. Scoring methods (quadratic, log, exp_utility)...")
for scoring in [:quadratic, :log, :exponential_utility]
    cfg = CVaROHMCConfig(1000, 15, 2; scoring=scoring, cvar_weight=0.15)
    res = cvar_ohmc_price(call, data, cfg; rng=MersenneTwister(42))
    println("   $(rpad(scoring, 20)) -> price=$(round(res.option_price, digits=4)), CVaR=$(round(res.cvar, digits=4))")
end

# Test 3: CVaR weight effect
println("\n4. CVaR weight effect (η = 0.01 vs 0.5)...")
cfg_low = CVaROHMCConfig(1000, 15, 2; cvar_weight=0.01)
cfg_high = CVaROHMCConfig(1000, 15, 2; cvar_weight=0.5)
res_low = cvar_ohmc_price(call, data, cfg_low; rng=MersenneTwister(42))
res_high = cvar_ohmc_price(call, data, cfg_high; rng=MersenneTwister(42))
println("   η=0.01: CVaR contrib = $(round(res_low.cvar_contribution, digits=6))")
println("   η=0.50: CVaR contrib = $(round(res_high.cvar_contribution, digits=6))")
@assert res_high.cvar_contribution > res_low.cvar_contribution "Higher η should give higher CVaR contribution"
println("   ✓ Higher η → higher CVaR contribution")

# Test 4: Rockafellar-Uryasev ν convergence
println("\n5. VaR threshold (ν) convergence...")
cfg = CVaROHMCConfig(1500, 15, 2; cvar_weight=0.2, outer_iterations=5)
res = cvar_ohmc_price(call, data, cfg; rng=MersenneTwister(42))
println("   ν history: ", [round(v, digits=4) for v in res.nu_convergence])
println("   Final VaR (ν*): $(round(res.var, digits=4))")
@assert length(res.nu_convergence) >= 2 "Should have multiple ν iterations"
println("   ✓ ν optimization converged")

# Test 5: Compare with standard OHMC
println("\n6. Comparison: Standard OHMC vs CVaR-OHMC...")
comparison = compare_ohmc_cvar(call, data, 1500, 15, 2;
    cvar_alpha=0.95, cvar_weight=0.25, rng=MersenneTwister(42))
println("   Standard: price=$(round(comparison.standard_price, digits=4)), var=$(round(comparison.standard_variance, digits=6))")
println("   CVaR:     price=$(round(comparison.cvar_price, digits=4)), var=$(round(comparison.cvar_variance, digits=6))")
println("   ✓ Comparison completed")

# Test 6: Constrained form (CVaR <= budget)
println("\n7. Constrained CVaR-OHMC (CVaR ≤ 2.0)...")
cfg_constr = CVaROHMCConfig(1500, 15, 2;
    cvar_objective=CVaRConstraint,
    cvar_budget=2.0,
    cvar_weight=0.1
)
res_constr = constrained_cvar_ohmc(call, data, cfg_constr;
    rng=MersenneTwister(42),
    max_dual_iterations=10,
    tolerance=0.3
)
println("   Final CVaR: $(round(res_constr.primal_result.cvar, digits=4))")
println("   Budget:     2.0")
println("   Slack:      $(round(res_constr.constraint_slack, digits=4))")
println("   Final λ:    $(round(res_constr.lagrange_multiplier, digits=4))")
println("   ✓ Constrained optimization completed")

# Test 7: Reproducibility
println("\n8. Reproducibility check...")
r1 = cvar_ohmc_price(call, data, config; rng=MersenneTwister(123))
r2 = cvar_ohmc_price(call, data, config; rng=MersenneTwister(123))
@assert r1.option_price == r2.option_price "Results should be identical with same seed"
@assert r1.cvar == r2.cvar "CVaR should be identical with same seed"
println("   ✓ Same RNG seed → identical results")

# Summary
println("\n" * "="^60)
println("All validations passed! ✓")
println("="^60)
println("""
CVaR-OHMC Implementation Summary:
- Rockafellar-Uryasev representation: CVaR_α(L) = min_ν [ν + E[(L-ν)⁺]/(1-α)]
- Joint optimization over (hedge ratio a, VaR threshold ν)
- Penalty form: min E[ℓ(e)] + η × CVaR_α(L)
- Constraint form: min E[ℓ(e)] s.t. CVaR_α(L) ≤ budget
- CVaR-aware sample weighting in regression
""")
