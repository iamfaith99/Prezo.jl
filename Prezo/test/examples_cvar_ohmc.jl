#=
    Examples: CVaR-aware OHMC (Lagrangian CVaR inside OHMC)

Run from Prezo package root:
    julia --project=. test/examples_cvar_ohmc.jl

This demonstrates making OHMC risk-aware by incorporating CVaR directly
into the optimization objective, using the Rockafellar-Uryasev representation.
=#

using Prezo
using Statistics
using Random
using Printf

Random.seed!(42)

# =============================================================================
# 1. Standard OHMC vs CVaR-OHMC Comparison
# =============================================================================
println("="^70)
println("1. Standard OHMC vs CVaR-OHMC Comparison")
println("="^70)

data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
bs_price = price(call, BlackScholes(), data)
println("Black-Scholes price: $(round(bs_price, digits=4))")

# Standard OHMC
std_config = OHMCConfig(5000, 25, 3)
std_result = ohmc_price(call, data, std_config; rng=MersenneTwister(1))
println("\nStandard OHMC:")
println("  Price:     $(round(std_result.option_price, digits=4))")
println("  Variance:  $(round(std_result.hedged_portfolio_variance, digits=6))")
println("  95% CI:    ($(round(std_result.confidence_interval[1], digits=4)), $(round(std_result.confidence_interval[2], digits=4)))")

# CVaR-OHMC with penalty form
cvar_config = CVaROHMCConfig(5000, 25, 3;
    cvar_alpha=0.95,
    cvar_weight=0.2,
    cvar_objective=CVaRPenalty
)
cvar_result = cvar_ohmc_price(call, data, cvar_config; rng=MersenneTwister(1))
println("\nCVaR-OHMC (η=0.2, α=0.95):")
println("  Price:     $(round(cvar_result.option_price, digits=4))")
println("  Variance:  $(round(cvar_result.hedged_portfolio_variance, digits=6))")
println("  CVaR:      $(round(cvar_result.cvar, digits=4))")
println("  VaR:       $(round(cvar_result.var, digits=4))")
println("  Tail paths: $(length(cvar_result.expected_shortfall_paths)) / $(cvar_config.n_paths)")

# =============================================================================
# 2. Effect of CVaR Weight (η) on Tail Risk
# =============================================================================
println("\n" * "="^70)
println("2. Effect of CVaR Weight (η) on Tail Risk")
println("="^70)

println("\nη       | Price   | Variance | CVaR    | VaR     | Base Obj | Total Obj")
println("-"^75)

for eta in [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
    config = CVaROHMCConfig(3000, 20, 3; cvar_alpha=0.95, cvar_weight=eta)
    result = cvar_ohmc_price(call, data, config; rng=MersenneTwister(42))
    println(@sprintf("%.2f    | %.4f  | %.6f | %.4f  | %.4f  | %.6f | %.6f",
        eta, result.option_price, result.hedged_portfolio_variance,
        result.cvar, result.var, result.base_objective, result.total_objective))
end

# =============================================================================
# 3. Effect of CVaR Confidence Level (α)
# =============================================================================
println("\n" * "="^70)
println("3. Effect of CVaR Confidence Level (α)")
println("="^70)

println("\nα       | Price   | CVaR    | VaR     | Tail Mean | # Tail Paths")
println("-"^70)

for alpha in [0.80, 0.90, 0.95, 0.99]
    config = CVaROHMCConfig(3000, 20, 3; cvar_alpha=alpha, cvar_weight=0.2)
    result = cvar_ohmc_price(call, data, config; rng=MersenneTwister(42))
    println(@sprintf("%.2f    | %.4f  | %.4f  | %.4f  | %.4f    | %d",
        alpha, result.option_price, result.cvar, result.var,
        result.tail_loss_mean, length(result.expected_shortfall_paths)))
end

# =============================================================================
# 4. Loss Definition Comparison
# =============================================================================
println("\n" * "="^70)
println("4. Loss Definition Comparison")
println("="^70)

loss_defs = [
    (HedgingError, "Hedging Error"),
    (PortfolioPnL, "Portfolio P&L"),
    (Drawdown, "Drawdown")
]

println("\nLoss Def       | Price   | Variance | CVaR    | CVaR Contrib")
println("-"^65)

for (loss_def, name) in loss_defs
    config = CVaROHMCConfig(3000, 20, 3;
        cvar_alpha=0.95,
        cvar_weight=0.2,
        loss_definition=loss_def
    )
    result = cvar_ohmc_price(call, data, config; rng=MersenneTwister(42))
    println(@sprintf("%-14s | %.4f  | %.6f | %.4f  | %.6f",
        name, result.option_price, result.hedged_portfolio_variance,
        result.cvar, result.cvar_contribution))
end

# =============================================================================
# 5. Scoring Method Comparison (with CVaR)
# =============================================================================
println("\n" * "="^70)
println("5. Scoring Method Comparison (with CVaR)")
println("="^70)

scorings = [
    (:quadratic, "Quadratic"),
    (:log, "Log"),
    (:exponential_utility, "Exp Utility")
]

println("\nScoring        | Price   | Variance | CVaR    | Total Obj")
println("-"^60)

for (scoring, name) in scorings
    config = CVaROHMCConfig(3000, 20, 3;
        cvar_alpha=0.95,
        cvar_weight=0.2,
        scoring=scoring,
        risk_aversion=0.15
    )
    result = cvar_ohmc_price(call, data, config; rng=MersenneTwister(42))
    println(@sprintf("%-14s | %.4f  | %.6f | %.4f  | %.6f",
        name, result.option_price, result.hedged_portfolio_variance,
        result.cvar, result.total_objective))
end

# =============================================================================
# 6. compare_ohmc_cvar Utility
# =============================================================================
println("\n" * "="^70)
println("6. Standard vs CVaR-OHMC Comparison Utility")
println("="^70)

comparison = compare_ohmc_cvar(call, data, 3000, 20, 3;
    cvar_alpha=0.95, cvar_weight=0.25, rng=MersenneTwister(42))

println("\nStandard OHMC:")
println("  Price:    $(round(comparison.standard_price, digits=4))")
println("  Variance: $(round(comparison.standard_variance, digits=6))")
println("  CVaR (post-hoc): $(round(comparison.standard_cvar, digits=4))")

println("\nCVaR-OHMC:")
println("  Price:    $(round(comparison.cvar_price, digits=4))")
println("  Variance: $(round(comparison.cvar_variance, digits=6))")
println("  CVaR:     $(round(comparison.cvar_cvar, digits=4))")

println("\nImprovements:")
println("  Variance reduction: $(round(comparison.variance_reduction_pct, digits=2))%")
println("  CVaR reduction:     $(round(comparison.cvar_reduction_pct, digits=2))%")

# =============================================================================
# 7. Adaptive CVaR Weight Selection
# =============================================================================
println("\n" * "="^70)
println("7. Adaptive CVaR Weight Selection")
println("="^70)

base_config = CVaROHMCConfig(2000, 15, 2; cvar_alpha=0.95, cvar_weight=0.1)
adaptive_config = AdaptiveCVaRConfig(
    base_config,
    0.01,   # eta_min
    1.0,    # eta_max
    8,      # eta_steps
    1.5     # target_cvar
)

println("Searching for optimal η to achieve target CVaR ≤ $(adaptive_config.target_cvar)...")
adaptive_result = adaptive_cvar_ohmc(call, data, adaptive_config; rng=MersenneTwister(42))

println("\nAdaptive selection result:")
println("  Optimal η:    $(round(adaptive_result.optimal_eta, digits=4))")
println("  Final score:  $(round(adaptive_result.final_score, digits=6))")
println("  Price:        $(round(adaptive_result.result.option_price, digits=4))")
println("  Achieved CVaR: $(round(adaptive_result.result.cvar, digits=4))")

# =============================================================================
# 8. Multi-Period CVaR (Path-Dependent Tail Risk)
# =============================================================================
println("\n" * "="^70)
println("8. Multi-Period CVaR (Path-Dependent Tail Risk)")
println("="^70)

base_config = CVaROHMCConfig(3000, 20, 2; cvar_alpha=0.95, cvar_weight=0.2)
multiperiod_config = MultiPeriodCVaRConfig(
    base_config,
    [0.25, 0.5, 0.75, 1.0],  # Quarterly evaluation times
    [0.2, 0.3, 0.3, 0.2]     # Weights (emphasize middle of life)
)

println("Evaluating CVaR at times: $(multiperiod_config.intermediate_times)")
println("With weights: $(multiperiod_config.time_weights)")

result = multiperiod_cvar_ohmc(call, data, multiperiod_config; rng=MersenneTwister(42))

println("\nMulti-period CVaR-OHMC result:")
println("  Price:            $(round(result.option_price, digits=4))")
println("  Terminal CVaR:    $(round(result.cvar, digits=4))")
println("  CVaR contribution: $(round(result.cvar_contribution, digits=6))")
println("  Total objective:  $(round(result.total_objective, digits=6))")

# =============================================================================
# 9. Constrained CVaR-OHMC (Lagrangian Dual)
# =============================================================================
println("\n" * "="^70)
println("9. Constrained CVaR-OHMC (CVaR ≤ Budget)")
println("="^70)

config = CVaROHMCConfig(2500, 15, 2;
    cvar_alpha=0.95,
    cvar_weight=0.1,          # Initial Lagrange multiplier
    cvar_objective=CVaRConstraint,
    cvar_budget=1.5           # Maximum allowed CVaR
)

println("Solving: min E[ℓ(e)] s.t. CVaR_0.95(L) ≤ $(config.cvar_budget)")

result = constrained_cvar_ohmc(call, data, config;
    rng=MersenneTwister(42),
    max_dual_iterations=15,
    dual_step_size=0.2,
    tolerance=0.1
)

println("\nConstrained optimization result:")
println("  Price:              $(round(result.primal_result.option_price, digits=4))")
println("  Final CVaR:         $(round(result.primal_result.cvar, digits=4))")
println("  Budget:             $(config.cvar_budget)")
println("  Constraint slack:   $(round(result.constraint_slack, digits=4)) (positive = feasible)")
println("  Final λ (dual var): $(round(result.lagrange_multiplier, digits=4))")
println("  Dual iterations:    $(length(result.dual_convergence))")

feasible = result.constraint_slack >= 0
println("  Feasible:           $(feasible ? "✓ Yes" : "✗ No")")

# =============================================================================
# 10. CVaR-OHMC with Transaction Costs
# =============================================================================
println("\n" * "="^70)
println("10. CVaR-OHMC with Transaction Costs")
println("="^70)

println("\nTC Rate | Price   | Variance | CVaR    | Base Obj | Total Obj")
println("-"^65)

for tc_rate in [0.0, 0.0005, 0.001, 0.002, 0.005]
    config = CVaROHMCConfig(2500, 20, 3;
        cvar_alpha=0.95,
        cvar_weight=0.2,
        tc_rate=tc_rate
    )
    result = cvar_ohmc_price(call, data, config; rng=MersenneTwister(42))
    println(@sprintf("%.4f  | %.4f  | %.6f | %.4f  | %.6f | %.6f",
        tc_rate, result.option_price, result.hedged_portfolio_variance,
        result.cvar, result.base_objective, result.total_objective))
end

# =============================================================================
# 11. ν (VaR) Convergence Diagnostics
# =============================================================================
println("\n" * "="^70)
println("11. ν (VaR Threshold) Convergence")
println("="^70)

config = CVaROHMCConfig(3000, 20, 3;
    cvar_alpha=0.95,
    cvar_weight=0.25,
    outer_iterations=6,
    nu_iterations=15
)

result = cvar_ohmc_price(call, data, config; rng=MersenneTwister(42))

println("\nν convergence across outer iterations:")
for (i, nu) in enumerate(result.nu_convergence)
    if i == 1
        println("  Initial: $(round(nu, digits=4))")
    else
        change = nu - result.nu_convergence[i-1]
        println("  Iter $(i-1):   $(round(nu, digits=4)) (Δ = $(round(change, digits=4)))")
    end
end
println("\nFinal VaR (ν*): $(round(result.var, digits=4))")
println("Final CVaR:     $(round(result.cvar, digits=4))")

# =============================================================================
# 12. Put Option with CVaR-OHMC
# =============================================================================
println("\n" * "="^70)
println("12. Put Option with CVaR-OHMC")
println("="^70)

put = EuropeanPut(100.0, 1.0)
bs_put = price(put, BlackScholes(), data)
println("Put Black-Scholes price: $(round(bs_put, digits=4))")

config = CVaROHMCConfig(3000, 20, 3; cvar_alpha=0.95, cvar_weight=0.2)
result = cvar_ohmc_price(put, data, config; rng=MersenneTwister(42))

println("\nCVaR-OHMC Put:")
println("  Price:    $(round(result.option_price, digits=4))")
println("  Variance: $(round(result.hedged_portfolio_variance, digits=6))")
println("  CVaR:     $(round(result.cvar, digits=4))")
println("  VaR:      $(round(result.var, digits=4))")

# =============================================================================
# Summary
# =============================================================================
println("\n" * "="^70)
println("Summary: Lagrangian CVaR inside OHMC")
println("="^70)
println("""
Key Takeaways:

1. CVaR-OHMC adds tail-risk awareness to hedge ratio optimization
2. Uses Rockafellar-Uryasev: CVaR_α(L) = min_ν [ν + E[(L-ν)⁺]/(1-α)]
3. Joint optimization over (hedge ratio a, VaR threshold ν)
4. Higher η → more conservative hedges (lower CVaR, possibly higher variance)
5. Higher α → focus on more extreme tail (95% vs 99% CVaR)
6. Penalty vs Constraint forms available via Lagrangian duality
7. Transaction costs naturally integrate into the objective
8. Multi-period CVaR captures drawdown risk during hedge lifetime
""")

println("\nDone.")
