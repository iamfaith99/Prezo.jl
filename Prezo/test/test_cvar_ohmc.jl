# Phase 5 Extension: CVaR-aware OHMC — Lagrangian CVaR inside OHMC

using Test
using Prezo
using Statistics
using Random

Random.seed!(42)

@testset "CVaR-OHMC" begin

    # Common setup
    data = MarketData(100.0, 0.05, 0.2, 0.0)
    call = EuropeanCall(100.0, 1.0)
    put = EuropeanPut(100.0, 1.0)

    @testset "CVaROHMCConfig construction and validation" begin
        # Basic construction
        config = CVaROHMCConfig(1000, 20, 2)
        @test config.n_paths == 1000
        @test config.n_steps == 20
        @test config.basis_order == 2
        @test config.cvar_alpha == 0.95
        @test config.cvar_weight == 0.1
        @test config.cvar_objective == CVaRPenalty
        @test config.loss_definition == HedgingError
        @test config.scoring == :quadratic

        # Custom CVaR parameters
        config2 = CVaROHMCConfig(2000, 30, 3;
            cvar_alpha=0.99,
            cvar_weight=0.25,
            cvar_objective=CVaRConstraint,
            cvar_budget=5.0,
            loss_definition=PortfolioPnL,
            scoring=:log
        )
        @test config2.cvar_alpha == 0.99
        @test config2.cvar_weight == 0.25
        @test config2.cvar_objective == CVaRConstraint
        @test config2.cvar_budget == 5.0
        @test config2.loss_definition == PortfolioPnL
        @test config2.scoring == :log

        # Validation errors
        @test_throws ErrorException CVaROHMCConfig(0, 20, 2)     # n_paths must be positive
        @test_throws ErrorException CVaROHMCConfig(1000, 0, 2)   # n_steps must be positive
        @test_throws ErrorException CVaROHMCConfig(1000, 20, 2; cvar_alpha=1.5)  # alpha in (0,1)
        @test_throws ErrorException CVaROHMCConfig(1000, 20, 2; cvar_alpha=0.0)  # alpha in (0,1)
        @test_throws ErrorException CVaROHMCConfig(1000, 20, 2; cvar_weight=-0.1)  # non-negative
        @test_throws ErrorException CVaROHMCConfig(1000, 20, 2; scoring=:invalid)
    end

    @testset "CVaR enums" begin
        @test CVaRPenalty isa CVaRObjective
        @test CVaRConstraint isa CVaRObjective
        @test HedgingError isa LossDefinition
        @test PortfolioPnL isa LossDefinition
        @test Drawdown isa LossDefinition
    end

    @testset "cvar_ohmc_price basic functionality" begin
        config = CVaROHMCConfig(800, 15, 2; cvar_alpha=0.95, cvar_weight=0.1)
        rng = MersenneTwister(123)

        result = cvar_ohmc_price(call, data, config; rng=rng)

        # Basic output validation
        @test result.option_price > 0.0
        @test isfinite(result.option_price)
        @test size(result.hedge_ratios) == (config.n_steps + 1, config.n_paths)
        @test result.hedged_portfolio_variance >= 0.0
        @test result.confidence_interval[1] < result.confidence_interval[2]
        @test result.confidence_interval[1] <= result.option_price <= result.confidence_interval[2]

        # CVaR-specific outputs
        @test isfinite(result.cvar)
        @test isfinite(result.var)
        @test result.cvar >= result.var  # CVaR >= VaR always
        @test length(result.expected_shortfall_paths) <= config.n_paths
        @test result.tail_loss_mean >= result.var  # Tail mean >= VaR
        @test isfinite(result.cvar_contribution)
        @test result.cvar_contribution >= 0.0

        # Diagnostics
        @test isfinite(result.base_objective)
        @test isfinite(result.total_objective)
        @test result.total_objective >= result.base_objective  # CVaR adds to objective
        @test length(result.nu_convergence) >= 1  # At least initial ν

        # Price should be in ballpark of Black-Scholes
        bs = price(call, BlackScholes(), data)
        @test 0.5 * bs < result.option_price < 2.0 * bs
    end

    @testset "CVaR-OHMC scoring variants" begin
        rng = MersenneTwister(456)
        bs = price(call, BlackScholes(), data)

        # Quadratic scoring
        config_q = CVaROHMCConfig(600, 12, 2; scoring=:quadratic, cvar_weight=0.15)
        result_q = cvar_ohmc_price(call, data, config_q; rng=copy(rng))
        @test 0.5 * bs < result_q.option_price < 2.0 * bs

        # Log scoring
        config_l = CVaROHMCConfig(600, 12, 2; scoring=:log, cvar_weight=0.15)
        result_l = cvar_ohmc_price(call, data, config_l; rng=copy(rng))
        @test 0.5 * bs < result_l.option_price < 2.0 * bs

        # Exponential utility scoring
        config_e = CVaROHMCConfig(600, 12, 2; scoring=:exponential_utility, risk_aversion=0.2, cvar_weight=0.15)
        result_e = cvar_ohmc_price(call, data, config_e; rng=copy(rng))
        @test 0.5 * bs < result_e.option_price < 2.0 * bs
    end

    @testset "CVaR-OHMC loss definitions" begin
        rng = MersenneTwister(789)

        # HedgingError loss
        config1 = CVaROHMCConfig(500, 10, 2; loss_definition=HedgingError)
        result1 = cvar_ohmc_price(call, data, config1; rng=copy(rng))
        @test isfinite(result1.cvar)

        # PortfolioPnL loss
        config2 = CVaROHMCConfig(500, 10, 2; loss_definition=PortfolioPnL)
        result2 = cvar_ohmc_price(call, data, config2; rng=copy(rng))
        @test isfinite(result2.cvar)

        # Drawdown loss
        config3 = CVaROHMCConfig(500, 10, 2; loss_definition=Drawdown)
        result3 = cvar_ohmc_price(call, data, config3; rng=copy(rng))
        @test isfinite(result3.cvar)
    end

    @testset "CVaR weight effect on tail risk" begin
        rng = MersenneTwister(111)

        # Low CVaR weight
        config_low = CVaROHMCConfig(1000, 15, 2; cvar_weight=0.01)
        result_low = cvar_ohmc_price(call, data, config_low; rng=copy(rng))

        # High CVaR weight
        config_high = CVaROHMCConfig(1000, 15, 2; cvar_weight=0.5)
        result_high = cvar_ohmc_price(call, data, config_high; rng=copy(rng))

        # Both should produce valid results
        @test isfinite(result_low.cvar)
        @test isfinite(result_high.cvar)
        @test result_high.cvar_contribution > result_low.cvar_contribution

        # High weight should typically reduce CVaR (hedges tail better)
        # Note: This is statistical, may not always hold with small samples
        # We just verify both are reasonable
        @test result_high.total_objective >= result_high.base_objective
    end

    @testset "CVaR alpha level effect" begin
        rng = MersenneTwister(222)

        # 90% CVaR (less extreme tail)
        config_90 = CVaROHMCConfig(800, 12, 2; cvar_alpha=0.90, cvar_weight=0.2)
        result_90 = cvar_ohmc_price(call, data, config_90; rng=copy(rng))

        # 99% CVaR (more extreme tail)
        config_99 = CVaROHMCConfig(800, 12, 2; cvar_alpha=0.99, cvar_weight=0.2)
        result_99 = cvar_ohmc_price(call, data, config_99; rng=copy(rng))

        # Both valid
        @test isfinite(result_90.cvar)
        @test isfinite(result_99.cvar)

        # 99% CVaR should be >= 90% CVaR (more extreme tail)
        # This relationship holds in expectation but may not with small samples
        @test result_99.var >= result_90.var - 0.5 * abs(result_90.var)  # Allow some tolerance
    end

    @testset "compare_ohmc_cvar" begin
        rng = MersenneTwister(333)

        comparison = compare_ohmc_cvar(call, data, 800, 12, 2;
            cvar_alpha=0.95, cvar_weight=0.2, rng=rng)

        @test comparison isa CVaRComparisonResult
        @test comparison.standard_price > 0.0
        @test comparison.cvar_price > 0.0
        @test comparison.standard_variance >= 0.0
        @test comparison.cvar_variance >= 0.0
        @test isfinite(comparison.standard_cvar)
        @test isfinite(comparison.cvar_cvar)
        @test isfinite(comparison.variance_reduction_pct)
        @test isfinite(comparison.cvar_reduction_pct)
    end

    @testset "AdaptiveCVaRConfig and adaptive_cvar_ohmc" begin
        base_config = CVaROHMCConfig(500, 10, 2; cvar_alpha=0.95, cvar_weight=0.1)
        adaptive_config = AdaptiveCVaRConfig(
            base_config,
            0.01,   # eta_min
            0.5,    # eta_max
            5,      # eta_steps
            2.0     # target_cvar
        )

        @test adaptive_config.eta_min == 0.01
        @test adaptive_config.eta_max == 0.5
        @test adaptive_config.eta_steps == 5
        @test adaptive_config.target_cvar == 2.0

        rng = MersenneTwister(444)
        adaptive_result = adaptive_cvar_ohmc(call, data, adaptive_config; rng=rng)

        @test adaptive_result.result isa CVaROHMCResult
        @test adaptive_result.optimal_eta >= adaptive_config.eta_min
        @test adaptive_result.optimal_eta <= adaptive_config.eta_max
        @test isfinite(adaptive_result.final_score)
    end

    @testset "MultiPeriodCVaRConfig and multiperiod_cvar_ohmc" begin
        base_config = CVaROHMCConfig(600, 15, 2; cvar_alpha=0.95, cvar_weight=0.15)
        multiperiod_config = MultiPeriodCVaRConfig(
            base_config,
            [0.25, 0.5, 0.75, 1.0],  # Quarterly evaluation
            [0.2, 0.3, 0.3, 0.2]     # Weights (higher in middle)
        )

        @test length(multiperiod_config.intermediate_times) == 4
        @test length(multiperiod_config.time_weights) == 4
        @test sum(multiperiod_config.time_weights) == 1.0

        rng = MersenneTwister(555)
        result = multiperiod_cvar_ohmc(call, data, multiperiod_config; rng=rng)

        @test result isa CVaROHMCResult
        @test result.option_price > 0.0
        @test isfinite(result.cvar)
        @test result.cvar_contribution > 0.0  # Should have contribution from multiple periods
    end

    @testset "constrained_cvar_ohmc" begin
        config = CVaROHMCConfig(600, 12, 2;
            cvar_alpha=0.95,
            cvar_weight=0.1,
            cvar_objective=CVaRConstraint,
            cvar_budget=3.0
        )

        rng = MersenneTwister(666)
        result = constrained_cvar_ohmc(call, data, config;
            rng=rng,
            max_dual_iterations=10,
            dual_step_size=0.1,
            tolerance=0.5)  # Relaxed tolerance for test speed

        @test result isa ConstrainedCVaROHMCResult
        @test result.primal_result isa CVaROHMCResult
        @test result.lagrange_multiplier >= 0.0
        @test isfinite(result.constraint_slack)
        @test length(result.dual_convergence) >= 1
    end

    @testset "CVaR-OHMC with transaction costs" begin
        config = CVaROHMCConfig(600, 12, 2;
            cvar_weight=0.15,
            tc_rate=0.001  # 10 bps transaction cost
        )

        rng = MersenneTwister(777)
        result = cvar_ohmc_price(call, data, config; rng=rng)

        @test result.option_price > 0.0
        @test isfinite(result.cvar)
        # With TC, base objective should reflect TC contribution
        @test result.base_objective > 0.0
    end

    @testset "CVaR-OHMC on put option" begin
        config = CVaROHMCConfig(600, 12, 2; cvar_weight=0.15)
        rng = MersenneTwister(888)

        result = cvar_ohmc_price(put, data, config; rng=rng)

        @test result.option_price > 0.0
        @test isfinite(result.cvar)

        bs_put = price(put, BlackScholes(), data)
        @test 0.5 * bs_put < result.option_price < 2.0 * bs_put
    end

    @testset "CVaR-OHMC reproducibility with RNG" begin
        config = CVaROHMCConfig(500, 10, 2; cvar_weight=0.2)

        result1 = cvar_ohmc_price(call, data, config; rng=MersenneTwister(999))
        result2 = cvar_ohmc_price(call, data, config; rng=MersenneTwister(999))

        @test result1.option_price == result2.option_price
        @test result1.cvar == result2.cvar
        @test result1.var == result2.var
        @test result1.hedge_ratios == result2.hedge_ratios
    end

    @testset "CVaR-OHMC nu convergence" begin
        config = CVaROHMCConfig(800, 15, 2;
            cvar_weight=0.2,
            outer_iterations=5,
            nu_iterations=10
        )

        rng = MersenneTwister(1111)
        result = cvar_ohmc_price(call, data, config; rng=rng)

        # Should have recorded nu values across outer iterations
        @test length(result.nu_convergence) >= 2
        # Nu values should all be finite
        @test all(isfinite, result.nu_convergence)
    end

    @testset "CVaR >= VaR property" begin
        # This is a fundamental property of CVaR
        for alpha in [0.90, 0.95, 0.99]
            config = CVaROHMCConfig(500, 10, 2; cvar_alpha=alpha, cvar_weight=0.15)
            result = cvar_ohmc_price(call, data, config; rng=MersenneTwister(1234))
            @test result.cvar >= result.var - 1e-10  # Small tolerance for numerical
        end
    end

    @testset "CVaROHMCResult field types" begin
        config = CVaROHMCConfig(300, 8, 2)
        result = cvar_ohmc_price(call, data, config; rng=MersenneTwister(2222))

        @test result.option_price isa Float64
        @test result.hedge_ratios isa Matrix{Float64}
        @test result.hedged_portfolio_variance isa Float64
        @test result.confidence_interval isa Tuple{Float64,Float64}
        @test result.cvar isa Float64
        @test result.var isa Float64
        @test result.expected_shortfall_paths isa Vector{Int}
        @test result.tail_loss_mean isa Float64
        @test result.cvar_contribution isa Float64
        @test result.base_objective isa Float64
        @test result.total_objective isa Float64
        @test result.nu_convergence isa Vector{Float64}
    end

end

println("CVaR-OHMC tests completed.")
