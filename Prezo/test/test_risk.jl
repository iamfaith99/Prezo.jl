"""
    test_risk.jl

Tests for Phase 6: Risk Management (VaR, CVaR, stress testing, scenario analysis, Kelly).
"""

using Test
using Prezo
using Statistics
using Random
using LinearAlgebra

Random.seed!(42)

@testset "Risk Management - Phase 6" begin

    @testset "VaR - Historical" begin
        returns = randn(1000) * 0.02  # ~2% daily vol
        
        var_95 = value_at_risk(returns, 0.95)
        var_99 = value_at_risk(returns, 0.99)
        
        @test var_95 > 0
        @test var_99 > var_95  # Higher confidence = higher VaR
        @test var_95 < 0.10  # Reasonable magnitude
    end

    @testset "VaR - Parametric" begin
        returns = randn(1000) * 0.02
        
        var_param = value_at_risk(returns, 0.95; method=ParametricVaR())
        var_hist = value_at_risk(returns, 0.95; method=HistoricalVaR())
        
        @test var_param > 0
        # Should be similar for normal returns
        @test abs(var_param - var_hist) / var_hist < 0.3
    end

    @testset "VaR - Monte Carlo" begin
        returns = randn(500) * 0.02
        
        var_mc = value_at_risk(returns, 0.95; method=MonteCarloVaR(5000))
        var_hist = value_at_risk(returns, 0.95)
        
        @test var_mc > 0
        @test abs(var_mc - var_hist) / var_hist < 0.5
    end

    @testset "VaR - Multi-period scaling" begin
        returns = randn(252) * 0.02
        
        var_1 = value_at_risk(returns, 0.95)
        var_10 = value_at_risk(returns, 0.95, 10)
        
        # Square-root-of-time scaling
        @test var_10 ≈ var_1 * sqrt(10) atol=1e-10
    end

    @testset "CVaR (Expected Shortfall)" begin
        returns = randn(1000) * 0.02
        
        var_95 = value_at_risk(returns, 0.95)
        cvar_95 = conditional_var(returns, 0.95)
        
        # CVaR >= VaR always
        @test cvar_95 >= var_95
        @test cvar_95 > 0
    end

    @testset "Portfolio VaR" begin
        returns = randn(252, 3) * 0.02
        weights = [0.4, 0.3, 0.3]
        
        result = portfolio_var(returns, weights, 0.95)
        
        @test result.var > 0
        @test result.cvar >= result.var
        @test result.confidence == 0.95
        @test length(result.component_var) == 3
    end

    @testset "Stress Testing - Single Scenario" begin
        equity = PortfolioExposure("Equity", 1_000_000.0, Dict(:equity => 1.0, :volatility => -0.05))
        bonds = PortfolioExposure("Bonds", 500_000.0, Dict(:rates => -5.0, :credit_spread => -2.0))
        
        result = stress_test([equity, bonds], CRISIS_2008)
        
        @test result.portfolio_pnl < 0  # Crisis = loss
        @test haskey(result.component_pnl, "Equity")
        @test haskey(result.component_pnl, "Bonds")
        @test result.portfolio_pnl_pct < 0
    end

    @testset "Stress Testing - Suite" begin
        equity = PortfolioExposure("Equity", 1_000_000.0, Dict(:equity => 1.0))
        
        scenarios = [CRISIS_2008, COVID_2020, RATE_SHOCK_UP, RATE_SHOCK_DOWN]
        results = stress_test_suite([equity], scenarios)
        
        @test length(results) == 4
        @test all(r -> r isa StressTestResult, results)
    end

    @testset "Stress Testing - Custom Scenario" begin
        exposure = PortfolioExposure("Test", 100_000.0, Dict(:factor_a => 0.5))
        scenario = HypotheticalScenario("Test Shock", Dict(:factor_a => -0.20))
        
        result = stress_test([exposure], scenario)
        
        expected_pnl = 100_000.0 * 0.5 * (-0.20)
        @test result.portfolio_pnl ≈ expected_pnl
    end

    @testset "Reverse Stress Test" begin
        equity = PortfolioExposure("Equity", 1_000_000.0, Dict(:equity => 1.0, :rates => -2.0))
        
        shocks = reverse_stress_test([equity], 0.20, [:equity, :rates])
        
        @test haskey(shocks, :equity)
        @test haskey(shocks, :rates)
        # Shocks should be bounded
        @test abs(shocks[:equity]) <= 0.5
        @test abs(shocks[:rates]) <= 0.5
    end

    @testset "Scenario Grid" begin
        ranges = Dict(
            :spot => [90.0, 100.0, 110.0],
            :vol => [0.15, 0.20, 0.25]
        )
        grid = scenario_grid(ranges)
        
        @test length(grid.scenarios) == 9  # 3 × 3
        @test length(grid.factors) == 2
    end

    @testset "Monte Carlo Scenarios" begin
        means = Dict(:spot => 100.0, :vol => 0.2)
        vols = Dict(:spot => 10.0, :vol => 0.05)
        
        scenarios = monte_carlo_scenarios(means, vols, 100; rng=MersenneTwister(42))
        
        @test length(scenarios) == 100
        @test all(s -> haskey(s, :spot) && haskey(s, :vol), scenarios)
        
        # Check statistics
        spots = [s[:spot] for s in scenarios]
        @test abs(mean(spots) - 100.0) < 5.0  # Within reasonable range
    end

    @testset "Scenario Analysis" begin
        scenarios = [Dict(:x => Float64(i)) for i in 1:10]
        valuation_fn(s) = s[:x]^2
        
        result = analyze_scenarios(valuation_fn, scenarios, 25.0)  # base at x=5
        
        @test length(result.values) == 10
        @test length(result.pnl) == 10
        @test result.values[5] ≈ 25.0  # x=5 -> 25
        @test result.pnl[5] ≈ 0.0  # No P&L at base
        @test haskey(result.statistics, :mean_pnl)
    end

    @testset "Sensitivity Table" begin
        valuation_fn(s) = s[:x] + s[:y]
        
        table = sensitivity_table(
            valuation_fn,
            :x, [1.0, 2.0, 3.0],
            :y, [10.0, 20.0]
        )
        
        @test size(table.matrix) == (3, 2)
        @test table.matrix[1, 1] ≈ 11.0  # x=1, y=10
        @test table.matrix[3, 2] ≈ 23.0  # x=3, y=20
    end

    @testset "Scenario Ladder" begin
        valuation_fn(s) = 2 * s[:x]
        
        ladder = scenario_ladder(valuation_fn, :x, [1.0, 2.0, 3.0, 4.0, 5.0])
        
        @test length(ladder) == 5
        @test ladder[1] ≈ 2.0
        @test ladder[5] ≈ 10.0
    end

    @testset "Kelly Fraction - Binary" begin
        # Fair coin with 2:1 payout
        f = kelly_fraction(0.5, 2.0)
        @test f ≈ 0.25  # p - q/b = 0.5 - 0.5/2 = 0.25
        
        # Edge case: no edge
        f_no_edge = kelly_fraction(0.5, 1.0)
        @test f_no_edge ≈ 0.0
        
        # Positive edge
        f_edge = kelly_fraction(0.6, 1.0)
        @test f_edge ≈ 0.2  # 0.6 - 0.4/1 = 0.2
    end

    @testset "Fractional Kelly" begin
        full = kelly_fraction(0.6, 1.5)
        half = fractional_kelly(0.6, 1.5, 0.5)
        quarter = fractional_kelly(0.6, 1.5, 0.25)
        
        @test half ≈ 0.5 * full
        @test quarter ≈ 0.25 * full
    end

    @testset "Kelly Continuous" begin
        # μ=10%, σ²=4%, r=2%
        f = kelly_continuous(0.10, 0.04, 0.02)
        @test f ≈ 2.0  # (0.10 - 0.02) / 0.04 = 2.0
        
        # From Sharpe ratio
        f_sharpe = kelly_from_sharpe(1.0, 0.20)
        @test f_sharpe ≈ 5.0  # 1.0 / 0.20 = 5
    end

    @testset "Kelly Portfolio" begin
        μ = [0.10, 0.15]
        Σ = [0.04 0.01; 0.01 0.09]
        
        result = kelly_portfolio(μ, Σ; risk_free_rate=0.02)
        
        @test length(result.weights) == 2
        @test result.expected_return > 0
        @test result.volatility > 0
        @test result.sharpe_ratio > 0
        @test result.kelly_leverage > 0
    end

    @testset "Fractional Kelly Portfolio" begin
        μ = [0.10, 0.12]
        Σ = [0.04 0.005; 0.005 0.06]
        
        full = kelly_portfolio(μ, Σ)
        half = fractional_kelly_portfolio(μ, Σ; fraction=0.5)
        
        @test half.weights ≈ 0.5 .* full.weights
        @test half.kelly_leverage ≈ 0.5 * full.kelly_leverage atol=1e-10
    end

    @testset "Kelly Growth Rate" begin
        # At optimal Kelly, growth rate should be positive
        f_opt = kelly_fraction(0.6, 1.0)
        g_opt = kelly_growth_rate(f_opt, 0.6, 1.0)
        @test g_opt > 0
        
        # At f=0, growth = 0
        g_zero = kelly_growth_rate(0.0, 0.6, 1.0)
        @test g_zero ≈ 0.0 atol=1e-10
        
        # Over-betting reduces growth
        g_over = kelly_growth_rate(0.8, 0.6, 1.0)
        @test g_over < g_opt
    end

    @testset "Optimal Bet Size" begin
        bankroll = 10_000.0
        kelly = 0.25
        
        bet = optimal_bet_size(bankroll, kelly)
        @test bet ≈ 2_500.0
        
        # With constraints
        bet_capped = optimal_bet_size(bankroll, kelly; max_bet=1_000.0)
        @test bet_capped ≈ 1_000.0
        
        # Negative Kelly -> no bet
        bet_neg = optimal_bet_size(bankroll, -0.1)
        @test bet_neg ≈ 0.0
    end

end
