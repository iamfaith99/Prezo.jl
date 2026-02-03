"""
    test_inference.jl

Tests for Phase 4: MLE, calibration, ABC.
"""

using Test
using Prezo
using Statistics
using Random
using Distributions
using LinearAlgebra

Random.seed!(42)

@testset "Inference - Phase 4" begin
    @testset "MLE" begin
        # Gaussian MLE: -log L = (n/2)*log(2π) + (n/2)*log(σ²) + sum((x-μ)²)/(2σ²)
        # params = [μ, σ]. We estimate μ and σ from data.
        n_obs = 100
        true_μ, true_σ = 2.0, 1.5
        data = randn(n_obs) .* true_σ .+ true_μ

        function loglik(params)
            μ, σ = params[1], max(params[2], 1e-6)
            -0.5 * n_obs * log(2π) - n_obs * log(σ) - sum((data .- μ) .^ 2) / (2 * σ^2)
        end

        mle = MLEProblem(
            loglik,
            [0.0, 1.0],
            [-10.0, 0.01],
            [10.0, 5.0],
        )
        sol = solve(mle)
        @test sol.converged
        @test length(sol.params) == 2
        @test sol.params[1] ≈ true_μ atol=0.3
        @test sol.params[2] ≈ true_σ atol=0.3
        se = standard_errors(sol)
        @test length(se) == 2
        @test all(isfinite.(se))
        @test all(se .> 0)
    end

    @testset "Calibration - OptionPricesTarget" begin
        # Calibrate Black-Scholes vol to synthetic option prices
        data = MarketData(100.0, 0.05, 0.22, 0.0)
        options = [
            EuropeanCall(90.0, 1.0),
            EuropeanCall(100.0, 1.0),
            EuropeanCall(110.0, 1.0),
        ]
        market_prices = [price(opt, BlackScholes(), data) for opt in options]
        target = OptionPricesTarget(options, market_prices)

        # price_fn(option, params): params = [vol]; use MarketData(spot, rate, params[1], div)
        function price_fn(opt, params)
            vol = max(params[1], 0.01)
            d = MarketData(100.0, 0.05, vol, 0.0)
            return price(opt, BlackScholes(), d)
        end

        result = calibrate_option_prices(
            price_fn,
            target,
            [0.2],
            [0.05],
            [0.5],
        )
        @test result.converged
        @test length(result.params) == 1
        @test result.params[1] ≈ 0.22 atol=0.02
        @test result.loss < 1e-6
    end

    @testset "ABC - Rejection" begin
        # Simulate from Normal(μ, σ); summary = [mean, std]; prior on (μ, σ)
        true_μ, true_σ = 0.0, 1.0
        observed = randn(200) .* true_σ .+ true_μ

        simulator(params) = randn(200) .* max(params[2], 0.1) .+ params[1]
        summary_stats(data) = [mean(data), std(data)]
        prior = product_distribution([Normal(0, 2), Uniform(0.3, 3.0)])

        method = RejectionABC(500, 0.15, prior)
        accepted, n_acc = abc_inference(
            method,
            simulator,
            summary_stats,
            euclidean_distance,
            observed;
            rng = MersenneTwister(123),
        )
        @test n_acc >= 1
        @test size(accepted, 2) == 2
        if n_acc >= 10
            @test abs(mean(accepted[:, 1]) - true_μ) < 0.5
            @test abs(mean(accepted[:, 2]) - true_σ) < 0.5
        end
    end

    @testset "euclidean_distance" begin
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0, 3.0]
        @test euclidean_distance(a, b) == 0.0
        @test euclidean_distance([0.0, 0.0], [3.0, 4.0]) == 5.0
    end
end
