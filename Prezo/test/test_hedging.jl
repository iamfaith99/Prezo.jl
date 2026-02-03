# Phase 5: Advanced Hedging — OHMC and delta hedging strategies

using Test
using Prezo
using Statistics
using Random

Random.seed!(123)

@testset "OHMC" begin
    data = MarketData(100.0, 0.05, 0.2, 0.0)
    call = EuropeanCall(100.0, 1.0)
    config = OHMCConfig(1000, 15, 2)

    r = ohmc_price(call, data, config; rng=MersenneTwister(42))

    @test r.option_price > 0.0
    @test isfinite(r.option_price)
    @test size(r.hedge_ratios) == (config.n_steps + 1, config.n_paths)
    @test r.hedged_portfolio_variance >= 0.0
    @test r.confidence_interval[1] < r.confidence_interval[2]
    @test r.confidence_interval[1] <= r.option_price <= r.confidence_interval[2]

    # OHMC price should be in the ballpark of Black-Scholes (same order)
    bs = price(call, BlackScholes(), data)
    @test 0.5 * bs < r.option_price < 2.0 * bs
end

@testset "OHMC scoring plug-in (quadratic / log / exponential_utility)" begin
    data = MarketData(100.0, 0.05, 0.2, 0.0)
    call = EuropeanCall(100.0, 1.0)
    rng = MersenneTwister(99)
    bs = price(call, BlackScholes(), data)

    rq = ohmc_price(call, data, OHMCConfig(800, 12, 2; scoring=:quadratic); rng=copy(rng))
    @test 0.5 * bs < rq.option_price < 2.0 * bs

    rl = ohmc_price(call, data, OHMCConfig(800, 12, 2; scoring=:log); rng=copy(rng))
    @test 0.5 * bs < rl.option_price < 2.0 * bs

    ru = ohmc_price(call, data, OHMCConfig(800, 12, 2; scoring=:exponential_utility, risk_aversion=0.1); rng=copy(rng))
    @test 0.5 * bs < ru.option_price < 2.0 * bs
end

@testset "Delta hedging strategies" begin
    option = EuropeanCall(100.0, 1.0)
    engine = BlackScholes()

    # Build a simple path: time, spot, rate, vol
    n = 11
    t = range(0.0, 1.0, length=n)
    spot = 100.0 .* exp.(0.05 .* t .+ 0.0 .* randn(n))
    hist = hcat(collect(t), spot, fill(0.05, n), fill(0.2, n))

    @testset "DiscreteDeltaHedge" begin
        strat = DiscreteDeltaHedge([0.25, 0.5, 0.75, 1.0], 0.001)
        perf = backtest_hedge(option, strat, hist, engine)
        @test perf.n_trades >= 2
        @test perf.total_cost >= 0.0
        @test perf.variance >= 0.0
        @test perf.max_drawdown >= 0.0
    end

    @testset "StopLossHedge" begin
        strat = StopLossHedge(0.15)
        perf = backtest_hedge(option, strat, hist, engine)
        @test perf.n_trades >= 1
        @test perf.total_cost >= 0.0
    end

    @testset "StaticHedge" begin
        strat = StaticHedge([EuropeanCall(100.0, 1.0)])
        perf = backtest_hedge(option, strat, hist, engine)
        @test perf.n_trades == 0
        @test perf.total_cost == 0.0
        @test isfinite(perf.total_pnl)
    end
end

@testset "HedgePerformance type" begin
    perf = HedgePerformance(0.1, 0.01, 0.05, 0.02, 5)
    @test perf.total_pnl == 0.1
    @test perf.total_cost == 0.01
    @test perf.n_trades == 5
end
