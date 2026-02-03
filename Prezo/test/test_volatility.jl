"""
    test_volatility.jl

Tests for Phase 2 volatility module: GARCH family (GARCH, EGARCH, GJRGARCH).
- Parameter validation and stationarity
- volatility_process, forecast, loglikelihood, simulate
- fit (MLE) and recovery from simulated data
"""

using Test
using Prezo
using Statistics
using Random
using Distributions

# Avoid ambiguity with StatsBase/Distributions (fit, loglikelihood)
const fit_garch = Prezo.fit
const loglikelihood_garch = Prezo.loglikelihood

Random.seed!(42)

@testset "Volatility Module - GARCH Family" begin
    @testset "GARCH(1,1) - Construction and validation" begin
        m = GARCH(0.0001, 0.08, 0.90)
        @test m.ω == 0.0001 && m.α == 0.08 && m.β == 0.90
        @test m isa GARCHModel
        @test m isa VolatilityModel

        @test_throws ArgumentError GARCH(-0.01, 0.1, 0.8)
        @test_throws ArgumentError GARCH(0.01, -0.1, 0.8)
        @test_throws ArgumentError GARCH(0.01, 0.1, 0.95)  # α+β >= 1
    end

    @testset "EGARCH - Construction" begin
        m = EGARCH(0.0, 0.1, -0.05, 0.9)
        @test m.β < 1
        @test_throws ArgumentError EGARCH(0.0, 0.1, 0.0, 1.0)
    end

    @testset "GJRGARCH - Construction" begin
        m = GJRGARCH(0.0001, 0.05, 0.1, 0.85)
        @test m isa GARCHModel
        @test_throws ArgumentError GJRGARCH(0.01, 0.5, 0.5, 0.5)  # α+γ/2+β >= 1
    end

    @testset "volatility_process" begin
        model = GARCH(0.0001, 0.08, 0.90)
        returns = randn(100) .* 0.01
        h = volatility_process(model, returns)
        @test length(h) == length(returns)
        @test all(h .> 0)
        # First value is unconditional variance
        unc = model.ω / (1 - model.α - model.β)
        @test isapprox(h[1], unc, rtol=1e-10)
    end

    @testset "loglikelihood" begin
        model = GARCH(0.0001, 0.08, 0.90)
        returns = randn(200) .* 0.01
        ll = loglikelihood_garch(model, returns)
        @test isfinite(ll)
        # Quasi log-likelihood (no -N/2*log(2π)) can be positive when h_t are small
    end

    @testset "forecast" begin
        model = GARCH(0.0001, 0.08, 0.90)
        returns = randn(50) .* 0.02
        f = forecast(model, returns, 5)
        @test length(f) == 5
        @test all(f .> 0)
        # With r=0 in future, h converges to ω/(1-β)
        h_inf = model.ω / (1 - model.β)
        @test f[5] ≈ h_inf atol=0.001
    end

    @testset "simulate" begin
        model = GARCH(0.0001, 0.08, 0.90)
        r, h = simulate(model, 500; seed=123)
        @test length(r) == 500 && length(h) == 500
        @test all(h .> 0)
        # Variance of returns should be in same order as unconditional variance
        unc = model.ω / (1 - model.α - model.β)
        @test 0.5 * unc < var(r) < 2.0 * unc
    end

    @testset "fit GARCH - recovery from simulated data" begin
        true_model = GARCH(0.0001, 0.08, 0.90)
        returns, _ = simulate(true_model, 2000; seed=42)
        fit_model = fit_garch(GARCH, returns)
        @test fit_model isa GARCH
        # Should be close to true params (allowing estimation error)
        @test isapprox(fit_model.ω, true_model.ω, rtol=0.5)
        @test isapprox(fit_model.α, true_model.α, rtol=0.3)
        @test isapprox(fit_model.β, true_model.β, rtol=0.05)
        @test fit_model.α + fit_model.β < 1
    end

    @testset "fit EGARCH" begin
        returns = randn(500) .* 0.01
        model = fit_garch(EGARCH, returns)
        @test model isa EGARCH
        @test abs(model.β) < 1
        h = volatility_process(model, returns)
        @test all(h .> 0)
    end

    @testset "fit GJRGARCH" begin
        returns = randn(500) .* 0.01
        model = fit_garch(GJRGARCH, returns)
        @test model isa GJRGARCH
        @test model.ω > 0
        @test model.α + model.γ / 2 + model.β < 1
    end

    @testset "fit - insufficient data" begin
        returns = randn(5)
        @test_throws ArgumentError fit_garch(GARCH, returns)
    end

    @testset "AGARCH" begin
        model = AGARCH(0.0001, 0.08, 0.5, 0.88)
        @test model isa GARCHModel
        returns = randn(200) .* 0.01
        h = volatility_process(model, returns)
        @test length(h) == 200 && all(h .> 0)
        fit_ag = fit_garch(AGARCH, returns)
        @test fit_ag isa AGARCH
        @test fit_ag.α + fit_ag.β < 1
    end

    @testset "Student-t innovations" begin
        model = GARCH(0.0001, 0.08, 0.90)
        returns = randn(100) .* 0.01
        ll_norm = loglikelihood_garch(model, returns; dist=Normal(0, 1))
        ll_t = loglikelihood_garch(model, returns; dist=TDist(5.0))
        @test isfinite(ll_norm) && isfinite(ll_t)
        r_sim, h_sim = Prezo.simulate(model, 50; seed=1, dist=TDist(6.0))
        @test length(r_sim) == 50 && length(h_sim) == 50
        @test all(h_sim .> 0)
    end

    @testset "RegimeGARCH" begin
        g1 = GARCH(0.0002, 0.10, 0.85)
        g2 = GARCH(0.0001, 0.05, 0.90)
        model = RegimeGARCH(g1, g2, 0.95, 0.95)
        @test model isa VolatilityModel
        returns = randn(150) .* 0.01
        h, pr1 = volatility_process(model, returns)
        @test length(h) == 150 && length(pr1) == 150
        @test all(0 .<= pr1 .<= 1) && all(h .> 0)
        f = forecast(model, returns, 3)
        @test length(f) == 3 && all(f .> 0)
        r_sim, h_sim, s_sim = simulate(model, 100; seed=42)
        @test length(r_sim) == 100 && length(s_sim) == 100
        @test all(s_sim .>= 1) && all(s_sim .<= 2)
    end

    @testset "DCC-GARCH" begin
        # returns: T x n (time × assets)
        Random.seed!(123)
        returns = randn(200, 3) .* 0.01
        model = fit_garch(DCCGARCH, returns)
        @test model isa DCCGARCH
        @test model isa VolatilityModel
        @test length(model.univariate_models) == 3
        @test model.a >= 0 && model.b >= 0 && model.a + model.b < 1
        @test size(model.Q_bar) == (3, 3)
        covs = Prezo.covariance_series(model, returns)
        @test length(covs) == 200
        @test all(size(H) == (3, 3) for H in covs)
        @test all(H -> isapprox(H, H'; atol=1e-10), covs)
        f = forecast(model, returns, 3)
        @test length(f) == 3
        @test all(size(H) == (3, 3) for H in f)
    end

    @testset "O-GARCH" begin
        Random.seed!(44)
        returns = randn(200, 4) .* 0.01
        model = fit_garch(OGARCH, returns; n_factors=3)
        @test model isa OGARCH
        @test model isa VolatilityModel
        @test size(model.loadings) == (4, 3)
        @test length(model.garch_models) == 3
        covs = Prezo.covariance_series(model, returns)
        @test length(covs) == 200
        @test all(size(H) == (4, 4) for H in covs)
        @test all(H -> isapprox(H, H'; atol=1e-10), covs)
    end

    @testset "Factor GARCH" begin
        Random.seed!(45)
        factors = randn(200, 2) .* 0.01
        B = [0.8 0.3; 0.2 0.9; 0.5 0.5]
        returns = (factors * B') .+ randn(200, 3) .* 0.005
        model = fit_garch(FactorGARCH, returns, factors)
        @test model isa FactorGARCH
        @test model isa VolatilityModel
        @test size(model.loadings) == (3, 2)
        @test length(model.garch_models) == 2
        @test length(model.resid_var) == 3
        covs = Prezo.covariance_series(model, returns, factors)
        @test length(covs) == 200
        @test all(size(H) == (3, 3) for H in covs)
        @test all(H -> isapprox(H, H'; atol=1e-10), covs)
    end

    @testset "Heston model" begin
        m = HestonModel(2.0, 0.04, 0.3, -0.5, 0.04)
        @test m isa VolatilityModel
        @test m.κ == 2.0 && m.θ == 0.04 && m.v₀ == 0.04
        @test_throws ArgumentError HestonModel(1.0, 0.04, 0.3, 1.5, 0.04)
        @test_throws ArgumentError HestonModel(1.0, -0.01, 0.3, 0.0, 0.04)
        spot, variance, returns = simulate_heston(m, 100, 1.0; S0=100.0, r=0.05, q=0.0, seed=42)
        @test length(spot) == 101 && length(variance) == 101 && length(returns) == 100
        @test all(spot .> 0) && all(variance .> 0)
        @test spot[1] == 100.0
        @test variance[1] == 0.04
    end

    @testset "Dupire local volatility" begin
        # Synthetic call grid: flat Black–Scholes so local vol ≈ constant
        data = MarketData(100.0, 0.05, 0.2, 0.0)
        strikes = [90.0, 100.0, 110.0]
        maturities = [0.5, 1.0]
        C = Float64[
            price(EuropeanCall(90.0, 0.5), BlackScholes(), data)
            price(EuropeanCall(100.0, 0.5), BlackScholes(), data)
            price(EuropeanCall(110.0, 0.5), BlackScholes(), data)
            price(EuropeanCall(90.0, 1.0), BlackScholes(), data)
            price(EuropeanCall(100.0, 1.0), BlackScholes(), data)
            price(EuropeanCall(110.0, 1.0), BlackScholes(), data)
        ]
        C = reshape(C, 3, 2)
        surf = dupire_local_vol(C, strikes, maturities, data)
        @test surf isa LocalVolSurface
        @test surf.strikes == Float64.(strikes) && surf.maturities == Float64.(maturities)
        @test size(surf.local_vols) == (3, 2)
        # Interior (1,2) and (2,2) etc should have finite local vol near 0.2
        interior_vols = [surf.local_vols[2, 1], surf.local_vols[2, 2]]
        @test all(isfinite, interior_vols)
        @test all(0.05 .< interior_vols .< 0.5)
    end
end
