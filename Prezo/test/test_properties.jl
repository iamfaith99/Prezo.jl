"""
    test_properties.jl

Property-based tests for option pricing: mathematical invariants that must hold
for any valid parameters. These complement unit tests by exercising random
inputs and checking relationships (put-call parity, bounds, monotonicity).

# Test Categories
- Put-call parity (European, Black-Scholes)
- Price bounds (call and put within no-arbitrage intervals)
- Monotonicity (price vs spot, strike, vol, time)
- Intrinsic value bounds
"""

using Test
using Prezo
using Random
using Distributions

Random.seed!(42)

# Number of random trials for property tests
const N_TRIALS = 100

@testset "Pricing Properties" begin
    @testset "Put-Call Parity (European, Black-Scholes)" begin
        for _ in 1:N_TRIALS
            spot = rand(Uniform(50.0, 150.0))
            strike = rand(Uniform(50.0, 150.0))
            vol = rand(Uniform(0.1, 0.5))
            rate = rand(Uniform(0.0, 0.1))
            expiry = rand(Uniform(0.1, 2.0))
            data = MarketData(spot, rate, vol, 0.0)
            call = EuropeanCall(strike, expiry)
            put = EuropeanPut(strike, expiry)
            call_price = price(call, BlackScholes(), data)
            put_price = price(put, BlackScholes(), data)
            lhs = call_price - put_price
            rhs = spot - strike * exp(-rate * expiry)
            @test isapprox(lhs, rhs, atol=1e-6)
        end
    end

    @testset "Price Bounds - European" begin
        for _ in 1:N_TRIALS
            spot = rand(Uniform(50.0, 150.0))
            strike = rand(Uniform(50.0, 150.0))
            vol = rand(Uniform(0.1, 0.5))
            rate = rand(Uniform(0.0, 0.1))
            expiry = rand(Uniform(0.1, 2.0))
            data = MarketData(spot, rate, vol, 0.0)
            call = EuropeanCall(strike, expiry)
            put = EuropeanPut(strike, expiry)
            call_price = price(call, BlackScholes(), data)
            put_price = price(put, BlackScholes(), data)
            # Call: 0 < C < S (no dividends)
            @test call_price > 0.0
            @test call_price < spot * 1.001  # allow tiny numerical tolerance
            # Put: 0 < P < K*exp(-r*T)
            @test put_price > 0.0
            @test put_price < strike * exp(-rate * expiry) * 1.001
        end
    end

    @testset "Intrinsic Value Bounds" begin
        for _ in 1:N_TRIALS
            spot = rand(Uniform(50.0, 150.0))
            strike = rand(Uniform(50.0, 150.0))
            vol = rand(Uniform(0.1, 0.5))
            rate = rand(Uniform(0.0, 0.1))
            expiry = rand(Uniform(0.1, 2.0))
            data = MarketData(spot, rate, vol, 0.0)
            call = EuropeanCall(strike, expiry)
            put = EuropeanPut(strike, expiry)
            call_price = price(call, BlackScholes(), data)
            put_price = price(put, BlackScholes(), data)
            intrinsic_call = max(0.0, spot - strike)
            intrinsic_put = max(0.0, strike * exp(-rate * expiry) - spot)
            @test call_price >= intrinsic_call - 1e-6
            @test put_price >= intrinsic_put - 1e-6
        end
    end

    @testset "Monotonicity in Spot" begin
        strike = 100.0
        expiry = 1.0
        vol = 0.2
        rate = 0.05
        call = EuropeanCall(strike, expiry)
        put = EuropeanPut(strike, expiry)
        spots = [80.0, 90.0, 100.0, 110.0, 120.0]
        call_prices = [price(call, BlackScholes(), MarketData(s, rate, vol, 0.0)) for s in spots]
        put_prices = [price(put, BlackScholes(), MarketData(s, rate, vol, 0.0)) for s in spots]
        # Call price increasing in spot
        for i in 1:(length(spots)-1)
            @test call_prices[i] <= call_prices[i+1] + 1e-10
        end
        # Put price decreasing in spot
        for i in 1:(length(spots)-1)
            @test put_prices[i] >= put_prices[i+1] - 1e-10
        end
    end

    @testset "Monotonicity in Strike" begin
        spot = 100.0
        expiry = 1.0
        vol = 0.2
        rate = 0.05
        data = MarketData(spot, rate, vol, 0.0)
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        call_prices = [price(EuropeanCall(k, expiry), BlackScholes(), data) for k in strikes]
        put_prices = [price(EuropeanPut(k, expiry), BlackScholes(), data) for k in strikes]
        # Call price decreasing in strike
        for i in 1:(length(strikes)-1)
            @test call_prices[i] >= call_prices[i+1] - 1e-10
        end
        # Put price increasing in strike
        for i in 1:(length(strikes)-1)
            @test put_prices[i] <= put_prices[i+1] + 1e-10
        end
    end

    @testset "Monotonicity in Volatility" begin
        spot = 100.0
        strike = 100.0
        expiry = 1.0
        rate = 0.05
        call = EuropeanCall(strike, expiry)
        put = EuropeanPut(strike, expiry)
        vols = [0.1, 0.2, 0.3, 0.5]
        call_prices = [price(call, BlackScholes(), MarketData(spot, rate, v, 0.0)) for v in vols]
        put_prices = [price(put, BlackScholes(), MarketData(spot, rate, v, 0.0)) for v in vols]
        for i in 1:(length(vols)-1)
            @test call_prices[i] <= call_prices[i+1] + 1e-10
            @test put_prices[i] <= put_prices[i+1] + 1e-10
        end
    end

    @testset "American >= European (same params)" begin
        data = MarketData(100.0, 0.05, 0.20, 0.0)
        engine = Prezo.Binomial(200)
        # Allow small tolerance: binomial is discrete so American can be marginally below European
        tol = 0.02
        for _ in 1:20
            strike = rand(Uniform(80.0, 120.0))
            expiry = rand(Uniform(0.25, 1.5))
            call_am = price(AmericanCall(strike, expiry), engine, data)
            call_eu = price(EuropeanCall(strike, expiry), BlackScholes(), data)
            put_am = price(AmericanPut(strike, expiry), engine, data)
            put_eu = price(EuropeanPut(strike, expiry), BlackScholes(), data)
            @test call_am >= call_eu - tol
            @test put_am >= put_eu - tol
        end
    end
end
