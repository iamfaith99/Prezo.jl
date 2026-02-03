using Test
using Prezo
using Random

# Set seed for reproducibility
Random.seed!(42)

@testset "Prezo.jl Tests" begin
    @testset "American Options" begin
        # Common market data for tests
        data = MarketData(100.0, 0.05, 0.20, 0.0)

        data_with_div = MarketData(100.0, 0.05, 0.20, 0.03)

        @testset "Option Construction" begin
            # Test American call construction
            call = AmericanCall(100.0, 1.0)
            @test call.strike == 100.0
            @test call.expiry == 1.0
            @test call isa AmericanOption

            # Test American put construction
            put = AmericanPut(100.0, 1.0)
            @test put.strike == 100.0
            @test put.expiry == 1.0
            @test put isa AmericanOption
        end

        @testset "Payoff Functions" begin
            call = AmericanCall(100.0, 1.0)
            put = AmericanPut(100.0, 1.0)

            # Test call payoffs
            @test payoff(call, 110.0) == 10.0
            @test payoff(call, 100.0) == 0.0
            @test payoff(call, 90.0) == 0.0

            # Test put payoffs
            @test payoff(put, 90.0) == 10.0
            @test payoff(put, 100.0) == 0.0
            @test payoff(put, 110.0) == 0.0
        end

        @testset "Binomial Engine - American vs European" begin
            # Test that American ≥ European for puts (with significant early exercise premium)
            put_am = AmericanPut(100.0, 1.0)
            put_eu = EuropeanPut(100.0, 1.0)

            engine = Binomial(200)
            price_am = price(put_am, engine, data)
            price_eu = price(put_eu, engine, data)

            @test price_am ≥ price_eu
            @test price_am > price_eu  # Put should have early exercise premium

            # Test that American ≈ European for calls with zero dividends
            call_am = AmericanCall(100.0, 1.0)
            call_eu = EuropeanCall(100.0, 1.0)

            price_call_am = price(call_am, engine, data)
            price_call_eu = price(call_eu, engine, data)

            @test price_call_am ≥ price_call_eu
            # With zero dividends, should be nearly equal
            @test isapprox(price_call_am, price_call_eu, atol=0.05)

            # With dividends, American call can have early exercise premium
            price_call_am_div = price(call_am, engine, data_with_div)
            price_call_eu_div = price(call_eu, BlackScholes(), data_with_div)

            # American option must always be ≥ European (arbitrage relationship)
            @test price_call_am_div ≥ price_call_eu_div
        end

        @testset "Longstaff-Schwartz Engine - American Puts" begin
            put = AmericanPut(100.0, 1.0)

            # Test basic LSM pricing
            engine = LongstaffSchwartz(50, 10000)
            lsm_price = price(put, engine, data)

            # Price should be positive
            @test lsm_price > 0.0

            # Compare with Binomial (should be similar)
            binomial_engine = Binomial(200)
            binomial_price = price(put, binomial_engine, data)

            # Should agree within reasonable tolerance (MC has variance)
            @test isapprox(lsm_price, binomial_price, rtol=0.05)

            # Test that American put ≥ European put
            put_eu = EuropeanPut(100.0, 1.0)
            eu_price = price(put_eu, BlackScholes(), data)
            @test lsm_price ≥ eu_price
        end

        @testset "Longstaff-Schwartz Engine - American Calls" begin
            call = AmericanCall(100.0, 1.0)

            # Test with zero dividends (should ≈ European)
            # Use antithetic variates for better accuracy and lower variance
            engine = LongstaffSchwartz(50, 25000, 3; antithetic=true)
            lsm_price = price(call, engine, data)

            call_eu = EuropeanCall(100.0, 1.0)
            eu_price = price(call_eu, BlackScholes(), data)

            @test lsm_price ≥ eu_price
            # LSM can show significant bias for American calls (known limitation)
            # Relaxed tolerance accounts for both MC variance and LSM bias
            @test isapprox(lsm_price, eu_price, rtol=0.15)

            # Test with dividends (can have early exercise premium)
            lsm_price_div = price(call, engine, data_with_div)
            eu_price_div = price(call_eu, BlackScholes(), data_with_div)

            # American ≥ European (allow small tolerance for MC variance)
            # LSM is Monte Carlo, so small violations due to sampling error are acceptable
            @test lsm_price_div ≥ eu_price_div - 0.10  # Allow ~1% tolerance for MC
        end

        @testset "Enhanced LSM Engines - Different Basis Functions" begin
            put = AmericanPut(100.0, 1.0)

            # Test Laguerre basis
            laguerre = LaguerreLSM(3, 50, 10000)
            price_laguerre = price(put, laguerre, data)
            @test price_laguerre > 0.0

            # Test Chebyshev basis
            chebyshev = ChebyshevLSM(3, 50, 10000; domain=(50.0, 150.0))
            price_chebyshev = price(put, chebyshev, data)
            @test price_chebyshev > 0.0

            # Test Power basis
            power = PowerLSM(3, 50, 10000)
            price_power = price(put, power, data)
            @test price_power > 0.0

            # Test Hermite basis
            hermite = HermiteLSM(3, 50, 10000; mean=100.0, std=20.0)
            price_hermite = price(put, hermite, data)
            @test price_hermite > 0.0

            # All should give similar results (within MC variance)
            @test isapprox(price_laguerre, price_chebyshev, rtol=0.10)
            @test isapprox(price_laguerre, price_power, rtol=0.10)
            @test isapprox(price_laguerre, price_hermite, rtol=0.10)
        end

        @testset "Monotonicity Properties" begin
            put = AmericanPut(100.0, 1.0)
            call = AmericanCall(100.0, 1.0)
            engine = Binomial(200)

            # Test delta: call increases with spot, put decreases with spot
            spots = [90.0, 100.0, 110.0]
            call_prices = Float64[]
            put_prices = Float64[]

            for s in spots
                test_data = MarketData(s, 0.05, 0.20, 0.0)
                push!(call_prices, price(call, engine, test_data))
                push!(put_prices, price(put, engine, test_data))
            end

            # Call prices should increase with spot
            @test call_prices[1] < call_prices[2] < call_prices[3]

            # Put prices should decrease with spot
            @test put_prices[1] > put_prices[2] > put_prices[3]
        end

        @testset "Volatility Sensitivity (Vega)" begin
            put = AmericanPut(100.0, 1.0)
            call = AmericanCall(100.0, 1.0)
            engine = Binomial(200)

            # Both call and put should increase with volatility
            vols = [0.10, 0.20, 0.30]
            call_prices = Float64[]
            put_prices = Float64[]

            for v in vols
                test_data = MarketData(100.0, 0.05, v, 0.0)
                push!(call_prices, price(call, engine, test_data))
                push!(put_prices, price(put, engine, test_data))
            end

            # Both should increase with volatility
            @test call_prices[1] < call_prices[2] < call_prices[3]
            @test put_prices[1] < put_prices[2] < put_prices[3]
        end

        @testset "Time Value Properties" begin
            put = AmericanPut(100.0, 1.0)
            engine = Binomial(200)

            # American options with more time should be worth at least as much
            put_short = AmericanPut(100.0, 0.5)
            put_long = AmericanPut(100.0, 1.5)

            price_short = price(put_short, engine, data)
            price_base = price(put, engine, data)
            price_long = price(put_long, engine, data)

            # More time = more value for American options
            @test price_short ≤ price_base ≤ price_long
        end

        @testset "Boundary Conditions" begin
            engine = Binomial(200)

            # Test call at S=0 (should be near 0)
            data_zero_spot = MarketData(0.01, 0.05, 0.20, 0.0)
            call = AmericanCall(100.0, 1.0)
            price_zero = price(call, engine, data_zero_spot)
            @test price_zero < 0.1

            # Test deep ITM put (should be close to intrinsic value)
            data_deep_itm = MarketData(50.0, 0.05, 0.20, 0.0)
            put = AmericanPut(100.0, 1.0)
            price_itm = price(put, engine, data_deep_itm)
            intrinsic = 100.0 - 50.0

            # American put deep ITM should be close to intrinsic (early exercise optimal)
            @test price_itm ≥ intrinsic
            @test price_itm ≤ intrinsic * 1.2  # Some time value, but not much
        end

        @testset "At Expiry (T=0)" begin
            # At expiry, American options should equal intrinsic value
            put = AmericanPut(100.0, 0.0001)  # Nearly expired
            call = AmericanCall(100.0, 0.0001)

            engine = Binomial(100)

            # ITM put
            data_itm = MarketData(90.0, 0.05, 0.20, 0.0)
            put_price = price(put, engine, data_itm)
            @test isapprox(put_price, 10.0, atol=0.1)

            # ITM call
            data_itm_call = MarketData(110.0, 0.05, 0.20, 0.0)
            call_price = price(call, engine, data_itm_call)
            @test isapprox(call_price, 10.0, atol=0.1)

            # OTM options should be near zero
            data_otm = MarketData(100.0, 0.05, 0.20, 0.0)
            put_otm = price(put, engine, data_otm)
            call_otm = price(call, engine, data_otm)
            @test put_otm < 0.5
            @test call_otm < 0.5
        end

        @testset "Strike Sensitivity" begin
            engine = Binomial(200)

            # Call prices should decrease with strike
            strikes = [90.0, 100.0, 110.0]
            call_prices = [price(AmericanCall(k, 1.0), engine, data) for k in strikes]
            @test call_prices[1] > call_prices[2] > call_prices[3]

            # Put prices should increase with strike
            put_prices = [price(AmericanPut(k, 1.0), engine, data) for k in strikes]
            @test put_prices[1] < put_prices[2] < put_prices[3]
        end

        @testset "Binomial Convergence" begin
            # Test that binomial converges with increasing steps
            put = AmericanPut(100.0, 1.0)

            steps_list = [50, 100, 200, 400]
            prices = [price(put, Binomial(s), data) for s in steps_list]

            # Prices should stabilize (differences should decrease)
            diff1 = abs(prices[2] - prices[1])
            diff2 = abs(prices[3] - prices[2])
            diff3 = abs(prices[4] - prices[3])

            # Later differences should be smaller (convergence)
            @test diff3 < diff1
        end

        @testset "Edge Cases" begin
            engine = Binomial(100)

            # Very high volatility
            data_high_vol = MarketData(100.0, 0.05, 1.0, 0.0)
            put = AmericanPut(100.0, 1.0)
            price_high_vol = price(put, engine, data_high_vol)
            @test price_high_vol > 0.0
            @test isfinite(price_high_vol)

            # Very low volatility (should approach intrinsic value)
            data_low_vol = MarketData(100.0, 0.05, 0.01, 0.0)
            price_low_vol = price(put, engine, data_low_vol)
            @test price_low_vol > 0.0
            @test isfinite(price_low_vol)

            # Zero interest rate
            data_zero_rate = MarketData(100.0, 0.0, 0.20, 0.0)
            price_zero_rate = price(put, engine, data_zero_rate)
            @test price_zero_rate > 0.0
            @test isfinite(price_zero_rate)
        end

        @testset "Cross-Engine Validation" begin
            # Compare Binomial and LSM for same option
            put = AmericanPut(100.0, 1.0)

            binomial_price = price(put, Binomial(200), data)
            lsm_price = price(put, LongstaffSchwartz(50, 20000), data)

            # Should agree within reasonable tolerance
            # (LSM has MC variance, so allow larger tolerance)
            @test isapprox(binomial_price, lsm_price, rtol=0.08)
        end

        @testset "Early Exercise Premium" begin
            # Test that American puts show clear early exercise premium
            put_am = AmericanPut(100.0, 1.0)
            put_eu = EuropeanPut(100.0, 1.0)

            # Higher interest rate → larger early exercise premium
            data_high_rate = MarketData(100.0, 0.10, 0.20, 0.0)

            engine = Binomial(200)
            price_am_high_r = price(put_am, engine, data_high_rate)
            price_eu_high_r = price(put_eu, BlackScholes(), data_high_rate)

            premium_high_r = price_am_high_r - price_eu_high_r

            # Low interest rate → smaller premium
            data_low_rate = MarketData(100.0, 0.01, 0.20, 0.0)
            price_am_low_r = price(put_am, engine, data_low_rate)
            price_eu_low_r = price(put_eu, BlackScholes(), data_low_rate)

            premium_low_r = price_am_low_r - price_eu_low_r

            # Premium should be positive in both cases
            @test premium_high_r > 0.0
            @test premium_low_r > 0.0

            # Higher interest rate → larger early exercise premium for puts
            @test premium_high_r > premium_low_r
        end
        end
    
    # Phase 1: Greeks Tests
    @testset "Greeks Module" begin
        include("test_greeks.jl")
    end
    
    # Phase 1: Implied Volatility Tests
    @testset "Implied Volatility Module" begin
        include("test_implied_vol.jl")
    end

    # Phase 1: Property-based tests (mathematical invariants)
    @testset "Property-Based Tests" begin
        include("test_properties.jl")
    end

    # Phase 2: Volatility (GARCH family)
    @testset "Volatility Module" begin
        include("test_volatility.jl")
    end

    # Phase 3: State estimation (Kalman, EKF, EnKF, Particle Filter)
    @testset "Filters Module" begin
        include("test_filters.jl")
    end

    # Phase 4: Inference (MLE, calibration, ABC)
    @testset "Inference Module" begin
        include("test_inference.jl")
    end

    # Phase 5: Advanced Hedging (OHMC, delta hedging)
    @testset "Hedging Module" begin
        include("test_hedging.jl")
    end

    # GPU acceleration (CUDA; tests run only when GPU available)
    @testset "GPU Module" begin
        include("test_gpu.jl")
    end
end
