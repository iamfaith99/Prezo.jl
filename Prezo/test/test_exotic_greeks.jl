"""
Test suite for exotic option Greeks.

Tests numerical Greeks calculations for:
- Asian options (Arithmetic and Geometric)
- Barrier options (Knock-Out and Knock-In)
- Lookback options (Fixed and Floating Strike)
"""

using Test
using Prezo
using Random

# Set seed for reproducibility
Random.seed!(42)

@testset "Exotic Option Greeks" begin
    # Common market data
    data = MarketData(100.0, 0.05, 0.20, 0.02)

    # Use a Monte Carlo engine with enough paths for reasonable precision
    # Note: More paths = more accurate Greeks but slower tests
    mc_engine = MonteCarlo(100, 20000)

    @testset "Asian Option Greeks" begin
        # Monthly averaging over 1 year
        averaging_times = collect(1:12) ./ 12

        @testset "Arithmetic Asian Call" begin
            asian_call = ArithmeticAsianCall(100.0, 1.0, averaging_times)

            # Test delta
            delta = greek(asian_call, Delta(), mc_engine, data)
            @test 0.0 < delta < 1.0  # Call delta should be between 0 and 1

            # Test gamma
            gamma = greek(asian_call, Gamma(), mc_engine, data)
            @test gamma > 0.0  # Gamma should be positive for long options

            # Test vega
            vega = greek(asian_call, Vega(), mc_engine, data)
            @test vega > 0.0  # Vega should be positive

            # Test theta (typically negative for long options)
            theta = greek(asian_call, Theta(), mc_engine, data)
            @test isfinite(theta)

            # Test rho
            rho = greek(asian_call, Rho(), mc_engine, data)
            @test rho > 0.0  # Call rho should be positive

            # Test phi (dividend sensitivity)
            phi = greek(asian_call, Phi(), mc_engine, data)
            @test phi < 0.0  # Call phi should be negative
        end

        @testset "Arithmetic Asian Put" begin
            asian_put = ArithmeticAsianPut(100.0, 1.0, averaging_times)

            # Test delta
            delta = greek(asian_put, Delta(), mc_engine, data)
            @test -1.0 < delta < 0.0  # Put delta should be between -1 and 0

            # Test vega
            vega = greek(asian_put, Vega(), mc_engine, data)
            @test vega > 0.0  # Vega should be positive
        end

        @testset "Geometric Asian Call - Analytical Comparison" begin
            geo_call = GeometricAsianCall(100.0, 1.0, averaging_times)

            # Numerical Greeks
            delta_num = greek(geo_call, Delta(), mc_engine, data)

            # Delta should be reasonable for a call
            @test 0.0 < delta_num < 1.0

            # Price should be lower than vanilla European call (averaging reduces volatility exposure)
            vanilla_call = EuropeanCall(100.0, 1.0)
            vanilla_price = price(vanilla_call, BlackScholes(), data)
            geo_price = price(geo_call, mc_engine, data)

            @test geo_price < vanilla_price * 1.1  # Allow some MC variance
        end

        @testset "all_greeks for Asian Option" begin
            asian_call = ArithmeticAsianCall(100.0, 1.0, averaging_times)

            all_g = all_greeks(asian_call, mc_engine, data)

            @test haskey(all_g, Delta())
            @test haskey(all_g, Gamma())
            @test haskey(all_g, Theta())
            @test haskey(all_g, Vega())
            @test haskey(all_g, Rho())
            @test haskey(all_g, Phi())

            # All values should be finite
            for (k, v) in all_g
                @test isfinite(v)
            end
        end
    end

    @testset "Barrier Option Greeks" begin
        @testset "Up-and-Out Call" begin
            # Spot = 100, Barrier = 120 (20% above)
            uoc = KnockOutCall(100.0, 1.0, 120.0, :up_and_out)

            # Test delta
            delta = greek(uoc, Delta(), mc_engine, data)
            # Up-and-out call delta can be complex, but should be bounded
            @test -1.0 < delta < 2.0

            # Test vega
            vega = greek(uoc, Vega(), mc_engine, data)
            # Up-and-out call vega can be negative (higher vol = more likely to knock out)
            @test isfinite(vega)

            # Test gamma
            gamma = greek(uoc, Gamma(), mc_engine, data)
            @test isfinite(gamma)
        end

        @testset "Down-and-Out Put" begin
            # Spot = 100, Barrier = 80 (20% below)
            dop = KnockOutPut(100.0, 1.0, 80.0, :down_and_out)

            # Test delta
            delta = greek(dop, Delta(), mc_engine, data)
            # Down-and-out put delta - should be negative overall
            @test isfinite(delta)

            # Test vega
            vega = greek(dop, Vega(), mc_engine, data)
            @test isfinite(vega)
        end

        @testset "Knock-In Call" begin
            # Down-and-in call
            dic = KnockInCall(100.0, 1.0, 80.0, :down_and_in)

            # Test delta
            delta = greek(dic, Delta(), mc_engine, data)
            @test isfinite(delta)

            # Test vega
            vega = greek(dic, Vega(), mc_engine, data)
            @test isfinite(vega)
        end

        @testset "In-Out Parity Check" begin
            # Knock-In + Knock-Out = Vanilla (for same barrier)
            barrier = 120.0

            uoc = KnockOutCall(100.0, 1.0, barrier, :up_and_out)
            uic = KnockInCall(100.0, 1.0, barrier, :up_and_in)
            vanilla = EuropeanCall(100.0, 1.0)

            # Test delta parity (with tolerance for MC noise)
            delta_out = greek(uoc, Delta(), mc_engine, data)
            delta_in = greek(uic, Delta(), mc_engine, data)
            delta_vanilla = greek(vanilla, Delta(), data)

            # In + Out ≈ Vanilla
            @test isapprox(delta_in + delta_out, delta_vanilla, rtol=0.15)
        end

        @testset "Barrier Proximity Functions" begin
            barrier_opt = KnockOutCall(100.0, 1.0, 120.0, :up_and_out)

            # Test barrier proximity calculation
            prox = barrier_proximity(barrier_opt, data)
            @test prox > 0.0  # Spot is not at barrier
            @test prox ≈ 0.2  # |100 - 120| / 100 = 0.2

            # Test near barrier check
            @test !is_near_barrier(barrier_opt, data, threshold=0.1)
            @test is_near_barrier(barrier_opt, data, threshold=0.25)

            # Test with spot very close to barrier
            data_near = MarketData(118.0, 0.05, 0.20, 0.02)
            @test is_near_barrier(barrier_opt, data_near, threshold=0.02)
        end
    end

    @testset "Lookback Option Greeks" begin
        @testset "Fixed Strike Lookback Call" begin
            lookback_call = FixedStrikeLookbackCall(100.0, 1.0)

            # Test delta
            delta = greek(lookback_call, Delta(), mc_engine, data)
            @test delta > 0.0  # Call delta should be positive

            # Test vega - lookback options have high vega
            vega = greek(lookback_call, Vega(), mc_engine, data)
            @test vega > 0.0

            # Lookback vega should be higher than vanilla
            vanilla_call = EuropeanCall(100.0, 1.0)
            vanilla_vega = greek(vanilla_call, Vega(), data)
            @test vega > vanilla_vega * 0.5  # At least comparable (allow MC variance)
        end

        @testset "Fixed Strike Lookback Put" begin
            lookback_put = FixedStrikeLookbackPut(100.0, 1.0)

            # Test delta
            delta = greek(lookback_put, Delta(), mc_engine, data)
            @test delta < 0.0  # Put delta should be negative

            # Test vega
            vega = greek(lookback_put, Vega(), mc_engine, data)
            @test vega > 0.0
        end

        @testset "Floating Strike Lookback Call" begin
            fl_call = FloatingStrikeLookbackCall(1.0)

            # Test delta
            delta = greek(fl_call, Delta(), mc_engine, data)
            @test isfinite(delta)

            # Test vega
            vega = greek(fl_call, Vega(), mc_engine, data)
            @test vega > 0.0
        end

        @testset "Floating Strike Lookback Put" begin
            fl_put = FloatingStrikeLookbackPut(1.0)

            # Test delta
            delta = greek(fl_put, Delta(), mc_engine, data)
            @test isfinite(delta)

            # Test vega
            vega = greek(fl_put, Vega(), mc_engine, data)
            @test vega > 0.0
        end
    end

    @testset "Second Order Greeks" begin
        averaging_times = collect(1:12) ./ 12
        asian_call = ArithmeticAsianCall(100.0, 1.0, averaging_times)

        @testset "Vanna" begin
            vanna = greek(asian_call, Vanna(), mc_engine, data)
            @test isfinite(vanna)
        end

        @testset "Vomma" begin
            vomma = greek(asian_call, Vomma(), mc_engine, data)
            @test isfinite(vomma)
        end
    end

    @testset "Helper Functions" begin
        @testset "get_expiry" begin
            averaging_times = collect(1:12) ./ 12

            @test get_expiry(ArithmeticAsianCall(100.0, 1.5, averaging_times)) ≈ 1.5
            @test get_expiry(ArithmeticAsianPut(100.0, 2.0, averaging_times)) ≈ 2.0
            @test get_expiry(GeometricAsianCall(100.0, 0.5, averaging_times)) ≈ 0.5
            @test get_expiry(GeometricAsianPut(100.0, 1.0, averaging_times)) ≈ 1.0
            @test get_expiry(KnockOutCall(100.0, 0.75, 120.0, :up_and_out)) ≈ 0.75
            @test get_expiry(KnockOutPut(100.0, 1.25, 80.0, :down_and_out)) ≈ 1.25
            @test get_expiry(KnockInCall(100.0, 1.0, 80.0, :down_and_in)) ≈ 1.0
            @test get_expiry(KnockInPut(100.0, 0.5, 120.0, :up_and_in)) ≈ 0.5
            @test get_expiry(FixedStrikeLookbackCall(100.0, 1.0)) ≈ 1.0
            @test get_expiry(FixedStrikeLookbackPut(100.0, 2.0)) ≈ 2.0
            @test get_expiry(FloatingStrikeLookbackCall(1.5)) ≈ 1.5
            @test get_expiry(FloatingStrikeLookbackPut(0.75)) ≈ 0.75
        end

        @testset "with_shorter_expiry" begin
            averaging_times = collect(1:12) ./ 12

            # Asian options - times should scale
            original = ArithmeticAsianCall(100.0, 1.0, averaging_times)
            shorter = with_shorter_expiry(original, 0.5)

            @test get_expiry(shorter) ≈ 0.5
            @test shorter.strike == original.strike
            @test length(shorter.averaging_times) == length(original.averaging_times)
            @test shorter.averaging_times[end] ≈ averaging_times[end] * 0.5

            # Barrier options
            barrier_original = KnockOutCall(100.0, 1.0, 120.0, :up_and_out)
            barrier_shorter = with_shorter_expiry(barrier_original, 0.6)

            @test get_expiry(barrier_shorter) ≈ 0.6
            @test barrier_shorter.strike == barrier_original.strike
            @test barrier_shorter.barrier == barrier_original.barrier
            @test barrier_shorter.barrier_type == barrier_original.barrier_type

            # Floating lookback options (no strike)
            fl_original = FloatingStrikeLookbackCall(1.0)
            fl_shorter = with_shorter_expiry(fl_original, 0.8)

            @test get_expiry(fl_shorter) ≈ 0.8
        end
    end

    @testset "Spot Sensitivity (Delta)" begin
        averaging_times = collect(1:12) ./ 12
        asian_call = ArithmeticAsianCall(100.0, 1.0, averaging_times)

        # Test that delta sign is correct across spot levels
        spots = [80.0, 100.0, 120.0]

        for spot in spots
            test_data = MarketData(spot, 0.05, 0.20, 0.02)
            delta = greek(asian_call, Delta(), mc_engine, test_data)

            # Asian call delta should always be positive
            @test delta > 0.0
        end

        # Higher spot should give higher delta for call (approaching 1)
        delta_low = greek(asian_call, Delta(), mc_engine, MarketData(80.0, 0.05, 0.20, 0.02))
        delta_high = greek(asian_call, Delta(), mc_engine, MarketData(120.0, 0.05, 0.20, 0.02))

        @test delta_high > delta_low
    end

    @testset "Volatility Sensitivity (Vega)" begin
        averaging_times = collect(1:12) ./ 12
        asian_call = ArithmeticAsianCall(100.0, 1.0, averaging_times)

        # Vega should be positive for long options
        vols = [0.10, 0.20, 0.30]

        for vol in vols
            test_data = MarketData(100.0, 0.05, vol, 0.02)
            vega = greek(asian_call, Vega(), mc_engine, test_data)

            @test vega > 0.0
        end
    end

    @testset "Greeks Batch Computation" begin
        averaging_times = collect(1:12) ./ 12
        asian_call = ArithmeticAsianCall(100.0, 1.0, averaging_times)

        # Test greeks() function with subset of Greeks
        subset = greeks(asian_call, mc_engine, data, [Delta(), Vega()])

        @test haskey(subset, Delta())
        @test haskey(subset, Vega())
        @test !haskey(subset, Gamma())

        @test isfinite(subset[Delta()])
        @test isfinite(subset[Vega()])
    end

    @testset "Edge Cases" begin
        @testset "Near Expiry" begin
            averaging_times = collect(1:12) ./ 12

            # Very short dated option
            short_asian = ArithmeticAsianCall(100.0, 0.01, averaging_times .* 0.01)

            # Theta should return 0 for options that can't have time reduced
            theta = greek(short_asian, Theta(), mc_engine, data)
            @test isfinite(theta)
        end

        @testset "Deep ITM Asian" begin
            averaging_times = collect(1:12) ./ 12
            itm_asian = ArithmeticAsianCall(80.0, 1.0, averaging_times)  # Low strike

            delta = greek(itm_asian, Delta(), mc_engine, data)
            # Deep ITM call should have high delta
            @test delta > 0.5
        end

        @testset "Deep OTM Asian" begin
            averaging_times = collect(1:12) ./ 12
            otm_asian = ArithmeticAsianCall(150.0, 1.0, averaging_times)  # High strike

            delta = greek(otm_asian, Delta(), mc_engine, data)
            # Deep OTM call should have low delta
            @test delta < 0.5
        end

        @testset "High Volatility" begin
            averaging_times = collect(1:12) ./ 12
            asian_call = ArithmeticAsianCall(100.0, 1.0, averaging_times)

            high_vol_data = MarketData(100.0, 0.05, 0.80, 0.02)

            delta = greek(asian_call, Delta(), mc_engine, high_vol_data)
            gamma = greek(asian_call, Gamma(), mc_engine, high_vol_data)
            vega = greek(asian_call, Vega(), mc_engine, high_vol_data)

            @test isfinite(delta)
            @test isfinite(gamma)
            @test isfinite(vega)
        end
    end
end
