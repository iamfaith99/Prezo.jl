"""
    test_implied_vol.jl

Comprehensive test suite for implied volatility calculations.

This test file validates:
1. Implied volatility round-trip (price → IV → price)
2. Solver convergence and accuracy
3. Put-call parity relationships in implied vol space
4. Volatility smile properties
5. Edge cases and error handling
6. Vectorized computations for option chains

# Test Categories
- Round-trip: Price → IV → price should match original
- Convergence: Solver iterations and tolerance
- Accuracy: Comparison across different solvers
- Properties: Volatility smile characteristics
- Edge cases: Deep ITM/OTM, short expiry, extreme vol
"""

using Test
using Prezo
using Random

Random.seed!(42)

@testset "Implied Volatility Module" begin

    # Common test data
    data = MarketData(100.0, 0.05, 0.20, 0.0)

    @testset "Round-Trip Tests" begin
        # Generate known price, compute IV, verify we get back same price
        call = EuropeanCall(100.0, 1.0)
        put = EuropeanPut(100.0, 1.0)

        # True price at σ = 20%
        true_vol = 0.20
        true_data = MarketData(100.0, 0.05, true_vol, 0.0)
        true_price_call = price(call, BlackScholes(), true_data)
        true_price_put = price(put, BlackScholes(), true_data)

        @testset "Newton-Raphson Round-Trip" begin
            # Recover implied vol from price
            iv_call = implied_vol(call, true_price_call, data, NewtonRaphson())
            @test isapprox(iv_call, true_vol, atol=1e-6)

            iv_put = implied_vol(put, true_price_put, data, NewtonRaphson())
            @test isapprox(iv_put, true_vol, atol=1e-6)

            # Verify: price with recovered IV should match
            check_data_call = MarketData(100.0, 0.05, iv_call, 0.0)
            check_price_call = price(call, BlackScholes(), check_data_call)
            @test isapprox(check_price_call, true_price_call, atol=1e-6)
        end

        @testset "Bisection Round-Trip" begin
            iv_call = implied_vol(call, true_price_call, data, Bisection())
            @test isapprox(iv_call, true_vol, atol=1e-6)

            iv_put = implied_vol(put, true_price_put, data, Bisection())
            @test isapprox(iv_put, true_vol, atol=1e-6)
        end

        @testset "HybridSolver Round-Trip" begin
            # Hybrid solver (default)
            iv_call = implied_vol(call, true_price_call, data, HybridSolver())
            @test isapprox(iv_call, true_vol, atol=1e-6)

            # Default method should also work
            iv_call_default = implied_vol(call, true_price_call, data)
            @test isapprox(iv_call_default, true_vol, atol=1e-6)
        end
    end

    @testset "Solver Accuracy Comparison" begin
        call = EuropeanCall(100.0, 1.0)
        target_price = 10.0  # Some market price

        iv_nr = implied_vol(call, target_price, data, NewtonRaphson())
        iv_bi = implied_vol(call, target_price, data, Bisection())
        iv_hy = implied_vol(call, target_price, data, HybridSolver())

        # All solvers should give similar results
        @test isapprox(iv_nr, iv_bi, rtol=0.01)
        @test isapprox(iv_nr, iv_hy, rtol=0.01)
        @test isapprox(iv_bi, iv_hy, rtol=0.01)
    end

    @testset "Put-Call Parity - Same IV" begin
        # For European options with same strike/expiry,
        # put-call parity implies they should have same implied vol
        strike = 100.0
        expiry = 1.0
        call = EuropeanCall(strike, expiry)
        put = EuropeanPut(strike, expiry)

        # Generate prices at some volatility
        vol = 0.25
        vol_data = MarketData(100.0, 0.05, vol, 0.0)
        call_price = price(call, BlackScholes(), vol_data)
        put_price = price(put, BlackScholes(), vol_data)

        # Compute IVs
        iv_call = implied_vol(call, call_price, data)
        iv_put = implied_vol(put, put_price, data)

        # Should be essentially identical
        @test isapprox(iv_call, iv_put, atol=1e-6)
    end

    @testset "Volatility Smile Properties" begin
        # Create a volatility smile (ITM and OTM have higher IV than ATM)
        atm_call = EuropeanCall(100.0, 1.0)
        itm_call = EuropeanCall(95.0, 1.0)
        otm_call = EuropeanCall(105.0, 1.0)

        # Simulate market prices with smile: 20% ATM, 22% ITM/OTM
        atm_price = price(atm_call, BlackScholes(), MarketData(100.0, 0.05, 0.20, 0.0))
        itm_price = price(itm_call, BlackScholes(), MarketData(100.0, 0.05, 0.22, 0.0))
        otm_price = price(otm_call, BlackScholes(), MarketData(100.0, 0.05, 0.22, 0.0))

        # Recover IVs
        iv_atm = implied_vol(atm_call, atm_price, data)
        iv_itm = implied_vol(itm_call, itm_price, data)
        iv_otm = implied_vol(otm_call, otm_price, data)

        # Verify smile shape
        @test isapprox(iv_atm, 0.20, atol=1e-6)
        @test isapprox(iv_itm, 0.22, atol=1e-6)
        @test isapprox(iv_otm, 0.22, atol=1e-6)
        @test iv_itm > iv_atm
        @test iv_otm > iv_atm
    end

    @testset "Moneyness Effects" begin
        # Deep ITM options: IV should be well-defined
        deep_itm_call = EuropeanCall(50.0, 1.0)
        deep_itm_price = price(deep_itm_call, BlackScholes(), data)
        iv_itm = implied_vol(deep_itm_call, deep_itm_price, data)
        @test 0.001 < iv_itm < 5.0

        # Deep OTM options: IV should be well-defined
        deep_otm_call = EuropeanCall(150.0, 1.0)
        deep_otm_price = price(deep_otm_call, BlackScholes(), data)
        iv_otm = implied_vol(deep_otm_call, deep_otm_price, data)
        @test 0.001 < iv_otm < 5.0

        # ATM options: Most reliable IV estimation
        atm_call = EuropeanCall(100.0, 1.0)
        atm_price = price(atm_call, BlackScholes(), data)
        iv_atm = implied_vol(atm_call, atm_price, data)
        @test isapprox(iv_atm, 0.20, atol=1e-6)
    end

    @testset "Time to Expiry Effects" begin
        # Short-dated options
        short_call = EuropeanCall(100.0, 0.1)
        short_price = price(short_call, BlackScholes(), data)
        iv_short = implied_vol(short_call, short_price, data)
        @test isapprox(iv_short, 0.20, atol=1e-6)

        # Long-dated options
        long_call = EuropeanCall(100.0, 3.0)
        long_price = price(long_call, BlackScholes(), data)
        iv_long = implied_vol(long_call, long_price, data)
        @test isapprox(iv_long, 0.20, atol=1e-6)
    end

    @testset "Dividend Effects" begin
        data_div = MarketData(100.0, 0.05, 0.20, 0.03)

        call = EuropeanCall(100.0, 1.0)
        call_price = price(call, BlackScholes(), data_div)
        iv_call = implied_vol(call, call_price, data_div)
        @test isapprox(iv_call, 0.20, atol=1e-6)

        put = EuropeanPut(100.0, 1.0)
        put_price = price(put, BlackScholes(), data_div)
        iv_put = implied_vol(put, put_price, data_div)
        @test isapprox(iv_put, 0.20, atol=1e-6)
    end

    @testset "Interest Rate Effects" begin
        # High rates
        data_high_rate = MarketData(100.0, 0.10, 0.20, 0.0)
        call = EuropeanCall(100.0, 1.0)
        call_price = price(call, BlackScholes(), data_high_rate)
        iv_call = implied_vol(call, call_price, data_high_rate)
        @test isapprox(iv_call, 0.20, atol=1e-6)

        # Zero rates
        data_zero_rate = MarketData(100.0, 0.0, 0.20, 0.0)
        put = EuropeanPut(100.0, 1.0)
        put_price = price(put, BlackScholes(), data_zero_rate)
        iv_put = implied_vol(put, put_price, data_zero_rate)
        @test isapprox(iv_put, 0.20, atol=1e-6)
    end

    @testset "Invalid Price Handling" begin
        call = EuropeanCall(100.0, 1.0)

        # Price too low (below intrinsic)
        iv_low = implied_vol(call, 0.1, data, Bisection())
        @test isnan(iv_low)

        # Price too high (above spot)
        iv_high = implied_vol(call, 200.0, data, Bisection())
        @test isnan(iv_high)
    end

    @testset "Vectorized Computation" begin
        # Create option chain
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        expiry = 1.0

        # Generate prices at 20% vol
        prices = Float64[]
        for strike in strikes
            call = EuropeanCall(strike, expiry)
            push!(prices, price(call, BlackScholes(), data))
        end

        # Compute IVs for entire chain
        ivs = implied_vol_chain(strikes, prices, expiry, true, data)

        @test length(ivs) == length(strikes)
        @test all(.!isnan.(ivs))

        # All should recover ~20% vol
        for iv in ivs
            @test isapprox(iv, 0.20, atol=1e-5)
        end
    end

    @testset "IV Statistics" begin
        # Create a volatility smile
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]

        # Simulate prices with smile: 25% wings, 20% ATM
        prices = Float64[]
        for (i, strike) in enumerate(strikes)
            vol = strike == 100.0 ? 0.20 : 0.25
            call = EuropeanCall(strike, 1.0)
            push!(prices, price(call, BlackScholes(),
                MarketData(100.0, 0.05, vol, 0.0)))
        end

        ivs = implied_vol_chain(strikes, prices, 1.0, true, data)
        stats = implied_vol_stats(ivs)

        @test stats.n_valid == 5
        @test 0.20 <= stats.minimum <= 0.26  # Allow small tolerance for numerical precision
        @test 0.20 <= stats.maximum <= 0.26  # Allow small tolerance for numerical precision
        @test stats.range > 0.04  # Smile range should be significant
    end

    @testset "Solver Configuration" begin
        # Test custom tolerances
        tight_solver = NewtonRaphson(tol=1e-10)
        loose_solver = NewtonRaphson(tol=1e-4)

        call = EuropeanCall(100.0, 1.0)
        target_price = 10.45

        iv_tight = implied_vol(call, target_price, data, tight_solver)
        iv_loose = implied_vol(call, target_price, data, loose_solver)

        # Both should give valid results
        @test 0.0 < iv_tight < 5.0
        @test 0.0 < iv_loose < 5.0

        # Test bisection bounds
        narrow_solver = Bisection(vol_low=0.1, vol_high=0.3)
        iv_narrow = implied_vol(call, target_price, data, narrow_solver)
        @test 0.0 < iv_narrow < 5.0
    end

    @testset "Extreme Volatility" begin
        # Very low volatility (essentially intrinsic value)
        data_low_vol = MarketData(100.0, 0.05, 0.05, 0.0)
        itm_call = EuropeanCall(90.0, 1.0)
        low_vol_price = price(itm_call, BlackScholes(), data_low_vol)
        iv_low = implied_vol(itm_call, low_vol_price, data_low_vol)
        @test isapprox(iv_low, 0.05, atol=1e-4)

        # Very high volatility
        data_high_vol = MarketData(100.0, 0.05, 1.0, 0.0)
        high_vol_price = price(itm_call, BlackScholes(), data_high_vol)
        iv_high = implied_vol(itm_call, high_vol_price, data_high_vol)
        @test isapprox(iv_high, 1.0, atol=1e-4)
    end

    @testset "Volatility Surface" begin
        # Build surface from price matrix (flat 20% vol)
        strikes = [90.0, 100.0, 110.0]
        expiries = [0.25, 0.5, 1.0]
        data = MarketData(100.0, 0.05, 0.20, 0.0)
        price_matrix = zeros(3, 3)
        for (i, k) in enumerate(strikes), (j, t) in enumerate(expiries)
            opt = EuropeanCall(k, t)
            price_matrix[i, j] = price(opt, BlackScholes(), data)
        end
        surf = build_implied_vol_surface(strikes, expiries, price_matrix, data)
        @test surf.is_call
        @test length(surf.strikes) == 3 && length(surf.expiries) == 3
        @test size(surf.iv_matrix) == (3, 3)
        @test all(.!isnan.(surf.iv_matrix))
        @test all(isapprox.(surf.iv_matrix, 0.20, atol=1e-5))

        # Interpolation on grid point
        @test isapprox(surface_iv(surf, 100.0, 0.5), 0.20, atol=1e-6)
        # Interpolation between grid points
        iv_mid = surface_iv(surf, 95.0, 0.375)
        @test 0.15 < iv_mid < 0.25
        # Outside grid returns NaN
        @test isnan(surface_iv(surf, 80.0, 0.5))
        @test isnan(surface_iv(surf, 100.0, 2.0))

        # Surface stats
        stats = surface_stats(surf)
        @test stats.n_valid == 9
        @test isapprox(stats.mean, 0.20, atol=1e-5)
        @test stats.range >= 0.0

        # Put surface
        put_price_matrix = similar(price_matrix)
        for (i, k) in enumerate(strikes), (j, t) in enumerate(expiries)
            opt = EuropeanPut(k, t)
            put_price_matrix[i, j] = price(opt, BlackScholes(), data)
        end
        surf_put = build_implied_vol_surface(strikes, expiries, put_price_matrix, data; is_call=false)
        @test !surf_put.is_call
        @test all(isapprox.(surf_put.iv_matrix, 0.20, atol=1e-5))
    end
end

@testset "Implied Volatility Type Hierarchy" begin
    @test NewtonRaphson() isa IVSolver
    @test Bisection() isa IVSolver
    @test HybridSolver() isa IVSolver
end

@testset "Helper Functions" begin
    call = EuropeanCall(100.0, 1.0)
    data = MarketData(100.0, 0.05, 0.20, 0.0)

    # Price bounds
    p_low, p_high = price_bounds(call, data)
    @test p_low < p_high
    @test p_low > 0.0

    # Valid price check
    is_valid, intrinsic, upper = is_valid_price(call, 10.0, data)
    @test is_valid

    is_valid_bad, _, _ = is_valid_price(call, 200.0, data)
    @test !is_valid_bad
end
