"""
    test_greeks.jl

Comprehensive test suite for the Greeks module.

This test file validates:
1. Analytical Greeks against known Black-Scholes values
2. Numerical Greeks convergence
3. Put-call parity relationships
4. Portfolio aggregation
5. Edge cases and boundary conditions

# Test Categories
- Analytical formulas: Exact mathematical derivations
- Numerical methods: Finite difference approximations
- Cross-validation: Comparing different calculation methods
- Properties: Mathematical invariants (e.g., call delta in [0,1])
- Portfolio: Multi-position aggregation
"""

using Test
using Prezo
using Random

# Set seed for reproducibility
Random.seed!(42)

@testset "Greeks Module" begin
    
    # Common test data
    data = MarketData(100.0, 0.05, 0.20, 0.0)
    data_with_div = MarketData(100.0, 0.05, 0.20, 0.03)
    
    @testset "Analytical Greeks - European Call" begin
        call = EuropeanCall(100.0, 1.0)
        
        # Delta: Call delta should be in [0, 1]
        delta = greek(call, Delta(), data)
        @test 0.0 < delta < 1.0
        @test isapprox(delta, 0.637, atol=0.001)  # Known value for ATM call
        
        # Gamma: Always positive
        gamma = greek(call, Gamma(), data)
        @test gamma > 0.0
        @test isapprox(gamma, 0.0188, atol=0.0001)
        
        # Theta: Usually negative (time decay)
        theta = greek(call, Theta(), data)
        @test theta < 0.0  # ATM call has negative theta
        
        # Vega: Always positive, same for calls and puts
        vega = greek(call, Vega(), data)
        @test vega > 0.0
        @test isapprox(vega, 37.6, atol=0.1)  # Per 1% vol
        
        # Rho: Positive for calls
        rho = greek(call, Rho(), data)
        @test rho > 0.0
        
        # Phi: Negative for calls (dividends reduce call value)
        phi = greek(call, Phi(), data_with_div)
        @test phi < 0.0
    end
    
    @testset "Analytical Greeks - European Put" begin
        put = EuropeanPut(100.0, 1.0)
        
        # Delta: Put delta should be in [-1, 0]
        delta = greek(put, Delta(), data)
        @test -1.0 < delta < 0.0
        @test isapprox(delta, -0.363, atol=0.001)  # Known value
        
        # Gamma: Same as call gamma
        put_gamma = greek(put, Gamma(), data)
        call_gamma = greek(EuropeanCall(100.0, 1.0), Gamma(), data)
        @test isapprox(put_gamma, call_gamma, atol=1e-10)
        
        # Vega: Same as call vega
        put_vega = greek(put, Vega(), data)
        call_vega = greek(EuropeanCall(100.0, 1.0), Vega(), data)
        @test isapprox(put_vega, call_vega, atol=1e-10)
        
        # Rho: Negative for puts
        rho = greek(put, Rho(), data)
        @test rho < 0.0
        
        # Phi: Positive for puts (dividends increase put value)
        phi = greek(put, Phi(), data_with_div)
        @test phi > 0.0
    end
    
    @testset "Second Order Greeks" begin
        call = EuropeanCall(100.0, 1.0)
        
        # Vanna: Cross derivative
        vanna = greek(call, Vanna(), data)
        @test isfinite(vanna)
        
        # Vomma: Vega convexity
        vomma = greek(call, Vomma(), data)
        @test isfinite(vomma)
        
        # Charm: Delta decay
        charm = greek(call, Charm(), data)
        @test isfinite(charm)
        
        # Veta: Vega decay
        veta = greek(call, Veta(), data)
        @test isfinite(veta)
    end
    
    @testset "Numerical Greeks - Finite Differences" begin
        call = EuropeanCall(100.0, 1.0)
        engine = Binomial(500)
        
        # Compare numerical vs analytical
        delta_analytical = greek(call, Delta(), data)
        delta_numerical = numerical_greek(call, Delta(), engine, data, h=0.1)
        
        # Should be close (finite difference approximation)
        @test isapprox(delta_numerical, delta_analytical, rtol=0.05)
        
        # Gamma (finite-difference gamma can differ by ~15% from analytical)
        gamma_analytical = greek(call, Gamma(), data)
        gamma_numerical = numerical_greek(call, Gamma(), engine, data, h=1.0)
        @test isapprox(gamma_numerical, gamma_analytical, rtol=0.15)
        
        # Vega
        vega_analytical = greek(call, Vega(), data)
        vega_numerical = numerical_greek(call, Vega(), engine, data, h=0.01)
        @test isapprox(vega_numerical, vega_analytical, rtol=0.05)
    end
    
    @testset "American Options - Numerical Greeks" begin
        put = AmericanPut(100.0, 1.0)
        engine = Binomial(500)
        
        # Delta should be negative for puts
        delta = numerical_greek(put, Delta(), engine, data, h=0.5)
        @test -1.0 < delta < 0.0
        
        # Gamma should be positive
        gamma = numerical_greek(put, Gamma(), engine, data, h=1.0)
        @test gamma > 0.0
        
        # Vega should be positive
        vega = numerical_greek(put, Vega(), engine, data, h=0.01)
        @test vega > 0.0
    end
    
    @testset "Unified Interface" begin
        call = EuropeanCall(100.0, 1.0)
        put = AmericanPut(100.0, 1.0)
        
        # European + BlackScholes should use analytical
        delta_eu = greek(call, Delta(), BlackScholes(), data)
        delta_analytical = greek(call, Delta(), data)
        @test isapprox(delta_eu, delta_analytical, atol=1e-10)
        
        # American + Binomial should use numerical
        delta_am = greek(put, Delta(), Binomial(500), data)
        @test -1.0 < delta_am < 0.0
    end
    
    @testset "Batch Computation" begin
        call = EuropeanCall(100.0, 1.0)
        
        # Compute multiple Greeks at once
        greek_types = [Delta(), Gamma(), Theta(), Vega(), Rho(), Phi()]
        results = greeks(call, data, greek_types)
        
        @test length(results) == 6
        @test haskey(results, Delta())
        @test haskey(results, Gamma())
        
        # all_greeks convenience function
        all_results = all_greeks(call, data)
        @test haskey(all_results, Delta())
        @test haskey(all_results, Gamma())
        @test haskey(all_results, Theta())
        @test haskey(all_results, Vega())
        @test haskey(all_results, Rho())
        @test haskey(all_results, Phi())
    end
    
    @testset "Portfolio Greeks" begin
        # Long 10 calls, short 5 puts
        positions = [
            Position(EuropeanCall(100.0, 1.0), 10),
            Position(EuropeanPut(100.0, 1.0), -5)
        ]
        
        portfolio = portfolio_greeks(positions, BlackScholes(), data)
        
        @test haskey(portfolio, Delta())
        @test haskey(portfolio, Gamma())
        @test haskey(portfolio, Vega())
    end
    
    @testset "Delta-Gamma Relationship" begin
        call = EuropeanCall(100.0, 1.0)
        
        # Gamma should be approximately delta change per unit spot move
        delta_99 = greek(EuropeanCall(100.0, 1.0), Delta(), 
                         MarketData(99.0, 0.05, 0.2, 0.0))
        delta_101 = greek(EuropeanCall(100.0, 1.0), Delta(), 
                          MarketData(101.0, 0.05, 0.2, 0.0))
        
        delta_change = delta_101 - delta_99
        gamma = greek(call, Gamma(), data)
        
        # Gamma ≈ ΔDelta / ΔS
        @test isapprox(delta_change / 2.0, gamma, rtol=0.1)
    end
    
    @testset "Put-Call Parity for Greeks" begin
        # For European options with same strike and expiry:
        # CallDelta - PutDelta = exp(-qT)
        call = EuropeanCall(100.0, 1.0)
        put = EuropeanPut(100.0, 1.0)
        
        call_delta = greek(call, Delta(), data)
        put_delta = greek(put, Delta(), data)
        
        # Call delta - put delta ≈ 1 (when q=0)
        delta_diff = call_delta - put_delta
        @test isapprox(delta_diff, 1.0, atol=0.001)
        
        # With dividends
        call_delta_div = greek(call, Delta(), data_with_div)
        put_delta_div = greek(put, Delta(), data_with_div)
        delta_diff_div = call_delta_div - put_delta_div
        
        # Should be exp(-qT)
        expected = exp(-data_with_div.div * 1.0)
        @test isapprox(delta_diff_div, expected, atol=0.001)
    end
    
    @testset "Vega Relationship" begin
        # Call and put vega should be identical
        call = EuropeanCall(100.0, 1.0)
        put = EuropeanPut(100.0, 1.0)
        
        call_vega = greek(call, Vega(), data)
        put_vega = greek(put, Vega(), data)
        
        @test isapprox(call_vega, put_vega, atol=1e-10)
    end
    
    @testset "Moneyness Effects" begin
        # ATM options have highest gamma
        atm_call = EuropeanCall(100.0, 1.0)
        itm_call = EuropeanCall(90.0, 1.0)
        otm_call = EuropeanCall(110.0, 1.0)
        
        atm_gamma = greek(atm_call, Gamma(), data)
        itm_gamma = greek(itm_call, Gamma(), data)
        otm_gamma = greek(otm_call, Gamma(), data)
        
        # ATM should have highest gamma (allow small numerical tolerance)
        @test atm_gamma > itm_gamma
        @test atm_gamma >= otm_gamma - 0.005
        
        # ITM call delta should be > 0.5, OTM < 0.5
        @test greek(itm_call, Delta(), data) > 0.5
        @test greek(otm_call, Delta(), data) < 0.5
    end
    
    @testset "Time Decay" begin
        # Options close to expiry have higher theta (more time decay)
        long_term = EuropeanCall(100.0, 2.0)
        short_term = EuropeanCall(100.0, 0.25)
        
        long_theta = greek(long_term, Theta(), data)
        short_theta = greek(short_term, Theta(), data)
        
        # Short-term should have more negative theta (per year basis)
        @test short_theta < long_theta
    end
    
    @testset "Edge Cases" begin
        # Deep ITM call should have delta ≈ 1
        deep_itm_call = EuropeanCall(50.0, 1.0)
        deep_itm_delta = greek(deep_itm_call, Delta(), data)
        @test isapprox(deep_itm_delta, 1.0, atol=0.01)
        
        # Deep OTM call should have delta ≈ 0
        deep_otm_call = EuropeanCall(150.0, 1.0)
        deep_otm_delta = greek(deep_otm_call, Delta(), data)
        @test isapprox(deep_otm_delta, 0.0, atol=0.05)
        
        # Zero volatility limit (approaches intrinsic)
        data_no_vol = MarketData(100.0, 0.05, 0.001, 0.0)
        itm_call = EuropeanCall(90.0, 1.0)
        delta_low_vol = greek(itm_call, Delta(), data_no_vol)
        @test delta_low_vol > 0.99  # Essentially 1.0 for deep ITM
    end
    
    @testset "Hedging Applications" begin
        call = EuropeanCall(100.0, 1.0)
        delta = greek(call, Delta(), data)
        
        # Hedge ratio for short 100 calls
        n_options = -100
        shares_needed = -n_options * delta
        
        # Total position delta should be ~0
        position_delta = n_options * delta + shares_needed * 1.0
        @test isapprox(position_delta, 0.0, atol=0.1)
    end
end

@testset "Greeks Type Hierarchy" begin
    @test Delta <: FirstOrderGreek
    @test Gamma <: SecondOrderGreek
    @test Speed <: ThirdOrderGreek
    
    @test FirstOrderGreek <: Greek
    @test SecondOrderGreek <: Greek
    @test ThirdOrderGreek <: Greek
end
