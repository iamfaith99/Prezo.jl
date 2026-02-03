"""
    exotic_greeks_examples.jl

Comprehensive examples demonstrating exotic option Greeks calculations.

This file showcases practical usage of Greeks for:
- Asian Options (Arithmetic and Geometric averages)
- Barrier Options (Knock-In and Knock-Out)
- Lookback Options (Fixed and Floating Strike)

Each example demonstrates:
1. Option construction
2. Greeks calculation
3. Interpretation of results
4. Comparison with vanilla options
"""

using Prezo
using Printf
using Random

Random.seed!(42)

println("="^80)
println("EXOTIC OPTION GREEKS - COMPREHENSIVE EXAMPLES")
println("="^80)

# Common market data
data = MarketData(100.0, 0.05, 0.20, 0.02)
println("\nMarket Parameters:")
println("  Spot Price (S):      \$$(data.spot)")
println("  Risk-Free Rate (r):  $(data.rate * 100)%")
println("  Volatility (σ):      $(data.vol * 100)%")
println("  Dividend Yield (q):  $(data.div * 100)%")

# Monte Carlo engine with sufficient paths for stable Greeks
mc_engine = MonteCarlo(100, 50000)
println("  MC Simulations:      $(mc_engine.reps) paths, $(mc_engine.steps) steps")

# ============================================================================
# Example 1: Asian Options
# ============================================================================

println("\n" * "="^80)
println("EXAMPLE 1: ASIAN OPTIONS")
println("="^80)

# Monthly averaging over 1 year
averaging_times = collect(1:12) ./ 12
asian_call = ArithmeticAsianCall(100.0, 1.0, averaging_times)
vanilla_call = EuropeanCall(100.0, 1.0)

println("\nAsian Call vs Vanilla Call Comparison:")
println("Strike: \$100, Expiry: 1 year, Monthly averaging")
println()

# Price comparison
asian_price = price(asian_call, mc_engine, data)
vanilla_price = price(vanilla_call, BlackScholes(), data)

println("PRICING:")
println("  Asian Call Price:    \$", @sprintf("%.4f", asian_price))
println("  Vanilla Call Price:  \$", @sprintf("%.4f", vanilla_price))
println("  Difference:          \$", @sprintf("%.4f", vanilla_price - asian_price))
println("  (Asian cheaper due to reduced volatility from averaging)")
println()

# Greeks comparison
println("GREEKS COMPARISON:")
println(@sprintf("%-12s %12s %12s %12s", "Greek", "Asian", "Vanilla", "Ratio"))
println("-"^50)

asian_greeks = all_greeks(asian_call, mc_engine, data)
vanilla_greeks = all_greeks(vanilla_call, data)

for greek_type in [Delta(), Gamma(), Vega(), Theta(), Rho()]
    asian_val = asian_greeks[greek_type]
    vanilla_val = vanilla_greeks[greek_type]
    ratio = abs(asian_val) / abs(vanilla_val)

    println(@sprintf("%-12s %12.4f %12.4f %12.2f%%",
        string(typeof(greek_type).name.name),
        asian_val, vanilla_val, ratio * 100))
end

println("\nKEY INSIGHTS:")
println("  • Delta: Asian has similar delta to vanilla (directional exposure)")
println("  • Gamma: Asian has lower gamma (less sensitive to spot moves)")
println("  • Vega:  Asian has ~60% of vanilla vega (averaging reduces vol sensitivity)")
println("  • Theta: Asian theta behavior depends on averaging schedule")

# ============================================================================
# Example 2: Barrier Options
# ============================================================================

println("\n" * "="^80)
println("EXAMPLE 2: BARRIER OPTIONS")
println("="^80)

# Up-and-Out Call (barrier above spot)
barrier_call = KnockOutCall(100.0, 1.0, 120.0, :up_and_out)
println("\nUp-and-Out Call:")
println("  Strike: \$100, Barrier: \$120 (20% above spot)")
println("  Current Spot: \$$(data.spot)")
println("  Barrier Proximity: $(round(barrier_proximity(barrier_call, data) * 100, digits=1))%")
println("  Near Barrier? $(is_near_barrier(barrier_call, data, threshold=0.15) ? "YES" : "NO")")
println()

barrier_price = price(barrier_call, mc_engine, data)
println("  Price: \$", @sprintf("%.4f", barrier_price))
println("  (Cheaper than vanilla due to knock-out risk)")
println()

barrier_greeks = all_greeks(barrier_call, mc_engine, data)

println("BARRIER OPTION GREEKS:")
println(@sprintf("  Delta:  %8.4f  (positive but suppressed by barrier)", barrier_greeks[Delta()]))
println(@sprintf("  Gamma:  %8.6f  (can be large near barrier)", barrier_greeks[Gamma()]))
println(@sprintf("  Vega:   %8.4f  (negative: higher vol = more knock-out risk)", barrier_greeks[Vega()]))
println(@sprintf("  Theta:  %8.4f  (time decay affected by barrier proximity)", barrier_greeks[Theta()]))

# Compare with vanilla
vanilla_delta = greek(vanilla_call, Delta(), data)
barrier_delta = barrier_greeks[Delta()]
delta_suppression = (1 - barrier_delta / vanilla_delta) * 100

println("\n  Delta Suppression: $(round(delta_suppression, digits=1))% vs vanilla")
println("  (Barrier limits upside participation)")

# Down-and-Out Put
dop = KnockOutPut(100.0, 1.0, 80.0, :down_and_out)
println("\n\nDown-and-Out Put:")
println("  Strike: \$100, Barrier: \$80 (20% below spot)")

dop_greeks = all_greeks(dop, mc_engine, data)
println(@sprintf("  Delta:  %8.4f  (negative, typical for puts)", dop_greeks[Delta()]))
println(@sprintf("  Vega:   %8.4f", dop_greeks[Vega()]))

# In-Out Parity Check
println("\n\nIN-OUT PARITY VERIFICATION:")
println("(Knock-In + Knock-Out = Vanilla)")

uoc = KnockOutCall(100.0, 1.0, 120.0, :up_and_out)
uic = KnockInCall(100.0, 1.0, 120.0, :up_and_in)

delta_out = greek(uoc, Delta(), mc_engine, data)
delta_in = greek(uic, Delta(), mc_engine, data)
delta_sum = delta_out + delta_in

println(@sprintf("  Knock-Out Delta: %8.4f", delta_out))
println(@sprintf("  Knock-In Delta:  %8.4f", delta_in))
println(@sprintf("  Sum:             %8.4f", delta_sum))
println(@sprintf("  Vanilla Delta:   %8.4f", vanilla_delta))
println(@sprintf("  Difference:      %8.4f (MC noise)", abs(delta_sum - vanilla_delta)))

# ============================================================================
# Example 3: Lookback Options
# ============================================================================

println("\n" * "="^80)
println("EXAMPLE 3: LOOKBACK OPTIONS")
println("="^80)

# Fixed Strike Lookback Call
lookback_call = FixedStrikeLookbackCall(100.0, 1.0)
println("\nFixed-Strike Lookback Call:")
println("  Strike: \$100, Expiry: 1 year")
println("  Payoff: max(0, S_max - K) where S_max = maximum price over life")
println()

lookback_price = price(lookback_call, mc_engine, data)
println("  Price: \$", @sprintf("%.4f", lookback_price))
println("  (More expensive than vanilla: captures best price)")
println()

lookback_greeks = all_greeks(lookback_call, mc_engine, data)

println("LOOKBACK GREEKS:")
println(@sprintf("  Delta:  %8.4f  (higher than vanilla)", lookback_greeks[Delta()]))
println(@sprintf("  Gamma:  %8.6f", lookback_greeks[Gamma()]))
println(@sprintf("  Vega:   %8.4f  (very high: benefits from large moves)", lookback_greeks[Vega()]))
println(@sprintf("  Theta:  %8.4f", lookback_greeks[Theta()]))

# Compare vega with vanilla
lookback_vega = lookback_greeks[Vega()]
vanilla_vega = greek(vanilla_call, Vega(), data)
vega_ratio = lookback_vega / vanilla_vega

println(@sprintf("\n  Vega Enhancement: %.1f%% vs vanilla", (vega_ratio - 1) * 100))
println("  (Lookback benefits more from volatility)")

# Floating Strike Lookback Call
fl_call = FloatingStrikeLookbackCall(1.0)
println("\n\nFloating-Strike Lookback Call:")
println("  Payoff: S_T - S_min (always in-the-money at expiry)")

fl_price = price(fl_call, mc_engine, data)
fl_greeks = all_greeks(fl_call, mc_engine, data)

println(@sprintf("  Price:  \$%.4f", fl_price))
println(@sprintf("  Delta:  %8.4f  (always positive)", fl_greeks[Delta()]))
println(@sprintf("  Vega:   %8.4f  (extreme vega sensitivity)", fl_greeks[Vega()]))

# ============================================================================
# Example 4: Risk Management with Exotic Greeks
# ============================================================================

println("\n" * "="^80)
println("EXAMPLE 4: RISK MANAGEMENT")
println("="^80)

println("\nScenario: Portfolio with 100 Asian calls")
portfolio_size = 100
asian_position_delta = portfolio_size * asian_greeks[Delta()]
asian_position_vega = portfolio_size * asian_greeks[Vega()]

println(@sprintf("  Portfolio Delta: %.2f", asian_position_delta))
println(@sprintf("  Portfolio Vega:  %.2f", asian_position_vega))
println()

# Delta hedge
shares_to_hedge = -asian_position_delta
println("DELTA HEDGING:")
println(@sprintf("  Action: %s %.0f shares of underlying",
    shares_to_hedge > 0 ? "BUY" : "SELL",
    abs(shares_to_hedge)))
println("  Result: Portfolio delta-neutral (directionally neutral)")
println()

# Gamma risk
asian_position_gamma = portfolio_size * asian_greeks[Gamma()]
println("GAMMA EXPOSURE:")
println(@sprintf("  Portfolio Gamma: %.4f", asian_position_gamma))
println("  P&L from 5% spot move: \$", @sprintf("%.2f", 0.5 * asian_position_gamma * (5.0)^2))
println("  (Convexity P&L from large moves)")

# ============================================================================
# Example 5: Sensitivity Analysis
# ============================================================================

println("\n" * "="^80)
println("EXAMPLE 5: SENSITIVITY ANALYSIS")
println("="^80)

println("\nAsian Call Delta across different spot levels:")
println(@sprintf("%-15s %12s %12s", "Spot", "Delta", "Delta/Spot"))
println("-"^40)

for spot in [80.0, 90.0, 100.0, 110.0, 120.0]
    test_data = MarketData(spot, 0.05, 0.20, 0.02)
    delta = greek(asian_call, Delta(), mc_engine, test_data)
    normalized_delta = delta

    println(@sprintf("\$%-14.2f %12.4f %12.4f", spot, delta, normalized_delta))
end

println("\nBarrier Call Vega at different volatility levels:")
println(@sprintf("%-15s %12s %12s", "Volatility", "Vega", "% of Vanilla"))
println("-"^40)

for vol in [0.10, 0.15, 0.20, 0.25, 0.30]
    test_data = MarketData(100.0, 0.05, vol, 0.02)
    barrier_vega = greek(barrier_call, Vega(), mc_engine, test_data)
    vanilla_vega_test = greek(vanilla_call, Vega(), test_data)
    vega_pct = (barrier_vega / vanilla_vega_test) * 100

    println(@sprintf("%-14.1f%% %12.4f %12.1f%%", vol * 100, barrier_vega, vega_pct))
end

# ============================================================================
# Example 6: Second Order Greeks
# ============================================================================

println("\n" * "="^80)
println("EXAMPLE 6: SECOND ORDER GREEKS")
println("="^80)

println("\nAsian Call - Cross-derivatives:")

# Vanna (delta sensitivity to vol)
vanna = greek(asian_call, Vanna(), mc_engine, data)
println(@sprintf("  Vanna:  %8.6f  (how delta changes with vol)", vanna))

# Vomma (vega convexity)
vomma = greek(asian_call, Vomma(), mc_engine, data)
println(@sprintf("  Vomma:  %8.6f  (vega sensitivity to vol)", vomma))

println("\nInterpretation:")
println("  • Vanna > 0: Delta increases as volatility rises")
println("  • Vomma > 0: Vega increases as volatility rises (convex vega)")

# ============================================================================
# Example 7: Greeks for Different Exotic Types
# ============================================================================

println("\n" * "="^80)
println("EXAMPLE 7: EXOTIC OPTIONS COMPARISON TABLE")
println("="^80)

# Create various exotic options
exotics = [
    ("Asian Call", ArithmeticAsianCall(100.0, 1.0, averaging_times)),
    ("Barrier Call", KnockOutCall(100.0, 1.0, 120.0, :up_and_out)),
    ("Lookback Call", FixedStrikeLookbackCall(100.0, 1.0)),
    ("Vanilla Call", EuropeanCall(100.0, 1.0))
]

println("\nAll options: Strike \$100, Expiry 1 year")
println()
println(@sprintf("%-15s %10s %10s %10s %10s", "Option Type", "Price", "Delta", "Gamma", "Vega"))
println("="^65)

for (name, option) in exotics
    if option isa EuropeanOption
        opt_price = price(option, BlackScholes(), data)
        opt_greeks = all_greeks(option, data)
    else
        opt_price = price(option, mc_engine, data)
        opt_greeks = all_greeks(option, mc_engine, data)
    end

    println(@sprintf("%-15s %10.4f %10.4f %10.6f %10.4f",
        name,
        opt_price,
        opt_greeks[Delta()],
        opt_greeks[Gamma()],
        opt_greeks[Vega()]))
end

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^80)
println("KEY TAKEAWAYS")
println("="^80)

println("""
1. ASIAN OPTIONS:
   • Lower volatility sensitivity than vanilla (averaging effect)
   • Delta similar to vanilla, but smoother behavior
   • Cheaper than vanilla due to reduced optionality

2. BARRIER OPTIONS:
   • Delta suppressed by knock-out risk
   • Vega can be negative (higher vol = more knock-out risk)
   • Greeks highly sensitive to barrier proximity
   • Use barrier_proximity() to monitor risk

3. LOOKBACK OPTIONS:
   • Higher delta and vega than vanilla
   • Extreme sensitivity to volatility
   • Most expensive exotic options (maximum optionality)

4. NUMERICAL GREEKS:
   • Use larger MC step sizes (1% of spot) for stability
   • Increase simulation paths (50,000+) for accuracy
   • Greeks subject to Monte Carlo noise
   • Vega and second-order Greeks are noisiest

5. RISK MANAGEMENT:
   • Delta-hedge with underlying
   • Gamma/Vega-hedge requires other options
   • Monitor barrier proximity for barrier options
   • Asian options good for reducing vega exposure

6. COMPUTATIONAL NOTES:
   • Exotic Greeks require Monte Carlo pricing
   • Each Greek calculation = multiple MC simulations
   • Batch calculations more efficient
   • Trade-off: accuracy vs computation time
""")

println("="^80)
println("End of Examples")
println("="^80)
