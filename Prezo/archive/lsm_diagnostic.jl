using Plots
using Prezo
using Printf
using Statistics
using LinearAlgebra

println("LSM Diagnostic Analysis: Understanding Performance Issues")
println("=" ^ 60)

# Focus on problematic cases
spot = 40.0
rate = 0.06
vol = 0.25
div = 0.0
expiry = 1.0

data = MarketData(spot, rate, vol, div)

# Test specific problematic strikes
test_strikes = [41.0, 43.0, 45.0, 47.0]  # ATM region where differences are large

println("Investigating ATM Put Options and Deep ITM Call Options")
println("\nTesting different LSM configurations:")

# Test different LSM configurations
lsm_configs = [
    (50, 10000, 2, "Low accuracy, 2nd order"),
    (100, 50000, 3, "Medium accuracy, 3rd order"),
    (100, 50000, 2, "Medium accuracy, 2nd order"),
    (200, 100000, 3, "High accuracy, 3rd order"),
    (100, 50000, 4, "Medium accuracy, 4th order")
]

bin_engine_reference = Binomial(1000)  # High accuracy reference

for (strike_idx, strike) in enumerate(test_strikes)
    println("\n" * "="^50)
    println("STRIKE: \$$(strike) (Moneyness: $(round(strike/spot, digits=3)))")
    println("="^50)

    # Create options
    american_put = AmericanPut(strike, expiry)
    american_call = AmericanCall(strike, expiry)
    european_put = EuropeanPut(strike, expiry)
    european_call = EuropeanCall(strike, expiry)

    # Reference prices
    euro_put_bs = price(european_put, BlackScholes(), data)
    euro_call_bs = price(european_call, BlackScholes(), data)
    american_put_bin = price(american_put, bin_engine_reference, data)
    american_call_bin = price(american_call, bin_engine_reference, data)

    println("Reference Prices:")
    @printf("European Put (BS):    \$%.6f\n", euro_put_bs)
    @printf("European Call (BS):   \$%.6f\n", euro_call_bs)
    @printf("American Put (Bin):   \$%.6f\n", american_put_bin)
    @printf("American Call (Bin):  \$%.6f\n", american_call_bin)
    @printf("Put Early Premium:    \$%.6f\n", american_put_bin - euro_put_bs)
    @printf("Call Early Premium:   \$%.6f (should be ~0)\n", american_call_bin - euro_call_bs)

    println("\nLSM Configuration Testing:")
    println("Steps\tPaths\tOrder\tPut Price\tPut Error\tCall Price\tCall Error\tDescription")
    println("-" ^ 95)

    for (steps, paths, order, desc) in lsm_configs
        lsm_engine = LongstaffSchwartz(steps, paths, order)

        american_put_lsm = price(american_put, lsm_engine, data)
        american_call_lsm = price(american_call, lsm_engine, data)

        put_error = abs(american_put_lsm - american_put_bin)
        call_error = abs(american_call_lsm - american_call_bin)

        @printf("%d\t%d\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%s\n",
                steps, paths, order, american_put_lsm, put_error,
                american_call_lsm, call_error, desc)
    end
end

# Deep dive into LSM mechanics for a specific problematic case
println("\n" * "="^60)
println("DEEP DIVE: LSM Mechanics Analysis")
println("="^60)

# Focus on strike 43 (problematic case)
problematic_strike = 43.0
american_put_problem = AmericanPut(problematic_strike, expiry)
american_call_problem = AmericanCall(problematic_strike, expiry)

# Custom LSM analysis with intermediate results
println("Analyzing LSM internals for Strike \$$(problematic_strike)...")

# We'll create a simplified version of the LSM algorithm to see what's happening
function analyze_lsm_internals(option, data, steps, reps)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data

    dt = expiry / steps
    disc = exp(-rate * dt)

    # Generate paths
    paths = asset_paths_col(MonteCarlo(steps, reps), spot, rate - div, vol, expiry)

    # Calculate payoffs at each time step
    println("\nPath Analysis:")
    @printf("Generated %d paths with %d time steps\n", reps, steps+1)
    @printf("Final asset prices: Min=%.2f, Max=%.2f, Mean=%.2f\n",
            minimum(paths[end,:]), maximum(paths[end,:]), mean(paths[end,:]))

    # Analyze early exercise decisions at a few time points
    exercise_analysis_times = [Int(round(steps*0.25)), Int(round(steps*0.5)), Int(round(steps*0.75)), steps]

    for t_idx in exercise_analysis_times
        t = exercise_analysis_times[end] - t_idx + 1  # Reverse for backward induction
        if t <= 1
            continue
        end

        S_t = paths[t, :]
        payoffs = payoff.(option, S_t)
        itm_paths = sum(payoffs .> 0)

        @printf("Time step %d (%.1f%% to expiry): %d/%d paths ITM, Avg payoff: %.4f\n",
                t, (t-1)/steps*100, itm_paths, reps, mean(payoffs[payoffs .> 0]))
    end

    return paths
end

# Analyze both puts and calls
println("\n--- American Put Analysis ---")
put_paths = analyze_lsm_internals(american_put_problem, data, 50, 10000)

println("\n--- American Call Analysis ---")
call_paths = analyze_lsm_internals(american_call_problem, data, 50, 10000)

# Check if call should never be exercised early (theory check)
european_call_problem = EuropeanCall(problematic_strike, expiry)
euro_call_price = price(european_call_problem, BlackScholes(), data)
theoretical_call_premium = max(0, spot - problematic_strike * exp(-rate * expiry))

println("\nTheoretical Analysis for Calls (Strike \$$(problematic_strike)):")
@printf("Current spot: \$%.2f\n", spot)
@printf("Discounted strike: \$%.2f\n", problematic_strike * exp(-rate * expiry))
@printf("Immediate exercise value: \$%.6f\n", max(0, spot - problematic_strike))
@printf("European call value: \$%.6f\n", euro_call_price)
@printf("Should calls ever be exercised early? %s\n",
        spot > problematic_strike * exp(-rate * expiry) ? "Possibly" : "NO - Never optimal!")

println("\nDiagnostic Summary:")
println("✓ Check basis function performance across different moneyness levels")
println("✓ Examine regression quality for ITM vs OTM options")
println("✓ Investigate simulation variance in LSM vs deterministic Binomial")
println("✓ Consider alternative basis functions for better stability")
println("✓ Calls showing early exercise premium indicates LSM calibration issue")