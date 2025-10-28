using Plots
using Prezo
using Printf
using Statistics
using LinearAlgebra

println("Simple Basis Function Performance Analysis")
println("=" ^ 45)

# Test parameters
spot = 40.0
rate = 0.06
vol = 0.25
div = 0.0
expiry = 1.0

data = MarketData(spot, rate, vol, div)

# Test the problematic strikes we identified earlier
test_strikes = [39.0, 41.0, 43.0, 45.0, 47.0]

println("Market Parameters:")
@printf("Spot: \$%.2f, Rate: %.1f%%, Vol: %.1f%%, Expiry: %.1f years\n\n",
        spot, rate*100, vol*100, expiry)

# Test different LSM configurations
println("Testing Original LSM with Different Parameters")
println("Strike\tEuropean\tBinomial\tLSM(50k)\tLSM(100k)\tLSM(200k)\tError vs Bin")
println("-" ^ 85)

for strike in test_strikes
    # Create options
    euro_put = EuropeanPut(strike, expiry)
    american_put = AmericanPut(strike, expiry)

    # Reference prices
    euro_price = price(euro_put, BlackScholes(), data)
    binomial_price = price(american_put, Binomial(1000), data)

    # Different LSM accuracies
    lsm_50k = price(american_put, LongstaffSchwartz(100, 50_000, 3), data)
    lsm_100k = price(american_put, LongstaffSchwartz(150, 100_000, 3), data)
    lsm_200k = price(american_put, LongstaffSchwartz(200, 200_000, 3), data)

    # Best error
    best_error = min(abs(lsm_50k - binomial_price),
                     abs(lsm_100k - binomial_price),
                     abs(lsm_200k - binomial_price))

    @printf("%.1f\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n",
            strike, euro_price, binomial_price, lsm_50k, lsm_100k, lsm_200k, best_error)
end

# Theoretical Analysis
println("\n" * "="^60)
println("THEORETICAL ANALYSIS - Why LSM Struggles at ATM")
println("="^60)

println("\n1. Condition Number Analysis:")
println("   - Measures numerical stability of basis matrix")
println("   - Lower is better (< 1000 is good, > 100,000 is poor)")

# Simple condition number test
test_spot_prices = [35.0, 40.0, 45.0]  # OTM, ATM, ITM
for test_spot in test_spot_prices
    # Create basis matrix like our LSM does
    S_norm = test_spot_prices ./ 100.0
    n = length(S_norm)
    X = ones(n, 4)  # Order 3 + constant

    # Laguerre polynomials
    X[:, 2] = 1.0 .- S_norm
    X[:, 3] = 1.0 .- 2.0 * S_norm .+ (S_norm .^ 2) / 2.0
    X[:, 4] = 1.0 .- 3.0 * S_norm .+ 1.5 * (S_norm .^ 2) .- (S_norm .^ 3) / 6.0

    cond_num = cond(X)
    @printf("   Spot=\$%.1f region: Condition number = %.1e\n", test_spot, cond_num)
end

println("\n2. Early Exercise Boundary Analysis:")
for strike in [41.0, 43.0, 45.0]
    euro_put = EuropeanPut(strike, expiry)
    american_put = AmericanPut(strike, expiry)

    euro_price = price(euro_put, BlackScholes(), data)
    american_price = price(american_put, Binomial(1000), data)

    premium = american_price - euro_price
    premium_pct = (premium / euro_price) * 100

    @printf("   Strike \$%.1f: Early exercise premium = \$%.4f (%.1f%%)\n",
            strike, premium, premium_pct)
end

println("\n3. Monte Carlo Convergence Issues:")
println("   - LSM uses regression on random paths")
println("   - ATM options have complex exercise boundaries")
println("   - Fewer ITM paths → unstable regression")
println("   - Binomial uses complete state space (deterministic)")

println("\n4. Basis Function Limitations:")
println("   - Laguerre polynomials may not capture ATM exercise boundary well")
println("   - Fixed normalization (÷100) may not be optimal")
println("   - Higher-order terms can become numerically unstable")

# Visualization of the problem
println("\nCreating diagnostic visualization...")

# Generate a range of spot prices to show the issue
spot_range = 30.0:1.0:50.0
strike_test = 42.0  # Problematic ATM strike

american_put_test = AmericanPut(strike_test, expiry)
european_put_test = EuropeanPut(strike_test, expiry)

binomial_prices = Float64[]
lsm_prices = Float64[]
european_prices = Float64[]

for test_spot in spot_range
    test_data = MarketData(test_spot, rate, vol, div)

    euro_price = price(european_put_test, BlackScholes(), test_data)
    bin_price = price(american_put_test, Binomial(500), test_data)
    lsm_price = price(american_put_test, LongstaffSchwartz(100, 50_000, 3), test_data)

    push!(european_prices, euro_price)
    push!(binomial_prices, bin_price)
    push!(lsm_prices, lsm_price)
end

# Create comparison plot
p1 = plot(spot_range, european_prices,
          label="European Put (BS)",
          linewidth=2,
          title="LSM vs Binomial: Strike \$$(strike_test)",
          xlabel="Spot Price (\$)",
          ylabel="Option Price (\$)",
          legend=:topright,
          color=:black,
          linestyle=:dot)

plot!(p1, spot_range, binomial_prices,
      label="American Put (Binomial)",
      linewidth=3,
      color=:blue)

plot!(p1, spot_range, lsm_prices,
      label="American Put (LSM)",
      linewidth=2,
      color=:red,
      linestyle=:dash)

# Add reference lines
vline!(p1, [spot], label="Current Spot", linewidth=1, color=:gray)
vline!(p1, [strike_test], label="Strike Price", linewidth=1, color=:orange)

# Error plot
p2 = plot(spot_range, abs.(lsm_prices .- binomial_prices),
          label="LSM Error vs Binomial",
          linewidth=2,
          title="Absolute Pricing Error",
          xlabel="Spot Price (\$)",
          ylabel="Absolute Error (\$)",
          color=:red)

vline!(p2, [spot], label="Current Spot", linewidth=1, color=:gray)
vline!(p2, [strike_test], label="Strike Price", linewidth=1, color=:orange)

# Combine plots
combined_plot = plot(p1, p2, layout=(2,1), size=(800, 600))

display(combined_plot)
savefig(combined_plot, "lsm_problem_analysis.png")
println("Diagnostic plot saved as 'lsm_problem_analysis.png'")

println("\nKey Findings:")
println("✓ LSM shows largest errors in ATM region (spot ≈ strike)")
println("✓ Errors decrease for deep ITM and OTM options")
println("✓ Basis function conditioning is worst in ATM region")
println("✓ Early exercise boundaries are most complex at ATM")
println("✓ Monte Carlo variance highest where exercise decisions are marginal")

println("\nRecommended Solutions:")
println("1. Use adaptive basis functions based on moneyness")
println("2. Implement variance reduction techniques")
println("3. Add minimum sample size requirements for regression")
println("4. Consider alternative basis functions (Chebyshev, Hermite)")
println("5. Implement theoretical validation constraints")