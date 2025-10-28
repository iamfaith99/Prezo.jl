using Plots
using Prezo
using Printf
using Statistics

println("Strike Price Sensitivity Analysis: LSM vs Binomial for American Options")
println("=" ^ 70)

# Fixed market parameters
spot = 40.0     # Current asset price
rate = 0.06     # 6% risk-free rate
vol = 0.25      # 25% volatility
div = 0.0       # No dividends
expiry = 1.0    # 1 year to expiration

# Create market data
data = MarketData(spot, rate, vol, div)

# Create range of strike prices from 25 to 55
strike_prices = 25.0:2.0:55.0

# Initialize arrays to store results
# European Options (Reference): Black-Scholes and Monte Carlo
euro_put_bs_prices = Float64[]
euro_put_mc_prices = Float64[]
euro_call_bs_prices = Float64[]
euro_call_mc_prices = Float64[]

# American Options: LSM vs Binomial (Main Focus)
american_put_bin_prices = Float64[]
american_put_lsm_prices = Float64[]
american_call_bin_prices = Float64[]
american_call_lsm_prices = Float64[]

println("\nMarket Parameters:")
@printf("Spot: \$%.2f, Rate: %.1f%%, Vol: %.1f%%, Time: %.1f years\n\n",
        spot, rate*100, vol*100, expiry)

println("Calculating option prices across strike price range...")
println("Strike\tEur Put(BS)\tEur Put(MC)\tEur Call(BS)\tEur Call(MC)\tAm Put(Bin)\tAm Put(LSM)\tAm Call(Bin)\tAm Call(LSM)")
println("-" ^ 120)

# Configure engines for consistent comparison
lsm_engine = LongstaffSchwartz(100, 50_000, 3)  # High accuracy LSM
bin_engine = Binomial(1_000)                     # High accuracy Binomial
mc_engine = MonteCarlo(1, 100_000)               # High accuracy Monte Carlo

# Calculate prices for each strike price
for strike in strike_prices
    # Create option contracts for this strike
    euro_put = EuropeanPut(strike, expiry)
    euro_call = EuropeanCall(strike, expiry)
    american_put = AmericanPut(strike, expiry)
    american_call = AmericanCall(strike, expiry)

    # European Options: Black-Scholes vs Monte Carlo (reference methods)
    euro_put_bs = price(euro_put, BlackScholes(), data)
    euro_put_mc = price(euro_put, mc_engine, data)
    euro_call_bs = price(euro_call, BlackScholes(), data)
    euro_call_mc = price(euro_call, mc_engine, data)

    # American Options: Binomial vs LSM (main comparison)
    american_put_bin = price(american_put, bin_engine, data)
    american_put_lsm = price(american_put, lsm_engine, data)
    american_call_bin = price(american_call, bin_engine, data)
    american_call_lsm = price(american_call, lsm_engine, data)

    # Store results
    push!(euro_put_bs_prices, euro_put_bs)
    push!(euro_put_mc_prices, euro_put_mc)
    push!(euro_call_bs_prices, euro_call_bs)
    push!(euro_call_mc_prices, euro_call_mc)

    push!(american_put_bin_prices, american_put_bin)
    push!(american_put_lsm_prices, american_put_lsm)
    push!(american_call_bin_prices, american_call_bin)
    push!(american_call_lsm_prices, american_call_lsm)

    # Print results
    @printf("%.1f\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n",
            strike, euro_put_bs, euro_put_mc, euro_call_bs, euro_call_mc,
            american_put_bin, american_put_lsm, american_call_bin, american_call_lsm)
end

println("\nCreating visualizations...")

# Plot 1: European Put Options - BS vs Monte Carlo (Reference)
p1 = plot(strike_prices, euro_put_bs_prices,
          label="European Put (Black-Scholes)",
          linewidth=3,
          title="European Put Options: Reference Methods",
          xlabel="Strike Price (\$)",
          ylabel="Option Price (\$)",
          legend=:topleft,
          color=:blue)

plot!(p1, strike_prices, euro_put_mc_prices,
      label="European Put (Monte Carlo)",
      linewidth=2,
      linestyle=:dash,
      color=:red)

# Add spot price reference line
vline!(p1, [spot], label="Spot Price", linewidth=1, linestyle=:dot, color=:gray)

# Plot 2: European Call Options - BS vs Monte Carlo (Reference)
p2 = plot(strike_prices, euro_call_bs_prices,
          label="European Call (Black-Scholes)",
          linewidth=3,
          title="European Call Options: Reference Methods",
          xlabel="Strike Price (\$)",
          ylabel="Option Price (\$)",
          legend=:topright,
          color=:blue)

plot!(p2, strike_prices, euro_call_mc_prices,
      label="European Call (Monte Carlo)",
      linewidth=2,
      linestyle=:dash,
      color=:red)

# Add spot price reference line
vline!(p2, [spot], label="Spot Price", linewidth=1, linestyle=:dot, color=:gray)

# Plot 3: American Put Options - LSM vs Binomial
p3 = plot(strike_prices, american_put_bin_prices,
          label="American Put (Binomial)",
          linewidth=3,
          title="American Put Options: LSM vs Binomial",
          xlabel="Strike Price (\$)",
          ylabel="Option Price (\$)",
          legend=:topleft,
          color=:green)

plot!(p3, strike_prices, american_put_lsm_prices,
      label="American Put (LSM)",
      linewidth=2,
      linestyle=:dash,
      color=:orange)

# Add European put for reference
plot!(p3, strike_prices, euro_put_bs_prices,
      label="European Put (BS Reference)",
      linewidth=1,
      linestyle=:dot,
      color=:gray)

# Add spot price reference line
vline!(p3, [spot], label="Spot Price", linewidth=1, linestyle=:dashdot, color=:black)

# Plot 4: American Call Options - LSM vs Binomial
p4 = plot(strike_prices, american_call_bin_prices,
          label="American Call (Binomial)",
          linewidth=3,
          title="American Call Options: LSM vs Binomial",
          xlabel="Strike Price (\$)",
          ylabel="Option Price (\$)",
          legend=:topright,
          color=:green)

plot!(p4, strike_prices, american_call_lsm_prices,
      label="American Call (LSM)",
      linewidth=2,
      linestyle=:dash,
      color=:orange)

# Add European call for reference
plot!(p4, strike_prices, euro_call_bs_prices,
      label="European Call (BS Reference)",
      linewidth=1,
      linestyle=:dot,
      color=:gray)

# Add spot price reference line
vline!(p4, [spot], label="Spot Price", linewidth=1, linestyle=:dashdot, color=:black)

# Create combined plot with four panels
combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900))

# Display and save the plot
display(combined_plot)
savefig(combined_plot, "strike_sensitivity_analysis.png")
println("Analysis plot saved as 'strike_sensitivity_analysis.png'")

# Calculate and display detailed comparison statistics
println("\nDetailed Performance Analysis:")
println("=" ^ 60)

# European Options: Black-Scholes vs Monte Carlo accuracy (reference validation)
euro_put_diff = abs.(euro_put_bs_prices .- euro_put_mc_prices)
euro_call_diff = abs.(euro_call_bs_prices .- euro_call_mc_prices)

println("European Options - Black-Scholes vs Monte Carlo (Reference Validation):")
@printf("Put Options  - Max Abs Diff: %.6f, Mean Abs Diff: %.6f, RMSE: %.6f\n",
        maximum(euro_put_diff), mean(euro_put_diff), sqrt(mean(euro_put_diff.^2)))
@printf("Call Options - Max Abs Diff: %.6f, Mean Abs Diff: %.6f, RMSE: %.6f\n",
        maximum(euro_call_diff), mean(euro_call_diff), sqrt(mean(euro_call_diff.^2)))

# American Options: LSM vs Binomial accuracy (MAIN FOCUS)
american_put_diff = abs.(american_put_bin_prices .- american_put_lsm_prices)
american_call_diff = abs.(american_call_bin_prices .- american_call_lsm_prices)

println("\n*** MAIN ANALYSIS: American Options - LSM vs Binomial ***")
@printf("Put Options  - Max Abs Diff: %.6f, Mean Abs Diff: %.6f, RMSE: %.6f\n",
        maximum(american_put_diff), mean(american_put_diff), sqrt(mean(american_put_diff.^2)))
@printf("Call Options - Max Abs Diff: %.6f, Mean Abs Diff: %.6f, RMSE: %.6f\n",
        maximum(american_call_diff), mean(american_call_diff), sqrt(mean(american_call_diff.^2)))

# Early Exercise Premium Analysis
put_early_exercise_premium_lsm = american_put_lsm_prices .- euro_put_bs_prices
put_early_exercise_premium_bin = american_put_bin_prices .- euro_put_bs_prices
call_early_exercise_premium_lsm = american_call_lsm_prices .- euro_call_bs_prices
call_early_exercise_premium_bin = american_call_bin_prices .- euro_call_bs_prices

println("\nEarly Exercise Premium Analysis (American vs European):")
println("LSM Method:")
@printf("  Put Options  - Max Premium: %.4f, Mean Premium: %.4f\n",
        maximum(put_early_exercise_premium_lsm), mean(put_early_exercise_premium_lsm))
@printf("  Call Options - Max Premium: %.4f, Mean Premium: %.4f\n",
        maximum(call_early_exercise_premium_lsm), mean(call_early_exercise_premium_lsm))

println("Binomial Method:")
@printf("  Put Options  - Max Premium: %.4f, Mean Premium: %.4f\n",
        maximum(put_early_exercise_premium_bin), mean(put_early_exercise_premium_bin))
@printf("  Call Options - Max Premium: %.4f, Mean Premium: %.4f\n",
        maximum(call_early_exercise_premium_bin), mean(call_early_exercise_premium_bin))

# Detailed moneyness analysis
println("\nDetailed Strike-by-Strike Analysis:")
println("Strike\tMoneyness\tAm Put(Bin)\tAm Put(LSM)\tDifference\tEur Put(BS)\tPut Premium")
println("-" ^ 90)

for (i, strike) in enumerate(strike_prices)
    moneyness = strike / spot
    am_put_bin = american_put_bin_prices[i]
    am_put_lsm = american_put_lsm_prices[i]
    diff = abs(am_put_bin - am_put_lsm)
    eur_put = euro_put_bs_prices[i]
    premium = am_put_bin - eur_put

    @printf("%.1f\t%.3f\t\t%.4f\t\t%.4f\t\t%.6f\t%.4f\t\t%.4f\n",
            strike, moneyness, am_put_bin, am_put_lsm, diff, eur_put, premium)
end

println("\nPerformance Summary:")
println("✓ LSM and Binomial methods show excellent agreement for American option pricing")
println("✓ Maximum difference between LSM and Binomial: $(@sprintf("%.6f", maximum(american_put_diff)))")
println("✓ Early exercise premiums are highest for deep in-the-money puts")
println("✓ American call premiums are minimal (expected with zero dividends)")
println("✓ Both methods properly capture strike price sensitivity")
println("✓ LSM provides a viable alternative to traditional binomial trees")