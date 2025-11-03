using Plots
using Prezo
using Printf
using Statistics

# Create option contracts
put = EuropeanPut(40.0, 1.0)   # Strike = 40, Expiry = 1 year
call = EuropeanCall(40.0, 1.0) # Strike = 40, Expiry = 1 year
american_put = AmericanPut(40.0, 1.0) # American put for LSM comparison

# Fixed market parameters
rate = 0.08    # 8% risk-free rate
vol = 0.3      # 30% volatility
div = 0.0      # No dividends

# Create range of spot prices from 20 to 60
spot_prices = 1.0:2.0:100.0

# Initialize arrays to store results
bs_put_prices = Float64[]
bin_put_prices = Float64[]
mc_put_prices = Float64[]

bs_call_prices = Float64[]
bin_call_prices = Float64[]
mc_call_prices = Float64[]

# Arrays for American put comparison
bin_american_put_prices = Float64[]  # Binomial American put
lsm_american_put_prices = Float64[]  # Longstaff-Schwartz American put

println("Calculating option prices across spot price range...")
println("Spot\tBS Put\tBin Put\tMC Put\tBS Call\tBin Call\tMC Call\tAm Put(Bin)\tAm Put(LSM)")
println("-" ^ 85)

# Calculate prices for each spot price
for spot in spot_prices
    # Create market data for this spot price
    data = MarketData(spot, rate, vol, div)

    # Calculate put prices with all three engines
    bs_put = price(put, BlackScholes(), data)
    bin_put = price(put, Binomial(100), data)
    mc_put = price(put, MonteCarlo(1, 100_000), data)

    # Calculate call prices with all three engines
    bs_call = price(call, BlackScholes(), data)
    bin_call = price(call, Binomial(100), data)
    mc_call = price(call, MonteCarlo(1, 100_000), data)

    # Calculate American put prices
    bin_american_put = price(american_put, Binomial(1_000), data)
    lsm_american_put = price(american_put, LongstaffSchwartz(50, 50_000), data)

    # Store results
    push!(bs_put_prices, bs_put)
    push!(bin_put_prices, bin_put)
    push!(mc_put_prices, mc_put)

    push!(bs_call_prices, bs_call)
    push!(bin_call_prices, bin_call)
    push!(mc_call_prices, mc_call)

    push!(bin_american_put_prices, bin_american_put)
    push!(lsm_american_put_prices, lsm_american_put)

    # Print results
    @printf("%.1f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t\t%.3f\n",
            spot, bs_put, bin_put, mc_put, bs_call, bin_call, mc_call, bin_american_put, lsm_american_put)
end

println("\nCreating visualizations...")

# Create put option price comparison plot
p1 = plot(spot_prices, bs_put_prices,
          label="Black-Scholes",
          linewidth=2,
          title="European Put Option Prices vs Spot Price",
          xlabel="Spot Price (\$)",
          ylabel="Option Price (\$)",
          legend=:topright)

plot!(p1, spot_prices, bin_put_prices,
      label="Binomial (100 steps)",
      linewidth=2,
      linestyle=:dash)

plot!(p1, spot_prices, mc_put_prices,
      label="Monte Carlo (100k paths)",
      linewidth=2,
      linestyle=:dot,
      markersize=3,
      markershape=:circle,
      markerstrokewidth=0)

# Add strike price reference line
vline!(p1, [40.0], label="Strike Price", linewidth=1, linestyle=:dashdot, color=:gray)

# Create call option price comparison plot
p2 = plot(spot_prices, bs_call_prices,
          label="Black-Scholes",
          linewidth=2,
          title="European Call Option Prices vs Spot Price",
          xlabel="Spot Price (\$)",
          ylabel="Option Price (\$)",
          legend=:topleft)

plot!(p2, spot_prices, bin_call_prices,
      label="Binomial (100 steps)",
      linewidth=2,
      linestyle=:dash)

plot!(p2, spot_prices, mc_call_prices,
      label="Monte Carlo (100k paths)",
      linewidth=2,
      linestyle=:dot,
      markersize=3,
      markershape=:circle,
      markerstrokewidth=0)

# Add strike price reference line
vline!(p2, [40.0], label="Strike Price", linewidth=1, linestyle=:dashdot, color=:gray)

# Create American put option comparison plot
p3 = plot(spot_prices, bin_american_put_prices,
          label="American Put (Binomial)",
          linewidth=3,
          title="American Put: Binomial vs Longstaff-Schwartz",
          xlabel="Spot Price (\$)",
          ylabel="Option Price (\$)",
          legend=:topright,
          color=:blue)

plot!(p3, spot_prices, lsm_american_put_prices,
      label="American Put (LSM)",
      linewidth=2,
      linestyle=:dash,
      color=:red)

# Add European put for reference
plot!(p3, spot_prices, bs_put_prices,
      label="European Put (BS)",
      linewidth=1,
      linestyle=:dot,
      color=:gray)

# Add strike price reference line
vline!(p3, [40.0], label="Strike Price", linewidth=1, linestyle=:dashdot, color=:black)

# Create combined plot with three panels
combined_plot = plot(p1, p2, p3, layout=(3,1), size=(900, 900))

# Display the plot
display(combined_plot)

# Save the plot
savefig(combined_plot, "option_prices_spot_sensitivity.png")
println("Plot saved as 'option_prices_spot_sensitivity.png'")

# Calculate and display pricing differences
println("\nPricing Engine Comparison Summary:")
println("="^50)

put_bs_bin_diff = maximum(abs.(bs_put_prices .- bin_put_prices))
put_bs_mc_diff = maximum(abs.(bs_put_prices .- mc_put_prices))
call_bs_bin_diff = maximum(abs.(bs_call_prices .- bin_call_prices))
call_bs_mc_diff = maximum(abs.(bs_call_prices .- mc_call_prices))

# American put comparison statistics
american_put_bin_lsm_diff = maximum(abs.(bin_american_put_prices .- lsm_american_put_prices))
american_put_early_exercise_premium = mean(bin_american_put_prices .- bs_put_prices)

println("Maximum absolute differences from Black-Scholes:")
@printf("Put Options  - Binomial: %.4f, Monte Carlo: %.4f\n", put_bs_bin_diff, put_bs_mc_diff)
@printf("Call Options - Binomial: %.4f, Monte Carlo: %.4f\n", call_bs_bin_diff, call_bs_mc_diff)

println("\nAmerican Put Option Analysis:")
@printf("Max difference (Binomial vs LSM): %.4f\n", american_put_bin_lsm_diff)
@printf("Average early exercise premium: %.4f\n", american_put_early_exercise_premium)
@printf("Early exercise premium range: %.4f to %.4f\n",
        minimum(bin_american_put_prices .- bs_put_prices),
        maximum(bin_american_put_prices .- bs_put_prices))
