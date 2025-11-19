using Plots
using Prezo
using Statistics

call = EuropeanCall(40.0, 1.0)
put = EuropeanPut(40.0, 1.0)

# Market data: spot=41.0, rate=0.08, vol=0.3, div=0.0
data = MarketData(41.0, 0.08, 0.3, 0.0)

M = 100_000
mc_prc = zeros(M)
mca_prc = zeros(M)
mcs_prc = zeros(M)

println("Running Monte Carlo study with $M iterations...")
println("This may take a few moments...")

for j in 1:M
    mc_prc[j] = price(call, MonteCarlo(1, 500), data)
    mca_prc[j] = price(call, MonteCarloAntithetic(1, 500), data)
    mcs_prc[j] = price(call, MonteCarloStratified(1, 500), data)

    # Progress indicator every 10,000 iterations
    if j % 10_000 == 0
        println("  Completed $j / $M iterations")
    end
end

println("\nMonte Carlo Study Results:")
println("─"^50)
println("Naive Monte Carlo:")
println("  Mean:     ", round(mean(mc_prc), digits=4))
println("  Std Dev:  ", round(std(mc_prc), digits=4))
println("  Variance: ", round(var(mc_prc), digits=6))

println("\nAntithetic Variates:")
println("  Mean:     ", round(mean(mca_prc), digits=4))
println("  Std Dev:  ", round(std(mca_prc), digits=4))
println("  Variance: ", round(var(mca_prc), digits=6))
println("  Variance Reduction: ", round((1 - var(mca_prc)/var(mc_prc)) * 100, digits=2), "%")

println("\nStratified Sampling:")
println("  Mean:     ", round(mean(mcs_prc), digits=4))
println("  Std Dev:  ", round(std(mcs_prc), digits=4))
println("  Variance: ", round(var(mcs_prc), digits=6))
println("  Variance Reduction: ", round((1 - var(mcs_prc)/var(mc_prc)) * 100, digits=2), "%")
println("─"^50)

# Create comparison plots
p1 = histogram(mc_prc,
    bins=50,
    alpha=0.5,
    label="Naive MC",
    xlabel="Option Price",
    ylabel="Frequency",
    title="Monte Carlo Methods Comparison (Overlaid)",
    legend=:topright,
    color=:blue)
histogram!(p1, mca_prc,
    bins=50,
    alpha=0.5,
    label="Antithetic",
    color=:red)
histogram!(p1, mcs_prc,
    bins=50,
    alpha=0.5,
    label="Stratified",
    color=:green)

# Side-by-side histograms
p2 = histogram(mc_prc,
    bins=50,
    alpha=0.7,
    label="",
    xlabel="Option Price",
    ylabel="Frequency",
    title="Naive Monte Carlo",
    color=:blue)

p3 = histogram(mca_prc,
    bins=50,
    alpha=0.7,
    label="",
    xlabel="Option Price",
    ylabel="Frequency",
    title="Antithetic Variates",
    color=:red)

p4 = histogram(mcs_prc,
    bins=50,
    alpha=0.7,
    label="",
    xlabel="Option Price",
    ylabel="Frequency",
    title="Stratified Sampling",
    color=:green)

# Combined plot
combined = plot(p1, plot(p2, p3, p4, layout=(1,3)),
    layout=(2,1),
    size=(1200, 800))

display(combined)

# Save the plot
savefig(combined, "mc_comparison.png")
println("\nPlot saved as 'mc_comparison.png'")
