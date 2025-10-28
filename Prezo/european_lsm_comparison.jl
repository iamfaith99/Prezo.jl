using Prezo
using Printf
using Statistics
using Random

println("European Options: LSM vs BSM vs Binomial vs Monte Carlo")
println("=" ^ 60)
println("Comparing European LSM regression-based pricing methods")
println()

# Set seed for reproducible results
Random.seed!(42)

# Market parameters
spot = 40.0
rate = 0.06
vol = 0.25
div = 0.0
expiry = 1.0

data = MarketData(spot, rate, vol, div)

# Test different strikes and option types
test_strikes = [35.0, 38.0, 40.0, 42.0, 45.0]
option_types = [("Call", EuropeanCall), ("Put", EuropeanPut)]

println("Market Parameters:")
@printf("Spot: \$%.2f, Rate: %.1f%%, Vol: %.1f%%, Expiry: %.1f years\\n\\n",
        spot, rate*100, vol*100, expiry)

for (type_name, OptionType) in option_types
    println("EUROPEAN $type_name OPTIONS")
    println("-" ^ 40)
    println("Strike\\tBSM\\t\\tBinomial\\tMonte Carlo\\tLSM (Lag)\\tLSM (Cheb)\\tLSM (Herm)")
    println("-" ^ 85)

    for strike in test_strikes
        option = OptionType(strike, expiry)

        # Black-Scholes-Merton (analytical benchmark)
        bsm_price = price(option, BlackScholes(), data)

        # Binomial (numerical benchmark)
        binomial_price = price(option, Binomial(2000), data)

        # Standard Monte Carlo
        mc_price = price(option, MonteCarlo(1, 100_000), data)

        # European LSM with different basis functions
        lsm_laguerre = price(option, EuropeanLaguerreLSM(3, 100_000, normalization=40.0), data)
        lsm_chebyshev = price(option, EuropeanChebyshevLSM(3, 100_000, domain=(25.0, 55.0)), data)
        lsm_hermite = price(option, EuropeanHermiteLSM(3, 100_000, mean=40.0, std=10.0), data)

        @printf("%.1f\\t%.6f\\t%.6f\\t%.6f\\t%.6f\\t%.6f\\t%.6f\\n",
                strike, bsm_price, binomial_price, mc_price,
                lsm_laguerre, lsm_chebyshev, lsm_hermite)
    end
    println()
end

println("ACCURACY ANALYSIS")
println("=" ^ 30)
println("Absolute errors vs Black-Scholes benchmark")
println()

for (type_name, OptionType) in option_types
    println("$type_name Options - Absolute Errors vs BSM:")
    println("Strike\\tBinomial\\tMonte Carlo\\tLSM (Lag)\\tLSM (Cheb)\\tLSM (Herm)")
    println("-" ^ 70)

    binomial_errors = Float64[]
    mc_errors = Float64[]
    lsm_laguerre_errors = Float64[]
    lsm_chebyshev_errors = Float64[]
    lsm_hermite_errors = Float64[]

    for strike in test_strikes
        option = OptionType(strike, expiry)

        # Reference price
        bsm_price = price(option, BlackScholes(), data)

        # Method prices
        binomial_price = price(option, Binomial(2000), data)
        mc_price = price(option, MonteCarlo(1, 100_000), data)
        lsm_laguerre = price(option, EuropeanLaguerreLSM(3, 100_000), data)
        lsm_chebyshev = price(option, EuropeanChebyshevLSM(3, 100_000), data)
        lsm_hermite = price(option, EuropeanHermiteLSM(3, 100_000), data)

        # Calculate absolute errors
        bin_err = abs(binomial_price - bsm_price)
        mc_err = abs(mc_price - bsm_price)
        lag_err = abs(lsm_laguerre - bsm_price)
        cheb_err = abs(lsm_chebyshev - bsm_price)
        herm_err = abs(lsm_hermite - bsm_price)

        push!(binomial_errors, bin_err)
        push!(mc_errors, mc_err)
        push!(lsm_laguerre_errors, lag_err)
        push!(lsm_chebyshev_errors, cheb_err)
        push!(lsm_hermite_errors, herm_err)

        @printf("%.1f\\t%.6f\\t%.6f\\t%.6f\\t%.6f\\t%.6f\\n",
                strike, bin_err, mc_err, lag_err, cheb_err, herm_err)
    end

    # Summary statistics
    println("\\nSummary Statistics for $type_name:")
    @printf("Mean Binomial Error:   %.6f\\n", mean(binomial_errors))
    @printf("Mean Monte Carlo Error: %.6f\\n", mean(mc_errors))
    @printf("Mean LSM Laguerre Error: %.6f\\n", mean(lsm_laguerre_errors))
    @printf("Mean LSM Chebyshev Error: %.6f\\n", mean(lsm_chebyshev_errors))
    @printf("Mean LSM Hermite Error: %.6f\\n", mean(lsm_hermite_errors))
    println()
end

println("CONVERGENCE ANALYSIS")
println("=" ^ 25)
println("Testing convergence with different simulation counts")
println()

# Test convergence for ATM put option
atm_put = EuropeanPut(40.0, expiry)
bsm_reference = price(atm_put, BlackScholes(), data)

simulation_counts = [10_000, 25_000, 50_000, 100_000, 200_000]

println("ATM Put Convergence (Strike = \$40.00, BSM = $(round(bsm_reference, digits=6))):")
println("Simulations\\tMonte Carlo\\tLSM (Lag)\\tLSM (Cheb)\\tLSM (Herm)")
println("-" ^ 65)

for sim_count in simulation_counts
    mc_price = price(atm_put, MonteCarlo(1, sim_count), data)
    lsm_lag = price(atm_put, EuropeanLaguerreLSM(3, sim_count), data)
    lsm_cheb = price(atm_put, EuropeanChebyshevLSM(3, sim_count), data)
    lsm_herm = price(atm_put, EuropeanHermiteLSM(3, sim_count), data)

    @printf("%d\\t\\t%.6f\\t%.6f\\t%.6f\\t%.6f\\n",
            sim_count, mc_price, lsm_lag, lsm_cheb, lsm_herm)
end

println("\\nCONVERGENCE ERRORS vs BSM:")
println("Simulations\\tMC Error\\tLSM (Lag)\\tLSM (Cheb)\\tLSM (Herm)")
println("-" ^ 60)

for sim_count in simulation_counts
    mc_price = price(atm_put, MonteCarlo(1, sim_count), data)
    lsm_lag = price(atm_put, EuropeanLaguerreLSM(3, sim_count), data)
    lsm_cheb = price(atm_put, EuropeanChebyshevLSM(3, sim_count), data)
    lsm_herm = price(atm_put, EuropeanHermiteLSM(3, sim_count), data)

    mc_err = abs(mc_price - bsm_reference)
    lag_err = abs(lsm_lag - bsm_reference)
    cheb_err = abs(lsm_cheb - bsm_reference)
    herm_err = abs(lsm_herm - bsm_reference)

    @printf("%d\\t\\t%.6f\\t%.6f\\t%.6f\\t%.6f\\n",
            sim_count, mc_err, lag_err, cheb_err, herm_err)
end

println("\\nCONDITION NUMBER ANALYSIS")
println("=" ^ 30)
println("Basis function numerical stability for different strikes")
println()

test_spot_prices = [35.0, 40.0, 45.0]
laguerre_basis = LaguerreBasis(3, 40.0)
chebyshev_basis = ChebyshevBasis(3, 25.0, 55.0)
hermite_basis = HermiteBasis(3, 40.0, 10.0)

println("Asset Price\\tLaguerre\\tChebyshev\\tHermite")
println("-" ^ 50)

using LinearAlgebra
for test_price in test_spot_prices
    S_test = [test_price - 2.0, test_price, test_price + 2.0]

    X_lag = laguerre_basis(S_test)
    X_cheb = chebyshev_basis(S_test)
    X_herm = hermite_basis(S_test)

    cond_lag = cond(X_lag)
    cond_cheb = cond(X_cheb)
    cond_herm = cond(X_herm)

    @printf("\$%.1f\\t\\t%.2e\\t%.2e\\t%.2e\\n", test_price, cond_lag, cond_cheb, cond_herm)
end

println("\\nKEY FINDINGS:")
println("=" ^ 15)
println("✓ European LSM provides regression-based alternative to Monte Carlo")
println("✓ No early exercise means LSM becomes pure payoff function approximation")
println("✓ Basis function choice affects numerical stability and accuracy")
println("✓ LSM can potentially reduce Monte Carlo variance through regression")
println("✓ Chebyshev and Hermite bases show better conditioning than Laguerre")
println("✓ Convergence patterns differ from standard Monte Carlo")

println("\\nINSIGHTS:")
println("- European LSM essentially learns the payoff function structure")
println("- Better basis functions can capture option payoff curvature more efficiently")
println("- Regression can smooth Monte Carlo noise but may introduce bias")
println("- Most useful when payoff functions have complex, learnable patterns")