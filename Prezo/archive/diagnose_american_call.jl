using Prezo

# Test data with dividends
data_with_div = MarketData(100.0, 0.05, 0.20, 0.03)
data_no_div = MarketData(100.0, 0.05, 0.20, 0.0)

# Options
call_am = AmericanCall(100.0, 1.0)
call_eu = EuropeanCall(100.0, 1.0)

println("="^60)
println("AMERICAN vs EUROPEAN CALL DIAGNOSIS")
println("="^60)

println("\n1. NO DIVIDENDS (q=0.0)")
println("   Theory: American ≈ European (early exercise never optimal)")
println("-"^60)

# Binomial
binomial_engine = Binomial(200)
am_price_no_div_binom = price(call_am, binomial_engine, data_no_div)
eu_price_no_div_binom = price(call_eu, binomial_engine, data_no_div)
eu_price_no_div_bs = price(call_eu, BlackScholes(), data_no_div)

println("Binomial Engine (200 steps):")
println("  American Call: \$", round(am_price_no_div_binom, digits=4))
println("  European Call: \$", round(eu_price_no_div_binom, digits=4))
println("  Difference:     \$", round(am_price_no_div_binom - eu_price_no_div_binom, digits=4))
println("  American ≥ European? ", am_price_no_div_binom ≥ eu_price_no_div_binom ? "✓" : "✗ VIOLATION!")

println("\nBlack-Scholes (European):")
println("  European Call: \$", round(eu_price_no_div_bs, digits=4))

println("\n" * "="^60)
println("2. WITH DIVIDENDS (q=0.03)")
println("   Theory: American ≥ European (early exercise may be optimal)")
println("-"^60)

# Binomial with dividends
am_price_div_binom = price(call_am, binomial_engine, data_with_div)
eu_price_div_binom = price(call_eu, binomial_engine, data_with_div)
eu_price_div_bs = price(call_eu, BlackScholes(), data_with_div)

println("Binomial Engine (200 steps):")
println("  American Call: \$", round(am_price_div_binom, digits=4))
println("  European Call: \$", round(eu_price_div_binom, digits=4))
println("  Difference:     \$", round(am_price_div_binom - eu_price_div_binom, digits=4))
println("  American ≥ European? ", am_price_div_binom ≥ eu_price_div_binom ? "✓" : "✗ VIOLATION!")

println("\nBlack-Scholes (European):")
println("  European Call: \$", round(eu_price_div_bs, digits=4))
println("  Difference from American: \$", round(am_price_div_binom - eu_price_div_bs, digits=4))
println("  American ≥ European? ", am_price_div_binom ≥ eu_price_div_bs ? "✓" : "✗ VIOLATION!")

println("\n" * "="^60)
println("3. LONGSTAFF-SCHWARTZ WITH DIVIDENDS")
println("-"^60)

lsm_engine = LongstaffSchwartz(50, 10000)
am_price_lsm = price(call_am, lsm_engine, data_with_div)

println("LSM Engine (50 steps, 10000 paths):")
println("  American Call: \$", round(am_price_lsm, digits=4))
println("  European (BS):  \$", round(eu_price_div_bs, digits=4))
println("  Difference:     \$", round(am_price_lsm - eu_price_div_bs, digits=4))
println("  American ≥ European? ", am_price_lsm ≥ eu_price_div_bs ? "✓" : "✗ VIOLATION!")

println("\n" * "="^60)
println("4. COMPARISON: Impact of Dividends")
println("-"^60)

println("European Call Prices:")
println("  No dividends (BS):   \$", round(eu_price_no_div_bs, digits=4))
println("  With dividends (BS): \$", round(eu_price_div_bs, digits=4))
println("  Difference:          \$", round(eu_price_no_div_bs - eu_price_div_bs, digits=4))
println("  (Dividends reduce call value)")

println("\nAmerican Call Prices:")
println("  No dividends (Bin):  \$", round(am_price_no_div_binom, digits=4))
println("  With dividends (Bin):\$", round(am_price_div_binom, digits=4))
println("  Difference:          \$", round(am_price_no_div_binom - am_price_div_binom, digits=4))

println("\n" * "="^60)
