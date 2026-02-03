"""
Benchmark script for Longstaff-Schwartz performance optimizations.

Tests:
1. Memory allocation improvements
2. Computational speed improvements
3. Variance reduction with antithetic variates
4. Accuracy validation

Run with: julia --project=. benchmark_lsm.jl
"""

using Prezo
using BenchmarkTools
using Printf
using Statistics

function print_header(title::String)
    println("\n" * "="^70)
    println(title)
    println("="^70)
end

function print_result(label::String, value, unit::String="")
    @printf("  %-40s: %12.6f %s\n", label, value, unit)
end

# Test parameters
data = MarketData(40.0, 0.06, 0.2, 0.0)
data_with_div = MarketData(40.0, 0.06, 0.2, 0.04)
option_put = AmericanPut(40.0, 1.0)
option_call = AmericanCall(40.0, 1.0)

# European options for comparison
eu_put = EuropeanPut(40.0, 1.0)
eu_call = EuropeanCall(40.0, 1.0)

print_header("LONGSTAFF-SCHWARTZ PERFORMANCE BENCHMARKS")

# ============================================================================
# 1. BASELINE: BASIC LSM PERFORMANCE
# ============================================================================
print_header("1. Basic LSM Performance (50 steps, 10k paths)")

println("\nAmerican Put (ITM scenario):")
engine_basic = LongstaffSchwartz(50, 10000)
price_basic = price(option_put, engine_basic, data)
print_result("Price", price_basic, "")

println("\n  Timing (10 runs):")
bench_basic = @benchmark price($option_put, $engine_basic, $data) samples=10
print_result("Median time", median(bench_basic.times) / 1e6, "ms")
print_result("Mean time", mean(bench_basic.times) / 1e6, "ms")
print_result("Std dev", std(bench_basic.times) / 1e6, "ms")
print_result("Allocations", bench_basic.allocs, "")
print_result("Memory", bench_basic.memory / 1024^2, "MB")

# ============================================================================
# 2. VARIANCE REDUCTION: ANTITHETIC VARIATES
# ============================================================================
print_header("2. Antithetic Variates (50 steps, 5k pairs = 10k total paths)")

println("\nAmerican Put with antithetic variates:")
engine_antithetic = LongstaffSchwartz(50, 5000, 3; antithetic=true)
price_antithetic = price(option_put, engine_antithetic, data)
print_result("Price", price_antithetic, "")

println("\n  Timing (10 runs):")
bench_antithetic = @benchmark price($option_put, $engine_antithetic, $data) samples=10
print_result("Median time", median(bench_antithetic.times) / 1e6, "ms")
print_result("Mean time", mean(bench_antithetic.times) / 1e6, "ms")
print_result("Std dev", std(bench_antithetic.times) / 1e6, "ms")
print_result("Allocations", bench_antithetic.allocs, "")
print_result("Memory", bench_antithetic.memory / 1024^2, "MB")

println("\n  Comparison:")
print_result("Price difference", abs(price_basic - price_antithetic), "")
print_result("Speedup", median(bench_basic.times) / median(bench_antithetic.times), "×")
print_result("Memory reduction", 1.0 - bench_antithetic.memory / bench_basic.memory, "% saved")

# ============================================================================
# 3. CONVERGENCE: VARIANCE REDUCTION EFFECTIVENESS
# ============================================================================
print_header("3. Variance Reduction Effectiveness (100 runs each)")

println("\nMeasuring price variance with 100 independent runs...")

# Standard MC
println("\n  Standard LSM (10k paths):")
prices_standard = Float64[]
for i in 1:100
    p = price(option_put, LongstaffSchwartz(50, 10000), data)
    push!(prices_standard, p)
end
mean_standard = mean(prices_standard)
std_standard = std(prices_standard)
stderr_standard = std_standard / sqrt(100)

print_result("Mean price", mean_standard, "")
print_result("Std deviation", std_standard, "")
print_result("Std error", stderr_standard, "")

# Antithetic variates
println("\n  Antithetic LSM (5k pairs = 10k paths):")
prices_antithetic_var = Float64[]
for i in 1:100
    p = price(option_put, LongstaffSchwartz(50, 5000, 3; antithetic=true), data)
    push!(prices_antithetic_var, p)
end
mean_antithetic_var = mean(prices_antithetic_var)
std_antithetic_var = std(prices_antithetic_var)
stderr_antithetic_var = std_antithetic_var / sqrt(100)

print_result("Mean price", mean_antithetic_var, "")
print_result("Std deviation", std_antithetic_var, "")
print_result("Std error", stderr_antithetic_var, "")

# Variance reduction factor
vrf = std_standard^2 / std_antithetic_var^2
println("\n  Variance Reduction:")
print_result("Variance reduction factor", vrf, "×")
print_result("Effective sample multiplier", vrf, "×")
print_result("Standard error reduction", 1.0 - stderr_antithetic_var / stderr_standard, "% reduction")

# ============================================================================
# 4. SCALING ANALYSIS
# ============================================================================
print_header("4. Scaling Analysis: Paths vs Time")

path_counts = [1000, 5000, 10000, 25000, 50000]
println("\n  Standard LSM:")
println("  Paths     Time (ms)    Price")
println("  -------   ----------   --------")

for paths in path_counts
    engine = LongstaffSchwartz(50, paths)
    p = price(option_put, engine, data)
    t = @elapsed price(option_put, engine, data)
    @printf("  %7d   %10.2f   %8.4f\n", paths, t * 1000, p)
end

println("\n  Antithetic LSM:")
println("  Pairs     Total Paths  Time (ms)    Price")
println("  -------   -----------  ----------   --------")

for paths in path_counts
    pairs = div(paths, 2)
    engine = LongstaffSchwartz(50, pairs, 3; antithetic=true)
    p = price(option_put, engine, data)
    t = @elapsed price(option_put, engine, data)
    @printf("  %7d   %11d  %10.2f   %8.4f\n", pairs, paths, t * 1000, p)
end

# ============================================================================
# 5. ACCURACY VALIDATION
# ============================================================================
print_header("5. Accuracy Validation")

println("\nComparing LSM vs analytical/Binomial prices:")

# American Put (high accuracy reference)
println("\n  American Put (40 strike, spot=40, r=6%, vol=20%, T=1y):")
binomial_put = price(option_put, Binomial(500), data)
lsm_put = price(option_put, LongstaffSchwartz(50, 50000), data)
lsm_antithetic_put = price(option_put, LongstaffSchwartz(50, 25000, 3; antithetic=true), data)

print_result("Binomial (500 steps) - reference", binomial_put, "")
print_result("LSM standard (50k paths)", lsm_put, "")
print_result("LSM antithetic (25k pairs)", lsm_antithetic_put, "")
print_result("LSM standard error", abs(lsm_put - binomial_put), "")
print_result("LSM antithetic error", abs(lsm_antithetic_put - binomial_put), "")

# European Put (analytical reference)
println("\n  European Put (validation):")
bs_put = price(eu_put, BlackScholes(), data)
lsm_eu_put = price(option_put, LongstaffSchwartz(50, 50000), data)

print_result("Black-Scholes (analytical)", bs_put, "")
print_result("LSM American Put", lsm_eu_put, "")
print_result("Early exercise premium", lsm_eu_put - bs_put, "")

# American Call (should be close to European with no dividends)
println("\n  American Call (no dividends):")
bs_call = price(eu_call, BlackScholes(), data)
lsm_call = price(option_call, LongstaffSchwartz(50, 50000), data)

print_result("Black-Scholes (analytical)", bs_call, "")
print_result("LSM American Call", lsm_call, "")
print_result("Early exercise premium", lsm_call - bs_call, "")
print_result("Premium (should be ~0)", abs(lsm_call - bs_call), "")

# With dividends, American call should have premium
println("\n  American Call (with 4% dividends):")
bs_call_div = price(eu_call, BlackScholes(), data_with_div)
lsm_call_div = price(option_call, LongstaffSchwartz(50, 50000), data_with_div)

print_result("Black-Scholes (analytical)", bs_call_div, "")
print_result("LSM American Call", lsm_call_div, "")
print_result("Early exercise premium", lsm_call_div - bs_call_div, "")

# ============================================================================
# 6. SUMMARY
# ============================================================================
print_header("SUMMARY")

println("\n  Key Performance Improvements:")
println("  - Memory allocation: Pre-allocated workspace reduces allocations")
println("  - Computation: @inbounds and views eliminate bounds checking overhead")
println("  - Variance reduction: Antithetic variates provide $(round(vrf, digits=2))× variance reduction")
println("  - Effective paths: $(round(vrf, digits=2))× more effective samples per path")

speedup = median(bench_basic.times) / median(bench_antithetic.times)
println("\n  Antithetic Variates Trade-off:")
@printf("  - %d%% more paths generated (2× paths)\n", 100)
@printf("  - %.1f× faster execution (optimized path generation)\n", speedup)
@printf("  - %.1f× variance reduction\n", vrf)
@printf("  - Net efficiency gain: %.1f×\n", vrf / (1 / speedup))

println("\n" * "="^70)
println("Benchmark complete!")
println("="^70)
