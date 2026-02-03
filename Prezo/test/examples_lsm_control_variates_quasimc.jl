#=
    LSM-specific control variates and quasi-Monte Carlo (Halton)

Demonstrates:
1. Control variates: American LSM + (BS European - European MC on same paths)
   for variance reduction (European put as control).
2. Quasi-MC: Halton low-discrepancy sequence for path generation, then LSM
   on those paths (same API via paths= keyword).

Run from Prezo package root (Prezo.jl/Prezo/):
    julia --project=. test/examples_lsm_control_variates_quasimc.jl

If you are already in Prezo/test/, run:
    julia --project=.. examples_lsm_control_variates_quasimc.jl
=#

using Prezo
using Statistics
using Random
using Distributions: Normal, quantile

# -----------------------------------------------------------------------------
# Halton sequence (in-script; no extra deps)
# -----------------------------------------------------------------------------

"""Van der Corput in base b for index i (0-based index -> [0,1])."""
function vandercorput(i::Int, b::Int)
    f = 1.0
    r = 0.0
    j = i
    while j > 0
        f /= b
        r += f * (j % b)
        j ÷= b
    end
    return r
end

# First 50 primes (for Halton dimensions up to 50)
const _HALTON_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229]

"""Halton sequence: n points in [0,1]^dim. Returns (dim, n) matrix."""
function halton_sequence(dim::Int, n::Int; skip::Int=0)
    bases = _HALTON_PRIMES[1:min(dim, length(_HALTON_PRIMES))]
    out = zeros(Float64, dim, n)
    for j in 1:n
        idx = skip + j - 1
        for d in 1:dim
            out[d, j] = vandercorput(idx, bases[d])
        end
    end
    return out
end

"""GBM paths using Halton sequence for normals: (steps+1) × reps. Optional skip for randomized QMC."""
function asset_paths_halton(steps::Int, reps::Int, spot::Float64, rate::Float64, vol::Float64, expiry::Float64; skip::Int=0)
    dt = expiry / steps
    nudt = (rate - 0.5 * vol^2) * dt
    sidt = vol * sqrt(dt)
    d = Normal(0.0, 1.0)
    u = halton_sequence(steps, reps; skip=skip)
    z = quantile.(Ref(d), u)
    paths = zeros(Float64, steps + 1, reps)
    paths[1, :] .= spot
    for t in 1:steps
        paths[t+1, :] .= paths[t, :] .* exp.(nudt .+ sidt .* z[t, :])
    end
    return paths
end

# -----------------------------------------------------------------------------
# 1. Control variates: American LSM + European control
# -----------------------------------------------------------------------------
println("=== 1. LSM with European put as control variate ===")
Random.seed!(42)
data = MarketData(100.0, 0.05, 0.2, 0.0)
american_put = AmericanPut(100.0, 1.0)
european_put = EuropeanPut(100.0, 1.0)
engine = LongstaffSchwartz(50, 10_000, 3)

# Single run with control variate
paths = asset_paths_col(MonteCarlo(50, 10_000), 100.0, 0.05, 0.2, 1.0)
american_lsm = price(american_put, engine, data; paths=paths)
european_mc = exp(-data.rate * 1.0) * mean(payoff.(Ref(european_put), paths[end, :]))
bs_european = price(european_put, BlackScholes(), data)
american_cv = american_lsm + (bs_european - european_mc)

println("  American LSM (raw):     $(round(american_lsm, digits=4))")
println("  European MC (same paths): $(round(european_mc, digits=4))")
println("  European BS (analytic):   $(round(bs_european, digits=4))")
println("  American CV (adjusted):   $(round(american_cv, digits=4))")

# Compare variance over multiple runs
n_runs = 30
american_raw = Float64[]
american_cv_runs = Float64[]
for _ in 1:n_runs
    local_paths = asset_paths_col(MonteCarlo(50, 10_000), 100.0, 0.05, 0.2, 1.0)
    a = price(american_put, engine, data; paths=local_paths)
    e_mc = exp(-data.rate * 1.0) * mean(payoff.(Ref(european_put), local_paths[end, :]))
    push!(american_raw, a)
    push!(american_cv_runs, a + (bs_european - e_mc))
end
println("  Over $n_runs runs:")
println("  Raw LSM:  mean=$(round(mean(american_raw), digits=4))  std=$(round(std(american_raw), digits=4))")
println("  CV LSM:   mean=$(round(mean(american_cv_runs), digits=4))  std=$(round(std(american_cv_runs), digits=4))")
println("  Variance ratio (raw/CV): $(round(var(american_raw)/max(1e-12, var(american_cv_runs)), digits=2))")

# -----------------------------------------------------------------------------
# 2. Quasi-Monte Carlo (Halton) paths + LSM
# -----------------------------------------------------------------------------
println("\n=== 2. LSM with Halton quasi-MC paths ===")
Random.seed!(123)
n_runs_qmc = 20
american_standard = Float64[]
american_halton = Float64[]
for run in 1:n_runs_qmc
    paths_std = asset_paths_col(MonteCarlo(50, 10_000), 100.0, 0.05, 0.2, 1.0)
    push!(american_standard, price(american_put, engine, data; paths=paths_std))
    # Randomized QMC: different Halton segment per run (skip = (run-1)*10_000)
    paths_halton = asset_paths_halton(50, 10_000, 100.0, 0.05, 0.2, 1.0; skip=(run - 1) * 10_000)
    push!(american_halton, price(american_put, engine, data; paths=paths_halton))
end
println("  Over $n_runs_qmc runs (50 steps, 10k paths); QMC uses randomized start (skip).")
println("  Standard MC LSM: mean=$(round(mean(american_standard), digits=4))  std=$(round(std(american_standard), digits=4))")
println("  Halton QMC LSM:  mean=$(round(mean(american_halton), digits=4))  std=$(round(std(american_halton), digits=4))")
v_std = var(american_standard)
v_qmc = max(1e-12, var(american_halton))
println("  Variance ratio (standard/QMC): $(round(v_std / v_qmc, digits=2))")

# -----------------------------------------------------------------------------
# 3. Combined: Halton paths + control variate
# -----------------------------------------------------------------------------
println("\n=== 3. Halton paths + control variate ===")
american_halton_cv = Float64[]
for run in 1:n_runs_qmc
    paths_h = asset_paths_halton(50, 10_000, 100.0, 0.05, 0.2, 1.0; skip=(run - 1) * 10_000)
    a = price(american_put, engine, data; paths=paths_h)
    e_mc = exp(-data.rate * 1.0) * mean(payoff.(Ref(european_put), paths_h[end, :]))
    push!(american_halton_cv, a + (bs_european - e_mc))
end
println("  Halton + CV: mean=$(round(mean(american_halton_cv), digits=4))  std=$(round(std(american_halton_cv), digits=4))")

println("\nDone.")