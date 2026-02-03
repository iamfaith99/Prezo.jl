#=
    Examples: Phase 5 — Advanced Hedging (OHMC and delta hedging strategies)

Run from Prezo package root:
    julia --project=. test/examples_hedging.jl
=#

using Prezo
using Statistics
using Random

Random.seed!(42)

# -----------------------------------------------------------------------------
# 1. Optimal Hedged Monte Carlo (OHMC) — scoring: quadratic / log / exponential_utility
# -----------------------------------------------------------------------------
println("=== 1. OHMC vs Black-Scholes (scoring plug-in) ===")
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
rng = MersenneTwister(1)
rq = ohmc_price(call, data, OHMCConfig(2000, 20, 3; scoring=:quadratic); rng=copy(rng))
rl = ohmc_price(call, data, OHMCConfig(2000, 20, 3; scoring=:log); rng=copy(rng))
ru = ohmc_price(call, data, OHMCConfig(2000, 20, 3; scoring=:exponential_utility, risk_aversion=0.1); rng=copy(rng))
bs = price(call, BlackScholes(), data)
println("  Quadratic:      $(round(rq.option_price, digits=4))")
println("  Log:            $(round(rl.option_price, digits=4))")
println("  Exp utility:    $(round(ru.option_price, digits=4))")
println("  Black-Scholes:  $(round(bs, digits=4))")
println("  95% CI (quad): ($(round(rq.confidence_interval[1], digits=4)), $(round(rq.confidence_interval[2], digits=4)))")

# -----------------------------------------------------------------------------
# 2. Delta hedging backtest — discrete rebalancing
# -----------------------------------------------------------------------------
println("\n=== 2. Discrete delta hedge backtest ===")
option = EuropeanCall(100.0, 1.0)
engine = BlackScholes()
n_obs = 21
t = range(0.0, 1.0, length=n_obs)
spot = 100.0 .* exp.(0.05 .* t .+ 0.2 .* sqrt.(max.(t, 1e-8)) .* randn(n_obs))
spot[1] = 100.0
hist = hcat(collect(t), spot, fill(0.05, n_obs), fill(0.2, n_obs))
strat = DiscreteDeltaHedge([0.25, 0.5, 0.75, 1.0], 0.001)
perf = backtest_hedge(option, strat, hist, engine)
println("  Total P&L:      $(round(perf.total_pnl, digits=4))")
println("  Transaction cost: $(round(perf.total_cost, digits=4))")
println("  N trades:       $(perf.n_trades)")

# -----------------------------------------------------------------------------
# 3. Stop-loss hedge
# -----------------------------------------------------------------------------
println("\n=== 3. Stop-loss delta hedge ===")
strat_sl = StopLossHedge(0.15)
perf_sl = backtest_hedge(option, strat_sl, hist, engine)
println("  Total P&L:      $(round(perf_sl.total_pnl, digits=4))")
println("  N trades:       $(perf_sl.n_trades)")

# -----------------------------------------------------------------------------
# 4. Static hedge (no rebalancing)
# -----------------------------------------------------------------------------
println("\n=== 4. Static hedge ===")
strat_static = StaticHedge([EuropeanCall(100.0, 1.0)])
perf_static = backtest_hedge(option, strat_static, hist, engine)
println("  Total P&L:      $(round(perf_static.total_pnl, digits=4))")
println("  N trades:       $(perf_static.n_trades)")

println("\nDone.")