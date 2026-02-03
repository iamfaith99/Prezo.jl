using Plots
using Prezo

call = EuropeanCall(40.0, 1.0)
put = EuropeanPut(40.0, 1.0)

# Market data: spot=41.0, rate=0.08, vol=0.3, div=0.0
data = MarketData(41.0, 0.08, 0.3, 0.0)

bsm_prc = price(put, BlackScholes(), data)
bsm_prc = round(bsm_prc, digits=2)
println("Black-Scholes-Merton price: $bsm_prc")

bin_prc = price(put, Binomial(100), data)
bin_prc = round(bin_prc, digits=2)
println("Binomial price: $bin_prc")

mc_prc = price(put, MonteCarlo(1, 5_000_000), data)
mc_prc = round(mc_prc, digits=2)
println("Monte Carlo price: $mc_prc")