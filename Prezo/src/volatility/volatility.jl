"""
    Volatility

Volatility modeling: GARCH family (univariate and multivariate), Heston stochastic
volatility, and Dupire local volatility.

# GARCH family (univariate)
- [`GARCH`](@ref), [`EGARCH`](@ref), [`GJRGARCH`](@ref), [`AGARCH`](@ref)
- `fit(GARCH, returns)`, `volatility_process`, `forecast`, `simulate`, `loglikelihood`
- Student-t: `loglikelihood(..., dist=TDist(ν))`, `simulate(..., dist=TDist(ν))`
- [`RegimeGARCH`](@ref): two-regime Markov-switching GARCH (`regime_garch.jl`)

# Multivariate GARCH
- [`DCCGARCH`](@ref): dynamic conditional correlation (`dcc_garch.jl`)
- [`OGARCH`](@ref): PCA-based orthogonal GARCH (`ogarch.jl`)
- [`FactorGARCH`](@ref): observed factors + GARCH (`factor_garch.jl`)

# Stochastic and local volatility
- [`HestonModel`](@ref), [`simulate_heston`](@ref): Heston SDE simulation (`heston.jl`)
- [`LocalVolSurface`](@ref), [`dupire_local_vol`](@ref): Dupire local vol from call prices (`local_vol.jl`)
"""
module Volatility

using Distributions
using Optim
using Statistics
using Random

include("garch.jl")
include("regime_garch.jl")
include("dcc_garch.jl")
include("ogarch.jl")
include("factor_garch.jl")
include("heston.jl")
include("local_vol.jl")

export VolatilityModel, GARCHModel
export GARCH, EGARCH, GJRGARCH, AGARCH, RegimeGARCH, DCCGARCH, OGARCH, FactorGARCH
export HestonModel, simulate_heston
export LocalVolSurface, dupire_local_vol
export fit, volatility_process, forecast, loglikelihood, simulate
export covariance_series

end # module Volatility
