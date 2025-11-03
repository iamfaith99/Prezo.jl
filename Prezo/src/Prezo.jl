module Prezo

# struct MarketData
#     rate::Float64
#     vol::Float64
#     div::Float64
# end

include("data.jl")
include("options.jl")
include("engines.jl")

export MarketData
export VanillaOption, EuropeanOption, AmericanOption
export EuropeanCall, EuropeanPut, AmericanCall, AmericanPut
export payoff
export BlackScholes, Binomial, MonteCarlo, LongstaffSchwartz
export asset_paths, asset_paths_col, asset_paths_ax, asset_paths_antithetic, plot_paths
export price
export BasisFunction, LaguerreBasis, ChebyshevBasis, PowerBasis, HermiteBasis
export EnhancedLongstaffSchwartz
export LaguerreLSM, ChebyshevLSM, PowerLSM, HermiteLSM
export EuropeanLongstaffSchwartz
export EuropeanLaguerreLSM, EuropeanChebyshevLSM
export EuropeanPowerLSM, EuropeanHermiteLSM
export validate_american_option_price

end # module Prezo
