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
export VanillaOption, EuropeanOption, EuropeanCall, EuropeanPut, AmericanOption, AmericanCall, AmericanPut, payoff
export BlackScholes, Binomial, MonteCarlo, LongstaffSchwartz, asset_paths, asset_paths_col, asset_paths_ax, plot_paths, price
export BasisFunction, LaguerreBasis, ChebyshevBasis, PowerBasis, HermiteBasis
export EnhancedLongstaffSchwartz, LaguerreLSM, ChebyshevLSM, PowerLSM, HermiteLSM
export EuropeanLongstaffSchwartz, EuropeanLaguerreLSM, EuropeanChebyshevLSM, EuropeanPowerLSM, EuropeanHermiteLSM
export validate_american_option_price

end # module Prezo
