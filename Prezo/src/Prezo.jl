module Prezo

# struct MarketData
#     rate::Float64
#     vol::Float64
#     div::Float64
# end

include("data.jl")
include("options.jl")
include("engines.jl")
include("paths.jl")
include("basis.jl")
include("lsm.jl")

# Include Greeks module
include("greeks/greeks.jl")
using .Greeks

# Include Implied Volatility module
include("implied_vol/implied_vol.jl")
using .ImpliedVol

# Include Volatility module (Phase 2: GARCH family)
include("volatility/volatility.jl")
using .Volatility

# Include Filters module (Phase 3: state estimation)
include("filters/filters.jl")
using .Filters
import .Filters: predict, update, filter_step, filter_data, smooth

# Include Inference module (Phase 4: MLE, calibration, ABC)
include("inference/inference.jl")
using .Inference

# Include Hedging module (Phase 5: OHMC, delta hedging)
include("hedging/hedging.jl")

# Include Risk module (Phase 6: VaR, CVaR, stress testing, scenario analysis, Kelly)
include("risk/risk.jl")
using .Risk

# Include GPU acceleration (Monte Carlo on CUDA)
include("gpu/gpu.jl")

export MarketData, MarketDataExt, DiscreteDividend
export present_value_dividends, adjusted_spot
export VanillaOption, EuropeanOption, AmericanOption
export EuropeanCall, EuropeanPut, AmericanCall, AmericanPut
export ExoticOption
export AsianOption, ArithmeticAsianCall, ArithmeticAsianPut, GeometricAsianCall, GeometricAsianPut
export BarrierOption, KnockOutCall, KnockOutPut, KnockInCall, KnockInPut, barrier_breached
export LookbackOption, FixedStrikeLookbackCall, FixedStrikeLookbackPut
export FloatingStrikeLookbackCall, FloatingStrikeLookbackPut
export payoff
export PricingEngine
export BlackScholes, Binomial, MonteCarlo, MonteCarloAntithetic, MonteCarloStratified, MonteCarloGPU, LongstaffSchwartz
export asset_paths, asset_paths_col, asset_paths_col_threaded, asset_paths_col_gpu, asset_paths_ax, asset_paths_antithetic, plot_paths
export price
export BasisFunction, LaguerreBasis, ChebyshevBasis, PowerBasis, HermiteBasis
export EnhancedLongstaffSchwartz
export LaguerreLSM, ChebyshevLSM, PowerLSM, HermiteLSM
export EuropeanLongstaffSchwartz
export EuropeanLaguerreLSM, EuropeanChebyshevLSM
export EuropeanPowerLSM, EuropeanHermiteLSM
export validate_american_option_price

# Greeks exports
export Greek, FirstOrderGreek, SecondOrderGreek, ThirdOrderGreek
export Delta, Gamma, Theta, Vega, Rho, Phi, RhoDiv
export Vanna, Charm, Vomma, Veta, Speed, Color, Ultima
export greek, greeks, all_greeks
export portfolio_greeks, Position
export numerical_greek, finite_difference
export hedge_ratio, is_delta_neutral, is_gamma_neutral
export pnl_attribution, scenario_analysis
export get_expiry, with_shorter_expiry
export barrier_proximity, is_near_barrier
export mc_optimal_step_spot, mc_optimal_step_vol, mc_optimal_step_rate

# Implied Volatility exports
export IVSolver, NewtonRaphson, Bisection, HybridSolver
export implied_vol, implied_vol_chain
export iv_objective, vega_for_iv
export is_valid_price, price_bounds
export implied_vol_stats
export ImpliedVolSurface, build_implied_vol_surface, surface_iv, surface_stats

# Volatility (Phase 2)
export VolatilityModel, GARCHModel
export GARCH, EGARCH, GJRGARCH, AGARCH, RegimeGARCH, DCCGARCH, OGARCH, FactorGARCH
export HestonModel, simulate_heston
export LocalVolSurface, dupire_local_vol
export fit, volatility_process, forecast, loglikelihood, simulate
export covariance_series

# Filters (Phase 3)
export StateSpaceFilter
export KalmanFilter, KalmanFilterState
export predict, update, filter_step, filter_data, smooth
export ExtendedKalmanFilter
export EnsembleKalmanFilter, EnsembleKalmanState
export ParticleFilter, ParticleState
export effective_sample_size
export systematic_resample, multinomial_resample, stratified_resample

# Inference (Phase 4)
export MLEProblem, MLESolution, solve, standard_errors
export CalibrationTarget, OptionPricesTarget, IVSurfaceTarget
export CalibrationMethod, LeastSquaresCalibration, RegularizedCalibration
export calibrate, calibrate_option_prices, CalibrationResult
export ABCMethod, RejectionABC
export MCMCABC, ABCSMC, RegressionABC, ABCModelChoice, HierarchicalABC
export abc_inference, abc_model_choice, euclidean_distance

# Hedging (Phase 5)
export OHMCConfig, OHMCResult, ohmc_price
export HedgingStrategy, DiscreteDeltaHedge, StopLossHedge, StaticHedge
export HedgePerformance, backtest_hedge

# Risk Management (Phase 6)
export VaRMethod, HistoricalVaR, ParametricVaR, MonteCarloVaR
export value_at_risk, conditional_var, portfolio_var, PortfolioVaR
export StressScenario, HistoricalScenario, HypotheticalScenario
export PortfolioExposure, StressTestResult
export stress_test, stress_test_suite, reverse_stress_test
export CRISIS_2008, COVID_2020, RATE_SHOCK_UP, RATE_SHOCK_DOWN
export ScenarioGrid, ScenarioAnalysisResult, SensitivityTable
export scenario_grid, monte_carlo_scenarios
export analyze_scenarios, sensitivity_table, scenario_ladder
export kelly_fraction, fractional_kelly
export kelly_continuous, kelly_from_sharpe
export KellyPortfolio, kelly_portfolio, fractional_kelly_portfolio
export kelly_growth_rate, kelly_drawdown_probability, optimal_bet_size

end # module Prezo
