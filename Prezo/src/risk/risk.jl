"""
    Risk

Risk management module for Prezo.jl.

# Value at Risk (VaR) and CVaR
- [`VaRMethod`](@ref): Abstract type for VaR methods
- [`HistoricalVaR`](@ref), [`ParametricVaR`](@ref), [`MonteCarloVaR`](@ref)
- [`value_at_risk`](@ref): Calculate VaR at confidence level
- [`conditional_var`](@ref): CVaR / Expected Shortfall
- [`portfolio_var`](@ref): Portfolio-level VaR with component decomposition

# Stress Testing
- [`StressScenario`](@ref), [`HistoricalScenario`](@ref), [`HypotheticalScenario`](@ref)
- Pre-defined scenarios: `CRISIS_2008`, `COVID_2020`, `RATE_SHOCK_UP`, `RATE_SHOCK_DOWN`
- [`stress_test`](@ref): Apply stress scenario to portfolio
- [`stress_test_suite`](@ref): Run multiple scenarios
- [`reverse_stress_test`](@ref): Find shocks for target loss

# Scenario Analysis
- [`scenario_grid`](@ref): Generate scenario grid
- [`monte_carlo_scenarios`](@ref): Monte Carlo scenario generation
- [`analyze_scenarios`](@ref): Analyze values across scenarios
- [`sensitivity_table`](@ref): 2D sensitivity analysis
- [`scenario_ladder`](@ref): 1D sensitivity analysis

# Kelly Criterion
- [`kelly_fraction`](@ref): Classic Kelly for binary outcomes
- [`fractional_kelly`](@ref): Reduced-risk Kelly
- [`kelly_continuous`](@ref): Kelly for Gaussian returns
- [`kelly_portfolio`](@ref): Multi-asset Kelly optimization
- [`kelly_growth_rate`](@ref): Expected log growth
"""
module Risk

using Statistics
using LinearAlgebra
using Random
using Distributions

include("var.jl")
include("stress_testing.jl")
include("scenario.jl")
include("kelly.jl")

# VaR exports
export VaRMethod, HistoricalVaR, ParametricVaR, MonteCarloVaR
export value_at_risk, conditional_var, portfolio_var, PortfolioVaR

# Stress testing exports
export StressScenario, HistoricalScenario, HypotheticalScenario
export PortfolioExposure, StressTestResult
export stress_test, stress_test_suite, reverse_stress_test
export CRISIS_2008, COVID_2020, RATE_SHOCK_UP, RATE_SHOCK_DOWN

# Scenario analysis exports
export ScenarioGrid, ScenarioAnalysisResult, SensitivityTable
export scenario_grid, monte_carlo_scenarios
export analyze_scenarios, sensitivity_table, scenario_ladder

# Kelly criterion exports
export kelly_fraction, fractional_kelly
export kelly_continuous, kelly_from_sharpe
export KellyPortfolio, kelly_portfolio, fractional_kelly_portfolio
export kelly_growth_rate, kelly_drawdown_probability, optimal_bet_size

end # module Risk
