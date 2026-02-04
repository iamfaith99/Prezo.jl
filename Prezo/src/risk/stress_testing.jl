"""
    Portfolio Stress Testing

Framework for applying stress scenarios to portfolios and measuring impact.
Supports historical scenarios (e.g., 2008 crisis) and hypothetical shocks.
"""

using Statistics

# ============================================================================
# Stress Scenario Types
# ============================================================================

"""
    StressScenario

Abstract type for stress test scenarios.
"""
abstract type StressScenario end

"""
    HistoricalScenario(name, shocks)

A historical stress scenario with predefined market shocks.

# Fields
- `name::String`: Scenario name (e.g., "2008 Financial Crisis")
- `shocks::Dict{Symbol, Float64}`: Market factor shocks (e.g., :equity => -0.40)
"""
struct HistoricalScenario <: StressScenario
    name::String
    shocks::Dict{Symbol, Float64}
end

"""
    HypotheticalScenario(name, shocks, description)

A hypothetical stress scenario for "what-if" analysis.

# Fields
- `name::String`: Scenario name
- `shocks::Dict{Symbol, Float64}`: Market factor shocks
- `description::String`: Scenario description
"""
struct HypotheticalScenario <: StressScenario
    name::String
    shocks::Dict{Symbol, Float64}
    description::String
end
HypotheticalScenario(name, shocks) = HypotheticalScenario(name, shocks, "")

# ============================================================================
# Pre-defined Historical Scenarios
# ============================================================================

"""
    CRISIS_2008

2008 Financial Crisis scenario: equity -40%, credit spreads +300bp, vol +100%.
"""
const CRISIS_2008 = HistoricalScenario(
    "2008 Financial Crisis",
    Dict(:equity => -0.40, :credit_spread => 0.03, :volatility => 1.0, :rates => -0.02)
)

"""
    COVID_2020

COVID-19 March 2020 crash: equity -35%, vol spike, flight to quality.
"""
const COVID_2020 = HistoricalScenario(
    "COVID-19 Crash (Mar 2020)",
    Dict(:equity => -0.35, :volatility => 2.0, :rates => -0.015, :oil => -0.60)
)

"""
    RATE_SHOCK_UP

Interest rate shock: +200bp parallel shift.
"""
const RATE_SHOCK_UP = HistoricalScenario(
    "Rate Shock +200bp",
    Dict(:rates => 0.02, :equity => -0.10, :credit_spread => 0.005)
)

"""
    RATE_SHOCK_DOWN

Interest rate shock: -100bp parallel shift.
"""
const RATE_SHOCK_DOWN = HistoricalScenario(
    "Rate Shock -100bp",
    Dict(:rates => -0.01, :equity => 0.05, :credit_spread => -0.002)
)

# ============================================================================
# Stress Test Execution
# ============================================================================

"""
    StressTestResult

Result of a stress test on a portfolio.

# Fields
- `scenario::StressScenario`: The scenario applied
- `portfolio_pnl::Float64`: Portfolio P&L under stress
- `portfolio_pnl_pct::Float64`: P&L as percentage of portfolio value
- `component_pnl::Dict{String, Float64}`: P&L by component/position
- `risk_metrics::Dict{Symbol, Float64}`: Risk metrics under stress
"""
struct StressTestResult
    scenario::StressScenario
    portfolio_pnl::Float64
    portfolio_pnl_pct::Float64
    component_pnl::Dict{String, Float64}
    risk_metrics::Dict{Symbol, Float64}
end

"""
    PortfolioExposure

Defines portfolio exposures to risk factors.

# Fields
- `name::String`: Portfolio or position name
- `value::Float64`: Current market value
- `sensitivities::Dict{Symbol, Float64}`: Sensitivity to each risk factor
"""
struct PortfolioExposure
    name::String
    value::Float64
    sensitivities::Dict{Symbol, Float64}
end

"""
    stress_test(exposures::Vector{PortfolioExposure}, scenario::StressScenario) -> StressTestResult

Apply a stress scenario to portfolio exposures and compute impact.

# Arguments
- `exposures`: Vector of portfolio exposures with factor sensitivities
- `scenario`: Stress scenario to apply

# Returns
`StressTestResult` with portfolio and component P&L.

# Examples
```julia
# Define exposures
equity_pos = PortfolioExposure("Equity", 1_000_000.0, Dict(:equity => 1.0, :volatility => -0.1))
bond_pos = PortfolioExposure("Bonds", 500_000.0, Dict(:rates => -5.0, :credit_spread => -2.0))

# Run stress test
result = stress_test([equity_pos, bond_pos], CRISIS_2008)
println("Portfolio P&L: ", result.portfolio_pnl)
println("P&L %: ", result.portfolio_pnl_pct * 100, "%")
```
"""
function stress_test(exposures::Vector{PortfolioExposure}, scenario::StressScenario)
    shocks = scenario.shocks
    
    total_value = sum(e.value for e in exposures)
    component_pnl = Dict{String, Float64}()
    total_pnl = 0.0
    
    for exp in exposures
        pnl = 0.0
        for (factor, shock) in shocks
            sensitivity = get(exp.sensitivities, factor, 0.0)
            pnl += exp.value * sensitivity * shock
        end
        component_pnl[exp.name] = pnl
        total_pnl += pnl
    end
    
    pnl_pct = total_value > 0 ? total_pnl / total_value : 0.0
    
    risk_metrics = Dict{Symbol, Float64}(
        :total_exposure => total_value,
        :stressed_value => total_value + total_pnl,
        :drawdown => -min(0.0, pnl_pct)
    )
    
    return StressTestResult(scenario, total_pnl, pnl_pct, component_pnl, risk_metrics)
end

"""
    stress_test_suite(exposures::Vector{PortfolioExposure}, 
                      scenarios::Vector{<:StressScenario}) -> Vector{StressTestResult}

Run multiple stress scenarios on a portfolio.

# Examples
```julia
scenarios = [CRISIS_2008, COVID_2020, RATE_SHOCK_UP]
results = stress_test_suite(exposures, scenarios)
for r in results
    println(r.scenario.name, ": ", round(r.portfolio_pnl_pct * 100, digits=2), "%")
end
```
"""
function stress_test_suite(exposures::Vector{PortfolioExposure}, 
                           scenarios::Vector{<:StressScenario})
    return [stress_test(exposures, s) for s in scenarios]
end

"""
    reverse_stress_test(exposures::Vector{PortfolioExposure}, target_loss::Float64,
                        factors::Vector{Symbol}; max_shock::Float64=0.5) -> Dict{Symbol, Float64}

Find the factor shocks needed to produce a target loss (reverse stress test).

Uses gradient descent to find shocks that produce the target portfolio loss.

# Arguments
- `exposures`: Portfolio exposures
- `target_loss`: Target loss as fraction of portfolio (e.g., 0.20 for 20% loss)
- `factors`: Risk factors to shock
- `max_shock`: Maximum absolute shock per factor

# Returns
Dictionary of factor shocks that produce approximately the target loss.
"""
function reverse_stress_test(exposures::Vector{PortfolioExposure}, target_loss::Float64,
                             factors::Vector{Symbol}; max_shock::Float64=0.5)
    total_value = sum(e.value for e in exposures)
    target_pnl = -target_loss * total_value
    
    # Compute aggregate sensitivities
    agg_sens = Dict{Symbol, Float64}()
    for f in factors
        sens = sum(e.value * get(e.sensitivities, f, 0.0) for e in exposures)
        agg_sens[f] = sens
    end
    
    # Simple proportional allocation of shocks
    total_sens = sum(abs(s) for s in values(agg_sens))
    shocks = Dict{Symbol, Float64}()
    
    if total_sens > 0
        for f in factors
            # Allocate shock proportional to sensitivity
            sens = agg_sens[f]
            if abs(sens) > 0
                shock = target_pnl * (abs(sens) / total_sens) / sens
                shocks[f] = clamp(shock, -max_shock, max_shock)
            else
                shocks[f] = 0.0
            end
        end
    end
    
    return shocks
end
