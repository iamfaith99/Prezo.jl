"""
    Scenario Analysis Framework

Tools for analyzing portfolio behavior under various market scenarios:
- Scenario generation (grid, Monte Carlo, historical)
- P&L distribution analysis
- Sensitivity tables (Greeks across scenarios)
"""

using Statistics
using Random

# ============================================================================
# Scenario Generation
# ============================================================================

"""
    ScenarioGrid

A grid of scenarios for sensitivity analysis.

# Fields
- `factors::Vector{Symbol}`: Risk factors varied
- `ranges::Dict{Symbol, Vector{Float64}}`: Values for each factor
- `scenarios::Vector{Dict{Symbol, Float64}}`: All scenario combinations
"""
struct ScenarioGrid
    factors::Vector{Symbol}
    ranges::Dict{Symbol, Vector{Float64}}
    scenarios::Vector{Dict{Symbol, Float64}}
end

"""
    scenario_grid(factor_ranges::Dict{Symbol, Vector{Float64}}) -> ScenarioGrid

Generate a grid of scenarios from factor ranges (Cartesian product).

# Arguments
- `factor_ranges`: Dict mapping factors to vectors of values to test

# Examples
```julia
ranges = Dict(
    :spot => [90.0, 95.0, 100.0, 105.0, 110.0],
    :vol => [0.15, 0.20, 0.25, 0.30]
)
grid = scenario_grid(ranges)
# 5 × 4 = 20 scenarios
```
"""
function scenario_grid(factor_ranges::Dict{Symbol, Vector{Float64}})
    factors = collect(keys(factor_ranges))
    ranges = [factor_ranges[f] for f in factors]
    
    # Cartesian product
    scenarios = Dict{Symbol, Float64}[]
    for combo in Iterators.product(ranges...)
        scenario = Dict{Symbol, Float64}()
        for (i, f) in enumerate(factors)
            scenario[f] = combo[i]
        end
        push!(scenarios, scenario)
    end
    
    return ScenarioGrid(factors, factor_ranges, scenarios)
end

"""
    monte_carlo_scenarios(factor_means::Dict{Symbol, Float64},
                          factor_vols::Dict{Symbol, Float64},
                          n_scenarios::Int;
                          correlation::Union{Matrix{Float64}, Nothing}=nothing,
                          rng::AbstractRNG=Random.GLOBAL_RNG) -> Vector{Dict{Symbol, Float64}}

Generate Monte Carlo scenarios for factors with optional correlation.

# Arguments
- `factor_means`: Expected values for each factor
- `factor_vols`: Volatilities for each factor
- `n_scenarios`: Number of scenarios to generate
- `correlation`: Optional correlation matrix
- `rng`: Random number generator

# Returns
Vector of scenario dictionaries.

# Examples
```julia
means = Dict(:spot => 100.0, :vol => 0.2, :rate => 0.05)
vols = Dict(:spot => 20.0, :vol => 0.05, :rate => 0.01)
scenarios = monte_carlo_scenarios(means, vols, 1000)
```
"""
function monte_carlo_scenarios(factor_means::Dict{Symbol, Float64},
                               factor_vols::Dict{Symbol, Float64},
                               n_scenarios::Int;
                               correlation::Union{Matrix{Float64}, Nothing}=nothing,
                               rng::AbstractRNG=Random.GLOBAL_RNG)
    factors = collect(keys(factor_means))
    n_factors = length(factors)
    
    # Generate correlated normals if correlation provided
    if correlation !== nothing
        size(correlation) == (n_factors, n_factors) || error("Correlation matrix size mismatch")
        # Cholesky decomposition
        L = cholesky(correlation).L
        Z = randn(rng, n_scenarios, n_factors) * L'
    else
        Z = randn(rng, n_scenarios, n_factors)
    end
    
    # Transform to factor values
    scenarios = Dict{Symbol, Float64}[]
    for i in 1:n_scenarios
        scenario = Dict{Symbol, Float64}()
        for (j, f) in enumerate(factors)
            μ = factor_means[f]
            σ = factor_vols[f]
            scenario[f] = μ + σ * Z[i, j]
        end
        push!(scenarios, scenario)
    end
    
    return scenarios
end

# ============================================================================
# Scenario Analysis
# ============================================================================

"""
    ScenarioAnalysisResult

Result of scenario analysis.

# Fields
- `scenarios::Vector{Dict{Symbol, Float64}}`: Scenarios evaluated
- `values::Vector{Float64}`: Portfolio/option values in each scenario
- `pnl::Vector{Float64}`: P&L relative to base case
- `statistics::Dict{Symbol, Float64}`: Summary statistics
"""
struct ScenarioAnalysisResult
    scenarios::Vector{Dict{Symbol, Float64}}
    values::Vector{Float64}
    pnl::Vector{Float64}
    statistics::Dict{Symbol, Float64}
end

"""
    analyze_scenarios(valuation_fn::Function, scenarios::Vector{Dict{Symbol, Float64}},
                      base_value::Float64) -> ScenarioAnalysisResult

Analyze portfolio/option values across scenarios.

# Arguments
- `valuation_fn`: Function that takes a scenario Dict and returns value
- `scenarios`: Vector of scenario dictionaries
- `base_value`: Base case value for P&L calculation

# Returns
`ScenarioAnalysisResult` with values, P&L, and statistics.

# Examples
```julia
# Option valuation function
function value_option(scenario)
    spot = scenario[:spot]
    vol = scenario[:vol]
    # ... pricing logic ...
    return price
end

base = value_option(Dict(:spot => 100.0, :vol => 0.2))
result = analyze_scenarios(value_option, grid.scenarios, base)
```
"""
function analyze_scenarios(valuation_fn::Function, 
                           scenarios::Vector{Dict{Symbol, Float64}},
                           base_value::Float64)
    n = length(scenarios)
    values = zeros(n)
    pnl = zeros(n)
    
    for (i, scenario) in enumerate(scenarios)
        values[i] = valuation_fn(scenario)
        pnl[i] = values[i] - base_value
    end
    
    stats = Dict{Symbol, Float64}(
        :mean_value => mean(values),
        :std_value => std(values),
        :min_value => minimum(values),
        :max_value => maximum(values),
        :mean_pnl => mean(pnl),
        :std_pnl => std(pnl),
        :min_pnl => minimum(pnl),
        :max_pnl => maximum(pnl),
        :var_95 => -quantile(pnl, 0.05),
        :cvar_95 => -mean(filter(x -> x <= quantile(pnl, 0.05), pnl))
    )
    
    return ScenarioAnalysisResult(scenarios, values, pnl, stats)
end

# ============================================================================
# Sensitivity Tables
# ============================================================================

"""
    SensitivityTable

A 2D sensitivity table for two-factor analysis.

# Fields
- `factor1::Symbol`: First factor (rows)
- `factor2::Symbol`: Second factor (columns)
- `values1::Vector{Float64}`: Values for factor 1
- `values2::Vector{Float64}`: Values for factor 2
- `matrix::Matrix{Float64}`: Result matrix (factor1 × factor2)
"""
struct SensitivityTable
    factor1::Symbol
    factor2::Symbol
    values1::Vector{Float64}
    values2::Vector{Float64}
    matrix::Matrix{Float64}
end

"""
    sensitivity_table(valuation_fn::Function, 
                      factor1::Symbol, values1::Vector{Float64},
                      factor2::Symbol, values2::Vector{Float64};
                      base_scenario::Dict{Symbol, Float64}=Dict{Symbol, Float64}()) -> SensitivityTable

Generate a 2D sensitivity table for two factors.

# Arguments
- `valuation_fn`: Function taking scenario Dict, returns value
- `factor1`, `values1`: First factor and its values (rows)
- `factor2`, `values2`: Second factor and its values (columns)
- `base_scenario`: Base values for other factors

# Returns
`SensitivityTable` with the value matrix.

# Examples
```julia
# Spot vs Vol sensitivity
table = sensitivity_table(
    price_option,
    :spot, collect(80.0:5.0:120.0),
    :vol, collect(0.10:0.05:0.40);
    base_scenario = Dict(:rate => 0.05, :expiry => 1.0)
)
```
"""
function sensitivity_table(valuation_fn::Function,
                           factor1::Symbol, values1::Vector{Float64},
                           factor2::Symbol, values2::Vector{Float64};
                           base_scenario::Dict{Symbol, Float64}=Dict{Symbol, Float64}())
    n1, n2 = length(values1), length(values2)
    matrix = zeros(n1, n2)
    
    for (i, v1) in enumerate(values1)
        for (j, v2) in enumerate(values2)
            scenario = copy(base_scenario)
            scenario[factor1] = v1
            scenario[factor2] = v2
            matrix[i, j] = valuation_fn(scenario)
        end
    end
    
    return SensitivityTable(factor1, factor2, values1, values2, matrix)
end

"""
    scenario_ladder(valuation_fn::Function, factor::Symbol, 
                    values::Vector{Float64}; base_scenario::Dict{Symbol, Float64}=Dict{Symbol, Float64}()) -> Vector{Float64}

Generate a 1D sensitivity ladder for a single factor.

# Examples
```julia
spot_ladder = scenario_ladder(price_option, :spot, collect(80.0:5.0:120.0))
```
"""
function scenario_ladder(valuation_fn::Function, factor::Symbol, 
                         values::Vector{Float64}; 
                         base_scenario::Dict{Symbol, Float64}=Dict{Symbol, Float64}())
    results = zeros(length(values))
    for (i, v) in enumerate(values)
        scenario = copy(base_scenario)
        scenario[factor] = v
        results[i] = valuation_fn(scenario)
    end
    return results
end
