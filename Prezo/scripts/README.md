# Prezo Development Scripts

This directory contains development, analysis, and validation scripts used during the development of Prezo.jl. These scripts are not part of the core package but are useful for testing, benchmarking, and exploring package functionality.

## Directory Structure

### `sensitivity/`

Scripts for analyzing option price sensitivity to various parameters.

- **`spot_sensitivity_test.jl`** - Analyzes how option prices change with spot price variations
- **`strike_sensitivity_test.jl`** - Analyzes option price behavior across different strike prices

**Usage:**
```bash
julia --project=.. sensitivity/spot_sensitivity_test.jl
julia --project=.. sensitivity/strike_sensitivity_test.jl
```

### `validation/`

Comprehensive validation and testing scripts for American option pricing.

- **`comprehensive_validation.jl`** - Full validation suite comparing multiple pricing engines
- **`american_options_test.jl`** - Specific tests for American option behavior and early exercise

**Usage:**
```bash
julia --project=.. validation/comprehensive_validation.jl
julia --project=.. validation/american_options_test.jl
```

### `lsm_analysis/`

In-depth analysis and testing of Longstaff-Schwartz Monte Carlo methods.

- **`basis_function_comparison_test.jl`** - Compares different polynomial basis functions (Laguerre, Chebyshev, Power, Hermite)
- **`european_lsm_analysis.jl`** - Analysis of LSM for European options
- **`european_lsm_comparison.jl`** - Compares European LSM with analytical Black-Scholes
- **`simple_basis_comparison.jl`** - Simplified basis function comparison
- **`test_atm_improvements.jl`** - Tests for at-the-money option pricing improvements
- **`test_enhanced_lsm.jl`** - Validation of enhanced LSM implementations

**Usage:**
```bash
julia --project=.. lsm_analysis/basis_function_comparison_test.jl
julia --project=.. lsm_analysis/european_lsm_analysis.jl
```

### `prototypes/`

Experimental and prototype code developed during feature exploration.

- **`improved_lsm_engines.jl`** - Prototype implementations of improved LSM engines
- **`julian_basis_functions.jl`** - Experimental Julian-style basis function designs

**Note:** Code in this directory may be experimental and not production-ready.

## Output

Scripts that generate plots or analysis results save their output to the `../output/` directory.

## Running Scripts

All scripts should be run from the `scripts/` directory using the parent project environment:

```bash
cd scripts
julia --project=.. <category>/<script_name>.jl
```

Or from the project root:

```bash
julia --project=. scripts/<category>/<script_name>.jl
```

## Adding New Scripts

When adding new development scripts:

1. Choose the appropriate subdirectory based on the script's purpose
2. Include a brief header comment explaining what the script does
3. Update this README with a description
4. Save any generated figures to `../output/`

## Categories

- **Sensitivity Analysis** → `sensitivity/`
- **Validation & Testing** → `validation/`
- **LSM Method Analysis** → `lsm_analysis/`
- **Experimental Code** → `prototypes/`

## Related Directories

- **`../archive/`** - Historical diagnostic scripts and bug reports
- **`../output/`** - Generated figures and analysis results
- **`../test/`** - Official package test suite
- **`../examples.jl`** - Main package examples
- **`../benchmark_lsm.jl`** - Performance benchmarks

## See Also

- Main package documentation: `../docs/`
- Package source code: `../src/`
- Test suite: `../test/runtests.jl`
- Benchmarks: `../benchmark_lsm.jl`
