# Julia Package Development & CI/CD

Expertise in Julia package development, testing, versioning, and continuous integration.

## Package Structure

### Standard Layout

```
Prezo/
├── Project.toml          # Package metadata, dependencies
├── Manifest.toml         # Locked dependency versions (gitignore for libraries)
├── README.md             # Package description
├── LICENSE              # MIT, BSD, Apache, etc.
├── .gitignore
├── src/
│   ├── Prezo.jl         # Main module file
│   ├── data.jl          # Market data types
│   ├── options.jl       # Option definitions
│   └── engines.jl       # Pricing engines
├── test/
│   └── runtests.jl      # Test entry point
├── docs/
│   ├── make.jl
│   └── src/
├── examples/
│   ├── basic_pricing.jl
│   └── american_options.jl
└── .github/
    └── workflows/
        ├── CI.yml
        └── Docs.yml
```

## Project.toml

### Essential Fields

```toml
name = "Prezo"
uuid = "b8def1d8-96d4-4d03-b88c-856be0d8edff"
version = "0.1.0"
authors = ["Your Name <email@example.com>"]

[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
julia = "1.9"
Distributions = "0.25"

[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Test"]
```

### Compat Bounds

**Be specific but not restrictive:**

```toml
[compat]
julia = "1.9"           # Minimum Julia version
Distributions = "0.25"  # Allow 0.25.x
LinearAlgebra = "1"     # Allow any 1.x
```

**Semantic versioning:**
- `"0.25"` = `"0.25.0 - 0.26.0"` (not including 0.26)
- `"1"` = `"1.0.0 - 2.0.0"` (not including 2.0)
- `"1.2.3"` = exactly 1.2.3

## Main Module File (src/Prezo.jl)

```julia
module Prezo

# Dependencies
using Distributions
using LinearAlgebra
using Statistics

# Exports - organized by category
export MarketData

export VanillaOption, EuropeanOption, AmericanOption
export EuropeanCall, EuropeanPut, AmericanCall, AmericanPut
export payoff

export BlackScholes, Binomial, MonteCarlo, LongstaffSchwartz
export price, greeks

# Include source files
include("data.jl")
include("options.jl")
include("engines.jl")

# Precompilation hints (optional but recommended)
if Base.VERSION >= v"1.9"
    include("precompile.jl")
end

end # module
```

## Testing

### test/runtests.jl

```julia
using Test
using Prezo

@testset "Prezo.jl" begin
    @testset "Market Data" begin
        include("test_data.jl")
    end

    @testset "Options" begin
        include("test_options.jl")
    end

    @testset "Black-Scholes" begin
        include("test_blackscholes.jl")
    end

    @testset "Binomial" begin
        include("test_binomial.jl")
    end

    @testset "Monte Carlo" begin
        include("test_montecarlo.jl")
    end

    @testset "American Options" begin
        include("test_american.jl")
    end
end
```

### Running Tests

```bash
# From package directory
julia --project -e 'using Pkg; Pkg.test()'

# Specific test file
julia --project test/test_blackscholes.jl

# With coverage
julia --project --code-coverage=user -e 'using Pkg; Pkg.test()'
```

## Continuous Integration

### .github/workflows/CI.yml

```yaml
name: CI

on:
  push:
    branches:
      - main
  pull_request:
  workflow_dispatch:

jobs:
  test:
    name: Julia ${{ matrix.version }} - ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        version:
          - '1.9'    # Minimum supported version
          - '1'      # Latest stable
        os:
          - ubuntu-latest
          - macos-latest
          - windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v1
        with:
          version: ${{ matrix.version }}
      - uses: julia-actions/cache@v1
      - uses: julia-actions/julia-buildpkg@v1
      - uses: julia-actions/julia-runtest@v1
      - uses: julia-actions/julia-processcoverage@v1
      - uses: codecov/codecov-action@v3
        with:
          files: lcov.info
```

### Code Coverage

Add badges to README.md:

```markdown
[![Coverage](https://codecov.io/gh/username/Prezo.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/username/Prezo.jl)
```

## Semantic Versioning

Follow SemVer: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes (0.x.y → 1.0.0, 1.x.y → 2.0.0)
- **MINOR**: New features, backward compatible (1.2.3 → 1.3.0)
- **PATCH**: Bug fixes (1.2.3 → 1.2.4)

### Version Bumping

```bash
# Edit Project.toml: version = "0.1.1"
git add Project.toml
git commit -m "Bump version to 0.1.1"
git tag v0.1.1
git push --tags
```

### Breaking Changes

Document in CHANGELOG.md:

```markdown
# Changelog

## v2.0.0 (2024-03-15)

### Breaking Changes
- `price()` now requires `MarketData` instead of individual parameters
- Removed deprecated `old_function()`

### Added
- New `greeks()` function for sensitivity analysis
- Support for American options with LSM

### Fixed
- Dividend handling in Black-Scholes (issue #42)
```

## Registering a Package

### With LocalRegistry (Development)

```julia
using LocalRegistry
register("Prezo")
```

### With General Registry (Public)

1. Create GitHub repo
2. Tag a version: `git tag v0.1.0 && git push --tags`
3. Comment on GitHub: `@JuliaRegistrator register`
4. Wait for CI checks
5. Merge the PR in General registry

### Requirements for Registration

- All tests pass
- Project.toml has version, uuid, compat bounds
- Has a LICENSE file
- README with basic description
- No absolute paths in code

## Precompilation

### src/precompile.jl

```julia
using PrecompileTools

@setup_workload begin
    # Setup code (runs at precompile time)
    data = MarketData(100.0, 0.05, 0.2, 0.0)
    option_call = EuropeanCall(100.0, 1.0)
    option_put = EuropeanPut(100.0, 1.0)

    @compile_workload begin
        # Code to precompile
        price(option_call, BlackScholes(), data)
        price(option_put, BlackScholes(), data)
        price(option_call, Binomial(50), data)
        price(option_put, Binomial(50), data)
    end
end
```

Reduces time-to-first-plot (TTFP) / time-to-first-execution.

## Benchmarking

### benchmark/benchmarks.jl

```julia
using BenchmarkTools
using Prezo

const SUITE = BenchmarkGroup()

# Market data
data = MarketData(100.0, 0.05, 0.2, 0.0)
option = EuropeanCall(100.0, 1.0)

# Black-Scholes benchmarks
SUITE["BlackScholes"] = @benchmarkable price($option, BlackScholes(), $data)

# Binomial benchmarks
for steps in [50, 100, 200]
    SUITE["Binomial"]["steps=$steps"] = @benchmarkable price($option, Binomial($steps), $data)
end

# Monte Carlo benchmarks
SUITE["MonteCarlo"] = BenchmarkGroup()
for paths in [1000, 10000]
    SUITE["MonteCarlo"]["paths=$paths"] = @benchmarkable price($option, MonteCarlo(100, $paths), $data)
end
```

### Running Benchmarks

```julia
using PkgBenchmark

# Run benchmarks
results = benchmarkpkg("Prezo")

# Compare against main branch
results = judge("Prezo", "main")

# Export results
export_markdown("benchmark_results.md", results)
```

## Dependency Management

### Adding Dependencies

```bash
# Activate package environment
julia --project

# Add dependency
] add Distributions

# Add test-only dependency
] add --extras Test
```

### Updating Dependencies

```bash
] update              # Update all
] update Distributions  # Update specific
```

### Removing Dependencies

```bash
] rm Distributions
```

### Checking Compatibility

```bash
] compat
```

## Package Extensions (Julia 1.9+)

For optional dependencies:

```toml
# Project.toml
[weakdeps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

[extensions]
PrezoCUDAExt = "CUDA"
```

```julia
# ext/PrezoCUDAExt/PrezoCUDAExt.jl
module PrezoCUDAExt

using Prezo
using CUDA

# GPU-accelerated implementations
function Prezo.price(option, engine::MonteCarlo, data::MarketData, ::CUDABackend)
    # CUDA implementation
end

end
```

## Development Workflow

### Working on Local Copy

```bash
# Develop mode (links to local directory)
] dev /path/to/Prezo.jl

# Make changes, test
julia --project=. test/runtests.jl

# Return to registered version
] free Prezo
```

### With Revise.jl (Recommended)

```julia
using Revise
using Prezo

# Make changes to source files
# Functions automatically reload!
```

## Common Pitfalls

### 1. Manifest.toml in Git

- **Libraries**: Add to `.gitignore`
- **Applications**: Commit for reproducibility

### 2. Circular Dependencies

```julia
# Bad: A depends on B, B depends on A
# Solution: Extract common code to C
```

### 3. Type Piracy

```julia
# Bad: extending Base.+ for types you don't own
Base.:+(::Float64, ::Float64) = ...  # DON'T DO THIS

# Good: own at least one type
Base.:+(::MyType, ::Float64) = ...
```

### 4. Globals in Tests

```julia
# Bad
data = MarketData(...)  # Global

@testset "Test" begin
    price(option, engine, data)  # Slower due to global
end

# Good
@testset "Test" begin
    data = MarketData(...)  # Local
    price(option, engine, data)
end
```

## CI Best Practices

### Cache Dependencies

```yaml
- uses: julia-actions/cache@v1
```

### Test on Multiple Platforms

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
    julia: ['1.9', '1']
```

### Fail Fast (or not)

```yaml
strategy:
  fail-fast: false  # Continue other tests if one fails
```

### Test Documentation Builds

```yaml
- name: Test docs build
  run: julia --project=docs docs/make.jl
```

## Release Checklist

Before releasing a new version:

1. ✓ All tests pass locally and on CI
2. ✓ Update CHANGELOG.md
3. ✓ Bump version in Project.toml
4. ✓ Update documentation if needed
5. ✓ Run benchmarks to check for regressions
6. ✓ Test with downstream packages
7. ✓ Create git tag
8. ✓ Push tag to GitHub
9. ✓ Create GitHub release with notes
10. ✓ Register with Julia General (if public)

## Further Reading

- Pkg.jl documentation: https://pkgdocs.julialang.org/
- PkgTemplates.jl: Bootstrap new packages
- BenchmarkTools.jl: Performance testing
- PrecompileTools.jl: Reduce latency
