# Julia Testing Skill

When working with Julia packages that need testing, follow these guidelines:

## Test Structure Setup

- Create proper test structure in `test/runtests.jl`
- Use Julia's standard testing framework with `@testset` and `@test`
- Add Test to `[extras]` and `[targets]` sections in Project.toml:
  ```toml
  [extras]
  Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

  [targets]
  test = ["Test"]
  ```

## Testing Conventions

- Organize tests in nested `@testset` blocks by functionality
- Use descriptive testset names: `@testset "Black-Scholes European Options" begin`
- Include both unit tests (individual functions) and integration tests (full workflows)
- Test with multiple numeric types: `Float64`, `Float32`, `BigFloat`
- Use `@test_throws` for expected error conditions

## Running Tests

- Run full test suite: `julia --project=. -e "using Pkg; Pkg.test()"`
- Run specific test file: `julia --project=. test/specific_test.jl`
- Activate package first: `julia --project=.`
- For packages like Prezo: `julia --project=Prezo -e "using Pkg; Pkg.test()"`

## Options Pricing Specific Tests

For financial derivatives packages like Prezo.jl, include:

### Mathematical Properties
- **Put-call parity**: `C - P ≈ S*exp(-q*T) - K*exp(-r*T)` for European options
- **Monotonicity**: Calls increase with spot, Puts decrease with spot
- **Volatility**: Both calls and puts increase with volatility
- **American vs European**: American option value ≥ European option value
- **Boundary conditions**:
  - Call value at S=0 should be 0
  - Put value at S→∞ should be 0
  - In-the-money options at expiry equal intrinsic value

### Numerical Accuracy
- Compare against known analytical solutions (Black-Scholes)
- Test convergence properties:
  - Binomial converges to Black-Scholes as steps increase
  - Monte Carlo error decreases as √(paths)
- Use reasonable tolerances: `@test value ≈ expected atol=0.01`

### Edge Cases
- Zero volatility (should give intrinsic value)
- Very high volatility
- Zero time to expiry
- Deep in/out of the money options
- Zero interest rate
- Zero dividend yield
- Negative strikes (should error or handle gracefully)

### Engine Comparison
- Cross-validate different pricing engines
- BlackScholes vs Binomial vs MonteCarlo for European options
- Ensure consistency within tolerance

## Test Organization

```julia
using Test
using Prezo

@testset "Prezo.jl Tests" begin
    @testset "Option Construction" begin
        # Test option types
    end

    @testset "Black-Scholes Engine" begin
        @testset "European Calls" begin
            # Specific call tests
        end
        @testset "European Puts" begin
            # Specific put tests
        end
    end

    @testset "Binomial Engine" begin
        # Binomial-specific tests
    end

    @testset "Monte Carlo Engine" begin
        # MC-specific tests
    end

    @testset "American Options" begin
        # Longstaff-Schwartz tests
    end

    @testset "Mathematical Properties" begin
        # Put-call parity, monotonicity, etc.
    end
end
```

## Best Practices

- Keep tests fast (use fewer MC paths for routine testing)
- Use fixtures or helper functions to create common test data
- Document what each test is validating
- Test both success and failure paths
- Avoid hardcoded "magic numbers" - calculate expected values or document sources
- Consider using `@test_broken` for known issues to track
- Add tests when fixing bugs to prevent regression

## Integration with Examples

- Existing `examples.jl` can be converted to tests
- Extract validation logic into test functions
- Keep examples for documentation, tests for validation
