# Numerical Methods & Validation

Expertise in numerical analysis, convergence testing, error estimation, and validation for computational finance.

## Convergence Analysis

### Understanding Convergence Rates

**Order of Convergence**: How fast error decreases as step size decreases.

```
Error ∝ h^p
```
where `h` is step size and `p` is convergence order.

**Common Orders:**
- **O(h)**: First-order (Euler method, basic finite differences)
- **O(h²)**: Second-order (Crank-Nicolson, centered differences)
- **O(1/√n)**: Monte Carlo (n = number of samples)
- **O(1/n)**: Quasi-Monte Carlo with good sequences

### Testing Convergence

**Method**: Compute solutions with varying step sizes and plot log(error) vs log(step size).

```julia
function test_convergence(
    analytical_solution::Float64,
    compute_numerical::Function,
    step_sizes::Vector{Float64}
)
    errors = Float64[]

    for h in step_sizes
        numerical = compute_numerical(h)
        error = abs(numerical - analytical_solution)
        push!(errors, error)
    end

    # Estimate convergence order
    # error ∝ h^p → log(error) ∝ p * log(h)
    log_h = log.(step_sizes)
    log_error = log.(errors)

    # Linear regression
    p = (log_error[end] - log_error[1]) / (log_h[end] - log_h[1])

    return errors, p
end
```

**Example: Binomial Convergence**
```julia
# Analytical Black-Scholes price
analytical = price(option, BlackScholes(), data)

# Test binomial convergence
steps_list = [50, 100, 200, 400, 800]
errors = Float64[]

for steps in steps_list
    binomial_price = price(option, Binomial(steps), data)
    push!(errors, abs(binomial_price - analytical))
end

# Plot or analyze errors
# Should see O(1/steps) or O(1/√steps) convergence
```

### Richardson Extrapolation

Accelerate convergence by combining solutions at different step sizes.

```julia
function richardson_extrapolation(f1, f2, h1, h2, order)
    # f1, f2: function values at step sizes h1, h2
    # order: convergence order (e.g., 2 for O(h²))
    ratio = (h1 / h2)^order
    return (ratio * f2 - f1) / (ratio - 1)
end

# Example: Improve binomial accuracy
price_50 = price(option, Binomial(50), data)
price_100 = price(option, Binomial(100), data)

# Assume O(1/steps) convergence
improved = richardson_extrapolation(price_50, price_100, 1/50, 1/100, 1)
```

## Error Bounds and Tolerances

### Absolute vs Relative Error

```julia
absolute_error = abs(computed - exact)
relative_error = abs(computed - exact) / abs(exact)
```

**When to use:**
- Absolute error: When exact value ≈ 0 or for debugging
- Relative error: For comparing accuracy across different scales

### Setting Tolerances

**Guidelines:**
- **Quick estimates**: atol=0.01, rtol=0.01 (1%)
- **Production**: atol=1e-6, rtol=1e-6 (0.0001%)
- **Research**: atol=1e-10, rtol=1e-10

**In tests:**
```julia
@test computed ≈ exact atol=1e-6
@test computed ≈ exact rtol=1e-6
@test isapprox(computed, exact, atol=1e-6, rtol=1e-6)
```

### Monte Carlo Error Bounds

**Standard Error:**
```julia
function monte_carlo_price_with_error(option, engine, data)
    (; reps) = engine
    payoffs = Vector{Float64}(undef, reps)

    # Generate payoffs
    for i in 1:reps
        path = generate_path(...)
        payoffs[i] = payoff(option, path[end])
    end

    # Statistics
    mean_payoff = mean(payoffs)
    std_payoff = std(payoffs)

    # Present value
    pv_mean = exp(-data.rate * option.expiry) * mean_payoff

    # Standard error of the mean
    stderr = std_payoff / sqrt(reps)
    pv_stderr = exp(-data.rate * option.expiry) * stderr

    # 95% confidence interval
    z = 1.96
    ci_lower = pv_mean - z * pv_stderr
    ci_upper = pv_mean + z * pv_stderr

    return (price=pv_mean, stderr=pv_stderr, ci=(ci_lower, ci_upper))
end
```

**MC Error Scales:**
- Error ∝ 1/√n
- To halve error: need 4× samples
- To reduce error by 10×: need 100× samples

## Numerical Stability

### Catastrophic Cancellation

Subtracting nearly equal numbers loses precision.

❌ **Bad:**
```julia
# For x near 0, exp(x) - 1 loses precision
function bad_approx(x)
    return (exp(x) - 1) / x
end
```

✅ **Good:**
```julia
# Use expm1(x) = exp(x) - 1, accurate for small x
function good_approx(x)
    return expm1(x) / x
end
```

### Overflow and Underflow

**Overflow**: Numbers > 1.8e308 (Float64 max)
**Underflow**: Numbers < 2.2e-308 (Float64 min)

**Common in finance:**
- `exp(large_number)` → overflow
- `exp(-(large rate) * (long time))` → underflow

**Solutions:**
```julia
# Instead of: exp(a) / (exp(a) + exp(b))
# Use log-sum-exp trick:
function stable_softmax(a, b)
    m = max(a, b)
    return exp(a - m) / (exp(a - m) + exp(b - m))
end

# For products of probabilities, work in log space
log_prob_total = sum(log.(probabilities))
prob_total = exp(log_prob_total)
```

### Condition Number

Measures sensitivity to input perturbations.

```julia
# Condition number of matrix A
using LinearAlgebra
κ = cond(A)  # κ = ||A|| * ||A^(-1)||
```

- κ ≈ 1: Well-conditioned (stable)
- κ > 10^10: Ill-conditioned (unstable)
- κ → ∞: Singular (non-invertible)

**In LSM regression**: Check condition number of basis matrix.

```julia
# Before solving: X * β = Y
κ = cond(X)
if κ > 1e10
    @warn "Regression matrix is ill-conditioned" κ
end
```

## Validation Techniques

### 1. Comparison with Analytical Solutions

```julia
@testset "Binomial vs Black-Scholes" begin
    # European option: Binomial should converge to BS
    option = EuropeanCall(100.0, 1.0)
    data = MarketData(100.0, 0.05, 0.2, 0.0)

    bs_price = price(option, BlackScholes(), data)
    binom_price = price(option, Binomial(500), data)

    @test isapprox(binom_price, bs_price, rtol=0.01)
end
```

### 2. Put-Call Parity

For European options: C - P = S*e^(-qT) - K*e^(-rT)

```julia
@testset "Put-Call Parity" begin
    call = EuropeanCall(100.0, 1.0)
    put = EuropeanPut(100.0, 1.0)

    call_price = price(call, engine, data)
    put_price = price(put, engine, data)

    (; spot, rate, div, expiry) = data
    (; strike) = call

    parity_lhs = call_price - put_price
    parity_rhs = spot * exp(-div * expiry) - strike * exp(-rate * expiry)

    @test isapprox(parity_lhs, parity_rhs, atol=0.01)
end
```

### 3. Monotonicity Tests

Prices should respect economic intuition.

```julia
@testset "Call Price Monotonicity" begin
    # Call price increases with spot
    spots = [90.0, 100.0, 110.0]
    prices = [price(call, engine, MarketData(S, r, σ, q)) for S in spots]

    @test issorted(prices)  # prices[1] < prices[2] < prices[3]

    # Call price increases with volatility
    vols = [0.1, 0.2, 0.3]
    prices = [price(call, engine, MarketData(S, r, σ, q)) for σ in vols]

    @test issorted(prices)

    # Call price decreases with strike
    strikes = [90.0, 100.0, 110.0]
    prices = [price(EuropeanCall(K, T), engine, data) for K in strikes]

    @test issorted(prices, rev=true)  # Decreasing
end
```

### 4. Boundary Conditions

Test extreme cases.

```julia
@testset "Boundary Conditions" begin
    # Call at S=0 should be ≈ 0
    data_zero = MarketData(0.01, 0.05, 0.2, 0.0)
    @test price(call, engine, data_zero) < 0.1

    # Deep ITM call ≈ intrinsic value
    data_itm = MarketData(200.0, 0.05, 0.2, 0.0)
    intrinsic = 200.0 - call.strike
    @test price(call, engine, data_itm) ≈ intrinsic atol=5.0

    # At expiry, option = intrinsic value
    call_expiring = EuropeanCall(100.0, 0.0001)
    @test price(call_expiring, engine, data_itm) ≈ intrinsic atol=0.1
end
```

### 5. Cross-Method Validation

Multiple methods should agree.

```julia
@testset "Cross-Method Validation" begin
    # All should give similar prices for European option
    bs_price = price(option, BlackScholes(), data)
    binom_price = price(option, Binomial(200), data)
    mc_price = price(option, MonteCarlo(100, 50000), data)

    # All within 1% of each other
    @test isapprox(binom_price, bs_price, rtol=0.01)
    @test isapprox(mc_price, bs_price, rtol=0.02)  # MC has more variance
end
```

### 6. Symmetry Properties

Some options have symmetry.

```julia
# Put-call symmetry for ATM options
@testset "ATM Symmetry" begin
    # For ATM (S=K) with q=0, call and put have similar time value
    data_atm = MarketData(100.0, 0.05, 0.2, 0.0)
    call = EuropeanCall(100.0, 1.0)
    put = EuropeanPut(100.0, 1.0)

    call_price = price(call, BlackScholes(), data_atm)
    put_price = price(put, BlackScholes(), data_atm)

    # With r>0, call > put (by forward difference)
    forward_diff = 100.0 * (1 - exp(-0.05 * 1.0))
    @test call_price - put_price ≈ forward_diff atol=0.01
end
```

## Variance Reduction (Monte Carlo)

### Antithetic Variates

Use Z and -Z together.

```julia
function mc_price_antithetic(option, data, n_pairs)
    sum_payoff = 0.0

    for i in 1:n_pairs
        Z = randn()

        # Path with Z
        S_up = generate_terminal_price(Z, data, option.expiry)
        payoff_up = payoff(option, S_up)

        # Path with -Z
        S_down = generate_terminal_price(-Z, data, option.expiry)
        payoff_down = payoff(option, S_down)

        # Average the pair
        sum_payoff += (payoff_up + payoff_down) / 2
    end

    mean_payoff = sum_payoff / n_pairs
    return exp(-data.rate * option.expiry) * mean_payoff
end
```

**Variance reduction**: Factor of 2-4× for smooth payoffs.

### Control Variates

Use a known solution to reduce variance.

```julia
function mc_price_control_variate(option::AmericanPut, data, n_paths)
    # Control: European put (known from Black-Scholes)
    eu_put = EuropeanPut(option.strike, option.expiry)
    control_exact = price(eu_put, BlackScholes(), data)

    american_payoffs = Float64[]
    european_payoffs = Float64[]

    for i in 1:n_paths
        path = generate_path(...)

        # American payoff (LSM or other)
        am_payoff = compute_american_payoff(path, option)
        push!(american_payoffs, am_payoff)

        # European payoff
        eu_payoff = payoff(eu_put, path[end])
        push!(european_payoffs, eu_payoff)
    end

    # Standard MC estimates
    am_mean = mean(american_payoffs)
    eu_mean = mean(european_payoffs)

    # Control variate adjustment
    covariance = cov(american_payoffs, european_payoffs)
    variance_control = var(european_payoffs)
    β = covariance / variance_control

    # Adjusted estimate
    am_adjusted = am_mean + β * (control_exact - eu_mean)

    return exp(-data.rate * option.expiry) * am_adjusted
end
```

## Regression Diagnostics (LSM)

### Check Regression Quality

```julia
function lsm_regression_diagnostics(X, Y)
    # Solve X * β = Y
    β = X \ Y

    # Predictions
    Ŷ = X * β

    # Residuals
    residuals = Y - Ŷ

    # R² (coefficient of determination)
    SS_res = sum(residuals.^2)
    SS_tot = sum((Y .- mean(Y)).^2)
    R² = 1 - SS_res / SS_tot

    # Condition number
    κ = cond(X)

    # Report
    @info "LSM Regression Diagnostics" R² κ

    if R² < 0.5
        @warn "Low R²: regression fit is poor" R²
    end

    if κ > 1e10
        @warn "Ill-conditioned basis matrix" κ
    end

    return (R²=R², condition=κ, residuals=residuals)
end
```

### Basis Function Selection

Too few: underfitting
Too many: overfitting

**Rule of thumb**: Start with 3-5 basis functions.

```julia
# Test multiple basis orders
for order in 2:6
    engine = LaguerreLSM(order, steps, paths)
    price_lsm = price(option, engine, data)
    println("Order $order: \$$price_lsm")
end

# Should stabilize around order 3-4
```

## Dealing with Discontinuities

### Barrier Options

Discontinuous payoffs cause convergence issues.

**Solutions:**
1. Increase time steps near barriers
2. Use continuity corrections
3. Use Brownian bridge for barrier checking

```julia
# Check if barrier hit using Brownian bridge
function barrier_hit_probability(S_prev, S_next, barrier, μ, σ, dt)
    # Probability that geometric Brownian motion hit barrier
    # between S_prev and S_next

    if (S_prev - barrier) * (S_next - barrier) < 0
        # Crossed barrier
        return 1.0
    end

    # Adjust for possible crossing between time steps
    # (Brownian bridge correction)
    log_barrier = log(barrier / S_prev)
    log_drift = (μ - 0.5 * σ^2) * dt
    log_next = log(S_next / S_prev)

    # Probability of hitting barrier
    p = exp(-2 * log_barrier * log_next / (σ^2 * dt))

    return min(1.0, p)
end
```

### Digital Options

Payoff has step function → poor MC convergence.

**Solutions:**
1. Use smoothed payoff
2. Increase sample size significantly
3. Use importance sampling
4. Use analytical methods (Black-Scholes for digitals)

## Quasi-Monte Carlo

Replace pseudo-random with low-discrepancy sequences.

```julia
using Sobol

function qmc_price(option, data, n_samples)
    # Sobol sequence generator
    seq = SobolSeq(1)  # 1 dimension

    sum_payoff = 0.0
    for i in 1:n_samples
        # Get next quasi-random number in [0,1]
        u = next!(seq)[1]

        # Convert to normal via inverse CDF
        Z = quantile(Normal(), u)

        # Generate price
        S_T = generate_terminal_price(Z, data, option.expiry)
        sum_payoff += payoff(option, S_T)
    end

    mean_payoff = sum_payoff / n_samples
    return exp(-data.rate * option.expiry) * mean_payoff
end
```

**Convergence**: O(1/n) or O(log(n)^d/n) vs O(1/√n) for MC.

## Testing Checklist

For any new pricing method:

1. ✓ Compare with analytical solution (if available)
2. ✓ Test convergence rate with varying step sizes
3. ✓ Verify put-call parity (European options)
4. ✓ Check American ≥ European
5. ✓ Test monotonicity (spot, vol, strike, time)
6. ✓ Verify boundary conditions
7. ✓ Cross-validate with other methods
8. ✓ Test edge cases (S=0, σ=0, T=0)
9. ✓ Profile performance
10. ✓ Document accuracy and speed trade-offs

## Further Reading

- Glasserman: "Monte Carlo Methods in Financial Engineering"
- Higham: "An Introduction to Financial Option Valuation"
- L'Ecuyer & Lemieux: "Variance Reduction via Lattice Rules"
- Longstaff & Schwartz: "Valuing American Options by Simulation" (LSM paper)
