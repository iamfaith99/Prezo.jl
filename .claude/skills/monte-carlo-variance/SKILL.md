# Monte Carlo Variance Reduction

Expertise in advanced variance reduction techniques for Monte Carlo option pricing.

## Why Variance Reduction Matters

Standard Monte Carlo error: **Error ∝ 1/√n**

To halve error → need 4× samples → 4× computational time

**Variance reduction can achieve same accuracy with 10-100× fewer samples.**

## Antithetic Variates

### Principle

Use pairs (Z, -Z) together. If f is monotonic, f(Z) and f(-Z) are negatively correlated.

### Implementation

```julia
function price_antithetic(
    option::EuropeanOption,
    data::MarketData,
    n_pairs::Int
)
    (; spot, rate, vol, div) = data
    (; expiry) = option

    drift = (rate - div - 0.5 * vol^2) * expiry
    vol_sqrt_T = vol * sqrt(expiry)
    disc = exp(-rate * expiry)

    sum_payoff = 0.0

    for i in 1:n_pairs
        Z = randn()

        # Path with +Z
        S_up = spot * exp(drift + vol_sqrt_T * Z)
        payoff_up = payoff(option, S_up)

        # Path with -Z
        S_down = spot * exp(drift - vol_sqrt_T * Z)
        payoff_down = payoff(option, S_down)

        # Average the pair
        sum_payoff += (payoff_up + payoff_down) / 2
    end

    return disc * sum_payoff / n_pairs
end
```

### Effectiveness

**Best for**: Smooth, monotonic payoffs (vanilla options)
**Variance reduction**: Factor of 2-4
**Works poorly for**: Non-monotonic payoffs (e.g., butterfly spread)

### Multi-step Extension

For path-dependent options, use antithetic paths:

```julia
function generate_antithetic_paths(
    spot::Float64,
    rate::Float64,
    vol::Float64,
    div::Float64,
    expiry::Float64,
    n_pairs::Int,
    n_steps::Int
)
    dt = expiry / n_steps
    drift = (rate - div - 0.5 * vol^2) * dt
    vol_sqrt_dt = vol * sqrt(dt)

    # Pre-allocate
    paths_up = Matrix{Float64}(undef, n_pairs, n_steps + 1)
    paths_down = Matrix{Float64}(undef, n_pairs, n_steps + 1)

    paths_up[:, 1] .= spot
    paths_down[:, 1] .= spot

    for step in 2:(n_steps + 1)
        for i in 1:n_pairs
            Z = randn()

            # Antithetic pair
            paths_up[i, step] = paths_up[i, step-1] * exp(drift + vol_sqrt_dt * Z)
            paths_down[i, step] = paths_down[i, step-1] * exp(drift - vol_sqrt_dt * Z)
        end
    end

    return paths_up, paths_down
end
```

## Control Variates

### Principle

Use a correlated random variable Y with known expectation E[Y].

**Adjusted estimator**: X̂ = X + c(Y - E[Y])

**Optimal c**: c* = -Cov(X,Y) / Var(Y)

**Variance reduction**: Var(X̂) = Var(X)(1 - ρ²_{X,Y})

### Implementation

```julia
function price_control_variate(
    option::AmericanPut,
    data::MarketData,
    engine::LongstaffSchwartz
)
    # Control: European put (known from Black-Scholes)
    eu_put = EuropeanPut(option.strike, option.expiry)
    control_exact = price(eu_put, BlackScholes(), data)

    # Generate paths
    (; reps, steps) = engine
    paths = generate_paths(data, expiry, reps, steps)

    # Compute payoffs
    american_payoffs = Float64[]
    european_payoffs = Float64[]

    for i in 1:reps
        # American payoff (using LSM logic)
        am_payoff = compute_american_payoff(paths[i, :], option, data)
        push!(american_payoffs, am_payoff)

        # European payoff
        eu_payoff = payoff(eu_put, paths[i, end])
        push!(european_payoffs, eu_payoff)
    end

    # Compute optimal coefficient
    β = -cov(american_payoffs, european_payoffs) / var(european_payoffs)

    # Standard estimates
    am_mean = mean(american_payoffs)
    eu_mean = mean(european_payoffs)

    # Control variate adjustment
    am_adjusted = am_mean + β * (control_exact - eu_mean)

    disc = exp(-data.rate * option.expiry)
    return disc * am_adjusted
end
```

### Multiple Control Variates

Use several correlated controls:

```julia
function price_multiple_controls(
    option::VanillaOption,
    data::MarketData,
    n_paths::Int,
    controls::Vector{Tuple{VanillaOption, Float64}}  # (control_option, exact_price)
)
    # Generate paths
    payoffs_target = Float64[]
    payoffs_controls = [Float64[] for _ in controls]

    for i in 1:n_paths
        path = generate_path(data, option.expiry)

        # Target payoff
        push!(payoffs_target, payoff(option, path[end]))

        # Control payoffs
        for (j, (ctrl_opt, _)) in enumerate(controls)
            push!(payoffs_controls[j], payoff(ctrl_opt, path[end]))
        end
    end

    # Build regression matrix
    n_controls = length(controls)
    X = hcat([payoffs_controls[j] for j in 1:n_controls]...)
    Y = payoffs_target

    # Optimal coefficients (via regression)
    β = X \ Y

    # Control variate adjustment
    target_mean = mean(payoffs_target)
    control_means = [mean(pc) for pc in payoffs_controls]
    control_exacts = [price for (_, price) in controls]

    adjustment = sum(β[j] * (control_exacts[j] - control_means[j])
                     for j in 1:n_controls)

    adjusted_mean = target_mean + adjustment

    disc = exp(-data.rate * option.expiry)
    return disc * adjusted_mean
end
```

## Importance Sampling

### Principle

Sample from a different distribution that concentrates on important regions.

**Original**: E[f(X)] where X ~ p(x)
**IS**: E[f(X)w(X)] where X ~ q(x), w(x) = p(x)/q(x)

### For Deep OTM Options

```julia
function price_importance_sampling(
    option::EuropeanCall,
    data::MarketData,
    n_paths::Int;
    drift_shift::Float64=0.5  # Shift towards ITM
)
    (; spot, rate, vol, div, strike, expiry) = (data..., option...)

    # Original drift (risk-neutral)
    μ_original = rate - div - 0.5 * vol^2

    # Shifted drift (towards strike)
    μ_shifted = μ_original + drift_shift * vol

    vol_sqrt_T = vol * sqrt(expiry)
    disc = exp(-rate * expiry)

    sum_weighted_payoff = 0.0

    for i in 1:n_paths
        Z = randn()

        # Sample from shifted distribution
        log_S_T = log(spot) + μ_shifted * expiry + vol_sqrt_T * Z
        S_T = exp(log_S_T)

        # Payoff
        poff = max(0, S_T - strike)

        # Likelihood ratio (weight)
        # w = p(Z | original) / p(Z | shifted)
        # For Gaussian: exp(-0.5 * [(Z - μ₁/σ)² - (Z - μ₂/σ)²])
        drift_diff = (μ_original - μ_shifted) * expiry / vol_sqrt_T
        weight = exp(Z * drift_diff - 0.5 * drift_diff^2)

        sum_weighted_payoff += poff * weight
    end

    return disc * sum_weighted_payoff / n_paths
end
```

**Choose shift**:
- For OTM call: positive drift (move towards higher prices)
- For OTM put: negative drift (move towards lower prices)
- Optimal shift ≈ vol * Φ⁻¹(P(ITM))

## Stratified Sampling

### Principle

Divide sample space into strata, sample proportionally from each.

### One-Dimensional

```julia
function price_stratified_1d(
    option::EuropeanOption,
    data::MarketData,
    n_strata::Int,
    samples_per_stratum::Int
)
    (; spot, rate, vol, div) = data
    (; expiry) = option

    drift = (rate - div - 0.5 * vol^2) * expiry
    vol_sqrt_T = vol * sqrt(expiry)
    disc = exp(-rate * expiry)

    sum_payoff = 0.0

    for stratum in 1:n_strata
        # Stratum bounds in uniform space
        u_low = (stratum - 1) / n_strata
        u_high = stratum / n_strata

        stratum_sum = 0.0

        for _ in 1:samples_per_stratum
            # Sample uniformly within stratum
            u = u_low + (u_high - u_low) * rand()

            # Convert to normal via inverse CDF
            Z = quantile(Normal(), u)

            # Generate price
            S_T = spot * exp(drift + vol_sqrt_T * Z)

            # Payoff
            stratum_sum += payoff(option, S_T)
        end

        sum_payoff += stratum_sum / samples_per_stratum
    end

    return disc * sum_payoff / n_strata
end
```

### Latin Hypercube Sampling

For multi-dimensional:

```julia
using LatinHypercubeSampling

function price_latin_hypercube(
    option::VanillaOption,
    data::MarketData,
    n_steps::Int,
    n_samples::Int
)
    # Generate LHS samples in [0,1]^n_steps
    plan = LHCoptim(n_samples, n_steps, 1000)[1]  # Optimize design

    # Scale to unit hypercube
    samples = scaleLHC(plan, [(0.0, 1.0) for _ in 1:n_steps])

    # Convert to normal
    Z_samples = [quantile(Normal(), samples[i, j])
                 for i in 1:n_samples, j in 1:n_steps]

    # Generate paths and price
    # ... (similar to standard MC but using Z_samples)
end
```

## Quasi-Monte Carlo

### Principle

Replace pseudo-random with low-discrepancy sequences.

**Error**: O((log n)^d / n) vs O(1/√n) for MC

### Sobol Sequences

```julia
using Sobol

function price_qmc_sobol(
    option::EuropeanOption,
    data::MarketData,
    n_samples::Int
)
    (; spot, rate, vol, div) = data
    (; expiry) = option

    drift = (rate - div - 0.5 * vol^2) * expiry
    vol_sqrt_T = vol * sqrt(expiry)
    disc = exp(-rate * expiry)

    # Sobol sequence
    seq = SobolSeq(1)  # 1-dimensional
    skip(seq, n_samples)  # Skip first samples (warmup)

    sum_payoff = 0.0

    for i in 1:n_samples
        # Get next quasi-random number
        u = next!(seq)[1]

        # Convert to normal
        Z = quantile(Normal(), u)

        # Generate price
        S_T = spot * exp(drift + vol_sqrt_T * Z)

        sum_payoff += payoff(option, S_T)
    end

    return disc * sum_payoff / n_samples
end
```

### Halton Sequences

```julia
function halton(index::Int, base::Int)
    result = 0.0
    f = 1.0 / base
    i = index

    while i > 0
        result += f * (i % base)
        i = div(i, base)
        f /= base
    end

    return result
end

function price_qmc_halton(
    option::EuropeanOption,
    data::MarketData,
    n_samples::Int
)
    # Similar to Sobol, but use Halton sequence
    # u_i = halton(i, 2)  # Base-2 Halton
end
```

## Moment Matching

### Principle

Adjust samples to match exact moments (mean, variance).

```julia
function moment_matched_normals(n::Int)
    # Generate n standard normals
    Z = randn(n)

    # Adjust to exact mean=0, var=1
    Z_adjusted = (Z .- mean(Z)) ./ std(Z)

    return Z_adjusted
end

function price_moment_matched(
    option::EuropeanOption,
    data::MarketData,
    n_paths::Int
)
    Z = moment_matched_normals(n_paths)

    # Use Z for pricing (same as standard MC)
    # ...
end
```

## Conditional Monte Carlo

### Principle

Condition on some variables, compute expectation analytically for others.

### Example: Asian Option with Geometric

```julia
function price_asian_conditional(
    option::AsianOption,  # Arithmetic average
    data::MarketData,
    n_paths::Int
)
    # Use geometric average as control
    # E[Arithmetic | Geometric] can be approximated

    # Generate paths, compute both averages
    arith_payoffs = Float64[]
    geom_payoffs = Float64[]

    for i in 1:n_paths
        path = generate_path(...)

        arith_avg = mean(path)
        geom_avg = exp(mean(log.(path)))

        push!(arith_payoffs, max(0, arith_avg - option.strike))
        push!(geom_payoffs, max(0, geom_avg - option.strike))
    end

    # Geometric has closed form
    geom_exact = price_geometric_asian(option, data)

    # Control variate adjustment
    # ...
end
```

## Combining Techniques

### Antithetic + Control Variates

```julia
function price_combined_av_cv(
    option::AmericanPut,
    data::MarketData,
    n_pairs::Int
)
    eu_put = EuropeanPut(option.strike, option.expiry)
    control_exact = price(eu_put, BlackScholes(), data)

    american_payoffs = Float64[]
    european_payoffs = Float64[]

    for i in 1:n_pairs
        # Generate antithetic pair
        Z = randn(n_steps)

        # Path with +Z
        path_up = generate_path_from_z(Z, ...)
        am_up = american_payoff(path_up, ...)
        eu_up = payoff(eu_put, path_up[end])

        # Path with -Z
        path_down = generate_path_from_z(-Z, ...)
        am_down = american_payoff(path_down, ...)
        eu_down = payoff(eu_put, path_down[end])

        # Average antithetic pairs
        push!(american_payoffs, (am_up + am_down) / 2)
        push!(european_payoffs, (eu_up + eu_down) / 2)
    end

    # Apply control variate
    β = -cov(american_payoffs, european_payoffs) / var(european_payoffs)
    am_mean = mean(american_payoffs)
    eu_mean = mean(european_payoffs)

    adjusted = am_mean + β * (control_exact - eu_mean)

    return exp(-data.rate * option.expiry) * adjusted
end
```

## Measuring Effectiveness

### Variance Reduction Factor

```julia
function compare_variance_reduction(
    option::VanillaOption,
    data::MarketData,
    n_paths::Int,
    n_runs::Int=100
)
    # Standard MC
    std_prices = [price_standard_mc(option, data, n_paths) for _ in 1:n_runs]
    std_variance = var(std_prices)

    # With variance reduction
    av_prices = [price_antithetic(option, data, n_paths÷2) for _ in 1:n_runs]
    av_variance = var(av_prices)

    # Variance reduction factor
    vrf = std_variance / av_variance

    println("Standard MC variance: $std_variance")
    println("Antithetic variance: $av_variance")
    println("Variance reduction factor: $vrf")
    println("Effective sample multiplier: $(vrf)x")

    return vrf
end
```

## When to Use What

| Technique | Best For | Variance Reduction | Complexity |
|-----------|----------|-------------------|------------|
| Antithetic | Smooth, monotonic | 2-4× | Low |
| Control Variate | Related known solution | 5-20× | Medium |
| Importance Sampling | Deep OTM, rare events | 10-100× | High |
| Stratified | Low dimension | 3-10× | Medium |
| Quasi-MC | Smooth, low dimension | Dimension-dependent | Low |
| Moment Matching | Any | 2-3× | Low |
| Conditional MC | Partial analytical | Varies | High |

## Practical Recommendations

1. **Always use antithetic**: Almost free, works for most options
2. **Use control variates** when related option has closed form
3. **Quasi-MC (Sobol)** for European options with <10 dimensions
4. **Importance sampling** only for deep OTM or barrier options
5. **Combine techniques**: Antithetic + Control often gives 10-50× reduction

## Further Reading

- Glasserman: "Monte Carlo Methods in Financial Engineering" (comprehensive)
- L'Ecuyer & Lemieux: "Variance Reduction via Lattice Rules"
- Broadie & Glasserman: "Pricing American Options by Simulation"
- Sobol: "Uniformly distributed sequences" (QMC theory)
