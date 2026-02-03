using Distributions: Normal, quantile
using Plots
using Random
using Base.Threads

"""
    asset_paths(engine::MonteCarlo, spot, rate, vol, expiry)

Generate asset price paths using geometric Brownian motion.

Returns a matrix of simulated asset prices with dimensions `(reps, steps+1)`.

# Arguments
- `engine::MonteCarlo`: Monte Carlo engine with steps and reps
- `spot`: Initial spot price
- `rate`: Risk-free interest rate
- `vol`: Volatility
- `expiry`: Time to expiration

# Returns
Matrix of size `(reps, steps+1)` where each row is a simulated path.

# Examples
```julia
engine = MonteCarlo(100, 1000)
paths = asset_paths(engine, 100.0, 0.05, 0.2, 1.0)
size(paths)  # (1000, 101)
```

See also: [`asset_paths_col`](@ref), [`asset_paths_ax`](@ref)
"""
function asset_paths(engine::MonteCarlo, spot, rate, vol, expiry)
    (; steps, reps) = engine

    dt = expiry / steps
    nudt = (rate - 0.5 * vol^2) * dt
    sidt = vol * sqrt(dt)
    paths = zeros(reps, steps + 1)
    paths[:, 1] .= spot

    @inbounds for i in 1:reps
        @inbounds for j in 2:steps+1
            z = randn()
            paths[i, j] = paths[i, j-1] * exp(nudt + sidt * z)
        end
    end

    return paths
end

"""
    asset_paths_col(engine::MonteCarlo, spot, rate, vol, expiry)

Generate asset price paths with column-major layout.

Returns a matrix of simulated asset prices with dimensions `(steps+1, reps)`.
This layout is preferred for time-based operations.

# Returns
Matrix of size `(steps+1, reps)` where each column is a simulated path.

See also: [`asset_paths`](@ref), [`asset_paths_ax`](@ref)
"""
function asset_paths_col(engine::MonteCarlo, spot, rate, vol, expiry)
    (; steps, reps) = engine

    dt = expiry / steps
    nudt = (rate - 0.5 * vol^2) * dt
    sidt = vol * sqrt(dt)
    paths = zeros(steps + 1, reps)
    paths[1, :] .= spot

    @inbounds for i in 1:reps
        @inbounds for j in 2:steps+1
            z = randn()
            paths[j, i] = paths[j-1, i] * exp(nudt + sidt * z)
        end
    end

    return paths
end

"""
    asset_paths_col_threaded(engine::MonteCarlo, spot, rate, vol, expiry)

Generate asset price paths with column-major layout using `Threads.@threads` over paths.

Same layout as [`asset_paths_col`](@ref): `(steps+1, reps)`. Use when `reps` is large
and `JULIA_NUM_THREADS` > 1. Each thread uses task-local RNG (Julia 1.7+).

# Examples
```julia
# Start Julia with: julia -t 4
paths = asset_paths_col_threaded(engine, 100.0, 0.05, 0.2, 1.0)
```
"""
function asset_paths_col_threaded(engine::MonteCarlo, spot, rate, vol, expiry)
    (; steps, reps) = engine
    dt = expiry / steps
    nudt = (rate - 0.5 * vol^2) * dt
    sidt = vol * sqrt(dt)
    paths = zeros(steps + 1, reps)
    paths[1, :] .= spot

    Threads.@threads for i in 1:reps
        @inbounds for j in 2:steps+1
            z = randn()
            paths[j, i] = paths[j-1, i] * exp(nudt + sidt * z)
        end
    end
    return paths
end

"""
    asset_paths_ax(engine::MonteCarlo, spot, rate, vol, expiry)

Generate asset price paths using axis-based iteration.

More idiomatic Julia implementation using `axes()` for iteration.

See also: [`asset_paths`](@ref), [`asset_paths_col`](@ref)
"""
function asset_paths_ax(engine::MonteCarlo, spot, rate, vol, expiry)
    (; steps, reps) = engine

    dt = expiry / steps
    nudt = (rate - 0.5 * vol^2) * dt
    sidt = vol * sqrt(dt)
    paths = zeros(steps + 1, reps)
    paths[1, :] .= spot

    @inbounds for j in axes(paths, 2), i in 2:last(axes(paths, 1))
        z = randn()
        paths[i, j] = paths[i-1, j] * exp(nudt + sidt * z)
    end

    return paths
end

"""
    asset_paths_antithetic(steps, reps, spot, rate, vol, expiry)

Generate asset price paths using antithetic variates for variance reduction.

For each random number Z, also generates a path with -Z, creating negatively
correlated paths that reduce Monte Carlo variance. Returns `2×reps` paths total.

# Arguments
- `steps::Int`: Number of time steps
- `reps::Int`: Number of path pairs (total paths = 2×reps)
- `spot`: Initial spot price
- `rate`: Risk-free rate
- `vol`: Volatility
- `expiry`: Time to expiration

# Returns
Matrix of size `(steps+1, 2×reps)` where paths `[i, j]` and `[i, j+reps]` are antithetic.

# Examples
```julia
paths = asset_paths_antithetic(100, 5000, 100.0, 0.05, 0.2, 1.0)
size(paths)  # (101, 10000)
```

See also: [`asset_paths_col`](@ref), [`MonteCarloAntithetic`](@ref)
"""
function asset_paths_antithetic(
    steps::Int,
    reps::Int,
    spot::Real,
    rate::Real,
    vol::Real,
    expiry::Real
)
    dt = expiry / steps
    nudt = (rate - 0.5 * vol^2) * dt
    sidt = vol * sqrt(dt)

    # Allocate for 2× paths (each pair is antithetic)
    paths = zeros(steps + 1, 2 * reps)
    paths[1, :] .= spot

    # Generate antithetic pairs
    @inbounds for j in 1:reps
        @inbounds for i in 2:steps+1
            z = randn()
            paths[i, j] = paths[i-1, j] * exp(nudt + sidt * z)
            paths[i, j+reps] = paths[i-1, j+reps] * exp(nudt - sidt * z)
        end
    end

    return paths
end

"""
    plot_paths(paths, num)

Visualize simulated asset price paths.

# Arguments
- `paths`: Matrix of simulated paths (from `asset_paths`)
- `num`: Number of paths to plot

# Examples
```julia
engine = MonteCarlo(100, 1000)
paths = asset_paths(engine, 100.0, 0.05, 0.2, 1.0)
plot_paths(paths, 10)
```

See also: [`asset_paths`](@ref), [`MonteCarlo`](@ref)
"""
function plot_paths(paths, num)
    steps = size(paths, 2) - 1

    plot(0:steps, paths[1, :], label="", legend=false)

    for i in 2:num
        plot!(0:steps, paths[i, :], label="", legend=false)
    end

    xaxis!("Time step")
    yaxis!("Asset price")
    title!("First $(num) Simulated Paths")
end

"""
    stratified_normal(N)

Generate stratified standard normal samples.

Returns a length-`N` vector of stratified draws from `Normal(0, 1)`.

See also: [`asset_paths_stratified`](@ref)
"""
function stratified_normal(N)
    d = Normal(0.0, 1.0)
    zhat = zeros(N)

    for i in 1:N
        u = rand()
        uhat = (i - 1 + u) / N
        zhat[i] = quantile(d, uhat)
    end

    return zhat
end

"""
    asset_paths_stratified(steps, reps, spot, rate, vol, expiry)

Generate asset price paths using stratified sampling for variance reduction.

# Returns
Matrix of size `(steps+1, reps)` where each column is a simulated path.

See also: [`stratified_normal`](@ref)
"""
function asset_paths_stratified(
    steps::Int,
    reps::Int,
    spot::Real,
    rate::Real,
    vol::Real,
    expiry::Real
)
    dt = expiry / steps
    nudt = (rate - 0.5 * vol^2) * dt
    sidt = vol * sqrt(dt)

    paths = zeros(steps + 1, reps)
    paths[1, :] .= spot

    @inbounds for i in 2:steps+1
        z = stratified_normal(reps)
        Random.shuffle!(z)

        @inbounds for j in 1:reps
            paths[i, j] = paths[i-1, j] * exp(nudt + sidt * z[j])
        end
    end

    return paths
end
