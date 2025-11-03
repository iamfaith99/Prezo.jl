# Julia Performance Optimization

Expertise in writing high-performance Julia code for numerical computing and financial applications.

## Core Principles

### 1. Type Stability is Critical

**Type stability**: A function is type-stable if the type of the output can be inferred from the types of the inputs.

**Check with `@code_warntype`:**
```julia
@code_warntype my_function(args...)
```
- Red/yellow warnings = type instability
- Goal: all blue output (inferred types)

**Common Issues:**

❌ **Bad - Type Unstable:**
```julia
function compute_price(use_high_vol::Bool)
    vol = use_high_vol ? 0.3 : 0.2  # Float64
    # ... later ...
    if use_high_vol
        return vol  # Returns Float64
    else
        return "low"  # Returns String - TYPE UNSTABLE!
    end
end
```

✅ **Good - Type Stable:**
```julia
function compute_price(vol::Float64)
    # Always returns Float64
    return calculate_value(vol)
end
```

### 2. Avoid Abstract Types in Containers

❌ **Bad - Abstract types:**
```julia
struct MarketData
    spot::AbstractFloat  # Abstract!
    rate::AbstractFloat
end

prices = Vector{AbstractFloat}()  # Abstract!
```

✅ **Good - Concrete types:**
```julia
struct MarketData{T<:AbstractFloat}
    spot::T
    rate::T
end

prices = Vector{Float64}()
```

### 3. Pre-allocate Arrays

**Key Rule**: Allocate once outside loops, fill inside.

❌ **Bad - Allocates every iteration:**
```julia
function monte_carlo_slow(n_paths, n_steps)
    results = Float64[]
    for i in 1:n_paths
        path = zeros(n_steps)  # ALLOCATION IN LOOP!
        for j in 1:n_steps
            path[j] = simulate_step()
        end
        push!(results, final_value(path))  # GROWING ARRAY!
    end
    return results
end
```

✅ **Good - Pre-allocated:**
```julia
function monte_carlo_fast(n_paths, n_steps)
    results = Vector{Float64}(undef, n_paths)  # Pre-allocate
    path = Vector{Float64}(undef, n_steps)     # Reuse buffer

    for i in 1:n_paths
        for j in 1:n_steps
            path[j] = simulate_step()
        end
        results[i] = final_value(path)
    end
    return results
end
```

### 4. Use Views Instead of Slices

Array slices create copies; views are references (zero-copy).

❌ **Bad - Copies data:**
```julia
function process_column(matrix, col)
    column = matrix[:, col]  # COPY!
    return sum(column)
end
```

✅ **Good - Zero-copy view:**
```julia
function process_column(matrix, col)
    column = @view matrix[:, col]  # VIEW!
    return sum(column)
end
```

**Broadcast with views:**
```julia
@views result = matrix[:, 1] .+ matrix[:, 2]
```

### 5. In-place Operations

Append `!` to function names for in-place operations.

❌ **Bad - Creates new array:**
```julia
result = result + 5.0
result = exp(result)
```

✅ **Good - In-place:**
```julia
result .+= 5.0
result .= exp.(result)  # Or use map!
```

**Broadcasting in-place:**
```julia
@. result = exp(-(rate * dt))  # Fuses operations, no temporaries
```

## Performance Macros

### @inbounds - Skip Bounds Checking

Use ONLY when you're certain indices are valid.

```julia
function sum_array(arr)
    total = 0.0
    @inbounds for i in 1:length(arr)
        total += arr[i]
    end
    return total
end
```

⚠️ **Warning**: Can cause segfaults if indices are wrong. Use carefully!

### @simd - SIMD Vectorization

Enables CPU vector instructions for simple loops.

```julia
function vectorized_multiply!(result, a, b)
    @simd for i in eachindex(result)
        @inbounds result[i] = a[i] * b[i]
    end
end
```

**Requirements for @simd:**
- Simple loop (no conditionals, no function calls)
- Iterations must be independent
- Use with `@inbounds` for best results

### @fastmath - Relaxed Floating-Point

Trades accuracy for speed (allows reordering, ignores NaN/Inf).

```julia
@fastmath begin
    result = sqrt(x^2 + y^2)
end
```

⚠️ **Use with caution**: Can change numerical results!

### @threads - Multi-threading

Parallelize independent iterations.

```julia
using Base.Threads

prices = Vector{Float64}(undef, n_options)
@threads for i in 1:n_options
    prices[i] = price_option(options[i])
end
```

**Requirements:**
- Start Julia with threads: `julia -t 4`
- Iterations must be independent
- Beware of race conditions

## Memory Allocation

### Track Allocations

```julia
# Time and allocation info
@time result = my_function()

# Detailed allocation tracking
@allocated result = my_function()

# Profile allocations
using Profile
@profile my_function()
Profile.print()
```

### Common Allocation Sources

1. **Growing arrays**: Use `sizehint!` or pre-allocate
2. **String concatenation**: Use `join()` or `IOBuffer`
3. **Closures**: Can allocate if they capture variables
4. **Abstract types**: Force runtime dispatch and boxing
5. **Non-constant globals**: Access through function arguments instead

### Heap vs Stack Allocation

**Stack** (fast): Small arrays, tuples
**Heap** (slower): Large arrays, mutable structs

Use `StaticArrays.jl` for small fixed-size arrays:
```julia
using StaticArrays

# Stack-allocated, super fast
v = SVector{3}(1.0, 2.0, 3.0)
m = SMatrix{3,3}(I)  # Identity matrix
```

## Function Specialization

Julia compiles specialized versions for each argument type combination.

### Avoid Type Parameters in Functions

❌ **Bad - Prevents specialization:**
```julia
function price(option::VanillaOption, data::MarketData)
    # Julia can't specialize on abstract types efficiently
end
```

✅ **Good - Allow specialization:**
```julia
function price(option::T, data::MarketData) where {T<:VanillaOption}
    # Julia creates specialized version for each concrete T
end
```

### Use Multiple Dispatch Effectively

```julia
# Specialized implementations
price(option::EuropeanCall, engine::BlackScholes, data) = ...
price(option::EuropeanCall, engine::MonteCarlo, data) = ...
price(option::AmericanPut, engine::Binomial, data) = ...
```

## Loop Optimization

### Loop Order Matters (Column-major)

Julia arrays are column-major (like Fortran, MATLAB).

❌ **Bad - Cache misses:**
```julia
for col in 1:n_cols
    for row in 1:n_rows
        matrix[row, col] = compute(row, col)
    end
end
```

✅ **Good - Cache-friendly:**
```julia
for col in 1:n_cols
    for row in 1:n_rows
        matrix[row, col] = compute(row, col)
    end
end
# Actually this is the same! The key is innermost loop varies rows.
```

**Correct pattern:**
```julia
# Iterate rows in inner loop (varies fastest)
for j in 1:n_cols
    for i in 1:n_rows
        matrix[i, j] = value
    end
end

# Or use eachcol
for col in eachcol(matrix)
    # Process column
end
```

### Loop Fusion with Broadcasting

Broadcasting automatically fuses operations.

❌ **Bad - Multiple passes:**
```julia
temp1 = a .+ b
temp2 = temp1 .* c
result = exp.(temp2)
```

✅ **Good - Single fused pass:**
```julia
result = @. exp((a + b) * c)
```

## Profiling and Benchmarking

### BenchmarkTools.jl

```julia
using BenchmarkTools

# Warm-up then benchmark
@benchmark my_function($arg1, $arg2)

# Interpolate variables with $
@btime price($option, $engine, $data)

# Compare implementations
suite = BenchmarkGroup()
suite["method1"] = @benchmarkable method1($data)
suite["method2"] = @benchmarkable method2($data)
results = run(suite)
```

### Profile.jl for Bottlenecks

```julia
using Profile

# Profile code
@profile for i in 1:1000
    my_function()
end

# View results
Profile.print()

# Flame graph (requires ProfileView.jl)
using ProfileView
ProfileView.view()
```

## Common Patterns in Financial Computing

### Monte Carlo Path Generation

✅ **Optimized pattern:**
```julia
function generate_paths!(
    paths::Matrix{Float64},
    S0::Float64,
    drift::Float64,
    vol::Float64,
    dt::Float64,
    rng::AbstractRNG
)
    n_paths, n_steps = size(paths)

    # Pre-compute constants
    drift_factor = exp(drift * dt)
    vol_sqrt_dt = vol * sqrt(dt)

    @threads for path_idx in 1:n_paths
        S = S0
        @inbounds paths[path_idx, 1] = S

        @simd for step in 2:n_steps
            Z = randn(rng)
            S *= drift_factor * exp(vol_sqrt_dt * Z)
            @inbounds paths[path_idx, step] = S
        end
    end

    return paths
end
```

### Binomial Tree with Pre-allocation

```julia
function binomial_tree!(
    tree::Matrix{Float64},
    S0::Float64,
    u::Float64,
    d::Float64,
    steps::Int
)
    @inbounds tree[1, 1] = S0

    for j in 2:steps+1
        @simd for i in 1:j
            if i == 1
                tree[i, j] = tree[i, j-1] * u
            elseif i == j
                tree[i, j] = tree[i-1, j-1] * d
            else
                tree[i, j] = tree[i-1, j-1] * d  # Same as u from [i,j-1]
            end
        end
    end
end
```

### Matrix Operations Optimization

Use LinearAlgebra views and BLAS:

```julia
using LinearAlgebra

# Use BLAS routines (highly optimized)
result = BLAS.gemv('N', A, x)  # Matrix-vector multiply

# Symmetric matrices
S = Symmetric(A)  # Tells Julia it's symmetric

# In-place solve
ldiv!(factorization, b)  # Solves in-place
```

## Type Piracy and Precompilation

### Avoid Type Piracy

Don't extend functions you don't own with types you don't own.

❌ **Bad - Type piracy:**
```julia
# You don't own Base.+ or Float64!
Base.:+(x::Float64, y::Float64) = x + y + 1.0
```

✅ **Good - Own the type or function:**
```julia
struct MyType
    value::Float64
end

Base.:+(x::MyType, y::MyType) = MyType(x.value + y.value)
```

### Precompilation

Help Julia precompile your package:

```julia
# In src/Prezo.jl
module Prezo

# ... exports and includes ...

# Precompilation directives
include("precompile.jl")

end
```

```julia
# In src/precompile.jl
using PrecompileTools

@setup_workload begin
    # Code to precompile
    data = MarketData(100.0, 0.05, 0.2, 0.0)
    option = EuropeanCall(100.0, 1.0)

    @compile_workload begin
        # This code will be precompiled
        price(option, BlackScholes(), data)
        price(option, Binomial(50), data)
    end
end
```

## Performance Anti-Patterns

### 1. Global Variables

❌ **Bad:**
```julia
global_data = MarketData(100.0, 0.05, 0.2, 0.0)

function compute_price()
    return price(option, engine, global_data)  # SLOW!
end
```

✅ **Good:**
```julia
function compute_price(data::MarketData)
    return price(option, engine, data)
end
```

### 2. Unnecessarily General Types

❌ **Bad:**
```julia
function sum_prices(prices::AbstractVector)
    return sum(prices)
end
```

✅ **Good (if you need the generality):**
```julia
function sum_prices(prices::AbstractVector{T}) where {T<:Real}
    return sum(prices)
end
```

### 3. Allocating Inside Tight Loops

❌ **Bad:**
```julia
for i in 1:n
    temp = zeros(m)  # ALLOCATION!
    # use temp...
end
```

✅ **Good:**
```julia
temp = zeros(m)  # Allocate once
for i in 1:n
    fill!(temp, 0.0)  # Reuse
    # use temp...
end
```

## Quick Wins Checklist

When optimizing existing code:

1. ✓ Run `@code_warntype` - fix red/yellow warnings
2. ✓ Check `@time` output - look for allocations
3. ✓ Add type parameters to structs with abstract fields
4. ✓ Pre-allocate arrays outside loops
5. ✓ Replace slices with `@view` or `@views`
6. ✓ Use `.=` instead of `=` for in-place operations
7. ✓ Fuse broadcasts with `@.`
8. ✓ Profile with `@profile` to find hotspots
9. ✓ Add `@inbounds` to hot loops (after verifying safety)
10. ✓ Consider `@threads` for embarrassingly parallel work

## Performance by Use Case

### Monte Carlo: Speed Priority
- Pre-allocate all arrays
- Use `@threads` for parallel paths
- Use `@simd` for path generation
- Consider GPU with CUDA.jl for massive parallelism

### Binomial Trees: Memory & Speed
- Pre-allocate tree matrix
- Use `@inbounds` in backward induction
- Consider in-place updates

### Greeks Calculation: Minimize Redundancy
- Reuse price calculations
- Vectorize finite differences
- Cache intermediate results

### Calibration: Fast Objective Functions
- Make objective function type-stable
- Pre-allocate work arrays
- Use ForwardDiff.jl for gradients (faster than finite difference)

## Further Reading

- Julia Performance Tips: https://docs.julialang.org/en/v1/manual/performance-tips/
- BenchmarkTools.jl: https://github.com/JuliaCI/BenchmarkTools.jl
- StaticArrays.jl: https://github.com/JuliaArrays/StaticArrays.jl
- ThreadsX.jl: Parallel extensions for common operations
