using Plots
using Prezo
using Printf
using Statistics
using LinearAlgebra

println("Julian Approaches to Basis Functions for LSM")
println("=" ^ 50)

# 1. ABSTRACT TYPE HIERARCHY (Very Julian!)
abstract type BasisFunction end

# Concrete types for different basis functions
struct LaguerreBasis <: BasisFunction
    order::Int
    normalization::Float64
end

struct ChebyshevBasis <: BasisFunction
    order::Int
    domain_min::Float64
    domain_max::Float64
end

struct PowerBasis <: BasisFunction
    order::Int
    normalization::Float64
end

struct HermiteBasis <: BasisFunction
    order::Int
    mean::Float64
    std::Float64
end

# 2. MULTIPLE DISPATCH (Core Julia Pattern)
# Each basis type has its own specialized method

function evaluate_basis(basis::LaguerreBasis, S::AbstractVector{T}) where T
    (; order, normalization) = basis
    n = length(S)
    X = ones(T, n, order + 1)

    # Normalize for numerical stability
    S_norm = S ./ normalization

    # Use broadcasting for vectorized operations (Very Julian!)
    for j in 1:order
        if j == 1
            X[:, j + 1] = @. 1.0 - S_norm
        elseif j == 2
            X[:, j + 1] = @. 1.0 - 2.0 * S_norm + (S_norm^2) / 2.0
        elseif j == 3
            X[:, j + 1] = @. 1.0 - 3.0 * S_norm + 1.5 * (S_norm^2) - (S_norm^3) / 6.0
        else
            # Recursive relation using broadcasting
            X[:, j + 1] = @. ((2 * j - 1 - S_norm) * X[:, j] - (j - 1) * X[:, j - 1]) / j
        end
    end

    return X
end

function evaluate_basis(basis::ChebyshevBasis, S::AbstractVector{T}) where T
    (; order, domain_min, domain_max) = basis
    n = length(S)
    X = ones(T, n, order + 1)

    # Map to [-1, 1] for Chebyshev polynomials
    S_mapped = @. 2.0 * (S - domain_min) / (domain_max - domain_min) - 1.0

    # Chebyshev polynomials using recurrence
    if order >= 1
        X[:, 2] = S_mapped
    end

    for j in 2:order
        X[:, j + 1] = @. 2.0 * S_mapped * X[:, j] - X[:, j - 1]
    end

    return X
end

function evaluate_basis(basis::PowerBasis, S::AbstractVector{T}) where T
    (; order, normalization) = basis
    n = length(S)
    X = ones(T, n, order + 1)

    S_norm = S ./ normalization

    # Simple power basis: [1, x, x^2, x^3, ...]
    for j in 1:order
        X[:, j + 1] = S_norm .^ j
    end

    return X
end

function evaluate_basis(basis::HermiteBasis, S::AbstractVector{T}) where T
    (; order, mean, std) = basis
    n = length(S)
    X = ones(T, n, order + 1)

    # Standardize inputs
    S_std = @. (S - mean) / std

    # Hermite polynomials (physicist's version)
    if order >= 1
        X[:, 2] = @. 2.0 * S_std
    end

    for j in 2:order
        X[:, j + 1] = @. 2.0 * S_std * X[:, j] - 2.0 * j * X[:, j - 1]
    end

    return X
end

# 3. TRAIT-BASED DESIGN (Advanced Julian Pattern)
abstract type BasisProperties end
struct Orthogonal <: BasisProperties end
struct NonOrthogonal <: BasisProperties end

# Dispatch on basis properties
basis_properties(::Type{LaguerreBasis}) = Orthogonal()
basis_properties(::Type{ChebyshevBasis}) = Orthogonal()
basis_properties(::Type{PowerBasis}) = NonOrthogonal()
basis_properties(::Type{HermiteBasis}) = Orthogonal()

# Different regression approaches based on orthogonality
function fit_continuation_value(X::Matrix, y::Vector, ::Orthogonal)
    # For orthogonal bases, can use more stable methods
    # QR decomposition is more stable for orthogonal systems
    F = qr(X)
    return F \ y
end

function fit_continuation_value(X::Matrix, y::Vector, ::NonOrthogonal)
    # For non-orthogonal bases, use standard least squares
    return X \ y
end

# Convenience method that dispatches based on basis type
function fit_continuation_value(basis::BasisFunction, X::Matrix, y::Vector)
    return fit_continuation_value(X, y, basis_properties(typeof(basis)))
end

# 4. FUNCTOR PATTERN (Very Julian!)
# Make basis functions callable
(basis::BasisFunction)(S::AbstractVector) = evaluate_basis(basis, S)

# 5. PARAMETRIC TYPES FOR PERFORMANCE
struct OptimizedLSM{B <: BasisFunction, T <: AbstractFloat}
    basis::B
    steps::Int
    reps::Int
    discount_factor::T
end

# Constructor with type inference
function OptimizedLSM(basis::BasisFunction, steps::Int, reps::Int, rate::T, dt::T) where T
    discount_factor = exp(-rate * dt)
    return OptimizedLSM{typeof(basis), T}(basis, steps, reps, discount_factor)
end

# 6. DEMONSTRATION OF DIFFERENT BASIS FUNCTIONS
println("Testing Different Basis Function Approaches")
println("-" ^ 45)

# Test data
spot_prices = [35.0, 37.0, 39.0, 41.0, 43.0, 45.0]
test_data = Float64.(spot_prices)

# Create different basis functions
bases = [
    LaguerreBasis(3, 100.0),
    ChebyshevBasis(3, 30.0, 50.0),
    PowerBasis(3, 40.0),
    HermiteBasis(3, 40.0, 5.0)
]

basis_names = ["Laguerre", "Chebyshev", "Power", "Hermite"]

for (i, (basis, name)) in enumerate(zip(bases, basis_names))
    println("\n$name Basis Functions:")
    X = basis(test_data)  # Using functor syntax!

    println("Spot Prices: ", test_data)
    println("Basis Matrix (first 3 columns):")
    for j in 1:size(X, 1)
        @printf("%.1f: [%.4f, %.4f, %.4f, ...]\n",
                test_data[j], X[j,1], X[j,2], X[j,3])
    end

    # Check conditioning
    cond_num = cond(X)
    @printf("Condition number: %.2e (lower is better)\n", cond_num)
end

# 7. INTEGRATION WITH EXISTING LSM
println("\n" * "="^50)
println("Improved LSM with Julian Basis Functions")
println("="^50)

# Enhanced LongstaffSchwartz struct
struct EnhancedLongstaffSchwartz{B <: BasisFunction}
    basis::B
    steps::Int
    reps::Int
end

# Constructor with basis function selection
function EnhancedLongstaffSchwartz(basis_type::Symbol, order::Int, steps::Int, reps::Int;
                                  normalization=100.0, domain=(30.0, 50.0))
    if basis_type == :laguerre
        basis = LaguerreBasis(order, normalization)
    elseif basis_type == :chebyshev
        basis = ChebyshevBasis(order, domain[1], domain[2])
    elseif basis_type == :power
        basis = PowerBasis(order, normalization)
    elseif basis_type == :hermite
        basis = HermiteBasis(order, 40.0, 5.0)  # Reasonable defaults
    else
        error("Unknown basis type: $basis_type")
    end

    return EnhancedLongstaffSchwartz(basis, steps, reps)
end

# Example usage
enhanced_engines = [
    EnhancedLongstaffSchwartz(:laguerre, 3, 100, 50000),
    EnhancedLongstaffSchwartz(:chebyshev, 3, 100, 50000),
    EnhancedLongstaffSchwartz(:power, 3, 100, 50000),
    EnhancedLongstaffSchwartz(:hermite, 3, 100, 50000)
]

println("Created enhanced LSM engines with different basis functions:")
for (i, engine) in enumerate(enhanced_engines)
    println("$(i). $(typeof(engine.basis).name.name) basis, order $(engine.basis.order)")
end

# 8. BROADCASTING AND VECTORIZATION EXAMPLES
println("\nJulian Broadcasting Examples:")
println("-" ^ 30)

# Instead of loops, use broadcasting
S_test = [35.0, 40.0, 45.0]
normalization = 100.0

# Non-Julian way (imperative)
println("Imperative style:")
S_norm_loop = similar(S_test)
for i in eachindex(S_test)
    S_norm_loop[i] = S_test[i] / normalization
end
println("Result: ", S_norm_loop)

# Julian way (declarative with broadcasting)
println("Julian broadcasting style:")
S_norm_broadcast = S_test ./ normalization
println("Result: ", S_norm_broadcast)

# More complex broadcasting
println("Complex broadcasting (Laguerre L₁):")
L1_broadcast = @. 1.0 - S_norm_broadcast
println("L₁(x) = 1 - x: ", L1_broadcast)

# 9. TYPE STABILITY DEMONSTRATION
function type_stable_basis(S::Vector{T}, order::Int) where T
    n = length(S)
    # Pre-allocate with correct type
    X = Matrix{T}(undef, n, order + 1)
    X[:, 1] .= one(T)  # Use one(T) instead of 1.0

    for j in 1:order
        X[:, j + 1] = S .^ j
    end

    return X
end

println("\nType Stability Example:")
S_float32 = Float32[35.0, 40.0, 45.0]
S_float64 = Float64[35.0, 40.0, 45.0]

X_32 = type_stable_basis(S_float32, 2)
X_64 = type_stable_basis(S_float64, 2)

println("Float32 input -> $(eltype(X_32)) output")
println("Float64 input -> $(eltype(X_64)) output")

println("\nJulian Basis Functions Summary:")
println("✓ Multiple dispatch for different basis types")
println("✓ Abstract types for common interface")
println("✓ Functor pattern for callable objects")
println("✓ Broadcasting for vectorized operations")
println("✓ Type stability for performance")
println("✓ Trait-based design for specialized algorithms")
println("✓ Parametric types for zero-cost abstractions")