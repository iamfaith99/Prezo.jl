using LinearAlgebra

"""
    BasisFunction

Abstract base type for polynomial basis functions used in LSM regression.

Subtypes include:
- [`LaguerreBasis`](@ref): Laguerre polynomials (recommended)
- [`ChebyshevBasis`](@ref): Chebyshev polynomials
- [`PowerBasis`](@ref): Simple power basis
- [`HermiteBasis`](@ref): Hermite polynomials
"""
abstract type BasisFunction end

"""
    LaguerreBasis(order, normalization)

Laguerre polynomial basis functions for LSM regression.

Laguerre polynomials are orthogonal and provide good numerical stability.
This is the recommended basis for most applications.

# Fields
- `order::Int`: Maximum polynomial order
- `normalization::Float64`: Normalization factor for asset prices

See also: [`EnhancedLongstaffSchwartz`](@ref), [`LaguerreLSM`](@ref)
"""
struct LaguerreBasis <: BasisFunction
    order::Int
    normalization::Float64
end

"""
    ChebyshevBasis(order, domain_min, domain_max)

Chebyshev polynomial basis functions for LSM regression.

Chebyshev polynomials are orthogonal on [-1, 1] and provide excellent
approximation properties. Asset prices are mapped to this domain.

# Fields
- `order::Int`: Maximum polynomial order
- `domain_min::Float64`: Minimum of expected price domain
- `domain_max::Float64`: Maximum of expected price domain

See also: [`EnhancedLongstaffSchwartz`](@ref), [`ChebyshevLSM`](@ref)
"""
struct ChebyshevBasis <: BasisFunction
    order::Int
    domain_min::Float64
    domain_max::Float64
end

"""
    PowerBasis(order, normalization)

Simple power basis (1, x, x², x³, ...) for LSM regression.

Simple but can suffer from numerical instability for higher orders
due to non-orthogonality.

# Fields
- `order::Int`: Maximum polynomial order
- `normalization::Float64`: Normalization factor for asset prices

See also: [`EnhancedLongstaffSchwartz`](@ref), [`PowerLSM`](@ref)
"""
struct PowerBasis <: BasisFunction
    order::Int
    normalization::Float64
end

"""
    HermiteBasis(order, mean, std)

Hermite polynomial basis functions for LSM regression.

Hermite polynomials are orthogonal with respect to Gaussian weight functions.
Asset prices are standardized before applying the polynomials.

# Fields
- `order::Int`: Maximum polynomial order
- `mean::Float64`: Mean for standardization
- `std::Float64`: Standard deviation for standardization

See also: [`EnhancedLongstaffSchwartz`](@ref), [`HermiteLSM`](@ref)
"""
struct HermiteBasis <: BasisFunction
    order::Int
    mean::Float64
    std::Float64
end

# Multiple dispatch for different basis types
function evaluate_basis(basis::LaguerreBasis, S::AbstractVector{T}) where {T}
    (; order, normalization) = basis
    n = length(S)
    X = ones(T, n, order + 1)

    S_norm = S ./ normalization

    for j in 1:order
        if j == 1
            X[:, j+1] = @. 1.0 - S_norm
        elseif j == 2
            X[:, j+1] = @. 1.0 - 2.0 * S_norm + (S_norm^2) / 2.0
        elseif j == 3
            X[:, j+1] = @. 1.0 - 3.0 * S_norm + 1.5 * (S_norm^2) - (S_norm^3) / 6.0
        else
            X[:, j+1] = @. ((2 * j - 1 - S_norm) * X[:, j] - (j - 1) * X[:, j-1]) / j
        end
    end

    return X
end

function evaluate_basis(basis::ChebyshevBasis, S::AbstractVector{T}) where {T}
    (; order, domain_min, domain_max) = basis
    n = length(S)
    X = ones(T, n, order + 1)

    S_mapped = @. 2.0 * (S - domain_min) / (domain_max - domain_min) - 1.0

    if order >= 1
        X[:, 2] = S_mapped
    end

    for j in 2:order
        X[:, j+1] = @. 2.0 * S_mapped * X[:, j] - X[:, j-1]
    end

    return X
end

function evaluate_basis(basis::PowerBasis, S::AbstractVector{T}) where {T}
    (; order, normalization) = basis
    n = length(S)
    X = ones(T, n, order + 1)

    S_norm = S ./ normalization

    for j in 1:order
        X[:, j+1] = S_norm .^ j
    end

    return X
end

function evaluate_basis(basis::HermiteBasis, S::AbstractVector{T}) where {T}
    (; order, mean, std) = basis
    n = length(S)
    X = ones(T, n, order + 1)

    S_std = @. (S - mean) / std

    if order >= 1
        X[:, 2] = @. 2.0 * S_std
    end

    for j in 2:order
        X[:, j+1] = @. 2.0 * S_std * X[:, j] - 2.0 * j * X[:, j-1]
    end

    return X
end

# Trait-based design for numerical stability
abstract type BasisProperties end
struct Orthogonal <: BasisProperties end
struct NonOrthogonal <: BasisProperties end

basis_properties(::Type{LaguerreBasis}) = Orthogonal()
basis_properties(::Type{ChebyshevBasis}) = Orthogonal()
basis_properties(::Type{PowerBasis}) = NonOrthogonal()
basis_properties(::Type{HermiteBasis}) = Orthogonal()

function fit_continuation_value(X::Matrix, y::Vector, ::Orthogonal)
    F = qr(X)
    return F \ y
end

function fit_continuation_value(X::Matrix, y::Vector, ::NonOrthogonal)
    return X \ y
end

function fit_continuation_value(basis::BasisFunction, X::Matrix, y::Vector)
    return fit_continuation_value(X, y, basis_properties(typeof(basis)))
end

# Functor pattern - make basis functions callable
(basis::BasisFunction)(S::AbstractVector) = evaluate_basis(basis, S)
