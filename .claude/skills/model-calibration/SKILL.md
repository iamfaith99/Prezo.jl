# Financial Model Calibration

Expertise in calibrating financial models to market data, implied volatility, and parameter estimation.

## Implied Volatility

### Newton-Raphson Method

Most efficient for Black-Scholes implied vol:

```julia
function implied_volatility_nr(
    option::EuropeanOption,
    market_price::Float64,
    data::MarketData;
    max_iter::Int=100,
    tol::Float64=1e-6
)
    # Initial guess: ATM approx
    σ = 0.2

    for iter in 1:max_iter
        # Price and vega at current σ
        data_σ = MarketData(data.spot, data.rate, σ, data.div)
        model_price = price(option, BlackScholes(), data_σ)
        gks = greeks(option, BlackScholes(), data_σ)

        # Price difference
        diff = model_price - market_price

        if abs(diff) < tol
            return σ
        end

        # Newton step: σ_new = σ - f(σ)/f'(σ)
        # f'(σ) = vega (but vega is per 1%, so multiply by 100)
        σ -= diff / (gks.vega * 100)

        # Bounds check
        σ = clamp(σ, 0.01, 5.0)
    end

    @warn "Implied vol did not converge" iter=max_iter
    return NaN
end
```

### Bisection Method (More Robust)

```julia
function implied_volatility_bisection(
    option::EuropeanOption,
    market_price::Float64,
    data::MarketData;
    vol_min::Float64=0.01,
    vol_max::Float64=5.0,
    tol::Float64=1e-6
)
    # Price functions at bounds
    price_at_vol(σ) = price(option, BlackScholes(),
                            MarketData(data.spot, data.rate, σ, data.div))

    # Check if solution exists in range
    p_min = price_at_vol(vol_min)
    p_max = price_at_vol(vol_max)

    if (market_price < p_min) || (market_price > p_max)
        @warn "Market price outside valid range"
        return NaN
    end

    # Bisection
    while (vol_max - vol_min) > tol
        vol_mid = (vol_min + vol_max) / 2
        p_mid = price_at_vol(vol_mid)

        if abs(p_mid - market_price) < tol
            return vol_mid
        end

        if (p_mid < market_price)
            vol_min = vol_mid
        else
            vol_max = vol_mid
        end
    end

    return (vol_min + vol_max) / 2
end
```

### Brent's Method (Combining Both)

Use Roots.jl:

```julia
using Roots

function implied_volatility_brent(
    option::EuropeanOption,
    market_price::Float64,
    data::MarketData
)
    price_diff(σ) = price(option, BlackScholes(),
                         MarketData(data.spot, data.rate, σ, data.div)) - market_price

    try
        return find_zero(price_diff, (0.01, 5.0), Roots.Brent())
    catch
        @warn "Implied vol calculation failed"
        return NaN
    end
end
```

## Volatility Surface Construction

### Market Data Structure

```julia
struct VolatilitySurfacePoint
    strike::Float64
    expiry::Float64
    implied_vol::Float64
    market_price::Float64
    bid::Float64
    ask::Float64
end

struct VolatilitySurface
    spot::Float64
    points::Vector{VolatilitySurfacePoint}
end
```

### Computing Surface from Market Prices

```julia
function build_volatility_surface(
    spot::Float64,
    rate::Float64,
    div::Float64,
    market_data::Vector{@NamedTuple{strike::Float64, expiry::Float64,
                                     call_price::Float64, put_price::Float64}}
)
    points = VolatilitySurfacePoint[]

    for data in market_data
        # Use call for OTM calls, put for OTM puts (better liquidity)
        if data.strike > spot
            # OTM call
            option = EuropeanCall(data.strike, data.expiry)
            market_price = data.call_price
        else
            # OTM put
            option = EuropeanPut(data.strike, data.expiry)
            market_price = data.put_price
        end

        # Compute implied vol
        mkt_data = MarketData(spot, rate, 0.2, div)  # Dummy vol
        impl_vol = implied_volatility_nr(option, market_price, mkt_data)

        if !isnan(impl_vol)
            point = VolatilitySurfacePoint(
                data.strike,
                data.expiry,
                impl_vol,
                market_price,
                0.0, 0.0  # Bid/ask if available
            )
            push!(points, point)
        end
    end

    return VolatilitySurface(spot, points)
end
```

### Interpolation

```julia
using Interpolations

function interpolate_vol_surface(surf::VolatilitySurface)
    # Extract grid
    strikes = [p.strike for p in surf.points]
    expiries = [p.expiry for p in surf.points]
    vols = [p.implied_vol for p in surf.points]

    # Create grid (assuming rectangular grid)
    unique_strikes = sort(unique(strikes))
    unique_expiries = sort(unique(expiries))

    # Reshape vols to matrix
    vol_matrix = reshape(vols, length(unique_strikes), length(unique_expiries))

    # Interpolate
    itp = interpolate(vol_matrix, BSpline(Cubic(Line(OnGrid()))))
    sitp = scale(itp, unique_strikes, unique_expiries)

    return sitp
end
```

## Model Calibration

### Objective Function

```julia
using Optim

struct CalibrationTarget
    option::VanillaOption
    market_price::Float64
    weight::Float64  # For weighted least squares
end

function calibration_objective(
    params::Vector{Float64},
    targets::Vector{CalibrationTarget},
    spot::Float64,
    rate::Float64,
    div::Float64,
    pricing_engine
)
    # Extract parameters (example: volatility and other model params)
    vol = params[1]
    # other_param = params[2], etc.

    sse = 0.0  # Sum of squared errors

    for target in targets
        data = MarketData(spot, rate, vol, div)
        model_price = price(target.option, pricing_engine, data)

        error = (model_price - target.market_price) * target.weight
        sse += error^2
    end

    return sse
end

function calibrate_model(
    targets::Vector{CalibrationTarget},
    spot::Float64,
    rate::Float64,
    div::Float64;
    initial_params::Vector{Float64}=[0.2],
    lower_bounds::Vector{Float64}=[0.01],
    upper_bounds::Vector{Float64}=[5.0]
)
    # Objective wrapper
    obj(params) = calibration_objective(params, targets, spot, rate, div, BlackScholes())

    # Optimize
    result = optimize(
        obj,
        lower_bounds,
        upper_bounds,
        initial_params,
        Fminbox(LBFGS()),
        Optim.Options(show_trace=true, iterations=1000)
    )

    if Optim.converged(result)
        return Optim.minimizer(result)
    else
        @warn "Calibration did not converge"
        return nothing
    end
end
```

### Heston Model Calibration

```julia
struct HestonParams
    v0::Float64      # Initial variance
    κ::Float64       # Mean reversion speed
    θ::Float64       # Long-term variance
    σ_v::Float64     # Vol of vol
    ρ::Float64       # Correlation
end

function heston_price(
    option::EuropeanCall,
    params::HestonParams,
    spot::Float64,
    rate::Float64,
    expiry::Float64
)
    # Heston formula (requires characteristic function)
    # This is a simplified placeholder
    # Real implementation uses FFT or numerical integration
    # See Gatheral's "The Volatility Surface"

    return heston_characteristic_function_price(
        spot, option.strike, expiry, rate,
        params.v0, params.κ, params.θ, params.σ_v, params.ρ
    )
end

function calibrate_heston(
    targets::Vector{CalibrationTarget},
    spot::Float64,
    rate::Float64
)
    # Initial guess
    initial = [
        0.04,  # v0
        2.0,   # κ
        0.04,  # θ
        0.3,   # σ_v
        -0.7   # ρ
    ]

    # Bounds
    lower = [0.001, 0.01, 0.001, 0.01, -0.999]
    upper = [1.0, 10.0, 1.0, 2.0, 0.999]

    # Objective
    function obj(params_vec)
        params = HestonParams(params_vec...)

        sse = 0.0
        for target in targets
            model_price = heston_price(target.option, params, spot, rate,
                                      target.option.expiry)
            error = model_price - target.market_price
            sse += error^2 * target.weight
        end

        return sse
    end

    # Optimize with constraints
    result = optimize(
        obj,
        lower,
        upper,
        initial,
        Fminbox(LBFGS()),
        Optim.Options(show_trace=true, iterations=10000)
    )

    params_opt = Optim.minimizer(result)
    return HestonParams(params_opt...)
end
```

## Local Volatility Model

### Dupire's Formula

```julia
function dupire_local_vol(
    strike::Float64,
    expiry::Float64,
    vol_surface_interpolator,
    spot::Float64,
    rate::Float64,
    div::Float64
)
    # σ_local²(K,T) = (∂C/∂T + (r-q)K∂C/∂K + qC) / (0.5K²∂²C/∂K²)

    # Get implied vol and derivatives from surface
    σ_impl = vol_surface_interpolator(strike, expiry)

    # Numerical derivatives (simplified)
    h_K = 1.0
    h_T = 0.01

    # ∂C/∂K (delta w.r.t strike)
    C_up = black_scholes_call(spot, strike + h_K, rate, div, σ_impl, expiry)
    C_down = black_scholes_call(spot, strike - h_K, rate, div, σ_impl, expiry)
    ∂C_∂K = (C_up - C_down) / (2 * h_K)

    # ∂²C/∂K² (gamma w.r.t strike)
    C_mid = black_scholes_call(spot, strike, rate, div, σ_impl, expiry)
    ∂²C_∂K² = (C_up - 2 * C_mid + C_down) / (h_K^2)

    # ∂C/∂T
    σ_T_plus = vol_surface_interpolator(strike, expiry + h_T)
    C_T_plus = black_scholes_call(spot, strike, rate, div, σ_T_plus, expiry + h_T)
    ∂C_∂T = (C_T_plus - C_mid) / h_T

    # Dupire formula
    numerator = ∂C_∂T + (rate - div) * strike * ∂C_∂K + div * C_mid
    denominator = 0.5 * strike^2 * ∂²C_∂K²

    σ_local² = numerator / denominator

    return sqrt(max(σ_local², 0.0))  # Ensure non-negative
end
```

## Jump Diffusion Calibration

### Merton Model

```julia
struct MertonJumpParams
    σ::Float64       # Diffusion volatility
    λ::Float64       # Jump intensity (jumps per year)
    μ_J::Float64     # Mean jump size
    σ_J::Float64     # Jump size volatility
end

function merton_jump_price(
    option::EuropeanCall,
    params::MertonJumpParams,
    spot::Float64,
    rate::Float64,
    div::Float64
)
    # Sum over possible number of jumps (Poisson distribution)
    # Each term uses adjusted Black-Scholes

    (; strike, expiry) = option
    (; σ, λ, μ_J, σ_J) = params

    max_jumps = 50
    price_sum = 0.0

    for n in 0:max_jumps
        # Probability of n jumps
        λ_prime = λ * (1 + μ_J)
        p_n = (λ_prime * expiry)^n * exp(-λ_prime * expiry) / factorial(n)

        # Adjusted parameters for n jumps
        σ_n = sqrt(σ^2 + n * σ_J^2 / expiry)
        r_n = rate - λ * μ_J + n * log(1 + μ_J) / expiry

        # Black-Scholes with adjusted params
        data_n = MarketData(spot, r_n, σ_n, div)
        price_n = price(option, BlackScholes(), data_n)

        price_sum += p_n * price_n
    end

    return price_sum
end
```

## Regularization and Constraints

### Tikhonov Regularization

Prevent overfitting:

```julia
function regularized_objective(
    params::Vector{Float64},
    targets::Vector{CalibrationTarget},
    λ_reg::Float64=0.01
)
    # Standard objective
    sse = calibration_objective(params, targets, ...)

    # Regularization term (L2 penalty)
    reg_term = λ_reg * sum(params.^2)

    return sse + reg_term
end
```

### Arbitrage-Free Constraints

For local volatility:

```julia
function check_arbitrage_free(σ_local_grid::Matrix{Float64})
    # Calendar spread: C(T1) ≤ C(T2) for T1 < T2
    # Butterfly spread: ∂²C/∂K² ≥ 0

    # Check local vol is positive
    if any(σ_local_grid .< 0)
        return false
    end

    # Check butterfly (simplified)
    for j in 1:size(σ_local_grid, 2)
        for i in 2:size(σ_local_grid, 1)-1
            butterfly = σ_local_grid[i-1, j] - 2*σ_local_grid[i, j] + σ_local_grid[i+1, j]
            if butterfly < -1e-6  # Small tolerance
                return false
            end
        end
    end

    return true
end
```

## Validation

### In-Sample vs Out-of-Sample

```julia
function split_calibration_validation(
    all_targets::Vector{CalibrationTarget};
    train_fraction::Float64=0.8
)
    n = length(all_targets)
    n_train = Int(floor(n * train_fraction))

    # Random split
    indices = shuffle(1:n)
    train_indices = indices[1:n_train]
    val_indices = indices[n_train+1:end]

    train_targets = all_targets[train_indices]
    val_targets = all_targets[val_indices]

    return train_targets, val_targets
end

function validate_calibration(
    params::Vector{Float64},
    validation_targets::Vector{CalibrationTarget},
    spot::Float64,
    rate::Float64,
    div::Float64
)
    errors = Float64[]

    for target in validation_targets
        data = MarketData(spot, rate, params[1], div)  # Simplified
        model_price = price(target.option, BlackScholes(), data)
        error = abs(model_price - target.market_price) / target.market_price

        push!(errors, error)
    end

    return (
        mean_error=mean(errors),
        max_error=maximum(errors),
        rmse=sqrt(mean(errors.^2))
    )
end
```

## Further Reading

- Gatheral: "The Volatility Surface" (comprehensive vol modeling)
- Cont & Tankov: "Financial Modelling with Jump Processes"
- Andersen & Piterbarg: "Interest Rate Modeling" (calibration techniques)
- Dupire (1994): "Pricing with a smile" (local volatility)
