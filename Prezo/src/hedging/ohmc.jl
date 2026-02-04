"""
    Optimal Hedged Monte Carlo (OHMC)

OHMC optimizes hedge ratios during Monte Carlo simulation. At each time step,
hedge ratio h_t(S_t) is chosen by a scoring/objective that can be:
- **quadratic**: minimize Var[V_{t+1} - h_t * S_{t+1}] (MSE / Brier-style)
- **log**: minimize E[log(1 + (V - h*S)^2)] via IRLS (log-score style)
- **exponential_utility**: minimize E[exp(-a*(V - h*S))] (CARA utility)

References: Potters et al. (2001), variance reduction for path-dependent options.
"""

using LinearAlgebra
using Statistics
using Random

"""
    OHMCConfig(n_paths, n_steps, basis_order; hedge_instrument=:underlying, scoring=:quadratic, risk_aversion=0.1)

Configuration for Optimal Hedged Monte Carlo.

# Fields
- `n_paths::Int`: Number of simulation paths
- `n_steps::Int`: Number of time steps per path
- `basis_order::Int`: Order of polynomial basis for hedge ratio regression
- `hedge_instrument::Symbol`: `:underlying` (hedge with stock; other variants reserved)
- `scoring::Symbol`: Objective for hedge ratio:
  - `:quadratic` — minimize E[(V - h*S)^2] (default; variance / MSE)
  - `:log` — minimize E[log(1 + (V - h*S)^2)] via IRLS
  - `:exponential_utility` — minimize E[exp(-a*(V - h*S))] (CARA; uses `risk_aversion`)
- `risk_aversion::Float64`: Risk aversion for `:exponential_utility` (default 0.1)

# Examples
```julia
config = OHMCConfig(10_000, 50, 3)
config_log = OHMCConfig(10_000, 50, 3; scoring=:log)
config_util = OHMCConfig(10_000, 50, 3; scoring=:exponential_utility, risk_aversion=0.2)
```
"""
struct OHMCConfig
    n_paths::Int
    n_steps::Int
    basis_order::Int
    hedge_instrument::Symbol
    scoring::Symbol           # :quadratic, :log, :exponential_utility
    risk_aversion::Float64    # used when scoring == :exponential_utility
    function OHMCConfig(n_paths::Int, n_steps::Int, basis_order::Int;
                        hedge_instrument::Symbol=:underlying,
                        scoring::Symbol=:quadratic,
                        risk_aversion::Float64=0.1)
        hedge_instrument in (:underlying,) || @warn "Only :underlying implemented; using it."
        scoring in (:quadratic, :log, :exponential_utility) ||
            error("scoring must be :quadratic, :log, or :exponential_utility")
        risk_aversion > 0 || error("risk_aversion must be positive")
        new(n_paths, n_steps, basis_order, :underlying, scoring, risk_aversion)
    end
end

"""
    OHMCResult(option_price, hedge_ratios, hedged_portfolio_variance, confidence_interval)

Result of OHMC pricing.

# Fields
- `option_price::Float64`: Estimated option value (mean of hedged portfolio at t=0)
- `hedge_ratios::Matrix{Float64}`: Time × Path; hedge_ratios[t+1, p] = shares at step t on path p
- `hedged_portfolio_variance::Float64`: Variance of terminal hedged portfolio (at t=0)
- `confidence_interval::Tuple{Float64, Float64}`: (lower, upper) 95% CI for option price
"""
struct OHMCResult
    option_price::Float64
    hedge_ratios::Matrix{Float64}  # (n_steps+1) × n_paths; row t+1 = hedge at step t
    hedged_portfolio_variance::Float64
    confidence_interval::Tuple{Float64, Float64}
end

"""
    ohmc_price(option::EuropeanOption, market_data::MarketData, config::OHMCConfig;
               rng::Union{AbstractRNG,Nothing}=nothing) -> OHMCResult

Price a European option using Optimal Hedged Monte Carlo.

1. Generate GBM paths (steps+1 × n_paths).
2. At expiry: V = payoff(S_T).
3. Backward: for each t, fit h_t(S_t) to minimize Var[V_{t+1} - h_t * S_{t+1}]
   via basis regression; then V_t = (V_{t+1} - h_t*S_{t+1})*exp(-r*dt) + h_t*S_t.
4. Option price = mean(V_0); variance and CI from V_0.

# Arguments
- `option::EuropeanOption`: Option to price
- `market_data::MarketData`: Spot, rate, vol, div
- `config::OHMCConfig`: OHMC configuration
- `rng`: Random number generator (optional)

# Returns
`OHMCResult` with option price, hedge ratios, hedged variance, and 95% CI.
"""
function ohmc_price(
    option::EuropeanOption,
    market_data::MarketData,
    config::OHMCConfig;
    rng::Union{AbstractRNG,Nothing}=nothing,
)
    rng = rng === nothing ? Random.GLOBAL_RNG : rng
    (; n_paths, n_steps, basis_order, scoring, risk_aversion) = config
    (; spot, rate, vol, div) = market_data
    (; strike, expiry) = option

    dt = expiry / n_steps
    nudt = (rate - div - 0.5 * vol^2) * dt
    sidt = vol * sqrt(dt)
    disc = exp(-rate * dt)

    # Paths: (n_steps+1) × n_paths; row t+1 = spot at step t
    paths = zeros(n_steps + 1, n_paths)
    paths[1, :] .= spot
    @inbounds for t in 1:n_steps
        z = randn(rng, n_paths)
        paths[t+1, :] = paths[t, :] .* exp.(nudt .+ sidt .* z)
    end

    # Basis: PowerBasis with normalization = spot for stability
    norm_S = max(spot, 1e-8)
    order = min(basis_order, 5)  # cap for stability
    n_basis = order + 1

    # Terminal option value (payoff at T); discount applied in rollback
    V = payoff.(Ref(option), paths[end, :])
    # Hedge ratios: (n_steps+1) × n_paths; row 1 = initial hedge, row t+1 = hedge at step t
    hedge_ratios = zeros(n_steps + 1, n_paths)
    # We'll fill hedge_ratios[t+1, :] when rolling back from t+1 to t

    for t in (n_steps - 1):-1:0
        S_t = paths[t+1, :]
        S_next = paths[t+2, :]
        Phi = _power_basis_matrix(S_t, order, norm_S)
        h = _ohmc_hedge_ratio(Phi, S_next, V, scoring, risk_aversion)
        hedge_ratios[t+2, :] = h
        V = (V .- h .* S_next) .* disc .+ h .* S_t
    end
    # Row 2 = hedge at step 0 (set when t=0 in loop). Replicate to row 1 for (time × path) layout.
    hedge_ratios[1, :] = hedge_ratios[2, :]

    option_price = mean(V)
    hedged_var = var(V)
    n = length(V)
    se = sqrt(hedged_var / n)
    ci = (option_price - 1.96 * se, option_price + 1.96 * se)

    return OHMCResult(option_price, hedge_ratios, hedged_var, ci)
end

# Internal: compute hedge ratio h = Phi * b from regression so that h ≈ dV/dS (delta).
# We fit V = Phi*a + (Phi*b)*S_next (intercept + slope in S_next); then h = Phi*b is the
# sensitivity (delta), not the level ratio V/S_next.
function _ohmc_hedge_ratio(Phi::Matrix{Float64}, S_next::Vector{Float64}, V::Vector{Float64},
                          scoring::Symbol, risk_aversion::Float64)
    n_paths, n_basis = size(Phi)
    reg = 1e-2
    # Design: V ≈ Phi*a + (Phi.*S_next)*b  =>  W*beta = V, W = [Phi  Phi.*S_next], beta = [a; b]
    W_slope = Phi .* S_next   # n_paths × n_basis
    W = [Phi W_slope]         # n_paths × (2*n_basis)

    if scoring == :quadratic
        # Min E[(V - a - b*S_next)^2] over a,b (in basis) => beta = (W'W + reg*I)^{-1} W'V
        A = W' * W + reg * I
        b_vec = W' * V
        beta = A \ b_vec
        b = beta[(n_basis + 1):end]
        return Phi * b
    end

    if scoring == :log
        # Min E[log(1 + (V - a - b*S_next)^2)] via IRLS
        beta = (W' * W + reg * I) \ (W' * V)
        for _ in 1:3
            r = V .- W * beta
            w = 1.0 ./ (1.0 .+ r .^ 2)
            Ww = W .* sqrt.(w)
            Vw = V .* sqrt.(w)
            A = Ww' * Ww + reg * I
            b_vec = Ww' * Vw
            beta = A \ b_vec
        end
        b = beta[(n_basis + 1):end]
        return Phi * b
    end

    # scoring == :exponential_utility: min E[exp(-a*(V - W*beta))]; Newton on full beta
    # Gradient g = a * W' * u, Hessian H = a^2 * W' * (u .* W), u = exp(-a*res)
    beta = (W' * W + reg * I) \ (W' * V)
    a_coef = risk_aversion
    for _ in 1:10
        res = V .- W * beta
        u = exp.(.-a_coef .* res)
        g = a_coef * (W' * u)
        H = (a_coef^2) * (W' * (u .* W)) + reg * I
        d_beta = H \ g
        beta = beta - d_beta
        if norm(d_beta) < 1e-8
            break
        end
    end
    b = beta[(n_basis + 1):end]
    return Phi * b
end

# Internal: power basis matrix (n_paths × (order+1)), first column ones, then S/S_norm, (S/S_norm)^2, ...
function _power_basis_matrix(S::AbstractVector, order::Int, norm_S::Float64)
    n = length(S)
    X = ones(eltype(S), n, order + 1)
    s = S ./ norm_S
    @inbounds for j in 1:order
        X[:, j+1] = s .^ j
    end
    return X
end
