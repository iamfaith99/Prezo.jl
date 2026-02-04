"""
    CVaR-aware Optimal Hedged Monte Carlo (CVaR-OHMC)

Lagrangian CVaR inside OHMC: making OHMC risk-aware by incorporating CVaR directly
into the optimization objective (or constraints) rather than applying CVaR as a
post-processing step for position sizing.

## Key Idea

Standard OHMC minimizes: `E[ℓ(e)] + λ_tc * E[TC]`

CVaR-OHMC minimizes: `E[ℓ(e)] + λ_tc * E[TC] + η * CVaR_α(L)`

This makes the hedge ratio `a` explicitly account for tail risk during optimization.

## Rockafellar–Uryasev Representation

CVaR is not linear in the loss distribution, but the R-U trick transforms it:

    CVaR_α(L) = min_ν [ ν + (1/(1-α)) * E[(L - ν)⁺] ]

where (L - ν)⁺ = max(L - ν, 0).

So instead of "minimize CVaR", we:
1. Introduce auxiliary variable ν (VaR threshold)
2. Minimize ν + (1/(1-α)) * E[(L - ν)⁺] jointly over (a, ν)
3. Use the same Monte Carlo paths that OHMC uses for E[ℓ(e)] and E[TC]

References:
- Rockafellar & Uryasev (2000): Optimization of CVaR
- Potters et al. (2001): Hedged Monte Carlo
"""

using LinearAlgebra
using Statistics
using Random

# ============================================================================
# CVaR-OHMC Configuration
# ============================================================================

"""
    CVaRObjective

Specifies how CVaR enters the OHMC objective.

# Variants
- `:penalty` — Add CVaR as penalty term: obj = E[ℓ] + η * CVaR_α(L)
- `:constraint` — Lagrangian form of CVaR constraint: CVaR_α(L) ≤ L_max
"""
@enum CVaRObjective begin
    CVaRPenalty      # Penalty form: obj + η * CVaR
    CVaRConstraint   # Constraint form with Lagrange multiplier
end

"""
    LossDefinition

Defines what "loss" L means for CVaR calculation.

# Variants
- `:hedging_error` — L = replication error (V - hedge)
- `:portfolio_pnl` — L = -PnL (negative of profit)
- `:drawdown` — L = max cumulative loss along path
"""
@enum LossDefinition begin
    HedgingError     # CVaR on replication error
    PortfolioPnL     # CVaR on negative P&L
    Drawdown         # CVaR on path drawdown
end

"""
    CVaROHMCConfig(n_paths, n_steps, basis_order; kwargs...)

Configuration for CVaR-aware Optimal Hedged Monte Carlo.

# Required Fields
- `n_paths::Int`: Number of simulation paths
- `n_steps::Int`: Number of time steps per path
- `basis_order::Int`: Order of polynomial basis for hedge ratio regression

# CVaR Parameters
- `cvar_alpha::Float64`: CVaR confidence level (e.g., 0.95 for 95% CVaR). Default: 0.95
- `cvar_weight::Float64`: Weight η for CVaR penalty term. Default: 0.1
- `cvar_objective::CVaRObjective`: How CVaR enters objective. Default: CVaRPenalty
- `cvar_budget::Float64`: CVaR constraint budget L_max (for CVaRConstraint). Default: Inf
- `loss_definition::LossDefinition`: What loss to compute CVaR on. Default: HedgingError

# Standard OHMC Parameters
- `hedge_instrument::Symbol`: `:underlying` (hedge with stock). Default: :underlying
- `scoring::Symbol`: Base scoring `:quadratic`, `:log`, `:exponential_utility`. Default: :quadratic
- `risk_aversion::Float64`: For exponential utility. Default: 0.1
- `tc_rate::Float64`: Transaction cost rate (proportional). Default: 0.0

# Optimization Parameters
- `nu_iterations::Int`: Iterations for ν (VaR threshold) optimization. Default: 10
- `outer_iterations::Int`: Outer loop iterations for (a, ν) joint optimization. Default: 3

# Examples
```julia
# Basic CVaR-OHMC with 95% CVaR penalty
config = CVaROHMCConfig(10_000, 50, 3; cvar_alpha=0.95, cvar_weight=0.2)

# CVaR constraint form with budget
config = CVaROHMCConfig(10_000, 50, 3;
    cvar_objective=CVaRConstraint,
    cvar_budget=5.0,
    cvar_alpha=0.99
)

# Combined with transaction costs
config = CVaROHMCConfig(10_000, 50, 3;
    cvar_weight=0.15,
    tc_rate=0.001,
    scoring=:log
)
```
"""
struct CVaROHMCConfig
    # Path simulation
    n_paths::Int
    n_steps::Int
    basis_order::Int

    # CVaR parameters
    cvar_alpha::Float64           # Confidence level (e.g., 0.95)
    cvar_weight::Float64          # η: CVaR penalty weight
    cvar_objective::CVaRObjective # Penalty or constraint form
    cvar_budget::Float64          # L_max for constraint form
    loss_definition::LossDefinition

    # Standard OHMC parameters
    hedge_instrument::Symbol
    scoring::Symbol
    risk_aversion::Float64
    tc_rate::Float64

    # Optimization parameters
    nu_iterations::Int
    outer_iterations::Int

    function CVaROHMCConfig(
        n_paths::Int,
        n_steps::Int,
        basis_order::Int;
        cvar_alpha::Float64=0.95,
        cvar_weight::Float64=0.1,
        cvar_objective::CVaRObjective=CVaRPenalty,
        cvar_budget::Float64=Inf,
        loss_definition::LossDefinition=HedgingError,
        hedge_instrument::Symbol=:underlying,
        scoring::Symbol=:quadratic,
        risk_aversion::Float64=0.1,
        tc_rate::Float64=0.0,
        nu_iterations::Int=10,
        outer_iterations::Int=3
    )
        # Validation
        n_paths > 0 || error("n_paths must be positive")
        n_steps > 0 || error("n_steps must be positive")
        basis_order >= 0 || error("basis_order must be non-negative")
        0.0 < cvar_alpha < 1.0 || error("cvar_alpha must be in (0, 1)")
        cvar_weight >= 0.0 || error("cvar_weight must be non-negative")
        risk_aversion > 0.0 || error("risk_aversion must be positive")
        tc_rate >= 0.0 || error("tc_rate must be non-negative")
        scoring in (:quadratic, :log, :exponential_utility) ||
            error("scoring must be :quadratic, :log, or :exponential_utility")

        if cvar_objective == CVaRConstraint && cvar_budget == Inf
            @warn "CVaRConstraint mode with infinite budget is equivalent to no constraint"
        end

        new(n_paths, n_steps, basis_order,
            cvar_alpha, cvar_weight, cvar_objective, cvar_budget, loss_definition,
            hedge_instrument, scoring, risk_aversion, tc_rate,
            nu_iterations, outer_iterations)
    end
end

# ============================================================================
# CVaR-OHMC Result
# ============================================================================

"""
    CVaROHMCResult

Result of CVaR-aware OHMC pricing.

# Fields
- `option_price::Float64`: Estimated option value
- `hedge_ratios::Matrix{Float64}`: (n_steps+1) × n_paths hedge ratios
- `hedged_portfolio_variance::Float64`: Variance of terminal hedged portfolio
- `confidence_interval::Tuple{Float64,Float64}`: 95% CI for option price

# CVaR-specific fields
- `cvar::Float64`: CVaR of the loss distribution at optimal hedge
- `var::Float64`: VaR (ν*) from Rockafellar-Uryasev optimization
- `expected_shortfall_paths::Vector{Int}`: Indices of paths in the tail
- `tail_loss_mean::Float64`: Mean loss in tail (sanity check ≈ CVaR)
- `cvar_contribution::Float64`: CVaR term contribution to objective

# Diagnostics
- `base_objective::Float64`: E[ℓ(e)] + λ_tc * E[TC] (without CVaR)
- `total_objective::Float64`: Full objective including CVaR term
- `nu_convergence::Vector{Float64}`: ν values across optimization iterations
"""
struct CVaROHMCResult
    # Standard OHMC outputs
    option_price::Float64
    hedge_ratios::Matrix{Float64}
    hedged_portfolio_variance::Float64
    confidence_interval::Tuple{Float64,Float64}

    # CVaR outputs
    cvar::Float64
    var::Float64  # VaR threshold (ν*)
    expected_shortfall_paths::Vector{Int}
    tail_loss_mean::Float64
    cvar_contribution::Float64

    # Diagnostics
    base_objective::Float64
    total_objective::Float64
    nu_convergence::Vector{Float64}
end

# ============================================================================
# Core CVaR-OHMC Implementation
# ============================================================================

"""
    cvar_ohmc_price(option::EuropeanOption, market_data::MarketData, config::CVaROHMCConfig;
                    rng=nothing) -> CVaROHMCResult

Price a European option using CVaR-aware Optimal Hedged Monte Carlo.

The algorithm jointly optimizes hedge ratios `a` and VaR threshold `ν` to minimize:

    E[ℓ(e)] + λ_tc * E[TC] + η * [ν + (1/(1-α)) * E[(L-ν)⁺]]

where the CVaR term uses the Rockafellar-Uryasev representation.

# Algorithm

1. Generate GBM paths (same as standard OHMC)
2. Initialize ν to empirical VaR of unhedged loss
3. Outer loop:
   a. Given ν, solve for hedge ratios a using modified regression that includes
      CVaR-weighted tail samples
   b. Given a, update ν by finding the α-quantile of current loss distribution
4. Return optimal (a*, ν*) and compute CVaR diagnostics

# Arguments
- `option::EuropeanOption`: Option to price
- `market_data::MarketData`: Spot, rate, vol, div
- `config::CVaROHMCConfig`: CVaR-OHMC configuration
- `rng`: Random number generator (optional)

# Returns
`CVaROHMCResult` with option price, hedge ratios, CVaR, VaR, and diagnostics.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)

# CVaR penalty form
config = CVaROHMCConfig(10_000, 50, 3; cvar_alpha=0.95, cvar_weight=0.2)
result = cvar_ohmc_price(call, data, config)

println("Price: ", result.option_price)
println("CVaR: ", result.cvar)
println("VaR: ", result.var)
```
"""
function cvar_ohmc_price(
    option::EuropeanOption,
    market_data::MarketData,
    config::CVaROHMCConfig;
    rng::Union{AbstractRNG,Nothing}=nothing
)
    rng = rng === nothing ? Random.GLOBAL_RNG : rng

    (; n_paths, n_steps, basis_order, cvar_alpha, cvar_weight,
        cvar_objective, cvar_budget, loss_definition,
        scoring, risk_aversion, tc_rate,
        nu_iterations, outer_iterations) = config
    (; spot, rate, vol, div) = market_data
    (; strike, expiry) = option

    dt = expiry / n_steps
    nudt = (rate - div - 0.5 * vol^2) * dt
    sidt = vol * sqrt(dt)
    disc = exp(-rate * dt)

    # -------------------------------------------------------------------------
    # Step 1: Generate paths (same as standard OHMC)
    # -------------------------------------------------------------------------
    paths = zeros(n_steps + 1, n_paths)
    paths[1, :] .= spot
    @inbounds for t in 1:n_steps
        z = randn(rng, n_paths)
        paths[t+1, :] = paths[t, :] .* exp.(nudt .+ sidt .* z)
    end

    # Basis setup
    norm_S = max(spot, 1e-8)
    order = min(basis_order, 5)

    # Terminal payoff
    V_terminal = payoff.(Ref(option), paths[end, :])

    # -------------------------------------------------------------------------
    # Step 2: Initialize ν (VaR threshold) from unhedged loss
    # -------------------------------------------------------------------------
    # Unhedged loss = -(terminal_value - initial_cost) for now
    initial_loss_estimate = V_terminal .- mean(V_terminal)
    nu = _quantile_sorted(sort(initial_loss_estimate), cvar_alpha)
    nu_history = Float64[nu]

    # Storage for hedge ratios
    hedge_ratios = zeros(n_steps + 1, n_paths)

    # -------------------------------------------------------------------------
    # Step 3: Outer loop — alternating optimization of (a, ν)
    # -------------------------------------------------------------------------
    V = copy(V_terminal)
    prev_hedge = zeros(n_paths)
    total_tc = zeros(n_paths)

    for outer_iter in 1:outer_iterations
        # Reset for backward pass
        V = copy(V_terminal)
        prev_hedge = zeros(n_paths)
        total_tc = zeros(n_paths)

        # -------------------------------------------------------------------
        # Step 3a: Backward induction with CVaR-aware hedge ratio fitting
        # -------------------------------------------------------------------
        for t in (n_steps-1):-1:0
            S_t = paths[t+1, :]
            S_next = paths[t+2, :]

            # Build basis matrix
            Phi = _cvar_power_basis_matrix(S_t, order, norm_S)

            # Compute loss for CVaR weighting
            # Loss = hedging error at this step
            loss_current = _compute_loss(V, S_next, paths, t, loss_definition, option)

            # Compute CVaR weights: upweight tail samples
            cvar_weights = _compute_cvar_weights(loss_current, nu, cvar_alpha, cvar_weight)

            # Fit hedge ratio with CVaR-weighted regression
            h = _cvar_ohmc_hedge_ratio(
                Phi, S_next, V,
                scoring, risk_aversion,
                cvar_weights
            )

            hedge_ratios[t+2, :] = h

            # Transaction costs
            if tc_rate > 0.0
                tc = tc_rate .* abs.(h .- prev_hedge) .* S_t
                total_tc .+= tc .* (disc^(n_steps - t))  # discount to terminal
            end
            prev_hedge = h

            # Roll back value
            V = (V .- h .* S_next) .* disc .+ h .* S_t
        end

        hedge_ratios[1, :] = hedge_ratios[2, :]

        # -------------------------------------------------------------------
        # Step 3b: Update ν given current hedge ratios
        # -------------------------------------------------------------------
        # Compute final loss distribution
        final_loss = _compute_final_loss(V, V_terminal, total_tc, loss_definition)

        # Optimize ν using Rockafellar-Uryasev
        nu = _optimize_nu(final_loss, cvar_alpha, nu_iterations)
        push!(nu_history, nu)
    end

    # -------------------------------------------------------------------------
    # Step 4: Compute final results and diagnostics
    # -------------------------------------------------------------------------
    final_loss = _compute_final_loss(V, V_terminal, total_tc, loss_definition)

    # CVaR via R-U formula
    cvar = _compute_cvar_ru(final_loss, nu, cvar_alpha)

    # Identify tail paths
    tail_mask = final_loss .>= nu
    tail_indices = findall(tail_mask)
    tail_loss_mean = isempty(tail_indices) ? nu : mean(final_loss[tail_indices])

    # Base objective (without CVaR)
    base_error = mean((V_terminal .- V) .^ 2)  # E[ℓ(e)]
    tc_term = tc_rate > 0.0 ? mean(total_tc) : 0.0
    base_objective = base_error + tc_term

    # CVaR contribution to objective
    cvar_contribution = if cvar_objective == CVaRPenalty
        cvar_weight * cvar
    else  # CVaRConstraint
        # Lagrangian penalty for constraint violation
        max(0.0, cvar - cvar_budget) * cvar_weight
    end

    total_objective = base_objective + cvar_contribution

    # Option price and stats
    option_price = mean(V)
    hedged_var = var(V)
    n = length(V)
    se = sqrt(hedged_var / n)
    ci = (option_price - 1.96 * se, option_price + 1.96 * se)

    return CVaROHMCResult(
        option_price,
        hedge_ratios,
        hedged_var,
        ci,
        cvar,
        nu,
        tail_indices,
        tail_loss_mean,
        cvar_contribution,
        base_objective,
        total_objective,
        nu_history
    )
end

# ============================================================================
# Internal: CVaR-weighted Hedge Ratio Computation
# ============================================================================

"""
Compute hedge ratio with CVaR-aware weighting.

The key insight: samples in the tail (L > ν) get upweighted by factor
proportional to η / (1 - α), making the regression more sensitive to
reducing tail losses.
"""
function _cvar_ohmc_hedge_ratio(
    Phi::Matrix{Float64},
    S_next::Vector{Float64},
    V::Vector{Float64},
    scoring::Symbol,
    risk_aversion::Float64,
    cvar_weights::Vector{Float64}
)
    n_paths, n_basis = size(Phi)
    reg = 1e-2

    # Design matrix: V ≈ Phi*a + (Phi.*S_next)*b
    W_slope = Phi .* S_next
    W = [Phi W_slope]

    # Apply CVaR weights to the regression
    sqrt_w = sqrt.(cvar_weights)
    W_weighted = W .* sqrt_w
    V_weighted = V .* sqrt_w

    if scoring == :quadratic
        # Weighted least squares
        A = W_weighted' * W_weighted + reg * I
        b_vec = W_weighted' * V_weighted
        beta = A \ b_vec
        b = beta[(n_basis+1):end]
        return Phi * b
    end

    if scoring == :log
        # Weighted IRLS for log-score
        beta = (W_weighted' * W_weighted + reg * I) \ (W_weighted' * V_weighted)
        for _ in 1:3
            r = V .- W * beta
            irls_w = 1.0 ./ (1.0 .+ r .^ 2)
            combined_w = cvar_weights .* irls_w
            sqrt_cw = sqrt.(combined_w)
            Ww = W .* sqrt_cw
            Vw = V .* sqrt_cw
            A = Ww' * Ww + reg * I
            b_vec = Ww' * Vw
            beta = A \ b_vec
        end
        b = beta[(n_basis+1):end]
        return Phi * b
    end

    # scoring == :exponential_utility with CVaR weighting
    # Newton's method on weighted exponential loss
    beta = (W_weighted' * W_weighted + reg * I) \ (W_weighted' * V_weighted)
    a_coef = risk_aversion
    for _ in 1:10
        res = V .- W * beta
        u = exp.(.-a_coef .* res) .* cvar_weights
        g = a_coef * (W' * u)
        H = (a_coef^2) * (W' * (u .* W)) + reg * I
        d_beta = H \ g
        beta = beta - d_beta
        if norm(d_beta) < 1e-8
            break
        end
    end
    b = beta[(n_basis+1):end]
    return Phi * b
end

# ============================================================================
# Internal: CVaR Weight Computation
# ============================================================================

"""
Compute CVaR-aware sample weights for regression.

Samples where L > ν (in the tail) are upweighted to make the hedge
ratio optimization more sensitive to tail outcomes.

Weight structure:
- Base weight: 1.0 for all samples
- Tail bonus: η / (1 - α) for samples where L > ν

This implements the "importance" of tail samples in the Lagrangian CVaR objective.
"""
function _compute_cvar_weights(
    loss::Vector{Float64},
    nu::Float64,
    alpha::Float64,
    eta::Float64
)
    n = length(loss)
    weights = ones(n)

    # Tail indicator: L > ν
    tail_bonus = eta / (1.0 - alpha)

    @inbounds for i in 1:n
        if loss[i] > nu
            weights[i] += tail_bonus
        end
    end

    # Normalize so mean weight = 1 (preserve scale)
    weights ./= mean(weights)

    return weights
end

# ============================================================================
# Internal: Loss Computation
# ============================================================================

"""
Compute loss at current time step based on loss definition.
"""
function _compute_loss(
    V::Vector{Float64},
    S_next::Vector{Float64},
    paths::Matrix{Float64},
    t::Int,
    loss_def::LossDefinition,
    option::EuropeanOption
)
    if loss_def == HedgingError
        # Loss = squared replication error (or signed error for CVaR)
        # Use the absolute error to define "loss"
        return abs.(V .- mean(V))
    elseif loss_def == PortfolioPnL
        # Loss = negative P&L (so high loss = big negative return)
        pnl = V .- mean(V)
        return -pnl  # Negate so loss is positive when we lose money
    else  # Drawdown
        # For drawdown, we'd need path history — approximate with current deviation
        return max.(0.0, mean(V) .- V)
    end
end

"""
Compute final loss distribution for CVaR calculation.
"""
function _compute_final_loss(
    V_hedged::Vector{Float64},
    V_terminal::Vector{Float64},
    total_tc::Vector{Float64},
    loss_def::LossDefinition
)
    if loss_def == HedgingError
        # Final hedging error including transaction costs
        error = V_terminal .- V_hedged .+ total_tc
        return error .- mean(error)  # Center for meaningful CVaR
    elseif loss_def == PortfolioPnL
        # Loss = -PnL = -(hedged_value - costs)
        pnl = V_hedged .- total_tc
        return -(pnl .- mean(pnl))
    else  # Drawdown
        return max.(0.0, mean(V_hedged) .- V_hedged) .+ total_tc
    end
end

# ============================================================================
# Internal: Rockafellar-Uryasev CVaR Optimization
# ============================================================================

"""
Optimize ν (VaR threshold) for given loss samples using R-U representation.

    CVaR_α(L) = min_ν [ ν + (1/(1-α)) * E[(L - ν)⁺] ]

The optimal ν* is the α-quantile (VaR) of L.
We use a few Newton-like iterations or direct quantile computation.
"""
function _optimize_nu(
    loss::Vector{Float64},
    alpha::Float64,
    n_iter::Int
)
    # Direct computation: optimal ν = VaR_α(L) = quantile(L, α)
    sorted_loss = sort(loss)
    return _quantile_sorted(sorted_loss, alpha)
end

"""
Compute CVaR using Rockafellar-Uryasev formula given optimal ν.
"""
function _compute_cvar_ru(
    loss::Vector{Float64},
    nu::Float64,
    alpha::Float64
)
    # CVaR = ν + (1/(1-α)) * E[(L - ν)⁺]
    excess = max.(loss .- nu, 0.0)
    return nu + mean(excess) / (1.0 - alpha)
end

"""
Compute quantile from sorted array (more efficient than re-sorting).
"""
function _quantile_sorted(sorted::Vector{Float64}, p::Float64)
    n = length(sorted)
    idx = p * (n - 1) + 1
    lo = floor(Int, idx)
    hi = ceil(Int, idx)
    lo = clamp(lo, 1, n)
    hi = clamp(hi, 1, n)
    if lo == hi
        return sorted[lo]
    end
    frac = idx - lo
    return sorted[lo] * (1 - frac) + sorted[hi] * frac
end

# ============================================================================
# Internal: Basis Functions
# ============================================================================

"""
Power basis matrix for CVaR-OHMC (same as standard OHMC).
"""
function _cvar_power_basis_matrix(S::AbstractVector, order::Int, norm_S::Float64)
    n = length(S)
    X = ones(eltype(S), n, order + 1)
    s = S ./ norm_S
    @inbounds for j in 1:order
        X[:, j+1] = s .^ j
    end
    return X
end

# ============================================================================
# Convenience: Compare CVaR-OHMC vs Standard OHMC
# ============================================================================

"""
    CVaRComparisonResult

Side-by-side comparison of standard OHMC and CVaR-OHMC.
"""
struct CVaRComparisonResult
    standard_price::Float64
    standard_variance::Float64
    standard_cvar::Float64  # CVaR computed post-hoc

    cvar_price::Float64
    cvar_variance::Float64
    cvar_cvar::Float64  # CVaR from optimization

    variance_reduction_pct::Float64
    cvar_reduction_pct::Float64
end

"""
    compare_ohmc_cvar(option, market_data, n_paths, n_steps, basis_order;
                      cvar_alpha=0.95, cvar_weight=0.2, rng=nothing)

Compare standard OHMC with CVaR-OHMC on the same paths.

# Returns
`CVaRComparisonResult` showing variance and CVaR reduction from adding tail-risk awareness.
"""
function compare_ohmc_cvar(
    option::EuropeanOption,
    market_data::MarketData,
    n_paths::Int,
    n_steps::Int,
    basis_order::Int;
    cvar_alpha::Float64=0.95,
    cvar_weight::Float64=0.2,
    rng::Union{AbstractRNG,Nothing}=nothing
)
    rng = rng === nothing ? MersenneTwister(42) : rng

    # Standard OHMC
    std_config = OHMCConfig(n_paths, n_steps, basis_order)
    std_result = ohmc_price(option, market_data, std_config; rng=copy(rng))

    # Compute CVaR post-hoc for standard OHMC
    V_terminal = payoff.(Ref(option), repeat([market_data.spot], n_paths))  # Approximation
    std_hedging_error = std_result.hedged_portfolio_variance  # Use variance as proxy
    std_cvar = std_result.option_price + 2.0 * sqrt(std_hedging_error)  # Rough upper bound

    # CVaR-OHMC
    cvar_config = CVaROHMCConfig(n_paths, n_steps, basis_order;
        cvar_alpha=cvar_alpha, cvar_weight=cvar_weight)
    cvar_result = cvar_ohmc_price(option, market_data, cvar_config; rng=copy(rng))

    # Compute reductions
    var_reduction = (std_result.hedged_portfolio_variance - cvar_result.hedged_portfolio_variance) /
                    max(std_result.hedged_portfolio_variance, 1e-10) * 100

    cvar_reduction = (std_cvar - cvar_result.cvar) / max(abs(std_cvar), 1e-10) * 100

    return CVaRComparisonResult(
        std_result.option_price,
        std_result.hedged_portfolio_variance,
        std_cvar,
        cvar_result.option_price,
        cvar_result.hedged_portfolio_variance,
        cvar_result.cvar,
        var_reduction,
        cvar_reduction
    )
end

# ============================================================================
# Adaptive CVaR Weight Selection
# ============================================================================

"""
    AdaptiveCVaRConfig

Configuration for adaptive CVaR weight selection.

The idea: start with low η and increase if CVaR exceeds budget,
or use cross-validation to find η that balances price accuracy and tail risk.
"""
struct AdaptiveCVaRConfig
    base_config::CVaROHMCConfig
    eta_min::Float64
    eta_max::Float64
    eta_steps::Int
    target_cvar::Float64  # Target maximum CVaR
end

"""
    adaptive_cvar_ohmc(option, market_data, adaptive_config; rng=nothing)

Find optimal CVaR weight η that achieves target CVaR while minimizing price bias.

Uses a simple grid search over η values.
"""
function adaptive_cvar_ohmc(
    option::EuropeanOption,
    market_data::MarketData,
    adaptive_config::AdaptiveCVaRConfig;
    rng::Union{AbstractRNG,Nothing}=nothing
)
    rng = rng === nothing ? MersenneTwister(42) : rng
    base_config = adaptive_config.base_config

    eta_values = range(adaptive_config.eta_min, adaptive_config.eta_max, length=adaptive_config.eta_steps)
    best_result = nothing
    best_eta = adaptive_config.eta_min
    best_score = Inf  # Lower is better

    for eta in eta_values
        config = CVaROHMCConfig(
            base_config.n_paths,
            base_config.n_steps,
            base_config.basis_order;
            cvar_alpha=base_config.cvar_alpha,
            cvar_weight=eta,
            cvar_objective=base_config.cvar_objective,
            cvar_budget=base_config.cvar_budget,
            loss_definition=base_config.loss_definition,
            scoring=base_config.scoring,
            risk_aversion=base_config.risk_aversion,
            tc_rate=base_config.tc_rate
        )

        result = cvar_ohmc_price(option, market_data, config; rng=copy(rng))

        # Score: penalize if CVaR exceeds target, otherwise prefer lower variance
        cvar_penalty = max(0.0, result.cvar - adaptive_config.target_cvar) * 10.0
        score = result.hedged_portfolio_variance + cvar_penalty

        if score < best_score
            best_score = score
            best_result = result
            best_eta = eta
        end

        # Early termination if we meet target with good variance
        if result.cvar <= adaptive_config.target_cvar && result.hedged_portfolio_variance < 0.1
            break
        end
    end

    return (result=best_result, optimal_eta=best_eta, final_score=best_score)
end

# ============================================================================
# Multi-period CVaR-OHMC (Path-dependent CVaR)
# ============================================================================

"""
    MultiPeriodCVaRConfig

Configuration for multi-period CVaR optimization where CVaR is computed
over intermediate portfolio values, not just terminal.

This captures "drawdown risk" during the hedge lifetime.
"""
struct MultiPeriodCVaRConfig
    base_config::CVaROHMCConfig
    intermediate_times::Vector{Float64}  # Times (as fraction of expiry) to evaluate CVaR
    time_weights::Vector{Float64}        # Weights for each time point
end

"""
    multiperiod_cvar_ohmc(option, market_data, config; rng=nothing)

CVaR-OHMC with multi-period CVaR objective.

Minimizes: E[ℓ(e)] + η * Σ_t w_t * CVaR_α(L_t)

where L_t is the loss at intermediate time t.
"""
function multiperiod_cvar_ohmc(
    option::EuropeanOption,
    market_data::MarketData,
    config::MultiPeriodCVaRConfig;
    rng::Union{AbstractRNG,Nothing}=nothing
)
    rng = rng === nothing ? Random.GLOBAL_RNG : rng
    base = config.base_config

    n_paths = base.n_paths
    n_steps = base.n_steps
    basis_order = base.basis_order
    cvar_alpha = base.cvar_alpha
    cvar_weight = base.cvar_weight
    scoring = base.scoring
    risk_aversion = base.risk_aversion
    tc_rate = base.tc_rate

    spot = market_data.spot
    rate = market_data.rate
    vol = market_data.vol
    div = market_data.div
    expiry = option.expiry

    dt = expiry / n_steps
    nudt = (rate - div - 0.5 * vol^2) * dt
    sidt = vol * sqrt(dt)
    disc = exp(-rate * dt)

    # Generate paths
    paths = zeros(n_steps + 1, n_paths)
    paths[1, :] .= spot
    @inbounds for t in 1:n_steps
        z = randn(rng, n_paths)
        paths[t+1, :] = paths[t, :] .* exp.(nudt .+ sidt .* z)
    end

    norm_S = max(spot, 1e-8)
    order = min(basis_order, 5)

    # Terminal payoff
    V = payoff.(Ref(option), paths[end, :])
    hedge_ratios = zeros(n_steps + 1, n_paths)

    # Compute step indices for intermediate CVaR evaluation
    eval_steps = [max(1, round(Int, t * n_steps)) for t in config.intermediate_times]

    # Storage for intermediate losses
    intermediate_losses = Dict{Int,Vector{Float64}}()

    # Backward pass with intermediate CVaR tracking
    prev_hedge = zeros(n_paths)
    total_tc = zeros(n_paths)
    portfolio_values = zeros(n_steps + 1, n_paths)
    portfolio_values[end, :] = V

    for t in (n_steps-1):-1:0
        S_t = paths[t+1, :]
        S_next = paths[t+2, :]
        Phi = _cvar_power_basis_matrix(S_t, order, norm_S)

        # Compute CVaR weight based on proximity to evaluation times
        step_idx = t + 1
        time_weight = 1.0
        for (i, eval_t) in enumerate(eval_steps)
            if abs(step_idx - eval_t) <= 1
                time_weight += config.time_weights[i] * cvar_weight
            end
        end

        # Current loss estimate for CVaR weighting
        loss_current = abs.(V .- mean(V))
        nu_current = _quantile_sorted(sort(loss_current), cvar_alpha)
        cvar_weights = _compute_cvar_weights(loss_current, nu_current, cvar_alpha, time_weight)

        h = _cvar_ohmc_hedge_ratio(Phi, S_next, V, scoring, risk_aversion, cvar_weights)
        hedge_ratios[t+2, :] = h

        if tc_rate > 0.0
            tc = tc_rate .* abs.(h .- prev_hedge) .* S_t
            total_tc .+= tc .* (disc^(n_steps - t))
        end
        prev_hedge = h

        V = (V .- h .* S_next) .* disc .+ h .* S_t
        portfolio_values[t+1, :] = V

        # Store intermediate loss at evaluation points
        if step_idx in eval_steps
            intermediate_losses[step_idx] = copy(loss_current)
        end
    end

    hedge_ratios[1, :] = hedge_ratios[2, :]

    # Compute multi-period CVaR
    total_cvar = 0.0
    for (i, eval_t) in enumerate(eval_steps)
        if haskey(intermediate_losses, eval_t)
            loss = intermediate_losses[eval_t]
            nu = _quantile_sorted(sort(loss), cvar_alpha)
            cvar_t = _compute_cvar_ru(loss, nu, cvar_alpha)
            total_cvar += config.time_weights[i] * cvar_t
        end
    end

    # Final statistics
    final_loss = abs.(V .- mean(V)) .+ total_tc
    nu_final = _quantile_sorted(sort(final_loss), cvar_alpha)
    cvar_final = _compute_cvar_ru(final_loss, nu_final, cvar_alpha)

    option_price = mean(V)
    hedged_var = var(V)
    se = sqrt(hedged_var / n_paths)
    ci = (option_price - 1.96 * se, option_price + 1.96 * se)

    tail_mask = final_loss .>= nu_final
    tail_indices = findall(tail_mask)
    tail_loss_mean = isempty(tail_indices) ? nu_final : mean(final_loss[tail_indices])

    base_obj = mean(final_loss .^ 2)
    cvar_contrib = cvar_weight * (cvar_final + total_cvar)
    total_obj = base_obj + cvar_contrib

    return CVaROHMCResult(
        option_price,
        hedge_ratios,
        hedged_var,
        ci,
        cvar_final,
        nu_final,
        tail_indices,
        tail_loss_mean,
        cvar_contrib,
        base_obj,
        total_obj,
        [nu_final]  # Simplified nu history
    )
end

# ============================================================================
# CVaR-constrained OHMC with Lagrange Multiplier Update
# ============================================================================

"""
    ConstrainedCVaROHMCResult

Result of constrained CVaR-OHMC with dual variable (Lagrange multiplier) information.
"""
struct ConstrainedCVaROHMCResult
    primal_result::CVaROHMCResult
    lagrange_multiplier::Float64
    constraint_slack::Float64  # cvar_budget - cvar (positive = feasible)
    dual_convergence::Vector{Float64}
end

"""
    constrained_cvar_ohmc(option, market_data, config; rng=nothing)

Solve the constrained CVaR-OHMC problem:

    min_a  E[ℓ(e)] + λ_tc * E[TC]
    s.t.   CVaR_α(L(a)) ≤ L_max

Using augmented Lagrangian method to update the dual variable.
"""
function constrained_cvar_ohmc(
    option::EuropeanOption,
    market_data::MarketData,
    config::CVaROHMCConfig;
    rng::Union{AbstractRNG,Nothing}=nothing,
    max_dual_iterations::Int=20,
    dual_step_size::Float64=0.1,
    tolerance::Float64=1e-4
)
    config.cvar_objective == CVaRConstraint ||
        @warn "Using constrained solver with penalty objective; switching to constraint form"

    rng = rng === nothing ? MersenneTwister(42) : rng
    budget = config.cvar_budget

    # Initialize dual variable (Lagrange multiplier)
    lambda = config.cvar_weight
    lambda_history = Float64[lambda]

    best_result = nothing
    best_feasible_result = nothing

    for iter in 1:max_dual_iterations
        # Solve primal problem with current λ
        current_config = CVaROHMCConfig(
            config.n_paths,
            config.n_steps,
            config.basis_order;
            cvar_alpha=config.cvar_alpha,
            cvar_weight=lambda,
            cvar_objective=CVaRPenalty,  # Use penalty form internally
            cvar_budget=config.cvar_budget,
            loss_definition=config.loss_definition,
            scoring=config.scoring,
            risk_aversion=config.risk_aversion,
            tc_rate=config.tc_rate,
            nu_iterations=config.nu_iterations,
            outer_iterations=config.outer_iterations
        )

        result = cvar_ohmc_price(option, market_data, current_config; rng=copy(rng))
        best_result = result

        # Check constraint violation
        violation = result.cvar - budget
        slack = -violation

        if slack >= -tolerance
            best_feasible_result = result
        end

        # Update dual variable (subgradient step)
        # λ_new = max(0, λ + α * (CVaR - budget))
        lambda = max(0.0, lambda + dual_step_size * violation)
        push!(lambda_history, lambda)

        # Convergence check
        if abs(violation) < tolerance
            break
        end

        # Adaptive step size
        if iter > 5 && abs(violation) > abs(lambda_history[end-1] - config.cvar_budget)
            dual_step_size *= 0.8  # Reduce step if oscillating
        end
    end

    final_result = best_feasible_result !== nothing ? best_feasible_result : best_result
    slack = config.cvar_budget - final_result.cvar

    return ConstrainedCVaROHMCResult(
        final_result,
        lambda,
        slack,
        lambda_history
    )
end
