# Exotic Options & Path-Dependent Products

Expertise in pricing exotic and path-dependent options beyond vanilla European and American options.

## Barrier Options

Options with knockout or knock-in features.

### Types

- **Knock-Out**: Option becomes worthless if barrier hit
- **Knock-In**: Option activates if barrier hit
- **Up-and-Out**: Barrier above spot
- **Down-and-Out**: Barrier below spot
- **Up-and-In**: Barrier above spot
- **Down-and-In**: Barrier below spot

### Monte Carlo Implementation

```julia
struct BarrierOption <: VanillaOption
    strike::Float64
    expiry::Float64
    barrier::Float64
    barrier_type::Symbol  # :up_out, :down_out, :up_in, :down_in
    option_type::Symbol   # :call, :put
end

function price_barrier_mc(
    option::BarrierOption,
    data::MarketData,
    n_paths::Int,
    n_steps::Int
)
    (; spot, rate, vol, div) = data
    (; strike, expiry, barrier, barrier_type, option_type) = option

    dt = expiry / n_steps
    drift = (rate - div - 0.5 * vol^2) * dt
    vol_sqrt_dt = vol * sqrt(dt)
    disc = exp(-rate * expiry)

    payoff_sum = 0.0

    for i in 1:n_paths
        S = spot
        barrier_hit = false

        # Generate path and check barrier
        for step in 1:n_steps
            Z = randn()
            S *= exp(drift + vol_sqrt_dt * Z)

            # Check barrier
            if (barrier_type in (:up_out, :up_in) && S >= barrier) ||
               (barrier_type in (:down_out, :down_in) && S <= barrier)
                barrier_hit = true
                break
            end
        end

        # Compute payoff based on barrier type
        if barrier_type in (:up_out, :down_out)
            # Knock-out: pay if barrier NOT hit
            if !barrier_hit
                poff = option_type == :call ? max(0, S - strike) : max(0, strike - S)
                payoff_sum += poff
            end
        else  # Knock-in
            # Pay if barrier WAS hit
            if barrier_hit
                poff = option_type == :call ? max(0, S - strike) : max(0, strike - S)
                payoff_sum += poff
            end
        end
    end

    return disc * payoff_sum / n_paths
end
```

### Brownian Bridge Correction

More accurate barrier checking:

```julia
function barrier_hit_probability_brownian_bridge(
    S_prev::Float64,
    S_next::Float64,
    barrier::Float64,
    drift::Float64,
    vol::Float64,
    dt::Float64
)
    # Probability that path hit barrier between S_prev and S_next

    # If already crossed, certain hit
    if (S_prev <= barrier <= S_next) || (S_next <= barrier <= S_prev)
        return 1.0
    end

    # Brownian bridge adjustment
    a = log(barrier / S_prev)
    b = log(S_next / S_prev)

    # Probability of hitting barrier
    p = exp(-2 * a * b / (vol^2 * dt))

    return min(1.0, max(0.0, p))
end
```

## Asian Options

Payoff depends on average price over time.

### Types

- **Average Price Call**: payoff = max(0, Average - K)
- **Average Strike Call**: payoff = max(0, S_T - Average)
- **Arithmetic Average**: (S_1 + S_2 + ... + S_n) / n
- **Geometric Average**: (S_1 * S_2 * ... * S_n)^(1/n)

### Implementation

```julia
struct AsianOption <: VanillaOption
    strike::Float64
    expiry::Float64
    averaging_type::Symbol  # :arithmetic, :geometric
    option_type::Symbol     # :call, :put
    payoff_type::Symbol     # :average_price, :average_strike
end

function price_asian_mc(
    option::AsianOption,
    data::MarketData,
    n_paths::Int,
    n_steps::Int
)
    (; spot, rate, vol, div) = data
    (; strike, expiry, averaging_type, option_type, payoff_type) = option

    dt = expiry / n_steps
    drift = (rate - div - 0.5 * vol^2) * dt
    vol_sqrt_dt = vol * sqrt(dt)
    disc = exp(-rate * expiry)

    payoff_sum = 0.0

    for i in 1:n_paths
        S = spot
        prices = Float64[S]

        # Generate path
        for step in 1:n_steps
            Z = randn()
            S *= exp(drift + vol_sqrt_dt * Z)
            push!(prices, S)
        end

        # Compute average
        if averaging_type == :arithmetic
            avg = mean(prices)
        else  # geometric
            avg = exp(mean(log.(prices)))
        end

        # Compute payoff
        if payoff_type == :average_price
            poff = option_type == :call ? max(0, avg - strike) : max(0, strike - avg)
        else  # average_strike
            S_T = prices[end]
            poff = option_type == :call ? max(0, S_T - avg) : max(0, avg - S_T)
        end

        payoff_sum += poff
    end

    return disc * payoff_sum / n_paths
end
```

### Geometric Asian (Closed Form)

```julia
function price_geometric_asian_call(
    strike::Float64,
    expiry::Float64,
    data::MarketData
)
    (; spot, rate, vol, div) = data

    # Adjusted parameters for geometric average
    σ_adj = vol / sqrt(3)
    b_adj = 0.5 * (rate - div - vol^2 / 6)

    # Use Black-Scholes with adjusted parameters
    d1 = (log(spot / strike) + (b_adj + 0.5 * σ_adj^2) * expiry) / (σ_adj * sqrt(expiry))
    d2 = d1 - σ_adj * sqrt(expiry)

    Φ_d1 = norm_cdf(d1)
    Φ_d2 = norm_cdf(d2)

    price = spot * exp((b_adj - rate) * expiry) * Φ_d1 -
            strike * exp(-rate * expiry) * Φ_d2

    return price
end
```

## Lookback Options

Payoff depends on maximum or minimum price over time.

```julia
struct LookbackOption <: VanillaOption
    expiry::Float64
    lookback_type::Symbol  # :fixed_strike, :floating_strike
    option_type::Symbol    # :call, :put
    strike::Union{Float64, Nothing}  # For fixed strike only
end

function price_lookback_mc(
    option::LookbackOption,
    data::MarketData,
    n_paths::Int,
    n_steps::Int
)
    (; spot, rate, vol, div) = data
    (; expiry, lookback_type, option_type, strike) = option

    dt = expiry / n_steps
    drift = (rate - div - 0.5 * vol^2) * dt
    vol_sqrt_dt = vol * sqrt(dt)
    disc = exp(-rate * expiry)

    payoff_sum = 0.0

    for i in 1:n_paths
        S = spot
        S_max = spot
        S_min = spot

        # Generate path and track extrema
        for step in 1:n_steps
            Z = randn()
            S *= exp(drift + vol_sqrt_dt * Z)
            S_max = max(S_max, S)
            S_min = min(S_min, S)
        end

        # Compute payoff
        if lookback_type == :fixed_strike
            if option_type == :call
                poff = max(0, S_max - strike)
            else
                poff = max(0, strike - S_min)
            end
        else  # floating_strike
            if option_type == :call
                poff = S - S_min  # Always positive
            else
                poff = S_max - S  # Always positive
            end
        end

        payoff_sum += poff
    end

    return disc * payoff_sum / n_paths
end
```

## Digital (Binary) Options

Discontinuous payoff: fixed amount if condition met.

```julia
struct DigitalOption <: VanillaOption
    strike::Float64
    expiry::Float64
    payout::Float64
    option_type::Symbol  # :call, :put
end

function price_digital_mc(
    option::DigitalOption,
    data::MarketData,
    n_paths::Int
)
    (; spot, rate, vol, div) = data
    (; strike, expiry, payout, option_type) = option

    drift = (rate - div - 0.5 * vol^2) * expiry
    vol_sqrt_T = vol * sqrt(expiry)
    disc = exp(-rate * expiry)

    hits = 0

    for i in 1:n_paths
        Z = randn()
        S_T = spot * exp(drift + vol_sqrt_T * Z)

        if (option_type == :call && S_T > strike) ||
           (option_type == :put && S_T < strike)
            hits += 1
        end
    end

    probability = hits / n_paths
    return disc * payout * probability
end

# Analytical formula
function price_digital_call_analytical(
    strike::Float64,
    expiry::Float64,
    payout::Float64,
    data::MarketData
)
    (; spot, rate, vol, div) = data

    d2 = (log(spot / strike) + (rate - div - 0.5 * vol^2) * expiry) / (vol * sqrt(expiry))

    return payout * exp(-rate * expiry) * norm_cdf(d2)
end
```

## Chooser Options

Holder chooses call or put at intermediate time.

```julia
struct ChooserOption <: VanillaOption
    strike::Float64
    expiry::Float64
    choice_time::Float64  # When to choose
end

function price_chooser(
    option::ChooserOption,
    data::MarketData
)
    (; strike, expiry, choice_time) = option
    (; spot, rate, vol, div) = data

    # At choice time, holder picks max(call, put)
    # Closed form: Chooser = Call(K,T) + Put(K*exp(-r(T-t)), t)
    # This is a simplified approximation

    # Full expiry call
    call = EuropeanCall(strike, expiry)
    call_price = price(call, BlackScholes(), data)

    # Put to choice time with adjusted strike
    K_adj = strike * exp(-rate * (expiry - choice_time))
    put = EuropeanPut(K_adj, choice_time)
    put_price = price(put, BlackScholes(), data)

    return call_price + put_price
end
```

## Bermudan Options

Discrete exercise dates (between European and American).

```julia
struct BermudanOption <: VanillaOption
    strike::Float64
    expiry::Float64
    exercise_dates::Vector{Float64}  # Allowed exercise times
    option_type::Symbol  # :call, :put
end

function price_bermudan_lsm(
    option::BermudanOption,
    data::MarketData,
    n_paths::Int
)
    # Similar to Longstaff-Schwartz but only regress at exercise dates
    (; strike, expiry, exercise_dates, option_type) = option
    (; spot, rate, vol, div) = data

    # Sort exercise dates
    ex_dates = sort(exercise_dates)
    n_dates = length(ex_dates)

    # Generate paths to all exercise dates
    paths = generate_paths_to_dates(spot, rate, vol, div, ex_dates, n_paths)

    # Initialize cash flows at expiry
    S_T = paths[:, end]
    if option_type == :call
        cash_flows = max.(0, S_T .- strike)
    else
        cash_flows = max.(0, strike .- S_T)
    end

    # Backward induction through exercise dates
    for i in (n_dates-1):-1:1
        t = ex_dates[i]
        t_next = i == n_dates ? expiry : ex_dates[i+1]
        dt = t_next - t

        S_t = paths[:, i]

        # Exercise value
        if option_type == :call
            exercise_value = max.(0, S_t .- strike)
        else
            exercise_value = max.(0, strike .- S_t)
        end

        # Continuation value (regress)
        itm = exercise_value .> 0
        if sum(itm) > 10  # Need enough ITM paths
            X = hcat(ones(sum(itm)), S_t[itm], S_t[itm].^2)
            Y = cash_flows[itm] * exp(-rate * dt)
            β = X \ Y
            continuation = X * β

            # Update cash flows: exercise or continue
            exercise_now = exercise_value[itm] .> continuation
            cash_flows[itm] = ifelse.(exercise_now, exercise_value[itm],
                                      cash_flows[itm] * exp(-rate * dt))
        else
            # Not enough ITM paths, just discount
            cash_flows .*= exp(-rate * dt)
        end
    end

    # Discount from first exercise date to today
    discount = exp(-rate * ex_dates[1])
    return mean(cash_flows) * discount
end
```

## Basket Options

Multiple underlying assets.

```julia
struct BasketOption
    strikes::Vector{Float64}
    weights::Vector{Float64}
    expiry::Float64
    basket_type::Symbol  # :average, :best_of, :worst_of
    option_type::Symbol  # :call, :put
end

function price_basket_mc(
    option::BasketOption,
    spots::Vector{Float64},
    rate::Float64,
    vols::Vector{Float64},
    corr_matrix::Matrix{Float64},
    n_paths::Int
)
    (; weights, expiry, basket_type, option_type, strikes) = option

    n_assets = length(spots)
    dt = expiry
    disc = exp(-rate * expiry)

    # Cholesky decomposition for correlated normals
    L = cholesky(corr_matrix).L

    payoff_sum = 0.0

    for i in 1:n_paths
        # Generate correlated random numbers
        Z = randn(n_assets)
        Z_corr = L * Z

        # Terminal prices
        S_T = similar(spots)
        for j in 1:n_assets
            drift = (rate - 0.5 * vols[j]^2) * dt
            S_T[j] = spots[j] * exp(drift + vols[j] * sqrt(dt) * Z_corr[j])
        end

        # Compute basket value
        if basket_type == :average
            basket_value = sum(weights .* S_T) / sum(weights)
            strike = strikes[1]
        elseif basket_type == :best_of
            basket_value = maximum(S_T)
            strike = strikes[1]
        else  # worst_of
            basket_value = minimum(S_T)
            strike = strikes[1]
        end

        # Payoff
        if option_type == :call
            poff = max(0, basket_value - strike)
        else
            poff = max(0, strike - basket_value)
        end

        payoff_sum += poff
    end

    return disc * payoff_sum / n_paths
end
```

## Rainbow Options

Multiple assets with complex payoffs.

```julia
# Best-of-two call: max(Call on S1, Call on S2)
function price_best_of_two_call(
    strikes::Tuple{Float64, Float64},
    expiry::Float64,
    spots::Tuple{Float64, Float64},
    vols::Tuple{Float64, Float64},
    correlation::Float64,
    rate::Float64,
    n_paths::Int
)
    disc = exp(-rate * expiry)
    dt = expiry

    payoff_sum = 0.0

    for i in 1:n_paths
        # Correlated normals
        Z1 = randn()
        Z2 = correlation * Z1 + sqrt(1 - correlation^2) * randn()

        # Terminal prices
        S1_T = spots[1] * exp((rate - 0.5 * vols[1]^2) * dt + vols[1] * sqrt(dt) * Z1)
        S2_T = spots[2] * exp((rate - 0.5 * vols[2]^2) * dt + vols[2] * sqrt(dt) * Z2)

        # Payoffs from each option
        payoff1 = max(0, S1_T - strikes[1])
        payoff2 = max(0, S2_T - strikes[2])

        # Best of two
        poff = max(payoff1, payoff2)

        payoff_sum += poff
    end

    return disc * payoff_sum / n_paths
end
```

## Practical Tips

### Variance Reduction for Exotics

- **Antithetic variates**: Very effective for path-dependent options
- **Control variates**: Use simpler related option (e.g., geometric Asian for arithmetic)
- **Importance sampling**: For barrier options, bias towards barrier
- **Stratification**: For digital options, stratify around strike

### Convergence Issues

- **Barriers**: Need many time steps (daily or finer)
- **Digitals**: Discontinuous payoff → slow convergence
- **Lookbacks**: Need fine time grid to capture extrema

### Testing Exotics

```julia
@testset "Exotic Option Properties" begin
    # Barrier: out + in = vanilla
    data = MarketData(100.0, 0.05, 0.2, 0.0)
    call = EuropeanCall(100.0, 1.0)

    barrier_up_out = BarrierOption(100.0, 1.0, 120.0, :up_out, :call)
    barrier_up_in = BarrierOption(100.0, 1.0, 120.0, :up_in, :call)

    vanilla_price = price(call, MonteCarlo(100, 10000), data)
    out_price = price_barrier_mc(barrier_up_out, data, 10000, 100)
    in_price = price_barrier_mc(barrier_up_in, data, 10000, 100)

    @test out_price + in_price ≈ vanilla_price rtol=0.05
end
```

## Further Reading

- Haug: "The Complete Guide to Option Pricing Formulas" (comprehensive exotic formulas)
- Wilmott: Exotic options chapters
- Zhang: "The Mathematics of Financial Derivatives" (path-dependent options)
