# Julia Visualization for Finance

Expertise in creating financial visualizations using Plots.jl and other Julia plotting libraries.

## Plots.jl Basics

### Setup

```julia
using Plots

# Set default backend
gr()  # GR (fast, default)
# or
plotly()  # Interactive
pyplot()  # Matplotlib backend
```

## Option Payoff Diagrams

### Single Option Payoff

```julia
function plot_payoff(option::VanillaOption, spot_range=nothing)
    (; strike) = option

    # Default range: 50% around strike
    if isnothing(spot_range)
        spot_range = range(strike * 0.5, strike * 1.5, length=100)
    end

    payoffs = [payoff(option, S) for S in spot_range]

    plot(
        spot_range,
        payoffs,
        label=string(typeof(option)),
        xlabel="Spot Price",
        ylabel="Payoff",
        title="Option Payoff Diagram",
        linewidth=2,
        legend=:best,
        grid=true
    )

    # Add zero line
    hline!([0], linestyle=:dash, linewidth=1, label="", color=:black)

    # Mark strike
    vline!([strike], linestyle=:dot, linewidth=1, label="Strike", color=:red)
end
```

### Strategy Payoff (Multiple Positions)

```julia
function plot_strategy_payoff(
    positions::Vector{Tuple{VanillaOption, Int}},  # (option, quantity)
    spot_range
)
    total_payoff = zeros(length(spot_range))

    p = plot(
        xlabel="Spot Price",
        ylabel="Profit/Loss",
        title="Strategy Payoff",
        legend=:best,
        grid=true
    )

    for (option, quantity) in positions
        payoffs = [payoff(option, S) * quantity for S in spot_range]
        total_payoff .+= payoffs

        plot!(p, spot_range, payoffs,
              label="$quantity × $(typeof(option))",
              linewidth=1.5,
              alpha=0.7)
    end

    # Plot total
    plot!(p, spot_range, total_payoff,
          label="Total",
          linewidth=3,
          color=:black)

    # Zero line
    hline!(p, [0], linestyle=:dash, linewidth=1, label="", color=:gray)

    return p
end

# Example: Bull Call Spread
call_long = EuropeanCall(100.0, 1.0)
call_short = EuropeanCall(110.0, 1.0)

strategy = [(call_long, 1), (call_short, -1)]
spot_range = 80:0.5:120

plot_strategy_payoff(strategy, spot_range)
```

## Price Surfaces

### Price vs Spot and Time

```julia
function plot_price_surface(
    option::VanillaOption,
    engine::PricingEngine,
    base_data::MarketData;
    spot_range=nothing,
    time_range=nothing
)
    (; strike) = option
    (; spot, rate, vol, div) = base_data

    # Defaults
    if isnothing(spot_range)
        spot_range = range(strike * 0.5, strike * 1.5, length=50)
    end
    if isnothing(time_range)
        time_range = range(0.01, option.expiry, length=50)
    end

    # Compute prices
    prices = zeros(length(spot_range), length(time_range))

    for (i, S) in enumerate(spot_range)
        for (j, T) in enumerate(time_range)
            data_ij = MarketData(S, rate, vol, div)
            opt_ij = typeof(option)(strike, T)
            prices[i, j] = price(opt_ij, engine, data_ij)
        end
    end

    # 3D surface plot
    surface(
        spot_range,
        time_range,
        prices',
        xlabel="Spot Price",
        ylabel="Time to Expiry",
        zlabel="Option Price",
        title="$(typeof(option)) Price Surface",
        camera=(30, 30),
        colorbar=true
    )
end
```

### Heatmap Version

```julia
function plot_price_heatmap(
    option::VanillaOption,
    engine::PricingEngine,
    base_data::MarketData;
    spot_range=nothing,
    time_range=nothing
)
    # ... compute prices same as above ...

    heatmap(
        spot_range,
        time_range,
        prices',
        xlabel="Spot Price",
        ylabel="Time to Expiry",
        title="$(typeof(option)) Price Heatmap",
        color=:viridis,
        colorbar=true
    )
end
```

## Greeks Visualization

### Greeks vs Spot Price

```julia
function plot_greeks_vs_spot(
    option::VanillaOption,
    engine::BlackScholes,
    base_data::MarketData;
    spot_range=nothing
)
    (; strike) = option

    if isnothing(spot_range)
        spot_range = range(strike * 0.7, strike * 1.3, length=100)
    end

    # Compute all Greeks
    deltas = Float64[]
    gammas = Float64[]
    vegas = Float64[]
    thetas = Float64[]

    for S in spot_range
        data_S = MarketData(S, base_data.rate, base_data.vol, base_data.div)
        gks = greeks(option, engine, data_S)

        push!(deltas, gks.delta)
        push!(gammas, gks.gamma)
        push!(vegas, gks.vega)
        push!(thetas, gks.theta)
    end

    # Create subplots
    p1 = plot(spot_range, deltas, title="Delta", ylabel="Δ", legend=false)
    vline!(p1, [strike], linestyle=:dot, color=:red)

    p2 = plot(spot_range, gammas, title="Gamma", ylabel="Γ", legend=false)
    vline!(p2, [strike], linestyle=:dot, color=:red)

    p3 = plot(spot_range, vegas, title="Vega", ylabel="ν", legend=false)
    vline!(p3, [strike], linestyle=:dot, color=:red)

    p4 = plot(spot_range, thetas, title="Theta", ylabel="Θ", legend=false)
    vline!(p4, [strike], linestyle=:dot, color=:red)

    plot(p1, p2, p3, p4, layout=(2,2), size=(800, 600),
         plot_title="Greeks for $(typeof(option))")
end
```

### Delta-Gamma Ladder

```julia
function plot_delta_gamma_ladder(
    option::VanillaOption,
    engine::BlackScholes,
    base_data::MarketData;
    n_levels::Int=21
)
    (; spot, strike) = (base_data.spot, option.strike)

    spot_levels = range(spot * 0.8, spot * 1.2, length=n_levels)

    prices = Float64[]
    deltas = Float64[]
    gammas = Float64[]

    for S in spot_levels
        data_S = MarketData(S, base_data.rate, base_data.vol, base_data.div)
        prc = price(option, engine, data_S)
        gks = greeks(option, engine, data_S)

        push!(prices, prc)
        push!(deltas, gks.delta)
        push!(gammas, gks.gamma)
    end

    # Create table plot
    p1 = plot(spot_levels, prices, label="Price", ylabel="Price",
              linewidth=2, legend=:best)
    vline!(p1, [spot], label="Current Spot", linestyle=:dash)

    p2 = plot(spot_levels, deltas, label="Delta", ylabel="Δ",
              linewidth=2, color=:red, legend=:best)
    vline!(p2, [spot], label="Current Spot", linestyle=:dash)

    p3 = plot(spot_levels, gammas, label="Gamma", ylabel="Γ",
              linewidth=2, color=:green, xlabel="Spot Price", legend=:best)
    vline!(p3, [spot], label="Current Spot", linestyle=:dash)

    plot(p1, p2, p3, layout=(3,1), size=(700, 800))
end
```

## Volatility Surface

### 3D Surface

```julia
function plot_volatility_surface(
    vol_surface::VolatilitySurface;
    strike_range=nothing,
    expiry_range=nothing
)
    strikes = [p.strike for p in vol_surface.points]
    expiries = [p.expiry for p in vol_surface.points]
    vols = [p.implied_vol for p in vol_surface.points]

    # Group by strike and expiry for meshgrid
    unique_strikes = sort(unique(strikes))
    unique_expiries = sort(unique(expiries))

    # Create grid
    vol_grid = zeros(length(unique_strikes), length(unique_expiries))

    for p in vol_surface.points
        i = findfirst(==(p.strike), unique_strikes)
        j = findfirst(==(p.expiry), unique_expiries)
        vol_grid[i, j] = p.implied_vol
    end

    # Plot
    surface(
        unique_strikes,
        unique_expiries,
        vol_grid',
        xlabel="Strike",
        ylabel="Time to Expiry",
        zlabel="Implied Volatility",
        title="Volatility Surface",
        camera=(45, 45),
        colorbar=true,
        color=:plasma
    )
end
```

### Volatility Smile (Fixed Expiry)

```julia
function plot_volatility_smile(
    vol_surface::VolatilitySurface,
    expiry::Float64;
    tolerance::Float64=0.01
)
    # Filter points near the expiry
    relevant = filter(p -> abs(p.expiry - expiry) < tolerance, vol_surface.points)

    if isempty(relevant)
        error("No data points found near expiry $expiry")
    end

    strikes = [p.strike for p in relevant]
    vols = [p.implied_vol for p in relevant]

    # Sort by strike
    perm = sortperm(strikes)
    strikes = strikes[perm]
    vols = vols[perm]

    plot(
        strikes,
        vols * 100,  # Convert to percentage
        xlabel="Strike",
        ylabel="Implied Volatility (%)",
        title="Volatility Smile (T=$(round(expiry, digits=2)))",
        linewidth=2,
        marker=:circle,
        markersize=4,
        legend=false,
        grid=true
    )

    # Mark ATM
    vline!([vol_surface.spot], label="ATM", linestyle=:dash, color=:red)
end
```

## Monte Carlo Path Visualization

### Sample Paths

```julia
function plot_sample_paths(
    spot::Float64,
    rate::Float64,
    vol::Float64,
    div::Float64,
    expiry::Float64,
    n_paths::Int=50,
    n_steps::Int=100
)
    dt = expiry / n_steps
    times = 0:dt:expiry

    p = plot(
        xlabel="Time",
        ylabel="Spot Price",
        title="Monte Carlo Sample Paths",
        legend=false,
        alpha=0.3
    )

    for _ in 1:n_paths
        S = spot
        path = [S]

        for step in 1:n_steps
            Z = randn()
            S *= exp((rate - div - 0.5 * vol^2) * dt + vol * sqrt(dt) * Z)
            push!(path, S)
        end

        plot!(p, times, path, color=:blue, linewidth=0.5)
    end

    # Add starting spot
    hline!(p, [spot], color=:red, linewidth=2, linestyle=:dash,
           label="Initial Spot")

    return p
end
```

### Convergence Plot

```julia
function plot_mc_convergence(
    option::VanillaOption,
    engine_generator::Function,  # n_paths -> MonteCarlo(steps, n_paths)
    data::MarketData,
    analytical_price::Float64;
    path_counts=[100, 500, 1000, 5000, 10000, 50000]
)
    estimates = Float64[]
    std_errors = Float64[]

    for n_paths in path_counts
        engine = engine_generator(n_paths)

        # Multiple runs to estimate std error
        prices = [price(option, engine, data) for _ in 1:10]

        push!(estimates, mean(prices))
        push!(std_errors, std(prices))
    end

    # Plot estimates with error bars
    p = plot(
        path_counts,
        estimates,
        yerror=1.96 * std_errors,  # 95% CI
        xlabel="Number of Paths",
        ylabel="Estimated Price",
        title="Monte Carlo Convergence",
        marker=:circle,
        markersize=5,
        linewidth=2,
        legend=false,
        xscale=:log10
    )

    # Analytical reference
    hline!(p, [analytical_price], linestyle=:dash, linewidth=2,
           color=:red, label="Analytical")

    return p
end
```

## Performance Comparison

### Benchmarking Plot

```julia
using BenchmarkTools

function plot_performance_comparison(
    option::EuropeanOption,
    data::MarketData;
    steps_range=[10, 20, 50, 100, 200, 500]
)
    times_binom = Float64[]
    times_mc = Float64[]

    for steps in steps_range
        # Binomial
        t_binom = @belapsed price($option, Binomial($steps), $data)
        push!(times_binom, t_binom * 1000)  # Convert to ms

        # Monte Carlo (10000 paths)
        t_mc = @belapsed price($option, MonteCarlo($steps, 10000), $data)
        push!(times_mc, t_mc * 1000)
    end

    plot(
        steps_range,
        [times_binom times_mc],
        xlabel="Number of Steps",
        ylabel="Time (ms)",
        title="Performance Comparison",
        label=["Binomial" "Monte Carlo"],
        linewidth=2,
        marker=[:circle :square],
        markersize=5,
        legend=:topleft,
        yscale=:log10,
        grid=true
    )
end
```

## Animations

### Time Decay Animation

```julia
using Plots

@gif for days_to_expiry in 365:-5:1
    T = days_to_expiry / 365
    option = EuropeanCall(100.0, T)

    spot_range = 80:0.5:120
    prices = [price(option, BlackScholes(), MarketData(S, 0.05, 0.2, 0.0))
              for S in spot_range]

    plot(
        spot_range,
        prices,
        xlabel="Spot Price",
        ylabel="Option Price",
        title="Call Option - Days to Expiry: $days_to_expiry",
        ylim=(0, 25),
        linewidth=2,
        legend=false,
        grid=true
    )

    vline!([100.0], linestyle=:dot, color=:red)
end fps=10
```

## Interactive Plots (Plotly)

```julia
using Plots
plotly()

function interactive_greeks(
    option::EuropeanCall,
    base_data::MarketData
)
    spot_range = 50:1:150
    vol_range = 0.1:0.05:0.5

    # Create meshgrid
    deltas = zeros(length(spot_range), length(vol_range))

    for (i, S) in enumerate(spot_range)
        for (j, σ) in enumerate(vol_range)
            data_ij = MarketData(S, base_data.rate, σ, base_data.div)
            gks = greeks(option, BlackScholes(), data_ij)
            deltas[i, j] = gks.delta
        end
    end

    surface(
        spot_range,
        vol_range * 100,  # Convert to %
        deltas',
        xlabel="Spot Price",
        ylabel="Volatility (%)",
        zlabel="Delta",
        title="Interactive Delta Surface (hover for values)",
        colorbar=true
    )
end
```

## Saving Plots

```julia
# Save as PNG
savefig("option_payoff.png")

# Save as PDF (vector)
savefig("option_payoff.pdf")

# Save as HTML (interactive)
plotly()
p = plot(...)
savefig(p, "interactive_plot.html")

# High DPI
plot(..., dpi=300)
savefig("high_res.png")
```

## Themes and Styling

```julia
# Use predefined themes
theme(:dark)
theme(:vibrant)
theme(:juno)

# Custom theme
custom_theme = PlotTheme(
    bg=:white,
    fg=:black,
    linewidth=2,
    markersize=4,
    gridcolor=:gray,
    gridlinewidth=0.5
)

plot(..., theme=custom_theme)

# Publication-ready settings
default(
    fontfamily="Computer Modern",
    framestyle=:box,
    grid=true,
    gridlinewidth=0.5,
    gridalpha=0.3,
    size=(600, 400),
    dpi=300
)
```

## Further Reading

- Plots.jl documentation: http://docs.juliaplots.org/
- PlotlyJS.jl for advanced interactivity
- Makie.jl for high-performance visualization
- StatsPlots.jl for statistical plots
