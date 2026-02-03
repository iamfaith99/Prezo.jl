"""
    Greek

Abstract base type for all option Greeks (sensitivities).

Greeks measure the sensitivity of option prices to various parameters.
They are essential for risk management, hedging, and trading strategies.

# Hierarchy
- `FirstOrderGreek` <: Greek - Sensitivity to single parameter
- `SecondOrderGreek` <: Greek - Second derivative or cross derivatives
- `ThirdOrderGreek` <: Greek - Third derivatives (for advanced risk management)

See also: [`Delta`](@ref), [`Gamma`](@ref), [`Vega`](@ref)
"""
abstract type Greek end

"""
    FirstOrderGreek <: Greek

Abstract type for first-order sensitivities (price sensitivities).

These measure how option price changes with respect to a single parameter:
- Spot price (Delta)
- Time to expiry (Theta)
- Volatility (Vega)
- Interest rate (Rho)
- Dividend yield (Phi)

See also: [`Delta`](@ref), [`Theta`](@ref), [`Vega`](@ref), [`Rho`](@ref)
"""
abstract type FirstOrderGreek <: Greek end

"""
    SecondOrderGreek <: Greek

Abstract type for second-order sensitivities (convexity and cross-derivatives).

These measure how first-order Greeks change with respect to parameters:
- Gamma: How Delta changes with spot
- Vanna: How Delta changes with volatility
- Vomma: How Vega changes with volatility

See also: [`Gamma`](@ref), [`Vanna`](@ref), [`Vomma`](@ref)
"""
abstract type SecondOrderGreek <: Greek end

"""
    ThirdOrderGreek <: Greek

Abstract type for third-order sensitivities.

These are rarely used in practice but important for:
- Understanding tail risks
- Advanced hedging strategies
- Stress testing

See also: [`Speed`](@ref), [`Color`](@ref)
"""
abstract type ThirdOrderGreek <: Greek end

## First Order Greeks

"""
    Delta() <: FirstOrderGreek

Price sensitivity to underlying asset price.

Mathematically: Δ = ∂V/∂S

# Interpretation
- Delta represents the hedge ratio (number of shares needed to hedge)
- For calls: 0 ≤ Δ ≤ 1 (positive delta, increases with spot)
- For puts: -1 ≤ Δ ≤ 0 (negative delta, decreases with spot)
- ATM options (spot ≈ strike) typically have |Δ| ≈ 0.5

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
delta = greek(call, Delta(), data)  # ≈ 0.6 for ATM call
```

# Hedging Application
To delta-hedge a short call position:
```julia
n_calls = -100  # Short 100 calls
delta = greek(call, Delta(), data)
shares_needed = -n_calls * delta  # Buy this many shares
```

See also: [`Gamma`](@ref), [`Charm`](@ref)
"""
struct Delta <: FirstOrderGreek end

"""
    Gamma() <: SecondOrderGreek

Delta sensitivity to underlying asset price (second derivative).

Mathematically: Γ = ∂²V/∂S² = ∂Δ/∂S

# Interpretation
- Gamma measures how fast Delta changes with spot price
- Always positive for long options (convexity)
- Highest for ATM options near expiration
- Important for dynamic hedging (delta rebalancing frequency)

# Trading Applications
- Long gamma: Benefits from large price moves (volatility trading)
- Short gamma: Benefits from small/no moves (time decay trading)
- Gamma scalping: Rebalance delta to capture profits from volatility

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
gamma = greek(call, Gamma(), data)

# Gamma profit: 0.5 * Gamma * (ΔS)²
price_move = 5.0
gamma_pnl = 0.5 * gamma * price_move^2
```

See also: [`Delta`](@ref), [`Speed`](@ref), [`Color`](@ref)
"""
struct Gamma <: SecondOrderGreek end

"""
    Theta() <: FirstOrderGreek

Price sensitivity to time decay (per year).

Mathematically: Θ = ∂V/∂τ (where τ = time to expiry)

# Interpretation
- Measures how much value is lost per year as expiration approaches
- Usually negative (options lose value over time)
- More negative for shorter-dated options
- Can be positive for deep ITM puts (time value of money)

# Time Decay Characteristics
- Theta accelerates as expiration approaches
- ATM options have highest theta (greatest time value)
- Weekends: Often excluded from theta calculations

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
theta_per_year = greek(call, Theta(), data)
theta_per_day = theta_per_year / 365  # Daily time decay
```

# Hedging Strategy
Portfolio with negative theta: Sell options to collect time decay (short gamma)
Portfolio with positive theta: Buy options (long gamma, pay for convexity)

See also: [`Vega`](@ref), [`Rho`](@ref)
"""
struct Theta <: FirstOrderGreek end

"""
    Vega() <: FirstOrderGreek

Price sensitivity to volatility (per 1% change in volatility).

Mathematically: ν = ∂V/∂σ

# Interpretation
- Measures P&L impact from 1% change in implied volatility
- Always positive (options benefit from higher volatility)
- Highest for ATM options
- Increases with time to expiration

# Units
By convention, vega is reported per 1% (0.01) volatility change, not per unit.
So if vega = 0.15, a 1% increase in vol increases option value by 0.15.

# Trading Applications
- Long vega: Profits from volatility increases (buy options)
- Short vega: Profits from volatility decreases (sell options)
- Vega hedging: Trade options to neutralize volatility exposure

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
vega = greek(call, Vega(), data)  # e.g., 0.38

# P&L from 5% vol increase
vol_change = 0.05  # 5%
pnl = vega * vol_change * 100  # = 0.38 * 5 = 1.90
```

See also: [`Vomma`](@ref), [`Vanna`](@ref), [`Veta`](@ref)
"""
struct Vega <: FirstOrderGreek end

"""
    Rho() <: FirstOrderGreek

Price sensitivity to interest rate (per 1% change in rate).

Mathematically: ρ = ∂V/∂r

# Interpretation
- Measures P&L impact from 1% change in risk-free rate
- Calls: Positive rho (benefit from higher rates)
- Puts: Negative rho (hurt by higher rates)
- Effect increases with time to expiration

# Mechanism
Higher rates → Lower present value of strike → Higher call value
Higher rates → Higher cost of carry → Lower put value

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
put = EuropeanPut(100.0, 1.0)

rho_call = greek(call, Rho(), data)   # Positive, e.g., 0.42
rho_put = greek(put, Rho(), data)     # Negative, e.g., -0.38
```

# Practical Note
Rho is usually small for short-dated options but becomes significant
for LEAPS (Long-term Equity Anticipation Securities).

See also: [`Phi`](@ref), [`Theta`](@ref)
"""
struct Rho <: FirstOrderGreek end

"""
    Phi() <: FirstOrderGreek

Price sensitivity to dividend yield (per 1% change in dividend).

Also known as `RhoDiv` or dividend rho.

Mathematically: φ = ∂V/∂q (where q = dividend yield)

# Interpretation
- Calls: Negative phi (hurt by dividends, spot drops on ex-div)
- Puts: Positive phi (benefit from dividends)
- Magnitude similar to Rho but opposite sign

# Examples
```julia
data_with_div = MarketData(100.0, 0.05, 0.2, 0.03)
call = EuropeanCall(100.0, 1.0)
put = EuropeanPut(100.0, 1.0)

phi_call = greek(call, Phi(), data_with_div)   # Negative
phi_put = greek(put, Phi(), data_with_div)     # Positive
```

See also: [`Rho`](@ref)
"""
struct Phi <: FirstOrderGreek end

# Alias for Phi
const RhoDiv = Phi

## Second Order Greeks

"""
    Vanna() <: SecondOrderGreek

Cross derivative: Delta sensitivity to volatility.

Mathematically: Vanna = ∂²V/∂S∂σ = ∂Δ/∂σ = ∂ν/∂S

# Interpretation
- Measures how delta changes when volatility changes
- Important for dynamic hedging (delta hedge drifts with vol moves)
- Key for volatility skew trading

# Sign and Magnitude
- Usually positive for calls, negative for puts
- Highest for slightly OTM options
- Zero for ATM options

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
vanna = greek(call, Vanna(), data)

# If vol increases by 2%, delta changes by:
delta_change = vanna * 0.02
```

See also: [`Delta`](@ref), [`Vega`](@ref)
"""
struct Vanna <: SecondOrderGreek end

"""
    Vomma() <: SecondOrderGreek

Second derivative of price with respect to volatility.

Mathematically: Vomma = ∂²V/∂σ² = ∂ν/∂σ

# Interpretation
- Measures convexity of vega (how vega changes with vol)
- Important for volatility of volatility trading
- High vomma means vega exposure is volatile

# Trading Application
If you are long options (positive vega) and vol increases:
- Positive vomma: Vega increases → you want vol to increase more
- Negative vomma: Vega decreases → you want vol to stabilize

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
vomma = greek(call, Vomma(), data)
```

See also: [`Vega`](@ref), [`Ultima`](@ref)
"""
struct Vomma <: SecondOrderGreek end

"""
    Charm() <: SecondOrderGreek

Cross derivative: Delta sensitivity to time.

Mathematically: Charm = ∂²V/∂S∂τ = ∂Δ/∂τ

Also known as "Delta decay" or "DdeltaDtime".

# Interpretation
- Measures how delta changes as time passes
- Critical for overnight risk management
- Ensures delta-neutral portfolios stay neutral

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
charm = greek(call, Charm(), data)

# Overnight delta drift (1 day)
delta_drift = charm / 365
```

See also: [`Delta`](@ref), [`Theta`](@ref), [`Color`](@ref)
"""
struct Charm <: SecondOrderGreek end

"""
    Veta() <: SecondOrderGreek

Cross derivative: Vega sensitivity to time.

Mathematically: Veta = ∂²V/∂σ∂τ = ∂ν/∂τ

# Interpretation
- Measures how vega decays as expiration approaches
- Important for calendar spread trading
- Helps understand vol term structure exposure

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
veta = greek(call, Veta(), data)
```

See also: [`Vega`](@ref), [`Theta`](@ref)
"""
struct Veta <: SecondOrderGreek end

## Third Order Greeks

"""
    Speed() <: ThirdOrderGreek

Third derivative of price with respect to spot.

Mathematically: Speed = ∂³V/∂S³ = ∂Γ/∂S

# Interpretation
- Measures how gamma changes with spot price
- Important for understanding tail risks
- Used in advanced risk management

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
speed = greek(call, Speed(), data)
```

See also: [`Gamma`](@ref)
"""
struct Speed <: ThirdOrderGreek end

"""
    Color() <: ThirdOrderGreek

Cross derivative: Gamma sensitivity to time.

Mathematically: Color = ∂³V/∂S²∂τ = ∂Γ/∂τ

Also known as "Gamma decay".

# Interpretation
- Measures how gamma changes as time passes
- Important for options near expiration
- Critical for gamma scalping strategies

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 0.1)  # Short-dated
color = greek(call, Color(), data)
```

See also: [`Gamma`](@ref), [`Charm`](@ref)
"""
struct Color <: ThirdOrderGreek end

"""
    Ultima() <: ThirdOrderGreek

Third derivative of price with respect to volatility.

Mathematically: Ultima = ∂³V/∂σ³ = ∂Vomma/∂σ

# Interpretation
- Measures how vomma changes with volatility
- Extremely advanced Greek for vol-of-vol trading
- Rarely used outside of exotic options

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
ultima = greek(call, Ultima(), data)
```

See also: [`Vomma`](@ref), [`Vega`](@ref)
"""
struct Ultima <: ThirdOrderGreek end
