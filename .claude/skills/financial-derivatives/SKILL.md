# Financial Derivatives Skill

Deep expertise in options pricing, numerical methods, and quantitative finance.

## Core Options Pricing Theory

### The Black-Scholes-Merton Framework

**Assumptions**:
- Constant volatility (σ) and risk-free rate (r)
- Lognormal asset price distribution
- Continuous trading, no transaction costs
- No arbitrage opportunities
- Dividends (q) paid continuously

**Risk-Neutral Pricing**:
- Asset drift = r - q (not the real-world drift μ)
- Discount expected payoff at risk-free rate

**Black-Scholes Formulas**:
```
Call = S*exp(-q*T)*Φ(d1) - K*exp(-r*T)*Φ(d2)
Put = K*exp(-r*T)*Φ(-d2) - S*exp(-q*T)*Φ(-d1)

where:
d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
d2 = d1 - σ√T
```

### Fundamental Relationships

**Put-Call Parity** (European options):
```
C - P = S*exp(-q*T) - K*exp(-r*T)
```
This is an arbitrage relationship - it MUST hold for consistent pricing.

**Early Exercise Premium** (American options):
```
American Value = European Value + Early Exercise Premium
```
- American calls with q=0: Early exercise never optimal → American ≈ European
- American calls with q>0: May exercise early to capture dividends
- American puts: Early exercise often optimal (time value of money on strike)

**Monotonicity Properties**:
- ∂C/∂S > 0 (call delta positive)
- ∂P/∂S < 0 (put delta negative)
- ∂C/∂σ > 0, ∂P/∂σ > 0 (vega always positive)
- ∂C/∂T ≥ 0, ∂P/∂T ≥ 0 for American (more time = more value)
- ∂C/∂K < 0, ∂P/∂K > 0 (call decreases with strike, put increases)

## Numerical Methods

### 1. Black-Scholes (Analytical)

**Advantages**:
- Exact solution (up to numerical error in Φ)
- Extremely fast O(1)
- No convergence issues

**Limitations**:
- European options only
- Requires closed-form solution
- Constant parameters assumption

**Implementation Notes**:
- Use cumulative normal distribution Φ via erf: `Φ(x) = (1 + erf(x/√2))/2`
- Watch for numerical instability when d1, d2 are large
- Handle edge cases: T=0, σ=0

### 2. Binomial Trees

**Method**:
- Discretize time into n steps
- Model asset as recombining tree: up by u, down by d
- Risk-neutral probability: p = (exp((r-q)Δt) - d)/(u - d)
- Backward induction from expiry

**Parameters**:
- Cox-Ross-Rubinstein: `u = exp(σ√Δt)`, `d = 1/u`
- Jarrow-Rudd: `u = exp((r-q-σ²/2)Δt + σ√Δt)`, `d = exp((r-q-σ²/2)Δt - σ√Δt)`

**Advantages**:
- Handles American options naturally
- Easy to understand and implement
- Handles dividends, barriers, other features

**Limitations**:
- Slow convergence: O(n²) complexity
- Oscillating convergence due to strike placement in tree
- Memory intensive for large n

**Convergence**:
- Converges to Black-Scholes as n→∞
- Typical values: n=100-500 for reasonable accuracy
- Error roughly O(1/n) or O(1/√n) depending on smoothing

### 3. Monte Carlo Simulation

**Method**:
- Simulate many asset price paths under risk-neutral measure
- Compute payoff for each path
- Average and discount: `price = exp(-rT) * mean(payoffs)`

**Path Generation** (Geometric Brownian Motion):
```
S(t+Δt) = S(t) * exp((r - q - σ²/2)Δt + σ√Δt*Z)
where Z ~ N(0,1)
```

**Advantages**:
- Handles complex path-dependent payoffs
- Easy to add features (stochastic vol, jumps)
- Embarrassingly parallel

**Limitations**:
- European or simple path-dependent only (not early exercise)
- Slow convergence: error ∝ 1/√paths
- High variance for OTM options
- Computationally expensive

**Variance Reduction**:
- Antithetic variates: use both Z and -Z
- Control variates: use known solution (e.g., Black-Scholes)
- Importance sampling for rare events
- Quasi-random numbers (Sobol, Halton sequences)

**Confidence Intervals**:
```
CI = mean ± z * std/√paths
```
where z=1.96 for 95% confidence.

### 4. Longstaff-Schwartz (Least Squares Monte Carlo)

**Method**:
- Generate forward Monte Carlo paths
- Backward induction like binomial
- At each time: regress continuation value on basis functions
- Compare immediate exercise vs continuation
- Update exercise decision

**Basis Functions** (for spot price S):
- Laguerre: L₀(S)=1, L₁(S)=1-S, L₂(S)=(2-4S+S²)/2, ...
- Power: 1, S, S², S³, ...
- Hermite: H₀(S)=1, H₁(S)=S, H₂(S)=S²-1, ...
- Chebyshev: T₀(S)=1, T₁(S)=S, T₂(S)=2S²-1, ...

**Advantages**:
- Handles American options via Monte Carlo
- Flexible for complex payoffs and state variables
- Reasonable accuracy with enough paths

**Limitations**:
- Computationally intensive (paths × steps)
- Regression introduces approximation error
- Basis function choice affects accuracy
- Biased estimator (upper bound, can be improved)

**Implementation Notes**:
- Typical: 10,000-100,000 paths, 50-100 steps
- Normalize spot prices before regression (scale to ~1)
- Use 2-5 basis functions (too many → overfitting)
- Only regress in-the-money paths (exercise value > 0)
- Laguerre polynomials generally work best

## Common Implementation Pitfalls

### Risk-Neutral Drift
❌ **Wrong**: `drift = μ` (real-world expected return)
✅ **Correct**: `drift = r - q - σ²/2` (risk-neutral, with Itô correction)

### Dividend Handling
- Dividends reduce forward price: `F = S*exp((r-q)T)`
- In Black-Scholes: multiply S terms by `exp(-q*T)`
- In trees/MC: reduce drift by q

### Volatility Units
- σ is annualized (e.g., σ=0.20 means 20% per year)
- Scale for time: σ√T or σ√Δt
- Convert basis points: 100 bp = 0.01

### Boundary Conditions
- Call at S=0: value is 0
- Call as S→∞: value ≈ S (deep ITM)
- Put at S=0: value is K*exp(-r*T) (certain payoff)
- Put as S→∞: value is 0
- At expiry: value = max(payoff, 0)

### Time to Expiry
- T=0: option worth intrinsic value only
- Ensure T>0 in formulas to avoid division by zero
- Handle T<0 gracefully (expired = 0 value)

### Negative or Invalid Inputs
- Negative volatility: invalid
- Negative strike: unusual but mathematically valid
- Negative spot: invalid for equity options
- Negative time: option expired

## Greeks (Sensitivities)

**First Order**:
- **Delta (Δ)**: ∂V/∂S - rate of change with spot price
- **Vega (ν)**: ∂V/∂σ - sensitivity to volatility
- **Theta (Θ)**: ∂V/∂t - time decay
- **Rho (ρ)**: ∂V/∂r - sensitivity to interest rate

**Second Order**:
- **Gamma (Γ)**: ∂²V/∂S² - convexity, rate of delta change

**Numerical Greeks**:
- Finite difference: `Delta ≈ (V(S+h) - V(S-h))/(2h)`
- Central differences more accurate than forward/backward
- Choose h carefully (balance truncation vs roundoff error)

## Model Extensions

### Stochastic Volatility
- Heston model: volatility follows CIR process
- SABR model: local volatility with stochastic component
- Captures volatility smile/skew

### Jump Diffusion
- Merton model: Poisson jumps + GBM
- Captures heavy tails, gap risk

### Local Volatility
- Dupire's formula: σ(S,t) varies by strike and time
- Calibrated to match market prices

### Interest Rate Models
- Short rate models (Vasicek, CIR, Hull-White)
- HJM framework for term structure

## Calibration and Market Data

**Implied Volatility**:
- Back out σ from market price using Black-Scholes
- Volatility smile: σ varies by strike
- Volatility surface: σ(K, T)

**Market Conventions**:
- Quotes often in implied vol, not price
- ATM (at-the-money): K ≈ F (forward price)
- Moneyness: K/S or K/F
- Delta-based strikes (25Δ, 10Δ) for FX options

## Validation and Testing

### Cross-Validation
- Compare multiple methods (BSM vs Tree vs MC)
- Should agree within numerical tolerance

### Analytical Benchmarks
- Use Black-Scholes as reference for European options
- Known special cases (T=0, σ=0, S=K)

### Convergence Studies
- Binomial: test multiple n values, ensure convergence
- Monte Carlo: plot error vs √paths, check CI coverage
- LSM: test different basis orders and path counts

### Sanity Checks
- Put-call parity must hold for European
- American ≥ European
- Price > intrinsic value (time value ≥ 0)
- Price < S for calls (can't be worth more than stock)
- Price < K*exp(-r*T) for puts (can't be worth more than strike PV)

## Implementation Best Practices

1. **Separate concerns**: Market data, option specification, pricing engine
2. **Use multiple dispatch**: `price(option, engine, data)`
3. **Vectorization**: Generate all paths at once, not in loops
4. **Type stability**: Ensure functions return consistent types
5. **Numerical stability**: Watch for overflow/underflow in exponentials
6. **Edge cases**: Handle T=0, σ=0, extreme strikes gracefully
7. **Random seeds**: Set seed for reproducible Monte Carlo results
8. **Performance**: Profile before optimizing, MC is embarrassingly parallel

## Further Reading

- **Hull**: "Options, Futures, and Other Derivatives" (standard textbook)
- **Glasserman**: "Monte Carlo Methods in Financial Engineering" (MC/LSM detail)
- **Wilmott**: "Paul Wilmott on Quantitative Finance" (PDEs, Greeks)
- **Shreve**: "Stochastic Calculus for Finance" (mathematical rigor)
