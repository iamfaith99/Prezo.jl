# Options

Prezo.jl supports European and American style options with call and put payoffs.

## Type Hierarchy

```
VanillaOption (abstract)
├── EuropeanOption (abstract)
│   ├── EuropeanCall
│   └── EuropeanPut
└── AmericanOption (abstract)
    ├── AmericanCall
    └── AmericanPut
```

## European Options

European options can only be exercised at expiration.

### European Call

A call option gives the holder the right to buy the underlying at the strike price.

```julia
# EuropeanCall(strike, expiry)
call = EuropeanCall(100.0, 1.0)
```

**Payoff**: `max(0, S - K)` where `S` is the spot price and `K` is the strike.

### European Put

A put option gives the holder the right to sell the underlying at the strike price.

```julia
# EuropeanPut(strike, expiry)
put = EuropeanPut(100.0, 1.0)
```

**Payoff**: `max(0, K - S)`

## American Options

American options can be exercised at any time up to and including expiration, making them more valuable than their European counterparts.

### American Call

```julia
# AmericanCall(strike, expiry)
am_call = AmericanCall(100.0, 1.0)
```

For non-dividend paying stocks, American calls have the same value as European calls since early exercise is suboptimal.

### American Put

```julia
# AmericanPut(strike, expiry)
am_put = AmericanPut(100.0, 1.0)
```

American puts always trade at a premium to European puts due to the early exercise feature.

## Option Parameters

### Strike Price
The price at which the option can be exercised.

```julia
option = EuropeanCall(105.0, 1.0)  # Strike = 105
```

### Expiry (Time to Maturity)
Time to expiration in years.

```julia
option = EuropeanCall(100.0, 0.5)   # 6 months
option = EuropeanCall(100.0, 1.0)   # 1 year
option = EuropeanCall(100.0, 2.0)   # 2 years
```

## Computing Payoffs

The `payoff` function computes the intrinsic value of an option at a given spot price:

```julia
call = EuropeanCall(100.0, 1.0)
put = EuropeanPut(100.0, 1.0)

# At different spot prices
payoff(call, 110.0)  # 10.0
payoff(call, 95.0)   # 0.0
payoff(put, 110.0)   # 0.0
payoff(put, 95.0)    # 5.0
```

## Moneyness

Options are classified by their moneyness:

- **In-the-Money (ITM)**: Positive intrinsic value
  - Call: S > K
  - Put: S < K
- **At-the-Money (ATM)**: S ≈ K
- **Out-of-the-Money (OTM)**: Zero intrinsic value
  - Call: S < K
  - Put: S > K
