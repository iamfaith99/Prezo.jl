# Dividend Handling Bug Report

## Summary

The test suite revealed that **American call options with dividends are priced lower than European call options**, violating the fundamental arbitrage relationship: **American ≥ European**.

After investigation, the root cause is **NOT** a bug in American option pricing, but rather **bugs in European option pricing** - specifically, the European implementations fail to properly account for dividends.

## Test Results

```
With Dividends (q=0.03):
  European Call (BS):    $10.4506
  European Call (Binom): $10.4544
  American Call (Binom): $8.6589   ✗ VIOLATION: Lower than European!
  American Call (LSM):   $8.8188   ✗ VIOLATION: Lower than European!

Without Dividends (q=0.0):
  European Call (BS):    $10.4506  (Same as above!)
  American Call (Binom): $10.4544  ✓ Correct: ≈ European
```

## Root Cause Analysis

### Bug #1: Black-Scholes Implementation (engines.jl:174-196)

**Current Implementation (INCORRECT):**
```julia
function price(option::EuropeanCall, engine::BlackScholes, data::MarketData)
    (; spot, rate, vol, div) = data

    d1 = (log(spot / strike) + (rate + 0.5 * vol^2) * expiry) / (vol * sqrt(expiry))
    d2 = d1 - vol * sqrt(expiry)

    price = spot * norm_cdf(d1) - strike * exp(-rate * expiry) * norm_cdf(d2)

    return price
end
```

**Issues:**
1. `div` is extracted but never used
2. The drift term should be `(rate - div)` not `rate`
3. The spot price should be multiplied by `exp(-div * expiry)`

**Correct Implementation:**
```julia
function price(option::EuropeanCall, engine::BlackScholes, data::MarketData)
    (; spot, rate, vol, div) = data

    d1 = (log(spot / strike) + (rate - div + 0.5 * vol^2) * expiry) / (vol * sqrt(expiry))
    d2 = d1 - vol * sqrt(expiry)

    price = spot * exp(-div * expiry) * norm_cdf(d1) - strike * exp(-rate * expiry) * norm_cdf(d2)

    return price
end
```

**Theory (from financial-derivatives skill):**
```
Call = S*exp(-q*T)*Φ(d1) - K*exp(-r*T)*Φ(d2)
where:
d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
d2 = d1 - σ√T
```

### Bug #2: Binomial Tree for European Options (engines.jl:76-103)

**Current Implementation (INCORRECT):**
```julia
function price(option::EuropeanOption, engine::Binomial, data::MarketData)
    (; spot, rate, vol, div) = data
    steps = engine.steps

    dt = expiry / steps
    u = exp(rate * dt + vol * sqrt(dt))
    d = exp(rate * dt - vol * sqrt(dt))
    pu = (exp(rate * dt) - d) / (u - d)
    # ... rest of implementation
end
```

**Issues:**
1. `div` is extracted but never used
2. Should use `(rate - div)` in tree construction, not `rate`

**Correct Implementation:**
```julia
function price(option::EuropeanOption, engine::Binomial, data::MarketData)
    (; spot, rate, vol, div) = data
    steps = engine.steps

    dt = expiry / steps
    u = exp((rate - div) * dt + vol * sqrt(dt))
    d = exp((rate - div) * dt - vol * sqrt(dt))
    pu = (exp((rate - div) * dt) - d) / (u - d)
    # ... rest of implementation
end
```

**Note:** The American binomial implementation (lines 105-145) **already does this correctly**!

### Bug #3: Black-Scholes Put Implementation (engines.jl:186-196)

Same issues as the Call implementation, plus:
- Line 188: `div` is not even extracted from `data`

## Why The Tests Failed

The tests appeared to show "American < European" violations, but what actually happened:

1. **European options are OVERPRICED** (dividends ignored → too high)
2. **American options are CORRECTLY PRICED** (dividends handled properly)
3. Result: American < Buggy European

This is why **both** Binomial and LSM showed the same failure - they were both being compared to incorrectly priced European options.

## Verification

The diagnosis is confirmed by:

1. **European call prices don't change with dividends:**
   - No div (q=0.0): $10.4506
   - With div (q=0.03): $10.4506 (should be ~$8.6)

2. **American call prices DO change with dividends:**
   - No div: $10.4544
   - With div: $8.6589 (correctly lower)

3. **Theory from financial-derivatives skill:**
   - Dividends reduce call value (stock price drift is reduced)
   - Risk-neutral drift = r - q, not r
   - Must account for present value of dividends

## Impact

This bug affects:
- All European option pricing when dividends > 0
- Black-Scholes engine
- Binomial engine (European options only)
- Any validation or comparison involving European options with dividends

American option pricing appears to be correct.

## Recommended Fix

Apply the corrections shown above to:
1. `price(option::EuropeanCall, engine::BlackScholes, ...)`
2. `price(option::EuropeanPut, engine::BlackScholes, ...)`
3. `price(option::EuropeanOption, engine::Binomial, ...)`

After fixing, the test suite should pass completely, with American ≥ European for all cases.

## Testing After Fix

The corrected European call with dividends should price around $8.6-8.7, making it:
- Lower than the no-dividend European ($10.45)
- Approximately equal to or less than American ($8.66-8.82)

---

*This bug was discovered using the julia-testing and financial-derivatives skills, which helped structure proper validation tests and identify the theoretical violations.*
