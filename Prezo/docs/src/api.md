# API Reference

```@meta
CurrentModule = Prezo
```

## Market Data

```@docs
MarketData
```

## Option Types

### Abstract Types

```@docs
VanillaOption
EuropeanOption
AmericanOption
```

### European Options

```@docs
EuropeanCall
EuropeanPut
```

### American Options

```@docs
AmericanCall
AmericanPut
```

### Payoff Functions

```@docs
payoff
```

## Pricing Engines

### Black-Scholes

```@docs
BlackScholes
```

### Binomial Tree

```@docs
Binomial
```

### Monte Carlo

```@docs
MonteCarlo
asset_paths
asset_paths_col
asset_paths_ax
plot_paths
```

### Longstaff-Schwartz (LSM)

```@docs
LongstaffSchwartz
```

## Basis Functions

```@docs
BasisFunction
LaguerreBasis
ChebyshevBasis
PowerBasis
HermiteBasis
```

## Enhanced LSM Engines

```@docs
EnhancedLongstaffSchwartz
LaguerreLSM
ChebyshevLSM
PowerLSM
HermiteLSM
```

## European LSM Variants

```@docs
EuropeanLongstaffSchwartz
EuropeanLaguerreLSM
EuropeanChebyshevLSM
EuropeanPowerLSM
EuropeanHermiteLSM
```

## Core Functions

```@docs
price
```

## Validation

```@docs
validate_american_option_price
```

## Index

```@index
```
