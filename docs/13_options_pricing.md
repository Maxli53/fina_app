# Options Pricing Models Documentation

## Overview

The Financial Platform provides comprehensive options pricing capabilities including Black-Scholes, Binomial, and Monte Carlo models, along with Greeks calculations, implied volatility, and portfolio risk analysis.

## Architecture

```
Options Pricing System
├── Pricing Models
│   ├── Black-Scholes (European)
│   ├── Binomial Tree (American/European)
│   └── Monte Carlo (Exotic/Path-dependent)
├── Greeks Calculations
│   ├── Delta, Gamma, Theta
│   └── Vega, Rho
├── Advanced Features
│   ├── Implied Volatility
│   ├── Volatility Surface
│   └── Portfolio VaR
└── API Integration
    ├── Single Option Pricing
    ├── Portfolio Analysis
    └── Strategy Analysis
```

## Pricing Models

### 1. Black-Scholes Model

**For European Options Only**

Mathematical Formula:
```
Call: C = S₀N(d₁) - Ke^(-rT)N(d₂)
Put:  P = Ke^(-rT)N(-d₂) - S₀N(-d₁)

Where:
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

**Implementation:**
```python
from app.services.options.options_pricing import (
    OptionsPricingService,
    OptionContract,
    MarketData,
    OptionType,
    ExerciseStyle
)

# Create option contract
option = OptionContract(
    underlying="AAPL",
    strike=150.0,
    expiry=datetime(2024, 6, 21),
    option_type=OptionType.CALL,
    exercise_style=ExerciseStyle.EUROPEAN
)

# Market data
market = MarketData(
    spot_price=155.0,
    risk_free_rate=0.05,
    volatility=0.25,
    dividend_yield=0.01
)

# Price option
pricing_service = OptionsPricingService()
result = pricing_service.price_option(option, market, model="black-scholes")

print(f"Option Price: ${result.price:.2f}")
print(f"Delta: {result.greeks.delta:.4f}")
```

### 2. Binomial Model

**For American and European Options**

Features:
- Handles early exercise (American options)
- Discrete time steps
- Convergence to Black-Scholes for European options

**Usage:**
```python
# American put option
american_put = OptionContract(
    underlying="SPY",
    strike=400.0,
    expiry=datetime(2024, 3, 15),
    option_type=OptionType.PUT,
    exercise_style=ExerciseStyle.AMERICAN
)

# Price with binomial model
result = pricing_service.price_option(
    american_put, 
    market, 
    model="binomial",
    steps=100  # Number of time steps
)
```

### 3. Monte Carlo Model

**For Exotic and Path-Dependent Options**

Features:
- Variance reduction techniques (antithetic variates, control variates)
- American option pricing via Longstaff-Schwartz
- Flexible for complex payoffs

**Advanced Usage:**
```python
# Price with Monte Carlo
result = pricing_service.price_option(
    option,
    market,
    model="monte-carlo",
    simulations=10000,
    time_steps=252,
    antithetic=True,
    control_variate=True
)

print(f"Price: ${result.price:.2f}")
print(f"Standard Error: ${result.additional_info['standard_error']:.4f}")
```

## Greeks Calculations

### The Greeks Explained

1. **Delta (Δ)**: Rate of change of option price with respect to underlying price
   - Call Delta: 0 to 1
   - Put Delta: -1 to 0

2. **Gamma (Γ)**: Rate of change of delta with respect to underlying price
   - Highest at-the-money
   - Same for calls and puts

3. **Theta (Θ)**: Time decay - change in option price with respect to time
   - Usually negative (options lose value over time)
   - Expressed per day

4. **Vega (ν)**: Sensitivity to volatility changes
   - Expressed per 1% change in volatility
   - Highest for at-the-money options

5. **Rho (ρ)**: Sensitivity to interest rate changes
   - Expressed per 1% change in interest rate

### Greeks Calculation Example

```python
# Get all Greeks
result = pricing_service.price_option(option, market)
greeks = result.greeks

print(f"""
Greeks for {option.underlying} {option.strike} {option.option_type.value}:
Delta: {greeks.delta:.4f}
Gamma: {greeks.gamma:.4f}
Theta: {greeks.theta:.4f} per day
Vega:  {greeks.vega:.4f} per 1% vol
Rho:   {greeks.rho:.4f} per 1% rate
""")
```

## Implied Volatility

### Calculate IV from Market Price

```python
# Calculate implied volatility
market_price = 8.50  # Observed option price

impl_vol = pricing_service.calculate_implied_volatility(
    option=option,
    market_price=market_price,
    market=market
)

print(f"Implied Volatility: {impl_vol:.1%}")
```

### API Endpoint

```bash
POST /api/options/implied-volatility
{
    "underlying": "AAPL",
    "strike": 150,
    "expiry": "2024-06-21",
    "option_type": "call",
    "market_price": 8.50,
    "spot_price": 155.0
}
```

## Volatility Surface

### Generate Volatility Surface

```python
# Get volatility surface
GET /api/options/volatility-surface/AAPL

Response:
{
    "surface": {
        "strikes": [140, 145, 150, 155, 160],
        "expiries": [30, 60, 90, 120, 180],
        "volatilities": [
            [0.28, 0.26, 0.25, 0.26, 0.28],  # 30 days
            [0.27, 0.25, 0.24, 0.25, 0.27],  # 60 days
            ...
        ],
        "sabr_parameters": {
            "alpha": 0.3,
            "beta": 0.5,
            "rho": -0.3,
            "nu": 0.4
        }
    }
}
```

### Volatility Smile Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Extract volatility smile for specific expiry
expiry_index = 2  # 90 days
strikes = surface["strikes"]
vols = surface["volatilities"][expiry_index]

plt.figure(figsize=(10, 6))
plt.plot(strikes, vols, 'b-', linewidth=2)
plt.scatter(spot_price, vols[strikes.index(spot_price)], 
           color='red', s=100, label='ATM')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.title('Volatility Smile - 90 Days to Expiry')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```

## Portfolio Pricing

### Price Multiple Options

```python
# Define portfolio positions
positions = [
    {
        "underlying": "AAPL",
        "strike": 150,
        "expiry": "2024-06-21",
        "option_type": "call",
        "quantity": 10
    },
    {
        "underlying": "AAPL",
        "strike": 145,
        "expiry": "2024-06-21",
        "option_type": "put",
        "quantity": -5  # Short position
    }
]

# Price portfolio
POST /api/options/portfolio/price
{
    "positions": positions,
    "model": "black-scholes"
}

Response:
{
    "portfolio": {
        "total_value": 12500,
        "portfolio_greeks": {
            "delta": 4.5,
            "gamma": 0.08,
            "theta": -45.2,
            "vega": 125.3,
            "rho": 23.1
        },
        "positions": [...]
    }
}
```

## Value at Risk (VaR)

### Portfolio VaR Calculation

```python
# Calculate VaR for options portfolio
POST /api/options/var
{
    "positions": positions,
    "confidence_level": 0.95,
    "horizon_days": 1,
    "simulations": 10000
}

Response:
{
    "var_analysis": {
        "var": 2500,  # 95% confidence 1-day VaR
        "cvar": 3200,  # Conditional VaR (Expected Shortfall)
        "current_value": 12500,
        "worst_loss": 4500,
        "best_gain": 1800
    }
}
```

## Option Chains

### Fetch Option Chain Data

```python
# Get option chain for symbol
GET /api/options/chains/AAPL?expiry=2024-06-21

Response:
{
    "chain": {
        "symbol": "AAPL",
        "spot_price": 155.0,
        "expiries": {
            "2024-06-21": {
                "calls": {
                    "150": {
                        "bid": 8.45,
                        "ask": 8.55,
                        "last": 8.50,
                        "volume": 1250,
                        "open_interest": 5000,
                        "implied_volatility": 0.25,
                        "greeks": {
                            "delta": 0.65,
                            "gamma": 0.015,
                            "theta": -0.08,
                            "vega": 0.22,
                            "rho": 0.15
                        }
                    }
                },
                "puts": {...}
            }
        }
    }
}
```

## Strategy Analysis

### Common Option Strategies

#### 1. Covered Call
```python
positions = [
    {"underlying": "AAPL", "quantity": 100},  # Long stock
    {"underlying": "AAPL", "strike": 160, "expiry": "2024-06-21", 
     "option_type": "call", "quantity": -1}  # Short call
]
```

#### 2. Protective Put
```python
positions = [
    {"underlying": "AAPL", "quantity": 100},  # Long stock
    {"underlying": "AAPL", "strike": 150, "expiry": "2024-06-21", 
     "option_type": "put", "quantity": 1}  # Long put
]
```

#### 3. Iron Condor
```python
positions = [
    {"strike": 145, "option_type": "put", "quantity": -1},   # Short put
    {"strike": 150, "option_type": "put", "quantity": 1},    # Long put
    {"strike": 160, "option_type": "call", "quantity": 1},   # Long call
    {"strike": 165, "option_type": "call", "quantity": -1}   # Short call
]
```

### Strategy P&L Analysis

```python
POST /api/options/strategies/analyze
{
    "strategy_type": "iron_condor",
    "legs": positions
}

Response:
{
    "strategy_analysis": {
        "max_profit": 250,
        "max_loss": 250,
        "breakeven_prices": [147.5, 162.5],
        "probability_of_profit": 0.68,
        "expected_value": 45,
        "pnl_profile": {
            "prices": [140, 145, 150, ..., 170],
            "pnl": [-250, -250, 0, ..., -250]
        }
    }
}
```

## Best Practices

### 1. Model Selection

**Use Black-Scholes when:**
- Pricing European options
- Need fast calculations
- Liquid markets with observable prices

**Use Binomial when:**
- Pricing American options
- Need early exercise features
- Discrete dividend payments

**Use Monte Carlo when:**
- Path-dependent options
- Complex payoff structures
- Multiple underlying assets

### 2. Parameter Estimation

**Volatility:**
- Use implied volatility from similar options
- Consider volatility term structure
- Account for volatility smile

**Interest Rates:**
- Use risk-free rate appropriate for option tenor
- Consider term structure
- Account for funding costs

**Dividends:**
- Include discrete dividends for American options
- Use dividend yield for continuous dividends
- Consider dividend uncertainty

### 3. Risk Management

**Position Limits:**
```python
# Check Greeks limits
if abs(portfolio_greeks["delta"]) > DELTA_LIMIT:
    raise RiskLimitExceeded("Delta limit breached")

if portfolio_greeks["vega"] > VEGA_LIMIT:
    raise RiskLimitExceeded("Vega limit breached")
```

**Hedging:**
```python
# Delta hedging example
hedge_shares = -portfolio_greeks["delta"] * 100
print(f"Hedge by {'buying' if hedge_shares > 0 else 'selling'} "
      f"{abs(hedge_shares):.0f} shares")
```

## Performance Optimization

### Caching Strategies

```python
# Cache option prices
@cache(ttl=60)  # 1 minute cache
def get_option_price(option_key, market_data):
    return pricing_service.price_option(option, market)

# Cache volatility surface
@cache(ttl=300)  # 5 minute cache
def get_volatility_surface(symbol):
    return calculate_vol_surface(symbol)
```

### Parallel Processing

```python
# Price multiple options in parallel
import asyncio

async def price_portfolio_async(positions):
    tasks = [
        price_option_async(pos) 
        for pos in positions
    ]
    results = await asyncio.gather(*tasks)
    return aggregate_results(results)
```

### GPU Acceleration

```python
# Use GPU for Monte Carlo simulations
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Run simulations on GPU
    prices = monte_carlo_gpu(option, market, device)
```

## Regulatory Compliance

### Best Execution

- Document pricing models used
- Maintain audit trail of calculations
- Regular model validation
- Benchmark against market prices

### Risk Disclosure

- Clear documentation of model assumptions
- Limitations of each pricing model
- Sensitivity analysis results
- Historical model performance

## Troubleshooting

### Common Issues

**1. Convergence Issues**
```python
# Increase iterations for implied volatility
impl_vol = pricing_service.calculate_implied_volatility(
    option, market_price, market,
    max_iterations=200,
    tolerance=1e-8
)
```

**2. Negative Time Value**
- Check input parameters
- Verify option hasn't expired
- Ensure positive volatility

**3. Unstable Greeks**
- Use finite difference with appropriate step size
- Implement smoothing for near-expiry options
- Consider analytical vs numerical methods

## API Reference

### Endpoints

1. **POST /api/options/price** - Price single option
2. **POST /api/options/implied-volatility** - Calculate IV
3. **POST /api/options/portfolio/price** - Price portfolio
4. **POST /api/options/var** - Calculate VaR
5. **GET /api/options/chains/{symbol}** - Get option chain
6. **GET /api/options/volatility-surface/{symbol}** - Get vol surface
7. **POST /api/options/strategies/analyze** - Analyze strategy

### Error Codes

- `400` - Invalid parameters
- `404` - Option not found
- `422` - Calculation error
- `500` - Server error

## Future Enhancements

1. **Exotic Options**: Barriers, lookbacks, Asians
2. **Multi-Asset Options**: Rainbow, basket options
3. **Stochastic Volatility**: Heston, SABR models
4. **Machine Learning**: IV prediction, pricing calibration
5. **Real-Time Greeks**: Streaming calculations