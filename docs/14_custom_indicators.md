# Custom Indicators Builder Documentation

## Overview

The Custom Indicators Builder allows users to create, test, backtest, and optimize their own technical indicators using a flexible formula-based system. It supports Python expressions, built-in functions, and indicator composition.

## Architecture

```
Custom Indicators System
├── Indicator Builder
│   ├── Formula Parser
│   ├── Parameter Management
│   └── Dependency Resolution
├── Built-in Library
│   ├── Trend Indicators
│   ├── Momentum Indicators
│   ├── Volatility Indicators
│   └── Volume Indicators
├── Testing Framework
│   ├── Backtesting Engine
│   ├── Parameter Optimization
│   └── Performance Metrics
└── API Integration
    ├── Create/Edit Indicators
    ├── Calculate Values
    └── Export/Import
```

## Creating Custom Indicators

### Basic Indicator Creation

```python
POST /api/indicators/create
{
    "name": "CustomMA",
    "type": "trend",
    "description": "Weighted moving average with custom weights",
    "formula": "data.rolling(window=period).apply(lambda x: np.average(x, weights=np.linspace(0.5, 1.0, len(x))))",
    "parameters": [
        {
            "name": "period",
            "type": "int",
            "default_value": 20,
            "min_value": 5,
            "max_value": 200,
            "description": "Rolling window period"
        }
    ]
}
```

### Formula Syntax

**Available Variables:**
- `data` - Primary price series (usually close)
- `high`, `low`, `open`, `close`, `volume` - OHLCV data
- `np` - NumPy functions
- `pd` - Pandas functions
- `talib` - TA-Lib functions
- `IndicatorLibrary` - Built-in indicators

**Example Formulas:**

1. **Simple Moving Average Crossover Signal**
```python
"IndicatorLibrary.sma(data, fast_period) - IndicatorLibrary.sma(data, slow_period)"
```

2. **Volume-Weighted RSI**
```python
"IndicatorLibrary.rsi(data * volume / volume.rolling(period).mean(), period)"
```

3. **Custom Momentum Oscillator**
```python
"(data - data.shift(period)) / data.shift(period) * 100"
```

4. **Multi-Output Indicator**
```python
"""
{
    'signal': data.pct_change().rolling(period).mean(),
    'upper': data + 2 * data.rolling(period).std(),
    'lower': data - 2 * data.rolling(period).std()
}
"""
```

## Built-in Indicator Library

### Trend Indicators

**Simple Moving Average (SMA)**
```python
IndicatorLibrary.sma(data, period=20)
```

**Exponential Moving Average (EMA)**
```python
IndicatorLibrary.ema(data, period=20)
```

### Momentum Indicators

**Relative Strength Index (RSI)**
```python
IndicatorLibrary.rsi(data, period=14)
```

**MACD**
```python
macd_result = IndicatorLibrary.macd(data, fast=12, slow=26, signal=9)
# Returns: {'macd': Series, 'signal': Series, 'histogram': Series}
```

**Stochastic Oscillator**
```python
stoch = IndicatorLibrary.stochastic(high, low, close, fastk_period=14)
# Returns: {'k': Series, 'd': Series}
```

### Volatility Indicators

**Bollinger Bands**
```python
bb = IndicatorLibrary.bollinger_bands(data, period=20, std_dev=2)
# Returns: {'upper': Series, 'middle': Series, 'lower': Series}
```

**Average True Range (ATR)**
```python
atr = IndicatorLibrary.atr(high, low, close, period=14)
```

### Volume Indicators

**On-Balance Volume (OBV)**
```python
obv = IndicatorLibrary.obv(close, volume)
```

**Volume-Weighted Average Price (VWAP)**
```python
vwap = IndicatorLibrary.vwap(high, low, close, volume)
```

## Calculating Indicators

### Single Indicator Calculation

```python
POST /api/indicators/calculate
{
    "indicator_name": "CustomMA",
    "symbol": "AAPL",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "parameters": {
        "period": 30
    }
}
```

**Response:**
```json
{
    "status": "success",
    "indicator": "CustomMA",
    "values": [
        {"date": "2024-01-15", "value": 175.23},
        {"date": "2024-01-16", "value": 175.45},
        ...
    ],
    "signals": [
        {"date": "2024-01-20", "signal": 1},  // Buy
        {"date": "2024-02-05", "signal": -1}, // Sell
        ...
    ],
    "statistics": {
        "mean": 176.5,
        "std": 3.2,
        "min": 170.1,
        "max": 182.3
    }
}
```

### Using Custom Indicators in Code

```python
from app.services.indicators.custom_indicators import CustomIndicatorBuilder
import pandas as pd

# Initialize builder
builder = CustomIndicatorBuilder()

# Create custom indicator
builder.create_indicator(
    name="MomentumFlow",
    indicator_type=IndicatorType.MOMENTUM,
    description="Custom momentum with volume weighting",
    formula="(data - data.shift(period)) * (volume / volume.rolling(period).mean())",
    parameters=[
        IndicatorParameter("period", int, 14, 5, 50)
    ]
)

# Load data
data = pd.DataFrame({
    'close': prices,
    'volume': volumes
})

# Calculate indicator
result = builder.calculate_indicator("MomentumFlow", data, period=20)
print(result.values)
```

## Backtesting Indicators

### Running Backtest

```python
POST /api/indicators/backtest
{
    "indicator_name": "CustomMA",
    "symbol": "AAPL",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "initial_capital": 10000,
    "parameters": {
        "period": 20
    }
}
```

**Response:**
```json
{
    "backtest": {
        "total_return": 0.235,
        "sharpe_ratio": 1.82,
        "max_drawdown": -0.12,
        "num_trades": 45,
        "win_rate": 0.58,
        "profit_factor": 1.75,
        "final_capital": 12350,
        "cumulative_returns": [...]
    }
}
```

### Backtest Metrics Explained

- **Total Return**: Overall percentage gain/loss
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss

## Parameter Optimization

### Optimize Indicator Parameters

```python
POST /api/indicators/optimize
{
    "indicator_name": "CustomMA",
    "symbol": "AAPL",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "param_ranges": {
        "period": [10, 50]
    },
    "objective": "sharpe_ratio"
}
```

**Optimization Process:**
1. Grid search over parameter space
2. Backtest each combination
3. Select parameters maximizing objective
4. Return best configuration

**Response:**
```json
{
    "best_parameters": {
        "period": 23
    },
    "best_score": 2.15,
    "optimization_results": {
        "tested_combinations": 40,
        "convergence_plot": [...],
        "parameter_sensitivity": {
            "period": {
                "impact": "high",
                "optimal_range": [20, 30]
            }
        }
    }
}
```

## Combining Indicators

### Create Composite Indicators

```python
POST /api/indicators/combine
{
    "name": "TrendMomentum",
    "indicators": [
        {"name": "CustomMA", "weight": 0.6},
        {"name": "RSI", "weight": 0.4}
    ],
    "combination_method": "weighted_average"
}
```

**Combination Methods:**
1. **Weighted Average**: Linear combination
2. **Voting**: Consensus signals
3. **Threshold**: All conditions must be met
4. **Custom Formula**: User-defined combination

### Advanced Combination Example

```python
# Create a multi-factor indicator
builder.create_indicator(
    name="MultiFactor",
    indicator_type=IndicatorType.CUSTOM,
    description="Combines trend, momentum, and volatility",
    formula="""
    trend_score = (data > IndicatorLibrary.sma(data, 50)).astype(int)
    momentum_score = (IndicatorLibrary.rsi(data, 14) > 50).astype(int)
    volatility_score = (IndicatorLibrary.atr(high, low, close, 14) < IndicatorLibrary.atr(high, low, close, 14).rolling(50).mean()).astype(int)
    
    (trend_score + momentum_score + volatility_score) / 3
    """,
    parameters=[],
    dependencies=["SMA", "RSI", "ATR"]
)
```

## Templates and Examples

### Get Indicator Templates

```python
GET /api/indicators/templates

Response:
[
    {
        "name": "Custom Moving Average",
        "type": "trend",
        "description": "Weighted moving average with custom weights",
        "formula": "...",
        "parameters": [...]
    },
    {
        "name": "Momentum Oscillator",
        "type": "momentum",
        "description": "Custom momentum oscillator",
        "formula": "...",
        "parameters": [...]
    }
]
```

### Popular Custom Indicators

#### 1. Adaptive Moving Average
```python
formula = """
efficiency_ratio = abs(data - data.shift(period)) / data.diff().abs().rolling(period).sum()
smoothing = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc) ** 2
ema = data.ewm(alpha=smoothing, adjust=False).mean()
ema
"""
```

#### 2. Volume-Price Trend (VPT)
```python
formula = """
price_change = data.pct_change()
vpt = (price_change * volume).cumsum()
vpt
"""
```

#### 3. Keltner Channels
```python
formula = """
{
    'middle': IndicatorLibrary.ema(close, period),
    'upper': IndicatorLibrary.ema(close, period) + multiplier * IndicatorLibrary.atr(high, low, close, period),
    'lower': IndicatorLibrary.ema(close, period) - multiplier * IndicatorLibrary.atr(high, low, close, period)
}
"""
```

## Exporting and Sharing

### Export Indicator as Code

```python
GET /api/indicators/CustomMA/code

Response:
{
    "code": """
'''
CustomMA - Weighted moving average with custom weights
Author: user123
Version: 1.0
Created: 2024-01-15
'''

import pandas as pd
import numpy as np

def customma(data: pd.Series, period: int = 20):
    '''
    Weighted moving average with custom weights
    
    Parameters:
        period: Rolling window period
    
    Returns:
        pd.Series: Indicator values
    '''
    
    return data.rolling(window=period).apply(
        lambda x: np.average(x, weights=np.linspace(0.5, 1.0, len(x)))
    )
""",
    "language": "python"
}
```

### Import Indicator

```python
POST /api/indicators/upload
Content-Type: multipart/form-data

file: indicator.py
```

## Best Practices

### 1. Formula Design

**Keep It Simple**
- Start with basic formulas
- Test incrementally
- Avoid over-optimization

**Performance Considerations**
- Use vectorized operations
- Minimize loops
- Cache intermediate results

### 2. Parameter Selection

**Reasonable Defaults**
```python
parameters = [
    IndicatorParameter(
        name="period",
        type=int,
        default_value=20,  # Common default
        min_value=2,       # Minimum viable
        max_value=200,     # Avoid overfitting
        description="Lookback period"
    )
]
```

### 3. Validation

**Test Edge Cases**
- Empty data
- Single data point
- Extreme parameter values
- Missing data (NaN handling)

**Example Validation**
```python
def validate_indicator(indicator_name, test_data):
    # Test with minimal data
    assert builder.calculate_indicator(
        indicator_name, 
        test_data[:5]
    ).values.notna().any()
    
    # Test with extreme parameters
    assert builder.calculate_indicator(
        indicator_name,
        test_data,
        period=2
    ).values.notna().sum() > 0
```

### 4. Signal Generation

**Clear Entry/Exit Signals**
```python
def generate_signals(values: pd.Series) -> pd.Series:
    signals = pd.Series(0, index=values.index)
    
    # Entry conditions
    signals[values > values.rolling(20).mean() + 2 * values.rolling(20).std()] = 1
    
    # Exit conditions
    signals[values < values.rolling(20).mean()] = -1
    
    return signals
```

## Troubleshooting

### Common Issues

**1. Formula Syntax Error**
```python
# Check formula validity
try:
    ast.parse(formula, mode='eval')
except SyntaxError as e:
    print(f"Invalid formula: {e}")
```

**2. Undefined Variables**
```python
# Ensure all variables are available
required_vars = extract_variables(formula)
available_vars = {'data', 'high', 'low', 'close', 'open', 'volume'}
missing = required_vars - available_vars
```

**3. Performance Issues**
```python
# Profile slow indicators
import cProfile

cProfile.run('builder.calculate_indicator(name, large_dataset)')
```

### Debug Mode

```python
# Enable debug logging
builder = CustomIndicatorBuilder(debug=True)

# Get detailed calculation info
result = builder.calculate_indicator(
    "CustomMA",
    data,
    debug_info=True
)
print(result.metadata['debug_info'])
```

## API Reference

### Endpoints

1. **GET /api/indicators/library** - List all indicators
2. **POST /api/indicators/create** - Create new indicator
3. **POST /api/indicators/calculate** - Calculate values
4. **POST /api/indicators/backtest** - Run backtest
5. **POST /api/indicators/optimize** - Optimize parameters
6. **POST /api/indicators/combine** - Combine indicators
7. **GET /api/indicators/{name}/code** - Export as code
8. **DELETE /api/indicators/{name}** - Delete indicator

### Error Handling

```python
try:
    result = builder.calculate_indicator("CustomIndicator", data)
except ValueError as e:
    # Invalid parameters
    handle_validation_error(e)
except SyntaxError as e:
    # Formula error
    handle_formula_error(e)
except Exception as e:
    # General error
    handle_general_error(e)
```

## Future Enhancements

1. **Machine Learning Integration**: Use ML to discover patterns
2. **Visual Builder**: Drag-and-drop indicator creation
3. **Community Marketplace**: Share and sell indicators
4. **Real-time Alerts**: Trigger notifications on signals
5. **Mobile Support**: Create indicators on mobile devices