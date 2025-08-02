"""
Custom Technical Indicators Builder
Allows users to create, test, and deploy custom technical indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass
from enum import Enum
import talib
import ast
import inspect
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class IndicatorType(Enum):
    """Types of technical indicators"""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    SUPPORT_RESISTANCE = "support_resistance"
    CUSTOM = "custom"


class DataField(Enum):
    """Available data fields"""
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    OHLC4 = "ohlc4"  # (O+H+L+C)/4
    HLC3 = "hlc3"    # (H+L+C)/3
    HL2 = "hl2"      # (H+L)/2


@dataclass
class IndicatorParameter:
    """Parameter definition for indicators"""
    name: str
    type: type
    default_value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: str = ""


@dataclass
class IndicatorDefinition:
    """Complete indicator definition"""
    name: str
    type: IndicatorType
    description: str
    parameters: List[IndicatorParameter]
    formula: str  # Python code or mathematical formula
    dependencies: List[str] = None  # Other indicators this depends on
    author: str = "system"
    version: str = "1.0"
    created_at: datetime = None


@dataclass
class IndicatorResult:
    """Result from indicator calculation"""
    name: str
    values: pd.Series
    signals: Optional[pd.Series] = None  # Buy/sell signals
    metadata: Dict[str, Any] = None


class IndicatorLibrary:
    """Built-in technical indicators library"""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        if len(data) > period:
            return pd.Series(talib.RSI(data.values, timeperiod=period), index=data.index)
        else:
            # Manual calculation for small datasets
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD indicator"""
        if len(data) > slow:
            macd_line, signal_line, histogram = talib.MACD(
                data.values, 
                fastperiod=fast, 
                slowperiod=slow, 
                signalperiod=signal
            )
            return {
                "macd": pd.Series(macd_line, index=data.index),
                "signal": pd.Series(signal_line, index=data.index),
                "histogram": pd.Series(histogram, index=data.index)
            }
        else:
            # Manual calculation
            ema_fast = data.ewm(span=fast, adjust=False).mean()
            ema_slow = data.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            return {
                "macd": macd_line,
                "signal": signal_line,
                "histogram": histogram
            }
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        if len(data) > period:
            upper, middle, lower = talib.BBANDS(
                data.values,
                timeperiod=period,
                nbdevup=std_dev,
                nbdevdn=std_dev,
                matype=0
            )
            return {
                "upper": pd.Series(upper, index=data.index),
                "middle": pd.Series(middle, index=data.index),
                "lower": pd.Series(lower, index=data.index)
            }
        else:
            # Manual calculation
            middle = data.rolling(window=period).mean()
            std = data.rolling(window=period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            return {
                "upper": upper,
                "middle": middle,
                "lower": lower
            }
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        if len(high) > period:
            return pd.Series(
                talib.ATR(high.values, low.values, close.values, timeperiod=period),
                index=high.index
            )
        else:
            # Manual calculation
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return true_range.rolling(window=period).mean()
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   fastk_period: int = 14, slowk_period: int = 3, slowd_period: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        if len(high) > fastk_period:
            slowk, slowd = talib.STOCH(
                high.values, low.values, close.values,
                fastk_period=fastk_period,
                slowk_period=slowk_period,
                slowd_period=slowd_period
            )
            return {
                "k": pd.Series(slowk, index=high.index),
                "d": pd.Series(slowd, index=high.index)
            }
        else:
            # Manual calculation
            lowest_low = low.rolling(window=fastk_period).min()
            highest_high = high.rolling(window=fastk_period).max()
            fastk = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            slowk = fastk.rolling(window=slowk_period).mean()
            slowd = slowk.rolling(window=slowd_period).mean()
            return {
                "k": slowk,
                "d": slowd
            }
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        if len(close) > 0:
            return pd.Series(talib.OBV(close.values, volume.values), index=close.index)
        else:
            # Manual calculation
            obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
            return obv
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def fibonacci_retracements(high: pd.Series, low: pd.Series, period: int = 50) -> Dict[str, float]:
        """Fibonacci Retracement Levels"""
        recent_high = high.tail(period).max()
        recent_low = low.tail(period).min()
        diff = recent_high - recent_low
        
        levels = {
            "0%": recent_high,
            "23.6%": recent_high - diff * 0.236,
            "38.2%": recent_high - diff * 0.382,
            "50%": recent_high - diff * 0.5,
            "61.8%": recent_high - diff * 0.618,
            "100%": recent_low
        }
        return levels


class CustomIndicatorBuilder:
    """Builder for creating custom technical indicators"""
    
    def __init__(self):
        self.indicators: Dict[str, IndicatorDefinition] = {}
        self.compiled_functions: Dict[str, Callable] = {}
        self._load_built_in_indicators()
    
    def _load_built_in_indicators(self):
        """Load built-in indicators"""
        # SMA
        self.indicators["SMA"] = IndicatorDefinition(
            name="SMA",
            type=IndicatorType.TREND,
            description="Simple Moving Average",
            parameters=[
                IndicatorParameter("period", int, 20, 1, 500, "Number of periods")
            ],
            formula="data.rolling(window=period).mean()",
            created_at=datetime.now()
        )
        
        # RSI
        self.indicators["RSI"] = IndicatorDefinition(
            name="RSI",
            type=IndicatorType.MOMENTUM,
            description="Relative Strength Index",
            parameters=[
                IndicatorParameter("period", int, 14, 2, 100, "Number of periods")
            ],
            formula="IndicatorLibrary.rsi(data, period)",
            created_at=datetime.now()
        )
        
        # MACD
        self.indicators["MACD"] = IndicatorDefinition(
            name="MACD",
            type=IndicatorType.MOMENTUM,
            description="Moving Average Convergence Divergence",
            parameters=[
                IndicatorParameter("fast", int, 12, 1, 50, "Fast EMA period"),
                IndicatorParameter("slow", int, 26, 20, 100, "Slow EMA period"),
                IndicatorParameter("signal", int, 9, 1, 50, "Signal line period")
            ],
            formula="IndicatorLibrary.macd(data, fast, slow, signal)",
            created_at=datetime.now()
        )
    
    def create_indicator(
        self,
        name: str,
        indicator_type: IndicatorType,
        description: str,
        formula: str,
        parameters: List[IndicatorParameter],
        dependencies: Optional[List[str]] = None,
        author: str = "user"
    ) -> IndicatorDefinition:
        """Create a new custom indicator"""
        
        # Validate formula
        if not self._validate_formula(formula, parameters):
            raise ValueError("Invalid formula syntax or undefined parameters")
        
        # Check dependencies
        if dependencies:
            for dep in dependencies:
                if dep not in self.indicators:
                    raise ValueError(f"Dependency '{dep}' not found")
        
        # Create indicator definition
        indicator = IndicatorDefinition(
            name=name,
            type=indicator_type,
            description=description,
            parameters=parameters,
            formula=formula,
            dependencies=dependencies or [],
            author=author,
            created_at=datetime.now()
        )
        
        # Compile the formula
        compiled_func = self._compile_formula(indicator)
        
        # Store indicator
        self.indicators[name] = indicator
        self.compiled_functions[name] = compiled_func
        
        logger.info(f"Created custom indicator: {name}")
        
        return indicator
    
    def _validate_formula(self, formula: str, parameters: List[IndicatorParameter]) -> bool:
        """Validate formula syntax and parameters"""
        try:
            # Parse the formula
            tree = ast.parse(formula, mode='eval')
            
            # Extract variable names
            variables = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    variables.add(node.id)
            
            # Check if all parameters are used
            param_names = {p.name for p in parameters}
            
            # Allow standard variables
            allowed_vars = param_names | {
                'data', 'high', 'low', 'close', 'open', 'volume',
                'pd', 'np', 'talib', 'IndicatorLibrary'
            }
            
            # Check for undefined variables
            undefined = variables - allowed_vars
            if undefined:
                logger.warning(f"Undefined variables in formula: {undefined}")
                return False
            
            return True
            
        except SyntaxError as e:
            logger.error(f"Formula syntax error: {e}")
            return False
    
    def _compile_formula(self, indicator: IndicatorDefinition) -> Callable:
        """Compile indicator formula into executable function"""
        
        # Build function signature
        params = ["data"]
        for param in indicator.parameters:
            params.append(f"{param.name}={param.default_value}")
        
        # Add additional data series parameters
        params.extend(["high=None", "low=None", "open=None", "volume=None"])
        
        func_signature = f"def {indicator.name}_func({', '.join(params)}):"
        
        # Build function body
        func_body = f"""
{func_signature}
    import pandas as pd
    import numpy as np
    import talib
    from app.services.indicators.custom_indicators import IndicatorLibrary
    
    result = {indicator.formula}
    return result
"""
        
        # Compile the function
        try:
            exec(func_body, globals())
            return globals()[f"{indicator.name}_func"]
        except Exception as e:
            logger.error(f"Failed to compile indicator {indicator.name}: {e}")
            raise
    
    def calculate_indicator(
        self,
        name: str,
        data: pd.DataFrame,
        **kwargs
    ) -> IndicatorResult:
        """Calculate indicator values"""
        
        if name not in self.indicators:
            raise ValueError(f"Indicator '{name}' not found")
        
        indicator = self.indicators[name]
        func = self.compiled_functions[name]
        
        # Prepare data
        if 'close' in data.columns:
            primary_data = data['close']
        else:
            primary_data = data.iloc[:, 0]  # Use first column
        
        # Calculate dependencies first
        dependency_results = {}
        if indicator.dependencies:
            for dep_name in indicator.dependencies:
                dep_result = self.calculate_indicator(dep_name, data, **kwargs)
                dependency_results[dep_name] = dep_result.values
        
        # Prepare parameters
        params = {
            'data': primary_data,
            'high': data.get('high') if 'high' in data else None,
            'low': data.get('low') if 'low' in data else None,
            'open': data.get('open') if 'open' in data else None,
            'volume': data.get('volume') if 'volume' in data else None
        }
        
        # Add user parameters
        for param in indicator.parameters:
            if param.name in kwargs:
                params[param.name] = kwargs[param.name]
        
        # Add dependency results
        params.update(dependency_results)
        
        # Calculate indicator
        try:
            result = func(**params)
            
            # Handle different return types
            if isinstance(result, pd.Series):
                values = result
                signals = None
            elif isinstance(result, dict):
                # Multi-output indicator (like MACD)
                values = result.get('main', result.get('macd', list(result.values())[0]))
                signals = result.get('signal')
            else:
                values = pd.Series(result, index=data.index)
                signals = None
            
            # Generate trading signals if applicable
            if signals is None and indicator.type in [IndicatorType.MOMENTUM, IndicatorType.TREND]:
                signals = self._generate_signals(values, indicator)
            
            return IndicatorResult(
                name=name,
                values=values,
                signals=signals,
                metadata={
                    'indicator_type': indicator.type.value,
                    'parameters': kwargs
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate indicator {name}: {e}")
            raise
    
    def _generate_signals(self, values: pd.Series, indicator: IndicatorDefinition) -> pd.Series:
        """Generate trading signals based on indicator values"""
        
        signals = pd.Series(0, index=values.index)
        
        if indicator.type == IndicatorType.MOMENTUM:
            # RSI-like signals
            if 'RSI' in indicator.name.upper():
                signals[values < 30] = 1   # Oversold - Buy
                signals[values > 70] = -1  # Overbought - Sell
            
            # MACD-like signals
            elif 'MACD' in indicator.name.upper():
                # Simplified - need both MACD and signal line
                signals[values > 0] = 1
                signals[values < 0] = -1
        
        elif indicator.type == IndicatorType.TREND:
            # Moving average signals
            if 'MA' in indicator.name.upper() or 'MOVING' in indicator.name.upper():
                # Price vs MA crossover
                # This requires price data - simplified for now
                signals[values.diff() > 0] = 1   # MA trending up
                signals[values.diff() < 0] = -1  # MA trending down
        
        return signals
    
    def backtest_indicator(
        self,
        name: str,
        data: pd.DataFrame,
        initial_capital: float = 10000,
        **kwargs
    ) -> Dict[str, Any]:
        """Backtest indicator performance"""
        
        # Calculate indicator
        result = self.calculate_indicator(name, data, **kwargs)
        
        if result.signals is None:
            return {
                "error": "No trading signals generated",
                "indicator_values": result.values
            }
        
        # Simple backtesting
        positions = result.signals.fillna(0)
        returns = data['close'].pct_change()
        
        # Calculate strategy returns
        strategy_returns = positions.shift(1) * returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # Calculate metrics
        total_return = cumulative_returns.iloc[-1] - 1
        sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        # Count trades
        trades = positions.diff().fillna(0)
        num_trades = len(trades[trades != 0])
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "num_trades": num_trades,
            "final_capital": initial_capital * (1 + total_return),
            "cumulative_returns": cumulative_returns,
            "indicator_values": result.values,
            "signals": result.signals
        }
    
    def optimize_parameters(
        self,
        name: str,
        data: pd.DataFrame,
        param_ranges: Dict[str, tuple],
        objective: str = "sharpe_ratio"
    ) -> Dict[str, Any]:
        """Optimize indicator parameters"""
        
        best_params = {}
        best_score = -np.inf
        
        # Grid search (simplified - use more sophisticated optimization in production)
        from itertools import product
        
        # Create parameter grid
        param_names = list(param_ranges.keys())
        param_values = [
            np.linspace(start, end, 10) if isinstance(start, float) 
            else range(start, end + 1, max(1, (end - start) // 10))
            for start, end in param_ranges.values()
        ]
        
        # Test each combination
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            
            try:
                # Backtest with these parameters
                backtest_result = self.backtest_indicator(name, data, **params)
                
                # Get objective value
                score = backtest_result.get(objective, -np.inf)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            
            except Exception as e:
                logger.debug(f"Parameter combination failed: {params}, {e}")
                continue
        
        # Run final backtest with best parameters
        final_backtest = self.backtest_indicator(name, data, **best_params)
        
        return {
            "best_parameters": best_params,
            "best_score": best_score,
            "objective": objective,
            "backtest_results": final_backtest
        }
    
    def export_indicator(self, name: str) -> str:
        """Export indicator as Python code"""
        
        if name not in self.indicators:
            raise ValueError(f"Indicator '{name}' not found")
        
        indicator = self.indicators[name]
        
        # Generate Python code
        code = f'''"""
{indicator.name} - {indicator.description}
Author: {indicator.author}
Version: {indicator.version}
Created: {indicator.created_at}
"""

import pandas as pd
import numpy as np
import talib

def {indicator.name.lower()}({', '.join([f"{p.name}: {p.type.__name__} = {p.default_value}" for p in indicator.parameters])}):
    """
    {indicator.description}
    
    Parameters:
    {chr(10).join([f"    {p.name}: {p.description}" for p in indicator.parameters])}
    
    Returns:
        pd.Series or dict: Indicator values
    """
    
    # Formula:
    # {indicator.formula}
    
    return {indicator.formula}
'''
        
        return code
    
    def save_indicator(self, name: str, filepath: str):
        """Save indicator to file"""
        code = self.export_indicator(name)
        with open(filepath, 'w') as f:
            f.write(code)
    
    def load_indicator(self, filepath: str) -> IndicatorDefinition:
        """Load indicator from file"""
        # Parse Python file and extract indicator definition
        # This is a simplified implementation
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract metadata from docstring
        # In production, use proper parsing
        name = filepath.split('/')[-1].replace('.py', '').upper()
        
        return self.create_indicator(
            name=name,
            indicator_type=IndicatorType.CUSTOM,
            description="Loaded from file",
            formula="# Loaded from " + filepath,
            parameters=[],
            author="file"
        )