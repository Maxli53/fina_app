import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.models.strategy import (
    StrategyConfig, BacktestConfig, BacktestResult, 
    PerformanceMetrics, SignalMethod
)
from app.services.data.yahoo_finance import YahooFinanceService

logger = logging.getLogger(__name__)


class BacktestingEngine:
    """Engine for backtesting trading strategies"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.yf_service = YahooFinanceService()
    
    async def run_backtest(
        self, 
        strategy_config: StrategyConfig, 
        backtest_config: BacktestConfig
    ) -> BacktestResult:
        """Run a complete strategy backtest"""
        
        def _backtest():
            try:
                logger.info(f"Starting backtest for strategy: {strategy_config.name}")
                
                # Fetch historical data
                price_data = self._fetch_backtest_data(strategy_config, backtest_config)
                
                # Generate signals
                signals = self._generate_signals(strategy_config, price_data)
                
                # Simulate trading
                positions, trades, returns_series = self._simulate_trading(
                    strategy_config, backtest_config, price_data, signals
                )
                
                # Calculate performance metrics
                performance_metrics = self._calculate_performance_metrics(
                    returns_series, trades, backtest_config
                )
                
                # Generate benchmark comparison
                benchmark_comparison = self._generate_benchmark_comparison(
                    returns_series, backtest_config
                )
                
                # Calculate risk analysis
                rolling_metrics = self._calculate_rolling_metrics(returns_series)
                drawdown_periods = self._identify_drawdown_periods(returns_series)
                var_history = self._calculate_var_history(returns_series)
                
                # Calculate execution statistics
                execution_stats = self._calculate_execution_stats(trades, backtest_config)
                
                return BacktestResult(
                    strategy_id=f"backtest_{datetime.now().timestamp()}",
                    config=backtest_config,
                    performance_metrics=performance_metrics,
                    returns_series=returns_series,
                    positions=positions,
                    trades=trades,
                    benchmark_comparison=benchmark_comparison,
                    rolling_metrics=rolling_metrics,
                    drawdown_periods=drawdown_periods,
                    var_history=var_history,
                    execution_stats=execution_stats,
                    slippage_impact=self._calculate_slippage_impact(trades, backtest_config),
                    transaction_costs=self._calculate_transaction_costs(trades, backtest_config)
                )
                
            except Exception as e:
                logger.error(f"Backtest failed: {str(e)}")
                raise
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _backtest)
    
    def _fetch_backtest_data(
        self, 
        strategy_config: StrategyConfig, 
        backtest_config: BacktestConfig
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for backtesting"""
        
        price_data = {}
        
        # Add some buffer for technical indicators
        start_date = backtest_config.start_date - timedelta(days=100)
        
        for symbol in strategy_config.symbols:
            try:
                # In a real implementation, this would be synchronous or use asyncio.run
                # For now, we'll simulate the data
                dates = pd.date_range(start=start_date, end=backtest_config.end_date, freq='D')
                
                # Generate realistic price data
                initial_price = 100
                returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
                prices = initial_price * np.exp(np.cumsum(returns))
                
                # Create OHLC data
                df = pd.DataFrame({
                    'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
                    'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
                    'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
                    'close': prices,
                    'volume': np.random.randint(100000, 1000000, len(dates))
                }, index=dates)
                
                # Ensure OHLC consistency
                df['high'] = np.maximum.reduce([df['open'], df['high'], df['low'], df['close']])
                df['low'] = np.minimum.reduce([df['open'], df['high'], df['low'], df['close']])
                
                price_data[symbol] = df
                
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {str(e)}")
                continue
        
        return price_data
    
    def _generate_signals(
        self, 
        strategy_config: StrategyConfig, 
        price_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Generate trading signals based on strategy configuration"""
        
        # Get the date range for signals
        start_date = max(df.index.min() for df in price_data.values())
        end_date = min(df.index.max() for df in price_data.values())
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=date_range)
        
        for signal_name, signal_config in strategy_config.signals.items():
            if signal_config.method == SignalMethod.IDTXL:
                signal_values = self._generate_idtxl_signals(signal_config, price_data, date_range)
            elif signal_config.method == SignalMethod.ML:
                signal_values = self._generate_ml_signals(signal_config, price_data, date_range)
            elif signal_config.method == SignalMethod.NN:
                signal_values = self._generate_nn_signals(signal_config, price_data, date_range)
            else:
                # Default to random signals for unknown methods
                signal_values = np.random.choice([-1, 0, 1], size=len(date_range), p=[0.3, 0.4, 0.3])
            
            signals[signal_name] = signal_values
        
        # Combine signals based on weights
        weighted_signals = np.zeros(len(date_range))
        for signal_name, signal_config in strategy_config.signals.items():
            if signal_name in signals.columns:
                weighted_signals += signals[signal_name] * signal_config.weight
        
        signals['combined_signal'] = weighted_signals
        
        # Apply threshold if configured
        threshold = 0.1  # Default threshold
        signals['position_signal'] = np.where(
            signals['combined_signal'] > threshold, 1,
            np.where(signals['combined_signal'] < -threshold, -1, 0)
        )
        
        return signals
    
    def _generate_idtxl_signals(
        self, 
        signal_config, 
        price_data: Dict[str, pd.DataFrame], 
        date_range: pd.DatetimeIndex
    ) -> np.ndarray:
        """Generate signals based on IDTxl analysis (simplified)"""
        
        # Simplified IDTxl signal generation
        # In reality, this would run actual IDTxl analysis
        
        # Use primary symbol's price data
        primary_symbol = list(price_data.keys())[0]
        prices = price_data[primary_symbol]['close'].reindex(date_range, method='ffill')
        
        # Generate signals based on price momentum and cross-symbol relationships
        returns = prices.pct_change(5)  # 5-day returns
        momentum_signal = np.where(returns > 0.02, 1, np.where(returns < -0.02, -1, 0))
        
        # Add some noise to simulate information-theoretic complexity
        noise = np.random.normal(0, 0.3, len(date_range))
        raw_signal = momentum_signal + noise
        
        # Normalize to [-1, 1] range
        return np.clip(raw_signal, -1, 1)
    
    def _generate_ml_signals(
        self, 
        signal_config, 
        price_data: Dict[str, pd.DataFrame], 
        date_range: pd.DatetimeIndex
    ) -> np.ndarray:
        """Generate signals based on ML predictions (simplified)"""
        
        # Simplified ML signal generation
        primary_symbol = list(price_data.keys())[0]
        df = price_data[primary_symbol].reindex(date_range, method='ffill')
        
        # Generate technical features
        df['sma_20'] = df['close'].rolling(20).mean()
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['bb_position'] = (df['close'] - df['sma_20']) / (df['close'].rolling(20).std() * 2)
        
        # Generate signals based on technical indicators
        ml_signal = np.where(
            (df['rsi'] < 30) & (df['bb_position'] < -0.5), 1,  # Oversold
            np.where((df['rsi'] > 70) & (df['bb_position'] > 0.5), -1, 0)  # Overbought
        )
        
        # Add model confidence simulation
        confidence = np.random.uniform(0.5, 1.0, len(date_range))
        return ml_signal * confidence
    
    def _generate_nn_signals(
        self, 
        signal_config, 
        price_data: Dict[str, pd.DataFrame], 
        date_range: pd.DatetimeIndex
    ) -> np.ndarray:
        """Generate signals based on neural network predictions (simplified)"""
        
        # Simplified NN signal generation
        primary_symbol = list(price_data.keys())[0]
        prices = price_data[primary_symbol]['close'].reindex(date_range, method='ffill')
        
        # Simulate sequence-based predictions
        sequence_length = 20
        signals = np.zeros(len(date_range))
        
        for i in range(sequence_length, len(date_range)):
            # Get price sequence
            price_sequence = prices.iloc[i-sequence_length:i].values
            
            # Normalize sequence
            price_sequence = (price_sequence - price_sequence.mean()) / price_sequence.std()
            
            # Simulate NN prediction (simplified pattern recognition)
            trend = np.polyfit(range(sequence_length), price_sequence, 1)[0]
            volatility = price_sequence.std()
            
            # Generate signal based on trend and volatility
            if trend > 0.1 and volatility < 1.0:
                signals[i] = 1  # Strong uptrend with low volatility
            elif trend < -0.1 and volatility < 1.0:
                signals[i] = -1  # Strong downtrend with low volatility
            else:
                signals[i] = 0  # No clear signal
        
        return signals
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _simulate_trading(
        self, 
        strategy_config: StrategyConfig, 
        backtest_config: BacktestConfig, 
        price_data: Dict[str, pd.DataFrame], 
        signals: pd.DataFrame
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Simulate trading based on signals"""
        
        positions = []
        trades = []
        returns_series = []
        
        # Initialize portfolio
        cash = backtest_config.initial_capital
        holdings = {symbol: 0 for symbol in strategy_config.symbols}
        portfolio_value = cash
        
        primary_symbol = strategy_config.symbols[0]
        prices = price_data[primary_symbol]['close']
        
        for date in signals.index:
            if date not in prices.index:
                continue
                
            current_price = prices[date]
            signal = signals.loc[date, 'position_signal']
            
            # Calculate current portfolio value
            portfolio_value = cash + sum(
                holdings[symbol] * price_data[symbol]['close'].loc[date] 
                for symbol in strategy_config.symbols 
                if symbol in price_data and date in price_data[symbol].index
            )
            
            # Position sizing based on strategy configuration
            if signal != 0:
                target_position_value = portfolio_value * strategy_config.risk_management.max_position_size * signal
                current_position_value = holdings[primary_symbol] * current_price
                
                position_change = target_position_value - current_position_value
                
                if abs(position_change) > portfolio_value * 0.01:  # Minimum trade size
                    shares_to_trade = position_change / current_price
                    
                    # Apply transaction costs
                    trade_cost = abs(position_change) * backtest_config.transaction_costs
                    
                    # Execute trade
                    if cash >= abs(position_change) + trade_cost:
                        holdings[primary_symbol] += shares_to_trade
                        cash -= position_change + trade_cost
                        
                        # Record trade
                        trades.append({
                            'date': date,
                            'symbol': primary_symbol,
                            'action': 'buy' if shares_to_trade > 0 else 'sell',
                            'shares': abs(shares_to_trade),
                            'price': current_price,
                            'value': abs(position_change),
                            'cost': trade_cost,
                            'signal_strength': abs(signal)
                        })
            
            # Record position
            positions.append({
                'date': date,
                'cash': cash,
                'holdings': holdings.copy(),
                'portfolio_value': portfolio_value,
                'primary_position': holdings[primary_symbol] * current_price
            })
            
            # Calculate returns
            if len(returns_series) > 0:
                previous_value = returns_series[-1]['portfolio_value']
                daily_return = (portfolio_value - previous_value) / previous_value
            else:
                daily_return = 0.0
            
            returns_series.append({
                'date': date.isoformat() if hasattr(date, 'isoformat') else str(date),
                'portfolio_value': portfolio_value,
                'daily_return': daily_return,
                'cumulative_return': (portfolio_value - backtest_config.initial_capital) / backtest_config.initial_capital
            })
        
        return positions, trades, returns_series
    
    def _calculate_performance_metrics(
        self, 
        returns_series: List[Dict], 
        trades: List[Dict], 
        backtest_config: BacktestConfig
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        if not returns_series:
            raise ValueError("No returns data available for performance calculation")
        
        # Extract returns
        daily_returns = np.array([r['daily_return'] for r in returns_series[1:]])  # Skip first day
        portfolio_values = np.array([r['portfolio_value'] for r in returns_series])
        
        # Basic performance metrics
        total_return = (portfolio_values[-1] - backtest_config.initial_capital) / backtest_config.initial_capital
        
        # Annualized metrics
        trading_days = len(daily_returns)
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        volatility = np.std(daily_returns) * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Downside deviation for Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown analysis
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = abs(np.min(drawdown))
        
        # Find max drawdown duration
        drawdown_periods = []
        in_drawdown = False
        start_idx = 0
        
        for i, dd in enumerate(drawdown):
            if dd < -0.01 and not in_drawdown:  # Start of drawdown
                in_drawdown = True
                start_idx = i
            elif dd >= -0.01 and in_drawdown:  # End of drawdown
                in_drawdown = False
                drawdown_periods.append(i - start_idx)
        
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # VaR calculations
        var_95 = np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0
        var_99 = np.percentile(daily_returns, 1) if len(daily_returns) > 0 else 0
        
        # Trading metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if self._is_winning_trade(t, returns_series)])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Trade analysis
        trade_returns = [self._calculate_trade_return(t, returns_series) for t in trades]
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            var_95=var_95,
            var_99=var_99,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor
        )
    
    def _is_winning_trade(self, trade: Dict, returns_series: List[Dict]) -> bool:
        """Determine if a trade was profitable (simplified)"""
        # Simplified: assume trades are profitable based on signal direction and subsequent price movement
        return np.random.random() > 0.4  # 60% win rate simulation
    
    def _calculate_trade_return(self, trade: Dict, returns_series: List[Dict]) -> float:
        """Calculate return for a specific trade (simplified)"""
        # Simplified trade return calculation
        return np.random.normal(0.02, 0.05)  # Random return simulation
    
    def _generate_benchmark_comparison(
        self, 
        returns_series: List[Dict], 
        backtest_config: BacktestConfig
    ) -> Dict[str, Any]:
        """Generate benchmark comparison (simplified)"""
        
        # Simulate benchmark performance
        benchmark_return = np.random.normal(0.08, 0.15)  # 8% annual return, 15% volatility
        strategy_return = returns_series[-1]['cumulative_return'] if returns_series else 0
        
        return {
            "benchmark_symbol": backtest_config.benchmark,
            "benchmark_return": benchmark_return,
            "strategy_return": strategy_return,
            "excess_return": strategy_return - benchmark_return,
            "tracking_error": 0.05,  # Simplified
            "information_ratio": (strategy_return - benchmark_return) / 0.05
        }
    
    def _calculate_rolling_metrics(self, returns_series: List[Dict]) -> Dict[str, List[float]]:
        """Calculate rolling performance metrics"""
        
        daily_returns = [r['daily_return'] for r in returns_series[1:]]
        window = 30  # 30-day rolling window
        
        rolling_sharpe = []
        rolling_volatility = []
        
        for i in range(window, len(daily_returns)):
            window_returns = daily_returns[i-window:i]
            vol = np.std(window_returns) * np.sqrt(252)
            ret = np.mean(window_returns) * 252
            sharpe = ret / vol if vol > 0 else 0
            
            rolling_volatility.append(vol)
            rolling_sharpe.append(sharpe)
        
        return {
            "rolling_sharpe": rolling_sharpe,
            "rolling_volatility": rolling_volatility
        }
    
    def _identify_drawdown_periods(self, returns_series: List[Dict]) -> List[Dict[str, Any]]:
        """Identify significant drawdown periods"""
        
        portfolio_values = [r['portfolio_value'] for r in returns_series]
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - peak) / peak
        
        drawdown_periods = []
        in_drawdown = False
        start_idx = 0
        
        for i, dd in enumerate(drawdown):
            if dd < -0.05 and not in_drawdown:  # 5% drawdown threshold
                in_drawdown = True
                start_idx = i
            elif dd >= -0.01 and in_drawdown:
                in_drawdown = False
                drawdown_periods.append({
                    "start_date": returns_series[start_idx]['date'],
                    "end_date": returns_series[i]['date'],
                    "duration_days": i - start_idx,
                    "max_drawdown": abs(min(drawdown[start_idx:i+1])),
                    "recovery_date": returns_series[i]['date']
                })
        
        return drawdown_periods
    
    def _calculate_var_history(self, returns_series: List[Dict]) -> List[Dict[str, Any]]:
        """Calculate VaR history"""
        
        daily_returns = [r['daily_return'] for r in returns_series[1:]]
        window = 30
        
        var_history = []
        for i in range(window, len(daily_returns)):
            window_returns = daily_returns[i-window:i]
            var_95 = np.percentile(window_returns, 5)
            var_99 = np.percentile(window_returns, 1)
            
            var_history.append({
                "date": returns_series[i+1]['date'],
                "var_95": var_95,
                "var_99": var_99
            })
        
        return var_history
    
    def _calculate_execution_stats(
        self, 
        trades: List[Dict], 
        backtest_config: BacktestConfig
    ) -> Dict[str, Any]:
        """Calculate execution statistics"""
        
        if not trades:
            return {"total_trades": 0}
        
        trade_sizes = [t['value'] for t in trades]
        
        return {
            "total_trades": len(trades),
            "avg_trade_size": np.mean(trade_sizes),
            "median_trade_size": np.median(trade_sizes),
            "largest_trade": max(trade_sizes),
            "smallest_trade": min(trade_sizes),
            "trades_per_day": len(trades) / max(1, (backtest_config.end_date - backtest_config.start_date).days)
        }
    
    def _calculate_slippage_impact(
        self, 
        trades: List[Dict], 
        backtest_config: BacktestConfig
    ) -> Dict[str, Any]:
        """Calculate slippage impact on performance"""
        
        total_slippage = sum(t['value'] * backtest_config.slippage for t in trades)
        
        return {
            "total_slippage_cost": total_slippage,
            "slippage_per_trade": total_slippage / len(trades) if trades else 0,
            "slippage_rate": backtest_config.slippage
        }
    
    def _calculate_transaction_costs(
        self, 
        trades: List[Dict], 
        backtest_config: BacktestConfig
    ) -> Dict[str, Any]:
        """Calculate transaction costs impact"""
        
        total_transaction_costs = sum(t.get('cost', 0) for t in trades)
        
        return {
            "total_transaction_costs": total_transaction_costs,
            "cost_per_trade": total_transaction_costs / len(trades) if trades else 0,
            "cost_rate": backtest_config.transaction_costs
        }