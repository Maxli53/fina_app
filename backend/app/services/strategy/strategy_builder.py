import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.models.strategy import (
    StrategyConfig, Strategy, StrategyOptimizationConfig, 
    StrategyOptimizationResult, SignalMethod, BacktestMetric
)
from app.services.analysis.idtxl_service import IDTxlService
from app.services.analysis.ml_service import MLService
from app.services.analysis.nn_service import NeuralNetworkService

logger = logging.getLogger(__name__)


class StrategyBuilderService:
    """Service for building and validating trading strategies"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.idtxl_service = IDTxlService()
        self.ml_service = MLService()
        self.nn_service = NeuralNetworkService()
    
    async def validate_strategy(self, config: StrategyConfig) -> Dict[str, Any]:
        """Validate strategy configuration"""
        errors = []
        warnings = []
        
        try:
            # Validate basic configuration
            if not config.name or len(config.name.strip()) == 0:
                errors.append("Strategy name is required")
            
            if not config.symbols:
                errors.append("At least one symbol is required")
            
            if not config.signals:
                errors.append("At least one signal source is required")
            
            # Validate signal weights sum to 1
            total_weight = sum(signal.weight for signal in config.signals.values())
            if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
                errors.append(f"Signal weights must sum to 1.0, got {total_weight}")
            
            # Validate risk management parameters
            if config.risk_management.max_position_size <= 0 or config.risk_management.max_position_size > 1:
                errors.append("Max position size must be between 0 and 1")
            
            if config.risk_management.stop_loss and config.risk_management.stop_loss >= 1:
                errors.append("Stop loss must be less than 1 (100%)")
            
            if config.risk_management.max_drawdown <= 0 or config.risk_management.max_drawdown > 1:
                errors.append("Max drawdown must be between 0 and 1")
            
            # Validate signal configurations
            for signal_name, signal_config in config.signals.items():
                if signal_config.weight < 0 or signal_config.weight > 1:
                    errors.append(f"Signal '{signal_name}' weight must be between 0 and 1")
                
                if signal_config.threshold and (signal_config.threshold < 0 or signal_config.threshold > 1):
                    errors.append(f"Signal '{signal_name}' threshold must be between 0 and 1")
            
            # Validate execution rules
            if config.execution_rules.min_holding_period and config.execution_rules.max_holding_period:
                if config.execution_rules.min_holding_period > config.execution_rules.max_holding_period:
                    errors.append("Minimum holding period cannot be greater than maximum holding period")
            
            # Check for potential issues
            if len(config.symbols) > 20:
                warnings.append("Large number of symbols may impact performance")
            
            if len(config.signals) > 5:
                warnings.append("Many signal sources may lead to over-fitting")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings
            }
            
        except Exception as e:
            logger.error(f"Strategy validation failed: {str(e)}")
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": warnings
            }
    
    async def validate_for_live_trading(self, strategy: Strategy) -> Dict[str, Any]:
        """Validate strategy is ready for live trading"""
        issues = []
        requirements = []
        
        try:
            # Check if strategy has been backtested
            if not strategy.backtest_results:
                issues.append("Strategy must be backtested before live trading")
            else:
                # Check backtest performance
                metrics = strategy.backtest_results.performance_metrics
                
                if metrics.sharpe_ratio < 0.5:
                    issues.append(f"Low Sharpe ratio: {metrics.sharpe_ratio:.2f} (recommended > 0.5)")
                
                if metrics.max_drawdown > 0.2:
                    issues.append(f"High max drawdown: {metrics.max_drawdown:.2f} (recommended < 0.2)")
                
                if metrics.total_trades < 10:
                    issues.append(f"Insufficient trade history: {metrics.total_trades} trades")
                
                if metrics.win_rate < 0.35:
                    issues.append(f"Low win rate: {metrics.win_rate:.2f} (recommended > 0.35)")
            
            # Check signal sources are properly configured
            for signal_name, signal_config in strategy.config.signals.items():
                if signal_config.method == SignalMethod.IDTXL:
                    if not signal_config.parameters.get('analysis_type'):
                        issues.append(f"IDTxl signal '{signal_name}' missing analysis_type parameter")
                
                elif signal_config.method == SignalMethod.ML:
                    if not signal_config.parameters.get('model_type'):
                        issues.append(f"ML signal '{signal_name}' missing model_type parameter")
                
                elif signal_config.method == SignalMethod.NN:
                    if not signal_config.parameters.get('architecture'):
                        issues.append(f"NN signal '{signal_name}' missing architecture parameter")
            
            # Check risk management
            if not strategy.config.risk_management.stop_loss:
                requirements.append("Consider adding stop loss for risk management")
            
            if not strategy.config.risk_management.var_limit:
                requirements.append("Consider adding VaR limit for risk management")
            
            return {
                "ready": len(issues) == 0,
                "issues": issues,
                "requirements": requirements
            }
            
        except Exception as e:
            logger.error(f"Live trading validation failed: {str(e)}")
            return {
                "ready": False,
                "issues": [f"Validation error: {str(e)}"],
                "requirements": requirements
            }
    
    async def check_rebacktest_needed(
        self, 
        backtest_result: Any, 
        new_config: StrategyConfig
    ) -> bool:
        """Check if strategy needs to be re-backtested due to config changes"""
        
        # Key parameters that require re-backtesting
        key_changes = [
            # Signal changes
            len(new_config.signals) != len(backtest_result.strategy_config.signals if hasattr(backtest_result, 'strategy_config') else {}),
            
            # Risk management changes
            new_config.risk_management.max_position_size != getattr(backtest_result, 'max_position_size', None),
            new_config.risk_management.stop_loss != getattr(backtest_result, 'stop_loss', None),
            
            # Symbol changes
            set(new_config.symbols) != set(getattr(backtest_result, 'symbols', [])),
            
            # Execution rule changes
            new_config.execution_rules.position_sizing != getattr(backtest_result, 'position_sizing', None),
            new_config.execution_rules.rebalance_frequency != getattr(backtest_result, 'rebalance_frequency', None)
        ]
        
        return any(key_changes)
    
    async def optimize_strategy(
        self, 
        base_config: StrategyConfig, 
        optimization_config: StrategyOptimizationConfig
    ) -> StrategyOptimizationResult:
        """Optimize strategy parameters"""
        
        def _optimize():
            try:
                logger.info(f"Starting strategy optimization with {optimization_config.optimization_method}")
                
                best_score = -np.inf
                best_parameters = {}
                evaluation_history = []
                
                # Generate parameter combinations based on optimization method
                if optimization_config.optimization_method == "grid_search":
                    param_combinations = self._generate_grid_search_params(optimization_config.parameter_ranges)
                elif optimization_config.optimization_method == "random_search":
                    param_combinations = self._generate_random_search_params(
                        optimization_config.parameter_ranges, 
                        optimization_config.max_evaluations
                    )
                else:  # bayesian
                    param_combinations = self._generate_bayesian_params(
                        optimization_config.parameter_ranges, 
                        optimization_config.max_evaluations
                    )
                
                # Evaluate each parameter combination
                for i, params in enumerate(param_combinations[:optimization_config.max_evaluations]):
                    if i % 10 == 0:
                        logger.info(f"Optimization progress: {i}/{min(len(param_combinations), optimization_config.max_evaluations)}")
                    
                    # Apply parameters to strategy config
                    test_config = self._apply_parameters_to_config(base_config, params)
                    
                    # Simulate backtest (simplified for now)
                    score, metrics = self._evaluate_strategy_config(test_config, optimization_config)
                    
                    evaluation_history.append({
                        "parameters": params,
                        "score": score,
                        "metrics": metrics,
                        "evaluation_id": i
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_parameters = params
                
                return StrategyOptimizationResult(
                    optimization_id=str(datetime.now().timestamp()),
                    strategy_id="temp",  # Will be set by caller
                    config=optimization_config,
                    best_parameters=best_parameters,
                    best_score=best_score,
                    best_metrics=evaluation_history[-1]["metrics"] if evaluation_history else {},
                    evaluation_history=evaluation_history,
                    total_evaluations=len(evaluation_history),
                    optimization_time=0.0,  # Will be calculated
                    convergence_info={
                        "method": optimization_config.optimization_method,
                        "final_score": best_score,
                        "improvement_over_baseline": best_score - evaluation_history[0]["score"] if evaluation_history else 0
                    },
                    completed_at=datetime.utcnow()
                )
                
            except Exception as e:
                logger.error(f"Strategy optimization failed: {str(e)}")
                raise
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _optimize)
    
    def _generate_grid_search_params(self, parameter_ranges: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate parameter combinations for grid search"""
        param_names = list(parameter_ranges.keys())
        param_values = []
        
        for param_name in param_names:
            param_range = parameter_ranges[param_name]
            if 'values' in param_range:
                param_values.append(param_range['values'])
            else:
                # Generate range based on min, max, step
                start = param_range['min']
                stop = param_range['max']
                step = param_range.get('step', (stop - start) / 10)
                param_values.append(list(np.arange(start, stop + step, step)))
        
        # Generate all combinations
        combinations = []
        import itertools
        for combo in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, combo)))
        
        return combinations
    
    def _generate_random_search_params(
        self, 
        parameter_ranges: Dict[str, Dict[str, Any]], 
        n_samples: int
    ) -> List[Dict[str, Any]]:
        """Generate random parameter combinations"""
        combinations = []
        
        for _ in range(n_samples):
            params = {}
            for param_name, param_range in parameter_ranges.items():
                if 'values' in param_range:
                    params[param_name] = np.random.choice(param_range['values'])
                else:
                    # Generate random value in range
                    min_val = param_range['min']
                    max_val = param_range['max']
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        params[param_name] = np.random.randint(min_val, max_val + 1)
                    else:
                        params[param_name] = np.random.uniform(min_val, max_val)
            combinations.append(params)
        
        return combinations
    
    def _generate_bayesian_params(
        self, 
        parameter_ranges: Dict[str, Dict[str, Any]], 
        n_samples: int
    ) -> List[Dict[str, Any]]:
        """Generate parameter combinations using Bayesian optimization (simplified)"""
        # For now, use random search as a placeholder
        # In a real implementation, this would use libraries like optuna or scikit-optimize
        return self._generate_random_search_params(parameter_ranges, n_samples)
    
    def _apply_parameters_to_config(
        self, 
        base_config: StrategyConfig, 
        parameters: Dict[str, Any]
    ) -> StrategyConfig:
        """Apply optimization parameters to strategy configuration"""
        # Deep copy the config and apply parameters
        config_dict = base_config.dict()
        
        for param_name, param_value in parameters.items():
            # Navigate nested parameters using dot notation
            if '.' in param_name:
                keys = param_name.split('.')
                current = config_dict
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = param_value
            else:
                config_dict[param_name] = param_value
        
        return StrategyConfig(**config_dict)
    
    def _evaluate_strategy_config(
        self, 
        config: StrategyConfig, 
        optimization_config: StrategyOptimizationConfig
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate a strategy configuration (simplified simulation)"""
        
        # Simplified evaluation - in reality this would run a full backtest
        # For now, return mock metrics based on configuration
        
        # Mock performance metrics
        base_return = 0.08
        base_volatility = 0.15
        base_sharpe = base_return / base_volatility
        
        # Adjust based on configuration complexity
        signal_penalty = len(config.signals) * 0.01  # Penalize over-complexity
        risk_bonus = config.risk_management.max_position_size * 0.05  # Reward appropriate risk
        
        mock_return = base_return - signal_penalty + risk_bonus + np.random.normal(0, 0.02)
        mock_volatility = base_volatility + np.random.normal(0, 0.02)
        mock_sharpe = mock_return / max(mock_volatility, 0.01)
        mock_drawdown = abs(np.random.normal(0.05, 0.02))
        
        # Calculate score based on optimization target
        if optimization_config.optimization_target == BacktestMetric.SHARPE_RATIO:
            score = mock_sharpe
        elif optimization_config.optimization_target == BacktestMetric.TOTAL_RETURN:
            score = mock_return
        elif optimization_config.optimization_target == BacktestMetric.MAX_DRAWDOWN:
            score = -mock_drawdown  # Minimize drawdown
        else:
            score = mock_sharpe  # Default to Sharpe ratio
        
        metrics = {
            "total_return": mock_return,
            "volatility": mock_volatility,
            "sharpe_ratio": mock_sharpe,
            "max_drawdown": mock_drawdown,
            "win_rate": 0.5 + np.random.normal(0, 0.1),
            "total_trades": int(50 + np.random.normal(0, 10))
        }
        
        return score, metrics
    
    async def generate_signal_recommendations(
        self, 
        symbols: List[str], 
        timeframe: str = "1d"
    ) -> Dict[str, Any]:
        """Generate recommendations for signal configurations"""
        
        # Analyze symbol correlations for IDTxl recommendations
        correlation_analysis = {
            "high_correlation_pairs": [],
            "low_correlation_pairs": [],
            "recommended_idtxl_variables": symbols[:5]  # Limit for performance
        }
        
        # ML feature recommendations
        ml_recommendations = {
            "recommended_features": [
                "return_5d", "return_20d", "volatility_20d", "rsi_14", "sma_ratio_20"
            ],
            "recommended_models": ["random_forest", "xgboost"],
            "recommended_target": "direction"
        }
        
        # NN architecture recommendations
        nn_recommendations = {
            "recommended_architecture": "lstm",
            "recommended_layers": [64, 32],
            "recommended_sequence_length": 60
        }
        
        return {
            "symbols": symbols,
            "timeframe": timeframe,
            "idtxl_recommendations": correlation_analysis,
            "ml_recommendations": ml_recommendations,
            "nn_recommendations": nn_recommendations,
            "risk_management_suggestions": {
                "max_position_size": 0.1,
                "stop_loss": 0.02,
                "max_drawdown": 0.15
            }
        }