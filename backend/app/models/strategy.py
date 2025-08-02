from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal, Union
from datetime import datetime
from enum import Enum


class StrategyType(str, Enum):
    SIGNAL_BASED = "signal_based"
    ML_PREDICTION = "ml_prediction"
    NN_ENSEMBLE = "nn_ensemble"
    INTEGRATED = "integrated"


class SignalMethod(str, Enum):
    IDTXL = "idtxl"
    ML = "ml"
    NN = "nn"


class PositionSizing(str, Enum):
    FIXED = "fixed"
    KELLY_CRITERION = "kelly_criterion"
    RISK_PARITY = "risk_parity"
    VOLATILITY_TARGET = "volatility_target"


class BacktestMetric(str, Enum):
    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"


class SignalConfig(BaseModel):
    """Configuration for a signal source"""
    method: SignalMethod
    weight: float = Field(..., ge=0, le=1)
    threshold: Optional[float] = Field(None, ge=0, le=1)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class RiskManagementConfig(BaseModel):
    """Risk management configuration"""
    max_position_size: float = Field(0.1, ge=0.01, le=1.0)
    stop_loss: Optional[float] = Field(None, ge=0.001, le=0.5)
    take_profit: Optional[float] = Field(None, ge=0.001, le=2.0)
    max_drawdown: float = Field(0.15, ge=0.05, le=0.5)
    var_limit: Optional[float] = Field(None, ge=0.001, le=0.1)
    daily_loss_limit: Optional[float] = Field(None, ge=0.001, le=0.1)


class ExecutionRules(BaseModel):
    """Strategy execution rules"""
    entry_conditions: List[str] = Field(default_factory=list)
    exit_conditions: List[str] = Field(default_factory=list)
    position_sizing: PositionSizing = PositionSizing.FIXED
    rebalance_frequency: Literal["intraday", "daily", "weekly", "monthly"] = "daily"
    min_holding_period: Optional[int] = Field(None, ge=1, description="Minimum holding period in periods")
    max_holding_period: Optional[int] = Field(None, ge=1, description="Maximum holding period in periods")


class StrategyConfig(BaseModel):
    """Complete strategy configuration"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    strategy_type: StrategyType
    symbols: List[str] = Field(..., min_items=1)
    signals: Dict[str, SignalConfig] = Field(..., min_items=1)
    risk_management: RiskManagementConfig
    execution_rules: ExecutionRules
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BacktestConfig(BaseModel):
    """Backtesting configuration"""
    start_date: datetime
    end_date: datetime
    initial_capital: float = Field(100000, ge=1000)
    benchmark: str = "SPY"
    transaction_costs: float = Field(0.001, ge=0, le=0.01)
    slippage: float = Field(0.0005, ge=0, le=0.01)
    commission: float = Field(0.0, ge=0, le=100)
    rebalance_frequency: Literal["daily", "weekly", "monthly"] = "daily"
    
    # Advanced settings
    margin_requirement: float = Field(1.0, ge=0.1, le=1.0)
    short_selling_allowed: bool = False
    max_leverage: float = Field(1.0, ge=0.1, le=10.0)


class PerformanceMetrics(BaseModel):
    """Strategy performance metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    var_99: float
    
    # Trading metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Additional metrics
    beta: Optional[float] = None
    alpha: Optional[float] = None
    information_ratio: Optional[float] = None
    tracking_error: Optional[float] = None


class BacktestResult(BaseModel):
    """Results from strategy backtesting"""
    strategy_id: str
    config: BacktestConfig
    performance_metrics: PerformanceMetrics
    returns_series: List[Dict[str, Any]]  # Date -> return data
    positions: List[Dict[str, Any]]  # Position history
    trades: List[Dict[str, Any]]  # Trade history
    benchmark_comparison: Dict[str, Any]
    
    # Risk analysis
    rolling_metrics: Dict[str, List[float]]
    drawdown_periods: List[Dict[str, Any]]
    var_history: List[Dict[str, Any]]
    
    # Execution details
    execution_stats: Dict[str, Any]
    slippage_impact: Dict[str, Any]
    transaction_costs: Dict[str, Any]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StrategyStatus(BaseModel):
    """Strategy execution status"""
    strategy_id: str
    name: str
    status: Literal["active", "paused", "stopped", "error"]
    created_at: datetime
    last_updated: datetime
    
    # Current state
    current_positions: Dict[str, float]
    current_pnl: float
    current_drawdown: float
    
    # Performance tracking
    total_trades_today: int
    total_pnl_today: float
    risk_metrics: Dict[str, float]
    
    # Signals
    latest_signals: Dict[str, Any]
    signal_strength: Dict[str, float]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Strategy(BaseModel):
    """Complete strategy definition"""
    id: str
    config: StrategyConfig
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None
    
    # Backtesting results
    backtest_results: Optional[BacktestResult] = None
    
    # Live trading status
    live_status: Optional[StrategyStatus] = None
    
    # Version control
    version: int = 1
    parent_strategy_id: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StrategyOptimizationConfig(BaseModel):
    """Configuration for strategy optimization"""
    optimization_target: BacktestMetric = BacktestMetric.SHARPE_RATIO
    parameter_ranges: Dict[str, Dict[str, Any]]  # parameter -> {min, max, step}
    max_evaluations: int = Field(100, ge=10, le=1000)
    cross_validation_folds: int = Field(5, ge=2, le=10)
    optimization_method: Literal["grid_search", "random_search", "bayesian"] = "bayesian"
    
    # Constraints
    min_trades: Optional[int] = Field(None, ge=1)
    max_drawdown_limit: Optional[float] = Field(None, ge=0.01, le=0.5)
    min_sharpe_ratio: Optional[float] = Field(None, ge=0)


class StrategyOptimizationResult(BaseModel):
    """Results from strategy optimization"""
    optimization_id: str
    strategy_id: str
    config: StrategyOptimizationConfig
    
    # Best results
    best_parameters: Dict[str, Any]
    best_score: float
    best_metrics: PerformanceMetrics
    
    # All evaluated parameters and scores
    evaluation_history: List[Dict[str, Any]]
    
    # Optimization metadata
    total_evaluations: int
    optimization_time: float
    convergence_info: Dict[str, Any]
    
    completed_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }