"""Trading models for live trading integration"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
import uuid


class BrokerType(str, Enum):
    """Supported broker types"""
    IBKR = "ibkr"
    IBKR_CP = "ibkr_cp"  # IBKR Client Portal API
    ALPACA = "alpaca"
    TD_AMERITRADE = "td_ameritrade"
    PAPER = "paper"  # Paper trading for testing


class DataProviderType(str, Enum):
    """Supported data providers"""
    IQFEED = "iqfeed"
    IBKR = "ibkr"
    YAHOO = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"


class ConnectionStatus(str, Enum):
    """Connection status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"
    AUTHENTICATED = "authenticated"


class OrderType(str, Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    MOC = "moc"  # Market on Close
    LOC = "loc"  # Limit on Close


class OrderSide(str, Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"


class OrderStatus(str, Enum):
    """Order execution status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(str, Enum):
    """Time in force for orders"""
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    GTD = "gtd"  # Good Till Date
    OPG = "opg"  # At the Opening
    CLO = "clo"  # At the Close


class ExecutionAlgorithm(str, Enum):
    """Execution algorithms"""
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"
    TWAP = "twap"  # Time Weighted Average Price
    VWAP = "vwap"  # Volume Weighted Average Price
    POV = "pov"    # Percentage of Volume
    IS = "is"      # Implementation Shortfall
    ICEBERG = "iceberg"
    SMART = "smart"  # Broker's smart routing


class IQFeedDataType(str, Enum):
    """IQFeed data types"""
    LEVEL1 = "level1"
    LEVEL2 = "level2"
    DERIVATIVE = "derivative"
    ADMIN = "admin"
    NEWS = "news"
    HISTORICAL_TICK = "historical_tick"
    HISTORICAL_INTERVAL = "historical_interval"
    HISTORICAL_DAILY = "historical_daily"


class IBKRClientPortalConfig(BaseModel):
    """IBKR Client Portal API configuration"""
    broker_type: BrokerType = BrokerType.IBKR_CP
    gateway_url: str = "https://localhost:5000/v1/api"
    
    # Authentication
    username: str
    account_id: str
    
    # SSL Configuration
    verify_ssl: bool = False  # Gateway uses self-signed cert
    cert_path: Optional[str] = None
    
    # Connection parameters
    connection_timeout: int = 30
    request_timeout: int = 10
    max_retries: int = 3
    
    # Session management
    reauthenticate_interval: int = 3600  # 1 hour
    keep_alive_interval: int = 60  # 1 minute
    
    # Risk limits
    max_daily_loss: float = 5000.0
    max_position_value: float = 100000.0
    max_order_size: float = 10000.0
    allowed_symbols: List[str] = Field(default_factory=list)
    blocked_symbols: List[str] = Field(default_factory=list)


class IQFeedConfig(BaseModel):
    """IQFeed data provider configuration"""
    provider_type: DataProviderType = DataProviderType.IQFEED
    host: str = "127.0.0.1"
    
    # Connection ports
    level1_port: int = 5009
    level2_port: int = 9200
    admin_port: int = 9300
    historical_port: int = 9100
    
    # Credentials
    login: str
    password: str
    product_id: str
    product_version: str = "1.0"
    
    # Connection settings
    connection_timeout: int = 30
    reconnect_attempts: int = 3
    heartbeat_interval: int = 5
    
    # Data settings
    symbols_to_watch: List[str] = Field(default_factory=list)
    request_updates: bool = True
    update_interval_ms: int = 100
    
    # Performance settings
    max_symbols: int = 500
    buffer_size: int = 65536
    use_compression: bool = False


class DataProviderConnection(BaseModel):
    """Data provider connection status"""
    provider_type: DataProviderType
    status: ConnectionStatus
    connected_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Connection metrics
    latency_ms: Optional[float] = None
    messages_received: int = 0
    data_points_received: int = 0
    
    # Service status (for IQFeed)
    level1_connected: bool = False
    level2_connected: bool = False
    historical_connected: bool = False
    admin_connected: bool = False


class BrokerConnection(BaseModel):
    """Broker connection status and info"""
    broker_type: BrokerType
    status: ConnectionStatus
    connected_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    account_info: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    
    # Account details
    account_id: Optional[str] = None
    account_type: Optional[str] = None  # cash, margin, portfolio
    base_currency: str = "USD"
    
    # Connection metrics
    latency_ms: Optional[float] = None
    messages_sent: int = 0
    messages_received: int = 0
    
    # IBKR specific
    session_token: Optional[str] = None
    is_authenticated: bool = False


class OrderRequest(BaseModel):
    """Order placement request"""
    symbol: str
    quantity: float
    side: OrderSide
    order_type: OrderType
    time_in_force: TimeInForce = TimeInForce.DAY
    
    # Price specifications
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_amount: Optional[float] = None
    trail_percent: Optional[float] = None
    
    # Execution parameters
    execution_algorithm: Optional[ExecutionAlgorithm] = None
    algorithm_params: Dict[str, Any] = Field(default_factory=dict)
    
    # Order metadata
    strategy_id: str
    signal_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    notes: Optional[str] = None
    
    # Risk controls
    max_slippage_percent: float = 0.01  # 1%
    reject_if_late: bool = False
    check_margin: bool = True
    
    # IBKR specific
    conid: Optional[int] = None  # Contract ID for IBKR
    sec_type: str = "STK"  # Security type
    exchange: str = "SMART"  # Exchange routing
    
    @validator('quantity')
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError("Quantity must be positive")
        return v
    
    @validator('limit_price')
    def validate_limit_price(cls, v, values):
        if v is not None and v <= 0:
            raise ValueError("Limit price must be positive")
        if values.get('order_type') == OrderType.LIMIT and v is None:
            raise ValueError("Limit price required for limit orders")
        return v


class Order(BaseModel):
    """Order information"""
    order_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    broker_order_id: Optional[str] = None
    
    # Order details from request
    symbol: str
    quantity: float
    filled_quantity: float = 0
    side: OrderSide
    order_type: OrderType
    status: OrderStatus = OrderStatus.PENDING
    
    # Prices
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    avg_fill_price: Optional[float] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    last_update: datetime = Field(default_factory=datetime.utcnow)
    
    # Execution details
    commission: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0
    total_cost: float = 0.0
    
    # Metadata
    strategy_id: str
    execution_algorithm: Optional[ExecutionAlgorithm] = None
    error_message: Optional[str] = None
    
    # IBKR specific
    conid: Optional[int] = None
    parent_id: Optional[str] = None
    
    def is_active(self) -> bool:
        """Check if order is still active"""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]
    
    def is_complete(self) -> bool:
        """Check if order is complete"""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]


class Fill(BaseModel):
    """Trade execution fill"""
    fill_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str
    broker_fill_id: Optional[str] = None
    
    # Execution details
    symbol: str
    quantity: float
    price: float
    side: OrderSide
    
    # Costs
    commission: float = 0.0
    exchange_fee: float = 0.0
    other_fees: float = 0.0
    
    # Timestamps
    executed_at: datetime
    reported_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Venue information
    exchange: Optional[str] = None
    liquidity_type: Optional[str] = None  # add, remove, passive, aggressive


class Position(BaseModel):
    """Live position information"""
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    
    # P&L
    unrealized_pnl: float
    realized_pnl: float = 0.0
    total_pnl: float
    pnl_percent: float
    
    # Position details
    side: str  # long or short
    opened_at: datetime
    last_update: datetime = Field(default_factory=datetime.utcnow)
    
    # Risk metrics
    position_risk: float = 0.0
    var_contribution: float = 0.0
    beta_adjusted_exposure: float = 0.0
    
    # Associated orders
    open_orders: List[str] = Field(default_factory=list)
    
    # IBKR specific
    conid: Optional[int] = None
    account_id: Optional[str] = None
    
    def is_profitable(self) -> bool:
        """Check if position is profitable"""
        return self.unrealized_pnl > 0


class PortfolioSnapshot(BaseModel):
    """Portfolio snapshot at a point in time"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    account_id: str
    
    # Values
    total_value: float
    cash_balance: float
    securities_value: float
    
    # P&L
    daily_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    
    # Positions
    positions: List[Position]
    position_count: int
    
    # Risk metrics
    portfolio_var: float = 0.0
    portfolio_beta: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    leverage: float = 1.0
    
    # Margin (if applicable)
    buying_power: float = 0.0
    margin_used: float = 0.0
    margin_available: float = 0.0
    
    # Performance
    daily_return: float = 0.0
    mtd_return: float = 0.0
    ytd_return: float = 0.0


class IQFeedMarketData(BaseModel):
    """IQFeed real-time market data"""
    symbol: str
    timestamp: datetime
    
    # Level 1 data
    last: float
    last_size: int
    total_volume: int
    bid: float
    bid_size: int
    ask: float
    ask_size: int
    
    # Extended quote data
    open: float
    high: float
    low: float
    close: float
    prev_close: float
    
    # Change data
    net_change: float
    percent_change: float
    
    # Trading info
    vwap: float
    open_interest: Optional[int] = None
    
    # Market state
    tick_direction: str  # up, down, unchanged
    exchange: str
    delay_minutes: int = 0
    market_open: bool = True
    
    # Additional IQFeed fields
    bid_tick: Optional[str] = None
    range_high: Optional[float] = None
    range_low: Optional[float] = None
    
    
class IQFeedHistoricalRequest(BaseModel):
    """IQFeed historical data request"""
    symbol: str
    data_type: IQFeedDataType
    
    # Time parameters
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    num_days: Optional[int] = None
    
    # Data parameters
    interval_seconds: Optional[int] = None  # For interval data
    max_datapoints: Optional[int] = None
    beginning_filter_time: Optional[str] = None  # HHmmSS
    ending_filter_time: Optional[str] = None  # HHmmSS
    
    # Options
    include_extended_hours: bool = False
    only_rth: bool = True  # Regular Trading Hours
    
    
class MarketDataSnapshot(BaseModel):
    """Unified market data snapshot"""
    symbol: str
    timestamp: datetime
    source: DataProviderType
    
    # Price data
    bid: float
    ask: float
    last: float
    mid: float
    
    # Size
    bid_size: int
    ask_size: int
    last_size: int
    
    # Volume
    volume: int
    vwap: float
    
    # Change
    change: float
    change_percent: float
    
    # Trading info
    open: float
    high: float
    low: float
    close: float
    prev_close: float
    
    # Market state
    halted: bool = False
    trading_status: str = "normal"


class ExecutionReport(BaseModel):
    """Execution quality report"""
    order_id: str
    symbol: str
    side: OrderSide
    
    # Execution metrics
    intended_quantity: float
    filled_quantity: float
    fill_rate: float
    
    # Price analysis
    arrival_price: float  # Price when order was placed
    avg_execution_price: float
    benchmark_price: float  # VWAP or other benchmark
    
    # Cost analysis
    implementation_shortfall: float
    market_impact: float
    timing_cost: float
    spread_cost: float
    total_cost: float
    
    # Execution details
    execution_time_seconds: float
    number_of_fills: int
    venues_used: List[str]
    
    # Quality scores
    execution_score: float  # 0-100
    price_improvement: float
    speed_score: float
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Alert(BaseModel):
    """Trading alert"""
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    alert_type: str  # risk_limit, execution_error, connection_lost, etc.
    severity: str  # info, warning, critical
    
    title: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    
    # Context
    strategy_id: Optional[str] = None
    symbol: Optional[str] = None
    order_id: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    
    # Actions
    suggested_actions: List[str] = Field(default_factory=list)
    auto_resolved: bool = False


class CircuitBreakerStatus(BaseModel):
    """Circuit breaker status"""
    breaker_id: str
    name: str
    status: str  # active, tripped, disabled
    
    # Configuration
    threshold: float
    current_value: float
    threshold_type: str  # loss, drawdown, volume, error_rate
    
    # State
    tripped_at: Optional[datetime] = None
    reset_at: Optional[datetime] = None
    trip_count: int = 0
    
    # Actions when tripped
    actions: List[str]  # halt_trading, reduce_size, alert_only
    affected_strategies: List[str] = Field(default_factory=list)
    
    def is_tripped(self) -> bool:
        """Check if circuit breaker is tripped"""
        return self.status == "tripped"


class TradingSession(BaseModel):
    """Trading session information"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Configuration
    broker_config: Optional[Union[IBKRClientPortalConfig, Dict[str, Any]]] = None
    data_provider_config: Optional[IQFeedConfig] = None
    active_strategies: List[str]
    mode: str = "live"  # live, paper, backtest
    
    # Session stats
    orders_placed: int = 0
    orders_filled: int = 0
    total_volume: float = 0.0
    total_commission: float = 0.0
    
    # P&L
    starting_balance: float
    current_balance: float = 0.0
    session_pnl: float = 0.0
    session_return: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    risk_events: int = 0
    circuit_breakers_tripped: int = 0
    
    # Status
    is_active: bool = True
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)