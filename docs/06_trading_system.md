# Trading System Documentation

## Overview

The trading system provides institutional-grade order management, execution, and risk control capabilities. It supports multiple brokers, sophisticated order types, and comprehensive risk management.

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Trading System                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Strategy   │  │    Risk     │  │   Order     │    │
│  │   Signals    │→ │  Validation │→ │  Creation   │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
│                                            ↓            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Position   │← │  Execution  │← │   Order     │    │
│  │  Update     │  │   Engine    │  │  Router     │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
│                                            ↓            │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Broker Connections                  │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐           │   │
│  │  │  IBKR  │  │ Alpaca │  │ Paper  │           │   │
│  │  └────────┘  └────────┘  └────────┘           │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Order Management

### Order Types

#### Market Orders
```python
{
    "symbol": "AAPL",
    "side": "buy",
    "quantity": 100,
    "order_type": "market",
    "time_in_force": "day"
}
```

#### Limit Orders
```python
{
    "symbol": "AAPL",
    "side": "buy",
    "quantity": 100,
    "order_type": "limit",
    "price": 175.00,
    "time_in_force": "gtc"
}
```

#### Stop Orders
```python
{
    "symbol": "AAPL",
    "side": "sell",
    "quantity": 100,
    "order_type": "stop",
    "stop_price": 170.00,
    "time_in_force": "day"
}
```

#### Stop-Limit Orders
```python
{
    "symbol": "AAPL",
    "side": "sell",
    "quantity": 100,
    "order_type": "stop_limit",
    "stop_price": 170.00,
    "limit_price": 169.50,
    "time_in_force": "day"
}
```

#### Bracket Orders
```python
{
    "symbol": "AAPL",
    "side": "buy",
    "quantity": 100,
    "order_type": "market",
    "bracket": {
        "take_profit": 180.00,
        "stop_loss": 170.00
    }
}
```

### Time in Force Options

- **DAY**: Order expires at market close
- **GTC**: Good Till Cancelled
- **IOC**: Immediate or Cancel
- **FOK**: Fill or Kill
- **GTD**: Good Till Date
- **OPG**: At the Open
- **CLS**: At the Close

### Order Lifecycle

```
Created → Pending → Submitted → Acknowledged → Working
   ↓         ↓          ↓            ↓           ↓
Rejected  Cancelled  Rejected    Cancelled   Filled/Partial
```

## Execution Algorithms

### TWAP (Time-Weighted Average Price)
```python
{
    "algorithm": "twap",
    "parameters": {
        "duration_minutes": 30,
        "slice_size": 100,
        "randomize": true
    }
}
```

### VWAP (Volume-Weighted Average Price)
```python
{
    "algorithm": "vwap",
    "parameters": {
        "participation_rate": 0.1,
        "max_participation": 0.25,
        "urgency": "medium"
    }
}
```

### Iceberg Orders
```python
{
    "algorithm": "iceberg",
    "parameters": {
        "display_quantity": 100,
        "total_quantity": 1000,
        "variance": 0.2
    }
}
```

## Risk Management

### Pre-Trade Risk Checks

#### Position Limits
```python
risk_config = {
    "max_position_size": 10000,
    "max_position_value": 100000,
    "max_portfolio_concentration": 0.20
}
```

#### Order Validation
```python
def validate_order(order: OrderRequest) -> ValidationResult:
    checks = [
        check_position_limits(order),
        check_buying_power(order),
        check_daily_loss_limit(order),
        check_symbol_restrictions(order),
        check_order_size_limits(order)
    ]
    
    return ValidationResult(
        approved=all(check.passed for check in checks),
        failed_checks=[check for check in checks if not check.passed]
    )
```

### Real-Time Risk Monitoring

#### Portfolio Risk Metrics
```python
{
    "portfolio_var_95": 5234.50,
    "portfolio_var_99": 7890.25,
    "expected_shortfall": 6500.00,
    "beta": 1.15,
    "correlation_risk": 0.82,
    "concentration_risk": {
        "AAPL": 0.25,
        "sector_technology": 0.45
    }
}
```

#### Position Risk
```python
{
    "symbol": "AAPL",
    "position_var": 1234.50,
    "max_loss_scenario": -2345.00,
    "stress_test_results": {
        "market_crash_10%": -17500.00,
        "volatility_spike": -3450.00
    }
}
```

### Circuit Breakers

#### Daily Loss Limit
```python
circuit_breaker = DailyLossCircuitBreaker(
    limit=-5000,
    action="halt_all_trading",
    reset_time="00:00 UTC"
)
```

#### Order Frequency Limit
```python
circuit_breaker = OrderFrequencyCircuitBreaker(
    max_orders_per_minute=10,
    max_orders_per_hour=200,
    action="reject_new_orders"
)
```

#### Position Concentration
```python
circuit_breaker = ConcentrationCircuitBreaker(
    max_single_position=0.25,
    max_sector_exposure=0.40,
    action="block_increases"
)
```

## Broker Integration

### Interactive Brokers (IBKR)

#### Connection Setup
```python
ibkr_config = {
    "gateway_url": "https://localhost:5000/v1/api",
    "username": os.getenv("IBKR_USERNAME"),
    "password": os.getenv("IBKR_PASSWORD"),
    "account_id": os.getenv("IBKR_ACCOUNT_ID"),
    "trading_mode": "live"  # or "paper"
}
```

#### Order Placement
```python
async def place_ibkr_order(order: OrderRequest) -> Order:
    # Convert to IBKR format
    ibkr_order = {
        "conid": await get_contract_id(order.symbol),
        "orderType": order.order_type.upper(),
        "side": order.side.upper(),
        "quantity": order.quantity,
        "tif": order.time_in_force.upper()
    }
    
    if order.order_type == "limit":
        ibkr_order["price"] = order.price
        
    response = await ibkr_client.place_order(ibkr_order)
    return convert_ibkr_response(response)
```

### IQFeed Market Data

#### Real-Time Quotes
```python
class IQFeedDataProvider:
    async def connect(self):
        self.quote_conn = QuoteConn()
        self.quote_conn.connect()
        
    async def subscribe_symbol(self, symbol: str):
        self.quote_conn.watch(symbol)
        
    async def get_quote(self, symbol: str) -> Quote:
        data = self.quote_conn.get_quote(symbol)
        return Quote(
            symbol=symbol,
            bid=data['bid'],
            ask=data['ask'],
            last=data['last'],
            volume=data['volume']
        )
```

#### Historical Data
```python
async def get_historical_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    interval: str = "1d"
) -> pd.DataFrame:
    hist_conn = HistoryConn()
    hist_conn.connect()
    
    if interval == "1d":
        data = hist_conn.request_daily_data_for_dates(
            symbol,
            start_date,
            end_date
        )
    else:
        data = hist_conn.request_intraday_data(
            symbol,
            interval,
            start_date,
            end_date
        )
    
    return pd.DataFrame(data)
```

## Position Management

### Position Tracking
```python
class Position:
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    
    @property
    def pnl_percent(self) -> float:
        return (self.unrealized_pnl / (self.avg_cost * abs(self.quantity))) * 100
```

### Portfolio Analytics
```python
class PortfolioAnalytics:
    def calculate_metrics(self, positions: List[Position]) -> Dict:
        return {
            "total_value": sum(p.market_value for p in positions),
            "total_pnl": sum(p.unrealized_pnl + p.realized_pnl for p in positions),
            "win_rate": self._calculate_win_rate(positions),
            "average_win": self._calculate_avg_win(positions),
            "average_loss": self._calculate_avg_loss(positions),
            "profit_factor": self._calculate_profit_factor(positions),
            "sharpe_ratio": self._calculate_sharpe_ratio(positions),
            "max_drawdown": self._calculate_max_drawdown(positions)
        }
```

## Performance Tracking

### Trade Analytics
```python
{
    "trade_id": "123456",
    "symbol": "AAPL",
    "entry_price": 175.00,
    "exit_price": 178.50,
    "quantity": 100,
    "pnl": 350.00,
    "pnl_percent": 2.0,
    "holding_period_days": 5,
    "commissions": 2.00,
    "slippage": 0.05
}
```

### Daily Performance
```python
{
    "date": "2024-01-25",
    "starting_balance": 100000.00,
    "ending_balance": 101234.50,
    "daily_pnl": 1234.50,
    "daily_return": 0.0123,
    "trades_count": 15,
    "winning_trades": 10,
    "losing_trades": 5,
    "commissions_paid": 30.00
}
```

## Advanced Features

### Multi-Account Support
```python
accounts = {
    "main": {
        "broker": "ibkr",
        "account_id": "U1234567",
        "allocation": 0.7
    },
    "secondary": {
        "broker": "alpaca",
        "account_id": "PA123456",
        "allocation": 0.3
    }
}
```

### Order Aggregation
```python
async def execute_aggregated_order(
    symbol: str,
    total_quantity: int,
    accounts: Dict[str, float]
) -> List[Order]:
    orders = []
    
    for account_id, allocation in accounts.items():
        account_quantity = int(total_quantity * allocation)
        order = await place_order(
            account_id=account_id,
            symbol=symbol,
            quantity=account_quantity
        )
        orders.append(order)
    
    return orders
```

### Smart Order Routing
```python
class SmartOrderRouter:
    def route_order(self, order: OrderRequest) -> BrokerSelection:
        # Analyze current market conditions
        market_data = self.get_market_data(order.symbol)
        
        # Check broker capabilities
        broker_scores = {}
        for broker in self.brokers:
            score = self.calculate_broker_score(
                broker,
                order,
                market_data
            )
            broker_scores[broker] = score
        
        # Select best broker
        best_broker = max(broker_scores, key=broker_scores.get)
        return BrokerSelection(
            broker=best_broker,
            reason=f"Best execution score: {broker_scores[best_broker]}"
        )
```

## Configuration

### Trading Configuration
```yaml
trading:
  default_broker: ibkr
  default_data_provider: iqfeed
  
  order_defaults:
    time_in_force: day
    extended_hours: false
    
  risk_management:
    pre_trade_checks: true
    position_limits:
      max_position_size: 10000
      max_position_value: 100000
      max_concentration: 0.20
    
    circuit_breakers:
      daily_loss_limit: -5000
      max_orders_per_minute: 10
      
  execution:
    smart_routing: true
    default_algorithm: twap
    slippage_model: linear
    
  monitoring:
    log_all_orders: true
    alert_on_rejects: true
    performance_tracking: true
```

## Error Handling

### Common Errors

#### Insufficient Buying Power
```python
{
    "error": "INSUFFICIENT_BUYING_POWER",
    "message": "Order requires $17,500 but only $15,000 available",
    "details": {
        "required": 17500,
        "available": 15000,
        "symbol": "AAPL",
        "quantity": 100
    }
}
```

#### Position Limit Exceeded
```python
{
    "error": "POSITION_LIMIT_EXCEEDED",
    "message": "Order would exceed maximum position size",
    "details": {
        "current_position": 9500,
        "order_quantity": 1000,
        "max_allowed": 10000
    }
}
```

#### Risk Check Failed
```python
{
    "error": "RISK_CHECK_FAILED",
    "message": "Order rejected by risk management",
    "failed_checks": [
        "concentration_limit",
        "daily_loss_approaching"
    ]
}
```

## Best Practices

### Order Management
1. Always use stop-loss orders
2. Implement position sizing rules
3. Use limit orders for better price control
4. Monitor partial fills
5. Handle connection failures gracefully

### Risk Management
1. Set appropriate position limits
2. Use circuit breakers
3. Monitor correlation risk
4. Implement drawdown controls
5. Regular risk audits

### Performance Optimization
1. Use appropriate order types
2. Consider market impact
3. Optimize execution timing
4. Monitor slippage
5. Track all costs

## Troubleshooting

### Connection Issues
```bash
# Check broker connection
curl -X GET https://localhost:5000/v1/api/iserver/auth/status

# Test market data feed
telnet localhost 5009

# Verify credentials
python scripts/test_broker_connection.py
```

### Order Failures
1. Check error logs
2. Verify market hours
3. Confirm symbol validity
4. Check account permissions
5. Review risk limits

### Performance Issues
1. Monitor API latency
2. Check network connectivity
3. Optimize order batching
4. Review execution algorithms
5. Analyze slippage patterns