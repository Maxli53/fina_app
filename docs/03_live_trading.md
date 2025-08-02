# Phase 3: Live Trading Integration

## Overview

Phase 3 implements comprehensive live trading capabilities, integrating with Interactive Brokers (IBKR) for order execution and IQFeed for professional market data. This phase provides a complete trading infrastructure with risk management, real-time monitoring, and fail-safe mechanisms.

## Core Components

### 1. Broker Integration (IBKR Client Portal API)

#### 1.1 Connection Management
- **Service**: `IBKRService` - Manages IBKR Client Portal API connections
- **Authentication**: Secure OAuth-based authentication with session management
- **Features**:
  - Automatic session keep-alive
  - Connection health monitoring
  - SSL certificate handling for gateway
  - Multi-account support

#### 1.2 Order Execution
- **Order Types**: Market, Limit, Stop, Stop-Limit, MOC, LOC, Trailing Stop
- **Time in Force**: DAY, GTC, IOC, FOK, GTD, OPG, CLO
- **Smart Routing**: IBKR's SMART algorithm for best execution
- **Execution Algorithms**:
  - VWAP (Volume Weighted Average Price)
  - TWAP (Time Weighted Average Price)
  - Adaptive algorithms
  - Iceberg orders

### 2. Market Data Integration (IQFeed)

#### 2.1 Real-time Data Streaming
- **Service**: `IQFeedService` - Professional market data feed
- **Data Types**:
  - Level 1 quotes (bid/ask/last)
  - Level 2 market depth
  - Time & Sales
  - News feeds
- **Performance**: 
  - Sub-millisecond latency
  - Up to 500 symbols concurrent streaming
  - Automatic reconnection handling

#### 2.2 Historical Data
- **Data Types**:
  - Daily bars
  - Intraday intervals (1-second to daily)
  - Tick data
- **Features**:
  - Date range queries
  - RTH/ETH filtering
  - Efficient data compression

### 3. Order Management System

#### 3.1 Centralized Order Manager
- **Service**: `OrderManager` - Core order execution and tracking
- **Features**:
  - Order lifecycle management
  - Risk validation before execution
  - Position tracking by strategy
  - Fill monitoring and reconciliation

#### 3.2 Risk Controls
- **Pre-trade Checks**:
  - Position size limits
  - Daily loss limits
  - Concentration limits
  - Margin requirements
- **Real-time Monitoring**:
  - P&L tracking
  - Drawdown monitoring
  - VaR calculations

### 4. Circuit Breakers & Fail-safes

#### 4.1 Built-in Circuit Breakers
1. **Daily Loss Limit**
   - Threshold: 5% daily loss
   - Action: Halt all trading
   - Reset: Manual or next trading day

2. **Order Rejection Rate**
   - Threshold: 20% rejection rate
   - Action: Reduce position sizes by 50%
   - Reset: After 30 minutes

3. **Position Concentration**
   - Threshold: 30% in single position
   - Action: Block new positions in symbol
   - Reset: When below 25%

#### 4.2 Monitoring & Alerts
- **Alert Types**:
  - Risk limit breaches
  - Connection failures
  - Execution errors
  - Circuit breaker trips
- **Delivery**: Real-time via API and WebSocket

## API Endpoints

### Trading Session Management
```
POST   /api/trading/session/start    - Start trading session
POST   /api/trading/session/end      - End trading session
GET    /api/trading/session/status   - Get session status
```

### Order Management
```
POST   /api/trading/orders/place     - Place new order
DELETE /api/trading/orders/{id}      - Cancel order
PUT    /api/trading/orders/{id}      - Modify order
GET    /api/trading/orders/{id}      - Get order status
GET    /api/trading/orders            - List orders
```

### Portfolio & Positions
```
GET    /api/trading/positions        - Get current positions
GET    /api/trading/portfolio        - Get portfolio snapshot
```

### Market Data
```
GET    /api/trading/market-data/{symbol}     - Get real-time quote
POST   /api/trading/market-data/subscribe    - Subscribe to symbols
DELETE /api/trading/market-data/subscribe/{symbol} - Unsubscribe
WS     /api/trading/ws/market-data           - WebSocket streaming
```

### Risk & Monitoring
```
GET    /api/trading/circuit-breakers  - Get circuit breaker status
POST   /api/trading/circuit-breakers/{id}/reset - Reset breaker
GET    /api/trading/statistics        - Get trading statistics
GET    /api/trading/alerts           - Get recent alerts
```

### Trading Controls
```
POST   /api/trading/trading/enable   - Enable trading
POST   /api/trading/trading/disable  - Disable trading
```

## Configuration

### Environment Variables
```bash
# IBKR Configuration
IBKR_USERNAME=your_username
IBKR_PASSWORD=your_password
IBKR_ACCOUNT_ID=your_account_id

# IQFeed Configuration  
IQFEED_LOGIN=your_login
IQFEED_PASSWORD=your_password
IQFEED_PRODUCT_ID=FINANCIAL_TIME_SERIES_PLATFORM
IQFEED_PRODUCT_VERSION=1.0
```

### Risk Limits (Configurable)
```python
{
    "max_daily_loss": 5000.0,      # Maximum daily loss in USD
    "max_position_value": 100000.0,  # Maximum position size
    "max_order_size": 10000.0,       # Maximum single order size
    "allowed_symbols": [],           # Empty = all allowed
    "blocked_symbols": []            # Symbols to block
}
```

## Usage Examples

### Starting a Trading Session
```python
# Start paper trading session
response = requests.post("/api/trading/session/start", json={
    "mode": "paper",
    "strategies": ["strategy_001", "strategy_002"]
})
```

### Placing an Order
```python
# Place a limit order
order_request = {
    "symbol": "AAPL",
    "quantity": 100,
    "side": "buy",
    "order_type": "limit",
    "limit_price": 150.00,
    "time_in_force": "day",
    "strategy_id": "strategy_001",
    "execution_algorithm": "vwap"
}

response = requests.post("/api/trading/orders/place", json=order_request)
```

### Monitoring Positions
```python
# Get current positions
positions = requests.get("/api/trading/positions").json()

# Get portfolio snapshot
portfolio = requests.get("/api/trading/portfolio").json()
```

### WebSocket Market Data
```javascript
const ws = new WebSocket('ws://localhost:8000/api/trading/ws/market-data');

// Subscribe to symbols
ws.send(JSON.stringify({
    action: 'subscribe',
    symbols: ['AAPL', 'GOOGL', 'MSFT']
}));

// Receive real-time updates
ws.onmessage = (event) => {
    const marketData = JSON.parse(event.data);
    console.log('Market update:', marketData);
};
```

## Performance Metrics

### Latency Targets
- Order placement: < 50ms
- Order cancellation: < 30ms
- Market data updates: < 10ms
- Position updates: < 100ms

### Throughput
- Orders per second: 10+
- Market data updates: 1000+ per second
- Concurrent strategies: 50+

### Reliability
- Uptime target: 99.9%
- Automatic reconnection
- Order persistence
- State recovery

## Security Considerations

### Authentication
- IBKR Client Portal OAuth
- IQFeed credential encryption
- Session token management
- API key rotation

### Network Security
- SSL/TLS encryption
- Certificate validation
- Firewall rules
- IP whitelisting

### Risk Controls
- Position limits enforced
- Daily loss limits
- Order rate limiting
- Emergency shutdown

## Deployment

### Prerequisites
1. IBKR account with API access enabled
2. IQFeed subscription (market data)
3. IBKR Client Portal Gateway running
4. IQConnect service running

### Docker Deployment
```yaml
services:
  backend:
    environment:
      - IBKR_USERNAME=${IBKR_USERNAME}
      - IBKR_PASSWORD=${IBKR_PASSWORD}
      - IQFEED_LOGIN=${IQFEED_LOGIN}
      - IQFEED_PASSWORD=${IQFEED_PASSWORD}
    ports:
      - "8000:8000"
```

### Production Checklist
- [ ] Configure production risk limits
- [ ] Set up monitoring and alerting
- [ ] Test circuit breakers
- [ ] Verify order execution in paper mode
- [ ] Configure backup data sources
- [ ] Set up automated backups
- [ ] Document emergency procedures

## Troubleshooting

### Common Issues

1. **IBKR Connection Failed**
   - Verify Client Portal Gateway is running
   - Check credentials in environment
   - Ensure account has API permissions

2. **IQFeed No Data**
   - Verify IQConnect is running
   - Check subscription status
   - Ensure symbols are valid

3. **Orders Rejected**
   - Check risk limits
   - Verify account permissions
   - Review margin requirements

### Debug Commands
```bash
# Check service connections
curl http://localhost:8000/api/trading/connections

# View circuit breaker status
curl http://localhost:8000/api/trading/circuit-breakers

# Get trading statistics
curl http://localhost:8000/api/trading/statistics
```

## Next Steps

### Phase 4: Advanced Features (Future)
- Multi-broker support (Alpaca, TD Ameritrade)
- Advanced execution algorithms
- Trade analytics and reporting
- Mobile app integration
- Cloud deployment with auto-scaling

### Enhancements
- Machine learning for execution optimization
- Sentiment-based trading signals
- Options trading support
- Crypto integration
- FIX protocol support