# API Reference Documentation

## Base URL
```
http://localhost:8000/api
```

## Authentication
Currently, the API is open for development. Production deployment will require JWT authentication.

## Response Format
All responses follow a consistent JSON format:
```json
{
    "status": "success|error",
    "data": {},
    "message": "Optional message",
    "timestamp": "2025-08-01T12:00:00Z"
}
```

---

## Health Check Endpoints

### GET /health
Check system health and component status.

**Response:**
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "components": {
        "database": "connected",
        "redis": "connected",
        "idtxl": "ready",
        "ml_service": "ready",
        "nn_service": "ready"
    }
}
```

### GET /health/gpu
Check GPU availability and status.

**Response:**
```json
{
    "gpu_available": true,
    "devices": [
        {
            "name": "NVIDIA GeForce RTX 3090",
            "memory_total": 24576,
            "memory_used": 2048
        }
    ]
}
```

---

## Data Service Endpoints

### GET /data/search
Search for financial symbols.

**Query Parameters:**
- `query` (required): Search term
- `limit`: Maximum results (default: 10)

**Example:**
```
GET /api/data/search?query=AAPL&limit=5
```

**Response:**
```json
{
    "results": [
        {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "exchange": "NASDAQ",
            "type": "equity"
        }
    ]
}
```

### GET /data/historical/{symbol}
Get historical price data for a symbol.

**Path Parameters:**
- `symbol`: Stock symbol (e.g., AAPL)

**Query Parameters:**
- `start_date`: Start date (YYYY-MM-DD)
- `end_date`: End date (YYYY-MM-DD)
- `interval`: Data interval (1d, 1h, 5m)

**Example:**
```
GET /api/data/historical/AAPL?start_date=2024-01-01&end_date=2024-12-31&interval=1d
```

**Response:**
```json
{
    "symbol": "AAPL",
    "data": [
        {
            "date": "2024-01-01",
            "open": 150.00,
            "high": 152.00,
            "low": 149.00,
            "close": 151.00,
            "volume": 50000000
        }
    ]
}
```

### GET /data/market-status
Get current market status.

**Response:**
```json
{
    "market": "US",
    "status": "open",
    "next_open": "2025-08-02T09:30:00-04:00",
    "next_close": "2025-08-01T16:00:00-04:00"
}
```

---

## Analysis Service Endpoints

### POST /analysis/idtxl
Run IDTxl information-theoretic analysis.

**Request Body:**
```json
{
    "data": {
        "AAPL": [...],
        "MSFT": [...],
        "GOOGL": [...]
    },
    "config": {
        "analysis_type": "transfer_entropy",
        "max_lag": 5,
        "estimator": "gaussian",
        "n_perm": 500
    }
}
```

**Response:**
```json
{
    "task_id": "task_123456",
    "status": "processing",
    "estimated_time": 60
}
```

### POST /analysis/ml
Run machine learning analysis.

**Request Body:**
```json
{
    "features": [[...]],
    "targets": [...],
    "config": {
        "model_type": "random_forest",
        "hyperparameter_optimization": true,
        "cross_validation_folds": 5,
        "test_size": 0.2
    }
}
```

**Response:**
```json
{
    "task_id": "task_789012",
    "status": "processing"
}
```

### POST /analysis/nn
Run neural network analysis.

**Request Body:**
```json
{
    "sequences": [[...]],
    "targets": [...],
    "config": {
        "model_type": "lstm",
        "layers": [128, 64, 32],
        "epochs": 100,
        "batch_size": 32,
        "use_gpu": true
    }
}
```

### GET /analysis/status/{task_id}
Get analysis task status.

**Response:**
```json
{
    "task_id": "task_123456",
    "status": "completed",
    "progress": 100,
    "result": {
        "analysis_type": "transfer_entropy",
        "results": {...}
    }
}
```

---

## Strategy Service Endpoints

### POST /strategy/create
Create a new trading strategy.

**Request Body:**
```json
{
    "name": "Multi-Signal Strategy",
    "description": "Combines IDTxl, ML, and NN signals",
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "signals": {
        "idtxl_signal": {
            "method": "idtxl",
            "weight": 0.3,
            "parameters": {...}
        },
        "ml_signal": {
            "method": "ml",
            "weight": 0.4,
            "parameters": {...}
        }
    },
    "risk_management": {
        "max_position_size": 0.1,
        "stop_loss": 0.02,
        "take_profit": 0.05
    }
}
```

### GET /strategy/{id}
Get strategy details.

### PUT /strategy/{id}
Update strategy configuration.

### DELETE /strategy/{id}
Delete a strategy.

### POST /strategy/{id}/backtest
Run strategy backtest.

**Request Body:**
```json
{
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "initial_capital": 100000,
    "transaction_costs": 0.001,
    "slippage": 0.0005
}
```

**Response:**
```json
{
    "task_id": "backtest_345678",
    "status": "processing"
}
```

### POST /strategy/{id}/optimize
Optimize strategy parameters.

**Request Body:**
```json
{
    "optimization_method": "bayesian",
    "metric": "sharpe_ratio",
    "n_trials": 100,
    "parameter_ranges": {
        "stop_loss": [0.01, 0.05],
        "take_profit": [0.02, 0.10]
    }
}
```

---

## Trading Service Endpoints

### POST /trading/session/start
Start a trading session.

**Request Body:**
```json
{
    "mode": "paper",
    "strategies": ["strategy_001", "strategy_002"],
    "broker_config": {
        "broker_type": "ibkr_cp",
        "username": "user123",
        "account_id": "DU123456"
    },
    "data_config": {
        "provider_type": "iqfeed",
        "symbols_to_watch": ["AAPL", "MSFT"]
    }
}
```

### POST /trading/session/end
End the current trading session.

### GET /trading/session/status
Get current session status.

### POST /trading/orders/place
Place a new order.

**Request Body:**
```json
{
    "symbol": "AAPL",
    "quantity": 100,
    "side": "buy",
    "order_type": "limit",
    "limit_price": 150.00,
    "time_in_force": "day",
    "strategy_id": "strategy_001",
    "execution_algorithm": "vwap"
}
```

**Response:**
```json
{
    "order_id": "ord_123456",
    "status": "submitted",
    "broker_order_id": "IB123456"
}
```

### DELETE /trading/orders/{order_id}
Cancel an order.

### PUT /trading/orders/{order_id}
Modify an existing order.

**Request Body:**
```json
{
    "new_quantity": 200,
    "new_limit_price": 149.50
}
```

### GET /trading/orders/{order_id}
Get order status and details.

### GET /trading/orders
List all orders.

**Query Parameters:**
- `strategy_id`: Filter by strategy
- `active_only`: Show only active orders
- `limit`: Maximum results
- `offset`: Pagination offset

### GET /trading/positions
Get current positions.

**Response:**
```json
{
    "positions": [
        {
            "symbol": "AAPL",
            "quantity": 100,
            "avg_cost": 145.50,
            "current_price": 150.00,
            "unrealized_pnl": 450.00,
            "pnl_percent": 3.09
        }
    ]
}
```

### GET /trading/portfolio
Get portfolio snapshot.

**Response:**
```json
{
    "account_id": "DU123456",
    "total_value": 105000.00,
    "cash_balance": 50000.00,
    "securities_value": 55000.00,
    "daily_pnl": 500.00,
    "unrealized_pnl": 1000.00,
    "positions": [...]
}
```

### GET /trading/market-data/{symbol}
Get real-time market data.

**Response:**
```json
{
    "symbol": "AAPL",
    "timestamp": "2025-08-01T10:30:00Z",
    "bid": 149.95,
    "ask": 150.05,
    "last": 150.00,
    "volume": 25000000,
    "change": 2.50,
    "change_percent": 1.69
}
```

### POST /trading/market-data/subscribe
Subscribe to real-time data.

**Request Body:**
```json
{
    "symbols": ["AAPL", "MSFT", "GOOGL"]
}
```

### WebSocket /trading/ws/market-data
WebSocket endpoint for streaming market data.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/trading/ws/market-data');
```

**Subscribe Message:**
```json
{
    "action": "subscribe",
    "symbols": ["AAPL", "MSFT"]
}
```

**Data Message:**
```json
{
    "type": "quote",
    "symbol": "AAPL",
    "data": {
        "bid": 149.95,
        "ask": 150.05,
        "last": 150.00,
        "timestamp": "2025-08-01T10:30:00.123Z"
    }
}
```

### GET /trading/circuit-breakers
Get circuit breaker status.

**Response:**
```json
{
    "daily_loss": {
        "status": "active",
        "threshold": 0.05,
        "current_value": 0.02,
        "tripped": false
    },
    "rejection_rate": {
        "status": "active",
        "threshold": 0.20,
        "current_value": 0.05,
        "tripped": false
    }
}
```

### POST /trading/circuit-breakers/{breaker_id}/reset
Reset a circuit breaker.

### POST /trading/trading/enable
Enable trading.

### POST /trading/trading/disable
Disable trading.

### GET /trading/statistics
Get trading statistics.

**Response:**
```json
{
    "total_orders": 150,
    "filled_orders": 145,
    "rejected_orders": 5,
    "total_volume": 50000,
    "total_commission": 150.00,
    "session": {
        "session_id": "sess_123456",
        "start_time": "2025-08-01T09:30:00Z",
        "duration_minutes": 120
    }
}
```

---

## Error Responses

### 400 Bad Request
```json
{
    "status": "error",
    "message": "Invalid request parameters",
    "errors": {
        "symbol": "Symbol is required",
        "quantity": "Quantity must be positive"
    }
}
```

### 401 Unauthorized
```json
{
    "status": "error",
    "message": "Authentication required"
}
```

### 404 Not Found
```json
{
    "status": "error",
    "message": "Resource not found"
}
```

### 500 Internal Server Error
```json
{
    "status": "error",
    "message": "Internal server error",
    "error_id": "err_123456"
}
```

---

## Rate Limiting

API rate limits (per minute):
- Data endpoints: 100 requests
- Analysis endpoints: 20 requests
- Trading endpoints: 100 requests
- WebSocket connections: 10 per IP

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1627890000
```

---

## Pagination

List endpoints support pagination:
```
GET /api/trading/orders?limit=20&offset=40
```

Response includes pagination metadata:
```json
{
    "data": [...],
    "pagination": {
        "total": 150,
        "limit": 20,
        "offset": 40,
        "has_next": true,
        "has_prev": true
    }
}
```

---

## Webhooks

Configure webhooks for event notifications:

### Events
- `order.filled`
- `order.rejected`
- `position.opened`
- `position.closed`
- `circuit_breaker.tripped`
- `analysis.completed`

### Webhook Payload
```json
{
    "event": "order.filled",
    "timestamp": "2025-08-01T10:30:00Z",
    "data": {
        "order_id": "ord_123456",
        "symbol": "AAPL",
        "quantity": 100,
        "price": 150.00
    }
}
```