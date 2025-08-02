# API Reference

## Overview

The Financial Time Series Analysis Platform API provides comprehensive endpoints for market data, analysis, trading, and portfolio management. All API endpoints are RESTful and return JSON responses.

## Base URL

```
Development: http://localhost:8000/api
Production: https://api.yourdomain.com/api
```

## Authentication

All API requests require authentication using JWT tokens.

### Obtaining a Token

```http
POST /api/auth/login
Content-Type: application/json

{
  "username": "demo",
  "password": "demo"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Using the Token

Include the token in the Authorization header:
```http
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
```

## Endpoints

### Health Check

#### System Health
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-25T10:30:00Z",
  "services": {
    "database": "healthy",
    "cache": "healthy",
    "market_data": "healthy",
    "trading": "healthy"
  }
}
```

### Market Data

#### Search Symbols
```http
GET /api/data/search?query=AAPL
```

**Parameters:**
- `query` (string, required): Search term

**Response:**
```json
{
  "results": [
    {
      "symbol": "AAPL",
      "name": "Apple Inc.",
      "exchange": "NASDAQ",
      "asset_class": "equity"
    }
  ]
}
```

#### Get Quote
```http
GET /api/data/quote/{symbol}
```

**Parameters:**
- `symbol` (string, required): Stock symbol

**Response:**
```json
{
  "symbol": "AAPL",
  "price": 175.50,
  "change": 2.35,
  "change_percent": 1.36,
  "volume": 45678900,
  "bid": 175.48,
  "ask": 175.52,
  "high": 176.80,
  "low": 173.20,
  "open": 174.00,
  "previous_close": 173.15,
  "timestamp": "2024-01-25T15:30:00Z"
}
```

#### Get Historical Data
```http
GET /api/data/historical/{symbol}?start_date=2024-01-01&end_date=2024-01-31&interval=1d
```

**Parameters:**
- `symbol` (string, required): Stock symbol
- `start_date` (string, required): Start date (YYYY-MM-DD)
- `end_date` (string, required): End date (YYYY-MM-DD)
- `interval` (string, optional): Data interval (1m, 5m, 15m, 30m, 1h, 1d)

**Response:**
```json
{
  "symbol": "AAPL",
  "interval": "1d",
  "data": [
    {
      "timestamp": "2024-01-01T00:00:00Z",
      "open": 173.00,
      "high": 175.50,
      "low": 172.80,
      "close": 174.90,
      "volume": 34567890
    }
  ]
}
```

#### Stream Market Data (WebSocket)
```javascript
// WebSocket connection
const ws = new WebSocket('ws://localhost:8000/ws');

// Subscribe to symbols
ws.send(JSON.stringify({
  "action": "subscribe",
  "symbols": ["AAPL", "MSFT", "GOOGL"]
}));

// Receive updates
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
  // {
  //   "type": "quote",
  //   "symbol": "AAPL",
  //   "price": 175.50,
  //   "timestamp": "2024-01-25T15:30:00Z"
  // }
};
```

### Analysis

#### Run IDTxl Analysis
```http
POST /api/analysis/idtxl
Content-Type: application/json

{
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "analysis_type": "transfer_entropy",
  "start_date": "2024-01-01",
  "end_date": "2024-01-31",
  "parameters": {
    "max_lag": 5,
    "estimator": "gaussian",
    "significance_level": 0.05,
    "permutations": 500
  },
  "use_gpu": true
}
```

**Response:**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "estimated_time_seconds": 30
}
```

#### Get Analysis Status
```http
GET /api/analysis/task/{task_id}
```

**Response:**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "result": {
    "transfer_entropy": {
      "AAPL->MSFT": {
        "value": 0.0234,
        "p_value": 0.001,
        "significant": true
      }
    },
    "mutual_information": {
      "AAPL-MSFT": 0.156
    },
    "processing_time_ms": 15234
  }
}
```

#### Run ML Analysis
```http
POST /api/analysis/ml
Content-Type: application/json

{
  "symbols": ["AAPL"],
  "model_type": "random_forest",
  "features": ["rsi", "macd", "volume_ratio", "price_change"],
  "target": "next_day_return",
  "start_date": "2023-01-01",
  "end_date": "2024-01-31",
  "parameters": {
    "n_estimators": 100,
    "max_depth": 10,
    "test_size": 0.2
  }
}
```

**Response:**
```json
{
  "task_id": "660e8400-e29b-41d4-a716-446655440001",
  "model_performance": {
    "accuracy": 0.65,
    "precision": 0.68,
    "recall": 0.62,
    "f1_score": 0.65,
    "feature_importance": {
      "rsi": 0.35,
      "macd": 0.28,
      "volume_ratio": 0.22,
      "price_change": 0.15
    }
  }
}
```

#### Run Neural Network Analysis
```http
POST /api/analysis/nn
Content-Type: application/json

{
  "symbols": ["AAPL"],
  "model_type": "lstm",
  "sequence_length": 30,
  "features": ["open", "high", "low", "close", "volume"],
  "target": "next_close",
  "start_date": "2023-01-01",
  "end_date": "2024-01-31",
  "parameters": {
    "hidden_units": 128,
    "dropout": 0.2,
    "epochs": 50,
    "batch_size": 32
  },
  "use_gpu": true
}
```

### Trading

#### Start Trading Session
```http
POST /api/trading/session/start
Content-Type: application/json

{
  "broker": "ibkr_cp",
  "data_provider": "iqfeed",
  "mode": "paper"  // or "live"
}
```

**Response:**
```json
{
  "session_id": "770e8400-e29b-41d4-a716-446655440002",
  "status": "active",
  "broker_connected": true,
  "data_feed_connected": true
}
```

#### Place Order
```http
POST /api/trading/orders
Content-Type: application/json

{
  "symbol": "AAPL",
  "side": "buy",
  "quantity": 100,
  "order_type": "limit",
  "price": 175.00,
  "time_in_force": "day",
  "stop_loss": 170.00,
  "take_profit": 180.00
}
```

**Response:**
```json
{
  "order_id": "880e8400-e29b-41d4-a716-446655440003",
  "status": "pending",
  "symbol": "AAPL",
  "side": "buy",
  "quantity": 100,
  "order_type": "limit",
  "price": 175.00,
  "created_at": "2024-01-25T15:30:00Z"
}
```

#### Get Order Status
```http
GET /api/trading/orders/{order_id}
```

**Response:**
```json
{
  "order_id": "880e8400-e29b-41d4-a716-446655440003",
  "status": "filled",
  "symbol": "AAPL",
  "side": "buy",
  "quantity": 100,
  "filled_quantity": 100,
  "avg_fill_price": 174.98,
  "commission": 1.00,
  "filled_at": "2024-01-25T15:31:23Z"
}
```

#### Cancel Order
```http
POST /api/trading/orders/{order_id}/cancel
```

**Response:**
```json
{
  "order_id": "880e8400-e29b-41d4-a716-446655440003",
  "status": "cancelled",
  "cancelled_at": "2024-01-25T15:32:00Z"
}
```

#### Get Positions
```http
GET /api/trading/positions
```

**Response:**
```json
{
  "positions": [
    {
      "symbol": "AAPL",
      "quantity": 100,
      "avg_cost": 174.98,
      "current_price": 175.50,
      "market_value": 17550.00,
      "unrealized_pnl": 52.00,
      "pnl_percent": 0.30
    }
  ],
  "total_value": 17550.00,
  "total_pnl": 52.00
}
```

### Portfolio

#### Get Portfolio Summary
```http
GET /api/portfolio/summary
```

**Response:**
```json
{
  "total_value": 125430.50,
  "cash_balance": 31358.00,
  "invested_amount": 94072.50,
  "total_pnl": 5234.75,
  "total_pnl_percent": 5.56,
  "daily_pnl": 342.15,
  "daily_pnl_percent": 0.27,
  "positions_count": 8,
  "win_rate": 0.68,
  "sharpe_ratio": 1.82,
  "max_drawdown": -0.085
}
```

#### Get Performance History
```http
GET /api/portfolio/performance?period=1M
```

**Parameters:**
- `period` (string, optional): Time period (1D, 1W, 1M, 3M, 1Y, ALL)

**Response:**
```json
{
  "period": "1M",
  "data": [
    {
      "date": "2024-01-01",
      "value": 120000.00,
      "pnl": 0.00,
      "benchmark": 100.00
    },
    {
      "date": "2024-01-02",
      "value": 120500.00,
      "pnl": 500.00,
      "benchmark": 100.42
    }
  ],
  "statistics": {
    "total_return": 0.0453,
    "benchmark_return": 0.0325,
    "alpha": 0.0128,
    "beta": 0.95,
    "volatility": 0.15
  }
}
```

### Strategy

#### Create Strategy
```http
POST /api/strategy/create
Content-Type: application/json

{
  "name": "Mean Reversion AAPL",
  "type": "mean_reversion",
  "symbols": ["AAPL"],
  "parameters": {
    "lookback_period": 20,
    "entry_z_score": -2.0,
    "exit_z_score": 0.0,
    "position_size": 100,
    "stop_loss": 0.03,
    "take_profit": 0.05
  },
  "risk_limits": {
    "max_position_size": 10000,
    "max_daily_loss": 500,
    "max_leverage": 1.0
  }
}
```

**Response:**
```json
{
  "strategy_id": "990e8400-e29b-41d4-a716-446655440004",
  "name": "Mean Reversion AAPL",
  "status": "created",
  "created_at": "2024-01-25T15:35:00Z"
}
```

#### Backtest Strategy
```http
POST /api/strategy/{strategy_id}/backtest
Content-Type: application/json

{
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "initial_capital": 100000,
  "commission": 0.001,
  "slippage": 0.0005
}
```

**Response:**
```json
{
  "backtest_id": "aa0e8400-e29b-41d4-a716-446655440005",
  "status": "completed",
  "results": {
    "total_return": 0.153,
    "sharpe_ratio": 1.82,
    "sortino_ratio": 2.15,
    "max_drawdown": -0.085,
    "win_rate": 0.685,
    "profit_factor": 2.34,
    "total_trades": 145,
    "winning_trades": 99,
    "losing_trades": 46,
    "avg_win": 234.50,
    "avg_loss": -145.30
  }
}
```

#### Deploy Strategy
```http
POST /api/strategy/{strategy_id}/deploy
Content-Type: application/json

{
  "mode": "paper",  // or "live"
  "capital_allocation": 10000,
  "start_immediately": true
}
```

### Risk Management

#### Get Risk Metrics
```http
GET /api/risk/metrics
```

**Response:**
```json
{
  "portfolio_var_95": 2340.50,
  "portfolio_var_99": 3450.75,
  "expected_shortfall": 2890.25,
  "current_exposure": 94072.50,
  "leverage_ratio": 0.75,
  "concentration_risk": {
    "AAPL": 0.25,
    "MSFT": 0.20,
    "GOOGL": 0.15
  },
  "correlation_matrix": {
    "AAPL": {
      "MSFT": 0.82,
      "GOOGL": 0.75
    }
  }
}
```

#### Check Risk Limits
```http
POST /api/risk/check
Content-Type: application/json

{
  "symbol": "AAPL",
  "side": "buy",
  "quantity": 1000,
  "order_type": "market"
}
```

**Response:**
```json
{
  "approved": false,
  "reasons": [
    "Position size exceeds limit (current: 8000, requested: 1000, limit: 8500)",
    "Concentration risk too high (would be 35% of portfolio)"
  ],
  "current_limits": {
    "max_position_size": 8500,
    "max_portfolio_concentration": 0.30,
    "daily_loss_limit": 5000,
    "current_daily_loss": -1234.50
  }
}
```

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "error": {
    "code": "INVALID_SYMBOL",
    "message": "Symbol INVALID not found",
    "details": {
      "field": "symbol",
      "value": "INVALID"
    }
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | Missing or invalid authentication token |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |

## Rate Limiting

API requests are rate limited based on your subscription tier:

- **Free**: 100 requests/minute
- **Basic**: 1000 requests/minute
- **Pro**: 10000 requests/minute
- **Enterprise**: Unlimited

Rate limit information is included in response headers:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1706189460
```

## Pagination

List endpoints support pagination:

```http
GET /api/trading/orders?page=1&limit=20
```

**Parameters:**
- `page` (integer, optional): Page number (default: 1)
- `limit` (integer, optional): Items per page (default: 20, max: 100)

**Response includes:**
```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 145,
    "pages": 8
  }
}
```

## Webhooks

Configure webhooks to receive real-time notifications:

```http
POST /api/webhooks
Content-Type: application/json

{
  "url": "https://your-server.com/webhook",
  "events": ["order.filled", "position.closed", "alert.triggered"],
  "secret": "your-webhook-secret"
}
```

## SDKs

Official SDKs are available for:
- Python: `pip install finplatform-sdk`
- JavaScript/TypeScript: `npm install @finplatform/sdk`
- Go: `go get github.com/finplatform/go-sdk`

## Support

- API Status: https://status.finplatform.com
- Documentation: https://docs.finplatform.com
- Support: api-support@finplatform.com