# System Orchestration Documentation

## Overview

The System Orchestrator is the central nervous system of the Financial Time Series Analysis Platform. It coordinates all components, manages workflows, monitors system health, and provides real-time updates to connected clients through WebSocket connections.

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      System Orchestrator                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │  Health Monitor  │  │ Workflow Engine │  │  Event Manager  ││
│  │                 │  │                 │  │                 ││
│  │ • Service Check │  │ • Analysis      │  │ • Broadcasting  ││
│  │ • Auto Recovery │  │ • Trading       │  │ • History       ││
│  │ • Degraded Mode │  │ • Rebalancing   │  │ • Filtering     ││
│  └─────────────────┘  └─────────────────┘  └─────────────────┘│
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │   Risk Monitor   │  │  Data Pipeline  │  │WebSocket Server ││
│  │                 │  │                 │  │                 ││
│  │ • VaR Tracking  │  │ • Market Data   │  │ • Real-time     ││
│  │ • Violations    │  │ • Analysis Queue│  │ • Bidirectional ││
│  │ • Circuit Break │  │ • Caching       │  │ • Multi-client  ││
│  └─────────────────┘  └─────────────────┘  └─────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### System States

```python
class SystemState(Enum):
    INITIALIZING = "initializing"  # System starting up
    READY = "ready"                # Initialized, not running
    RUNNING = "running"            # Fully operational
    DEGRADED = "degraded"          # Partial functionality
    MAINTENANCE = "maintenance"    # Maintenance mode
    ERROR = "error"                # Critical error state
    SHUTDOWN = "shutdown"          # Shutting down
```

## Workflows

### 1. Analysis to Trade Workflow

Automated workflow that performs analysis and executes trades based on signals.

```python
WorkflowType.ANALYSIS_TO_TRADE:
    1. Fetch Market Data
       - Retrieve latest price data
       - Validate data quality
       
    2. Run Analysis
       - IDTxl information theory
       - Machine learning models
       - Neural network predictions
       
    3. Generate Signals
       - Combine analysis results
       - Apply confidence thresholds
       
    4. Validate Signals
       - Risk checks
       - Market condition validation
       
    5. Execute Trades (optional)
       - Place orders
       - Monitor execution
```

### 2. Portfolio Rebalancing Workflow

Automatically rebalances portfolio based on target allocations.

```python
WorkflowType.PORTFOLIO_REBALANCE:
    1. Analyze Portfolio
       - Current positions
       - Market values
       - Concentration metrics
       
    2. Calculate Targets
       - Equal weight
       - Risk parity
       - Market cap weighted
       
    3. Generate Orders
       - Calculate differences
       - Create rebalance orders
       
    4. Execute Rebalance
       - Check market impact
       - Execute with algorithms
```

### 3. Strategy Optimization Workflow

Optimizes strategy parameters through backtesting.

```python
WorkflowType.STRATEGY_OPTIMIZATION:
    1. Select Strategies
       - Top performing
       - Candidate strategies
       
    2. Run Backtests
       - Parameter optimization
       - Multiple iterations
       
    3. Analyze Results
       - Performance metrics
       - Improvement analysis
       
    4. Update Parameters
       - Apply best parameters
       - Optional auto-deploy
```

## Health Monitoring

### Service Health Checks

The orchestrator continuously monitors all system components:

```python
# Health check interval: 30 seconds
Services monitored:
- Database connectivity
- Redis cache
- Market data feeds (IQFeed)
- Trading connections (IBKR)
- Analysis engines
- GPU availability
```

### Health States

- **HEALTHY**: Service operating normally
- **DEGRADED**: Service operational but impaired
- **UNHEALTHY**: Service experiencing issues
- **CRITICAL**: Service failure requiring intervention

### Automatic Recovery

When services fail, the orchestrator attempts recovery:

1. **Component Restart**: Restart failed service
2. **Connection Reset**: Re-establish connections
3. **Fallback Mode**: Switch to degraded operation
4. **Emergency Shutdown**: Last resort protection

## Risk Monitoring

### Real-time Risk Metrics

```python
{
    "portfolio_value": 125000.00,
    "var_95": 5234.50,          # Value at Risk (95%)
    "var_99": 7890.25,          # Value at Risk (99%)
    "expected_shortfall": 6500.00,
    "high_volatility": false
}
```

### Risk Violations

The system monitors for:
- VaR limit breaches
- Concentration limits
- Daily loss limits
- Volatility spikes

### Circuit Breakers

Automatic protection mechanisms:
- **Daily Loss Limit**: Halt trading at threshold
- **Position Concentration**: Block oversized positions
- **Order Frequency**: Prevent excessive trading

## WebSocket Communication

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8765');
```

### Message Types

#### System State Update
```json
{
    "type": "system_state",
    "data": {
        "state": "running",
        "active_workflows": ["health_monitoring", "data_pipeline"],
        "metrics": {
            "active_workflows": 2,
            "websocket_clients": 5,
            "uptime": "24h 35m 12s"
        }
    }
}
```

#### Health Update
```json
{
    "type": "health_update",
    "data": {
        "timestamp": "2024-01-25T10:30:00Z",
        "status": {
            "database": {
                "status": "healthy",
                "latency_ms": 5.2
            },
            "market_data": {
                "status": "healthy",
                "latency_ms": 12.3
            }
        }
    }
}
```

#### System Event
```json
{
    "type": "system_event",
    "data": {
        "timestamp": "2024-01-25T10:30:00Z",
        "event_type": "workflow_completed",
        "source": "orchestrator",
        "severity": "info",
        "data": {
            "workflow": "analysis_to_trade",
            "duration": 15.4
        }
    }
}
```

### Client Commands

#### Execute Workflow
```json
{
    "type": "execute_workflow",
    "workflow": "analysis_to_trade",
    "params": {
        "symbols": ["AAPL", "MSFT"],
        "timeframe": "1d"
    }
}
```

#### Subscribe to Market Data
```json
{
    "type": "subscribe",
    "symbols": ["AAPL", "MSFT", "GOOGL"]
}
```

## API Endpoints

### System Status
```http
GET /api/system/status
Authorization: Bearer {token}

Response:
{
    "state": "running",
    "uptime": "24h 35m 12s",
    "active_workflows": ["health_monitoring", "data_pipeline"],
    "metrics": {...},
    "connected_clients": 5,
    "event_count": 1247
}
```

### Execute Workflow
```http
POST /api/system/workflow/{workflow_type}
Authorization: Bearer {token}
Content-Type: application/json

{
    "parameters": {
        "symbols": ["AAPL", "MSFT"],
        "method": "risk_parity"
    }
}

Response:
{
    "status": "completed",
    "results": {...},
    "duration": 23.5
}
```

### System Control
```http
POST /api/system/control/{action}
Authorization: Bearer {token}

Actions: start, stop, restart

Response:
{
    "status": "started",
    "message": "System started successfully"
}
```

## Configuration

### System Configuration

```python
config = {
    "use_gpu": True,
    "health_check_interval": 30,      # seconds
    "risk_check_interval": 60,        # seconds
    "data_pipeline_interval": 1,      # seconds
    "websocket_port": 8765,
    "event_history_limit": 1000,
    "risk_limits": {
        "var_95_limit": 10000,
        "max_daily_loss": 5000,
        "max_concentration": 0.30
    },
    "emergency_close_positions": False
}
```

### Performance Tuning

```yaml
# Optimize for low latency
performance:
  health_check_interval: 10
  data_pipeline_interval: 0.5
  websocket_ping_interval: 5
  
# Optimize for stability
stability:
  health_check_interval: 60
  retry_attempts: 5
  recovery_delay: 30
```

## Frontend Integration

### System Status Component

```typescript
import { SystemStatus } from '@/components/SystemStatus';

// Display real-time system status
<SystemStatus />
```

### WebSocket Hook

```typescript
import { useWebSocket } from '@/hooks/useWebSocket';

const { socket, isConnected, sendMessage } = useWebSocket('ws://localhost:8765');

// Execute workflow
sendMessage({
  type: 'execute_workflow',
  workflow: 'analysis_to_trade',
  params: {}
});
```

## Monitoring and Observability

### Metrics Collection

The orchestrator collects:
- Operation counts and durations
- Error rates
- Resource utilization
- Client connections
- Event throughput

### Performance Metrics

```python
{
    "workflow_analysis_to_trade": {
        "count": 145,
        "total_time": 3456.7,
        "errors": 2,
        "last_run": "2024-01-25T10:30:00Z"
    }
}
```

### Event History

All system events are stored with:
- Timestamp
- Event type
- Source component
- Severity level
- Associated data

## Error Handling

### Recovery Strategies

1. **Automatic Retry**: Transient failures
2. **Component Restart**: Service failures
3. **Degraded Operation**: Partial functionality
4. **Manual Intervention**: Critical failures

### Error Propagation

```
Component Error → Health Check Failure → State Change → Event Broadcast → Client Notification
```

## Security Considerations

### Access Control
- JWT authentication required
- Role-based permissions
- API rate limiting

### Data Protection
- Encrypted WebSocket connections
- Secure credential storage
- Audit logging

## Best Practices

### Workflow Design
1. Keep workflows modular
2. Implement proper error handling
3. Add progress tracking
4. Include rollback mechanisms

### Health Monitoring
1. Set appropriate check intervals
2. Define clear health criteria
3. Implement gradual degradation
4. Log all state changes

### Risk Management
1. Configure conservative limits
2. Test circuit breakers
3. Monitor violation patterns
4. Regular limit reviews

## Troubleshooting

### Common Issues

#### WebSocket Connection Failed
```bash
# Check if orchestrator is running
curl http://localhost:8000/api/system/status

# Verify WebSocket port is open
netstat -an | grep 8765
```

#### Workflow Execution Failed
```python
# Check logs
tail -f logs/orchestrator.log

# Verify component health
GET /api/system/health
```

#### High Latency
1. Check network connectivity
2. Review health check intervals
3. Analyze workflow performance
4. Monitor resource usage

## Performance Optimization

### Batch Processing
```python
# Process multiple symbols together
await orchestrator.execute_workflow(
    WorkflowType.ANALYSIS_TO_TRADE,
    {"symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"]}
)
```

### Caching Strategy
- Market data: 60 seconds
- Analysis results: 1 hour
- Risk metrics: 5 minutes

### Resource Management
- Connection pooling
- Memory limits
- CPU throttling
- GPU allocation