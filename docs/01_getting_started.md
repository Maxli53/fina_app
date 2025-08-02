# Getting Started Guide

This guide will help you get up and running with the Financial Time Series Analysis Platform in under 30 minutes.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [First Login](#first-login)
4. [Running Your First Analysis](#running-your-first-analysis)
5. [Creating a Trading Strategy](#creating-a-trading-strategy)
6. [Placing Your First Trade](#placing-your-first-trade)
7. [Next Steps](#next-steps)

## Prerequisites

### System Requirements
- **OS**: Windows 10+, macOS 10.15+, or Linux
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 20GB free space
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

### Software Requirements
- Docker Desktop
- Git
- Web browser (Chrome, Firefox, Safari, or Edge)

### Account Requirements
- Demo account (provided) or
- IBKR account (for live trading)
- IQFeed subscription (for professional market data)

## Installation

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/yourorg/financial-platform.git
cd financial-platform
```

### Step 2: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your favorite editor
# For demo mode, you can use the default values
nano .env
```

### Step 3: Start the Platform

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs (optional)
docker-compose logs -f
```

### Step 4: Verify Installation

Open your browser and navigate to:
- Frontend: http://localhost:5173
- API Documentation: http://localhost:8000/docs

## First Login

### Using Demo Account

1. Navigate to http://localhost:5173
2. Click "Login"
3. Enter credentials:
   - Username: `demo`
   - Password: `demo`
4. Click "Sign In"

### Dashboard Overview

After login, you'll see the main dashboard with:

```
┌─────────────────────────────────────────────────────┐
│  Portfolio Overview                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │
│  │ Total Value │ │   P&L       │ │ Win Rate    │  │
│  │ $100,000    │ │ +$5,234     │ │   68%       │  │
│  └─────────────┘ └─────────────┘ └─────────────┘  │
│                                                     │
│  Performance Chart                                  │
│  ┌─────────────────────────────────────────────┐  │
│  │     📈 Portfolio vs Benchmark                │  │
│  └─────────────────────────────────────────────┘  │
│                                                     │
│  Open Positions                                     │
│  ┌─────────────────────────────────────────────┐  │
│  │ Symbol │ Quantity │ P&L    │ % Change      │  │
│  │ AAPL   │ 100      │ +$234  │ +2.3%        │  │
│  │ MSFT   │ 50       │ -$123  │ -1.2%        │  │
│  └─────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Running Your First Analysis

### Step 1: Navigate to Analysis

1. Click "Analysis" in the navigation menu
2. You'll see the analysis configuration page

### Step 2: Configure IDTxl Analysis

1. **Select Symbols**:
   - Search for "AAPL" and click to add
   - Search for "MSFT" and click to add
   - Search for "GOOGL" and click to add

2. **Set Date Range**:
   - Start Date: 30 days ago
   - End Date: Today

3. **Choose Analysis Type**:
   - Select "Transfer Entropy"
   - Max Lag: 5
   - Estimator: Gaussian

4. **Enable GPU** (if available):
   - Toggle "GPU Acceleration" to ON

### Step 3: Run Analysis

1. Click "Run Analysis"
2. Wait for results (typically 15-60 seconds)
3. View the results:
   - Information flow network
   - Significant connections
   - Statistical significance values

### Understanding Results

```
Transfer Entropy Results:
┌─────────────────────────────────┐
│ AAPL → MSFT: 0.0234 (p=0.001)  │
│ MSFT → GOOGL: 0.0189 (p=0.003) │
│ GOOGL → AAPL: 0.0156 (p=0.012) │
└─────────────────────────────────┘

Interpretation:
- Higher TE values indicate stronger information flow
- p-values < 0.05 indicate statistical significance
- Use these relationships to inform trading decisions
```

## Creating a Trading Strategy

### Step 1: Navigate to Strategies

1. Click "Strategies" in the navigation menu
2. Click "New Strategy"

### Step 2: Define Strategy Parameters

```python
# Example: Mean Reversion Strategy
{
  "name": "AAPL Mean Reversion",
  "type": "mean_reversion",
  "symbols": ["AAPL"],
  "parameters": {
    "lookback_period": 20,
    "entry_threshold": -2.0,  # Buy when 2 std below mean
    "exit_threshold": 0.0,    # Sell when returns to mean
    "position_size": 100,
    "stop_loss": 0.03         # 3% stop loss
  }
}
```

### Step 3: Backtest Strategy

1. Click "Run Backtest"
2. Set backtest period (e.g., last 1 year)
3. Review results:
   - Total Return: +15.3%
   - Sharpe Ratio: 1.82
   - Max Drawdown: -8.5%
   - Win Rate: 68.5%

### Step 4: Deploy Strategy

1. If satisfied with backtest results
2. Click "Deploy Strategy"
3. Choose "Paper Trading" for testing
4. Monitor performance in real-time

## Placing Your First Trade

### Step 1: Navigate to Trading

1. Click "Live Trading" in the navigation menu
2. Ensure trading session is active (green indicator)

### Step 2: Enter Order Details

1. **Select Symbol**: AAPL
2. **Choose Side**: Buy
3. **Order Type**: Market
4. **Quantity**: 100 shares
5. **Review estimated cost**: ~$17,500

### Step 3: Place Order

1. Click "Buy AAPL"
2. Review order confirmation
3. Click "Confirm Order"
4. Monitor order status in "Open Orders"

### Step 4: Monitor Position

- View in "Positions" section
- Real-time P&L updates
- Set alerts for price movements

## Next Steps

### 1. Explore Advanced Features

- **Machine Learning**: Try Random Forest or XGBoost models
- **Neural Networks**: Configure LSTM for price prediction
- **Portfolio Analysis**: Analyze entire portfolio risk

### 2. Customize Your Workspace

- Create custom dashboards
- Set up price alerts
- Configure risk limits

### 3. Learn More

- Read the [API Documentation](03_api_reference.md)
- Study [Analysis Engine](05_analysis_engine.md) details
- Review [Trading System](06_trading_system.md) features

### 4. Join the Community

- GitHub Discussions
- Discord Server
- Weekly Webinars

## Troubleshooting

### Common Issues

**Services won't start:**
```bash
# Check Docker is running
docker --version

# Check ports are available
netstat -an | grep -E "5173|8000|5432|6379"

# Restart services
docker-compose down
docker-compose up -d
```

**Can't login:**
```bash
# Check backend is running
curl http://localhost:8000/api/health

# Reset demo credentials
docker-compose exec backend python scripts/reset_demo.py
```

**Analysis fails:**
```bash
# Check GPU availability (if using GPU)
docker-compose exec backend nvidia-smi

# View analysis logs
docker-compose logs analysis-worker
```

## Getting Help

### Resources
- **Documentation**: http://localhost:8000/docs
- **Community Forum**: https://forum.finplatform.com
- **Email Support**: support@finplatform.com

### Quick Tips
- Start with paper trading before using real money
- Use small position sizes when learning
- Always set stop-loss orders
- Monitor system health regularly

## Conclusion

Congratulations! You've successfully:
- ✅ Installed the platform
- ✅ Logged in and explored the dashboard
- ✅ Run your first IDTxl analysis
- ✅ Created and backtested a strategy
- ✅ Placed your first trade

You're now ready to explore the full capabilities of the Financial Time Series Analysis Platform. Remember to start small, test thoroughly, and gradually increase complexity as you gain experience.

Happy trading! 🚀