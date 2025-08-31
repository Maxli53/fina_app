# Financial Time Series Analysis Platform - Technical Documentation

**Comprehensive technical documentation for the enterprise quantitative finance platform.**

---

## Table of Contents

1. [Platform Architecture](#platform-architecture)
2. [Quick Start & Installation](#quick-start--installation)
3. [Core Analysis Engine](#core-analysis-engine)
4. [API Reference](#api-reference)
5. [Trading System](#trading-system)
6. [Strategy Development](#strategy-development)
7. [Deployment & Production](#deployment--production)
8. [External Services Integration](#external-services-integration)
9. [Performance & Monitoring](#performance--monitoring)
10. [Security & Compliance](#security--compliance)
11. [Development & Testing](#development--testing)
12. [Troubleshooting](#troubleshooting)

---

## Platform Architecture

### System Overview

The Financial Time Series Analysis Platform is designed as a microservices architecture with five core components:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Frontend (React 18 + TypeScript)                 │
│                         [Fully Implemented]                         │
├─────────────────────────────────────────────────────────────────────┤
│                       API Gateway (FastAPI)                         │
│                        Port 8000 - Async                            │
├──────────────┬──────────────┬──────────────┬─────────────────────────┤
│   Analysis   │   Strategy   │   Trading    │    Data Services        │
│   Services   │   Services   │   Services   │                         │
├──────────────┼──────────────┼──────────────┼─────────────────────────┤
│ • IDTxl      │ • Builder    │ • IBKR API   │ • Yahoo Finance         │
│ • ML Models  │ • Backtesting│ • IQFeed     │ • Market Data Cache     │
│ • Neural Net │ • Risk Mgmt  │ • Orders     │ • Symbol Search         │
│ • GPU Accel  │ • Portfolio  │ • Positions  │ • Historical Data       │
└──────────────┴──────────────┴──────────────┴─────────────────────────┘
```

### Technology Stack

**Backend (Python)**
- **Framework**: FastAPI 0.104+ with async/await
- **Analysis**: IDTxl 1.5+, NumPy, SciPy, Pandas
- **Machine Learning**: scikit-learn, XGBoost, CuML (GPU)
- **Neural Networks**: TensorFlow 2.13+, PyTorch 2.0+
- **Database**: PostgreSQL 15+ with async SQLAlchemy
- **Cache**: Redis 7.0+ for high-performance caching
- **Message Queue**: Celery with Redis broker

**Frontend (TypeScript)**
- **Framework**: React 18+ with TypeScript 5.0+
- **UI Library**: Tailwind CSS 3.0+ with custom components
- **Charts**: Recharts for financial data visualization
- **State**: React Query + Context API
- **Real-time**: WebSocket integration
- **Build**: Vite with hot module replacement

**Infrastructure**
- **Containerization**: Docker 24.0+ and Docker Compose
- **Orchestration**: Kubernetes 1.28+ ready
- **Monitoring**: Prometheus + Grafana
- **Load Balancing**: NGINX reverse proxy
- **SSL/TLS**: Let's Encrypt certificates

### Directory Structure

```
Fina_platform/
├── backend/                    # FastAPI Application
│   ├── app/
│   │   ├── api/               # REST API endpoints
│   │   │   ├── analysis.py    # IDTxl, ML, NN endpoints
│   │   │   ├── strategy.py    # Strategy CRUD & backtesting
│   │   │   ├── trading.py     # Live trading & orders
│   │   │   └── data.py        # Market data & symbols
│   │   ├── models/            # Pydantic models
│   │   │   ├── analysis.py    # Analysis request/response
│   │   │   ├── trading.py     # Trading models
│   │   │   └── strategy.py    # Strategy models
│   │   ├── services/          # Business logic
│   │   │   ├── analysis/      # Analysis implementations
│   │   │   │   ├── idtxl_service.py
│   │   │   │   ├── ml_service.py
│   │   │   │   └── nn_service.py
│   │   │   ├── strategy/      # Strategy services
│   │   │   │   ├── strategy_builder.py
│   │   │   │   ├── backtesting_engine.py
│   │   │   │   └── risk_manager.py
│   │   │   ├── trading/       # Trading services
│   │   │   │   ├── ibkr_service.py
│   │   │   │   ├── iqfeed_service.py
│   │   │   │   └── order_manager.py
│   │   │   └── data/          # Data services
│   │   │       ├── yahoo_finance.py
│   │   │       └── market_data.py
│   │   ├── core/              # Core utilities
│   │   │   ├── config.py      # Configuration management
│   │   │   ├── database.py    # Database connections
│   │   │   └── security.py    # Authentication & JWT
│   │   └── utils/             # Utility functions
│   ├── main.py                # FastAPI app entry point
│   ├── requirements.txt       # Python dependencies
│   └── Dockerfile             # Container configuration
├── frontend/                   # React Application
│   ├── src/
│   │   ├── components/        # Reusable UI components
│   │   │   ├── charts/        # Chart components
│   │   │   ├── forms/         # Form components
│   │   │   └── ui/            # Basic UI elements
│   │   ├── pages/            # Page components
│   │   │   ├── Dashboard.tsx  # Portfolio overview
│   │   │   ├── Analysis.tsx   # Analysis configuration
│   │   │   ├── Trading.tsx    # Live trading interface
│   │   │   └── Strategy.tsx   # Strategy management
│   │   ├── services/         # API integration
│   │   │   ├── api.ts        # HTTP client configuration
│   │   │   ├── websocket.ts  # WebSocket client
│   │   │   └── auth.ts       # Authentication service
│   │   ├── hooks/            # Custom React hooks
│   │   ├── contexts/         # React contexts
│   │   └── types/            # TypeScript type definitions
│   ├── package.json          # Node.js dependencies
│   └── Dockerfile            # Frontend container
├── config/                    # Configuration files
│   ├── docker-compose.yml    # Development environment
│   ├── docker-compose.production.yml  # Production setup
│   └── nginx.conf            # NGINX configuration
├── k8s/                      # Kubernetes manifests
│   ├── development/          # Dev environment configs
│   └── production/           # Production configs
├── scripts/                  # Utility scripts
│   ├── dev.py               # Development helper
│   ├── deploy.py            # Deployment automation
│   └── backup.py            # Database backup
├── tests/                    # Test suites
│   ├── backend/             # Backend unit tests
│   ├── frontend/            # Frontend tests
│   └── integration/         # E2E tests
└── docs/                     # Additional documentation
    └── api/                 # API documentation assets
```

---

## Quick Start & Installation

### System Requirements

**Minimum Requirements**
- **OS**: Windows 10/11, macOS 12+, or Ubuntu 20.04+
- **Memory**: 8GB RAM (16GB recommended)
- **Storage**: 20GB free space
- **CPU**: 4 cores (8 cores recommended)
- **GPU**: Optional NVIDIA GPU with CUDA 11.8+ for acceleration

**Required Software**
- **Docker Desktop**: 24.0+ with Docker Compose
- **Python**: 3.11+ (for backend development)
- **Node.js**: 18+ with npm (for frontend development)
- **Git**: Version control

### Installation Steps

#### 1. Clone Repository
```bash
git clone https://github.com/yourorg/financial-platform.git
cd financial-platform
```

#### 2. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration (see Configuration section below)
# Required: Database, Redis, and trading credentials
```

#### 3. Start Development Environment
```bash
# One command starts all services
python scripts/dev.py start

# This will:
# - Build all Docker containers
# - Start PostgreSQL and Redis
# - Initialize database schema  
# - Start FastAPI backend (port 8000)
# - Start React frontend (port 5173)
```

#### 4. Verify Installation
```bash
# Check all services are running
python scripts/dev.py status

# Access services:
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Environment Configuration

Create `.env` file with required settings:

```env
# Database Configuration
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/financial_platform
DB_PASSWORD=your_secure_password

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Security Configuration
SECRET_KEY=your_secret_key_here  # Generate with: openssl rand -hex 32
JWT_SECRET_KEY=your_jwt_secret   # Generate with: openssl rand -hex 32
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Trading Credentials
IBKR_USERNAME=your_ibkr_username
IBKR_PASSWORD=your_ibkr_password
IBKR_ACCOUNT_ID=your_account_id
IBKR_GATEWAY_URL=https://localhost:5000/v1/api

IQFEED_LOGIN=your_iqfeed_login
IQFEED_PASSWORD=your_iqfeed_password
IQFEED_PRODUCT_ID=FINANCIAL_TIME_SERIES_PLATFORM
IQFEED_VERSION=1.0.0

# External API Keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
SERP_API_KEY=your_serpapi_key
CLAUDE_API_KEY=your_claude_key
OPENAI_API_KEY=your_openai_key

# Google Cloud Platform (Optional)
GOOGLE_APPLICATION_CREDENTIALS=./config/gcp-service-account.json
GOOGLE_CLOUD_PROJECT=your_project_id

# Application Configuration
ENVIRONMENT=development  # development/production
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# Trading Configuration
TRADING_MODE=paper  # paper/live (use paper for testing)
DEFAULT_ORDER_SIZE=100
MAX_POSITION_SIZE=10000
DAILY_LOSS_LIMIT=0.05  # 5% daily loss limit
```

### Development Commands

#### Backend Development
```bash
# Navigate to backend
cd backend

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Start development server
python main.py

# Run tests
pytest tests/

# Code formatting
black app/
isort app/

# Type checking
mypy app/
```

#### Frontend Development
```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Run tests
npm test

# Code linting
npm run lint
npm run typecheck

# Build for production
npm run build
```

#### Docker Commands
```bash
# Start all services
python scripts/dev.py start

# Stop all services
python scripts/dev.py stop

# View logs
python scripts/dev.py logs
python scripts/dev.py logs backend  # Specific service

# Restart services
python scripts/dev.py restart

# Clean up (removes containers and volumes)
python scripts/dev.py clean

# Rebuild containers
python scripts/dev.py build
```

---

## API Reference

### Base URL
```
Development: http://localhost:8000/api
Production: https://your-domain.com/api
```

### Authentication

**Development**: Open API for testing  
**Production**: JWT Bearer token required

```http
Authorization: Bearer <jwt_token>
```

**Get JWT Token**:
```http
POST /api/auth/login
Content-Type: application/json

{
    "username": "your_username", 
    "password": "your_password"
}
```

### Response Format

All API responses follow this consistent format:

```json
{
    "status": "success|error",
    "data": {},
    "message": "Optional message",
    "timestamp": "2025-08-31T12:00:00Z",
    "request_id": "uuid-string"
}
```

### Health Check Endpoints

#### GET `/health`
System health and component status check.

**Response**:
```json
{
    "status": "healthy",
    "version": "1.1.0",
    "environment": "development",
    "components": {
        "database": "connected",
        "redis": "connected", 
        "idtxl": "ready",
        "ml_service": "ready",
        "nn_service": "ready",
        "trading": "connected"
    },
    "system": {
        "cpu_usage": 45.2,
        "memory_usage": 68.1,
        "disk_usage": 32.8
    }
}
```

#### GET `/health/gpu`
GPU availability and status.

**Response**:
```json
{
    "gpu_available": true,
    "cuda_version": "11.8",
    "devices": [
        {
            "id": 0,
            "name": "NVIDIA GeForce RTX 4090",
            "memory_total": 24576,
            "memory_used": 2048,
            "memory_free": 22528,
            "utilization": 15.3
        }
    ]
}
```

### Data Service Endpoints

#### GET `/data/search`
Search for financial symbols with filtering options.

**Parameters**:
- `query` (required): Search term (symbol or company name)
- `limit`: Maximum results (default: 10, max: 100)
- `asset_type`: Filter by asset type (equity, etf, option, future)
- `exchange`: Filter by exchange (NYSE, NASDAQ, etc.)

**Example**:
```http
GET /api/data/search?query=AAPL&limit=5&asset_type=equity
```

**Response**:
```json
{
    "status": "success",
    "data": {
        "results": [
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "exchange": "NASDAQ",
                "asset_type": "equity",
                "currency": "USD",
                "market_cap": 2800000000000,
                "sector": "Technology"
            }
        ],
        "total_count": 1
    }
}
```

#### GET `/data/historical/{symbol}`
Retrieve historical price data for a symbol.

**Path Parameters**:
- `symbol`: Stock symbol (e.g., AAPL, MSFT)

**Query Parameters**:
- `start_date`: Start date (YYYY-MM-DD format)
- `end_date`: End date (YYYY-MM-DD format) 
- `interval`: Data interval (1d, 1h, 5m, 1m)
- `include_splits`: Include split adjustments (true/false)
- `include_dividends`: Include dividend data (true/false)

**Example**:
```http
GET /api/data/historical/AAPL?start_date=2024-01-01&end_date=2024-12-31&interval=1d
```

**Response**:
```json
{
    "status": "success", 
    "data": {
        "symbol": "AAPL",
        "interval": "1d",
        "data": [
            {
                "timestamp": "2024-01-02T00:00:00Z",
                "open": 187.15,
                "high": 188.54,
                "low": 183.92,
                "close": 185.64,
                "volume": 52564300,
                "adjusted_close": 185.64
            }
        ],
        "splits": [],
        "dividends": [
            {
                "date": "2024-02-16",
                "amount": 0.24
            }
        ]
    }
}
```

#### GET `/data/market-status`
Get current market status and trading hours.

**Response**:
```json
{
    "status": "success",
    "data": {
        "market_open": true,
        "next_open": "2025-09-01T09:30:00-04:00",
        "next_close": "2025-08-31T16:00:00-04:00", 
        "timezone": "America/New_York",
        "current_time": "2025-08-31T14:30:00-04:00",
        "trading_calendar": {
            "regular_hours": {
                "start": "09:30:00",
                "end": "16:00:00"
            },
            "extended_hours": {
                "pre_market": "04:00:00-09:30:00",
                "after_market": "16:00:00-20:00:00"
            }
        }
    }
}
```

### Analysis Endpoints

#### POST `/analysis/idtxl`
Run information-theoretic analysis using IDTxl.

**Request Body**:
```json
{
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "analysis_type": "transfer_entropy",
    "config": {
        "max_lag": 5,
        "estimator": "gaussian",
        "significance_level": 0.05,
        "permutations": 1000,
        "tau": 1
    },
    "data_config": {
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "interval": "1d"
    }
}
```

**Response**:
```json
{
    "status": "success",
    "data": {
        "task_id": "idtxl_20250831_143022_abc123",
        "status": "started",
        "estimated_duration": 45
    }
}
```

#### GET `/analysis/status/{task_id}`
Check analysis task status and retrieve results.

**Response (In Progress)**:
```json
{
    "status": "success",
    "data": {
        "task_id": "idtxl_20250831_143022_abc123",
        "status": "running",
        "progress": 65.3,
        "elapsed_time": 28.7,
        "estimated_remaining": 16.2
    }
}
```

**Response (Completed)**:
```json
{
    "status": "success",
    "data": {
        "task_id": "idtxl_20250831_143022_abc123", 
        "status": "completed",
        "results": {
            "transfer_entropy": {
                "AAPL->MSFT": {
                    "te": 0.0234,
                    "p_value": 0.003,
                    "significant": true
                },
                "MSFT->AAPL": {
                    "te": 0.0156,
                    "p_value": 0.045,
                    "significant": true
                }
            },
            "network_inference": {
                "edges": [
                    {"source": "AAPL", "target": "MSFT", "weight": 0.0234},
                    {"source": "MSFT", "target": "AAPL", "weight": 0.0156}
                ]
            },
            "statistics": {
                "total_connections": 2,
                "significant_connections": 2,
                "network_density": 0.33
            }
        },
        "execution_time": 44.8
    }
}
```

### Trading Endpoints

#### POST `/trading/orders`
Place a new trading order.

**Request Body**:
```json
{
    "symbol": "AAPL",
    "side": "buy",
    "quantity": 100,
    "order_type": "limit",
    "price": 185.50,
    "time_in_force": "DAY",
    "strategy_id": "strategy_123"
}
```

**Response**:
```json
{
    "status": "success",
    "data": {
        "order_id": "ord_20250831_143045_def456",
        "status": "pending",
        "symbol": "AAPL",
        "side": "buy", 
        "quantity": 100,
        "order_type": "limit",
        "price": 185.50,
        "created_at": "2025-08-31T14:30:45Z"
    }
}
```

#### GET `/trading/positions`
Get current trading positions.

**Query Parameters**:
- `strategy_id`: Filter by strategy (optional)
- `symbol`: Filter by symbol (optional)

**Response**:
```json
{
    "status": "success",
    "data": {
        "positions": [
            {
                "symbol": "AAPL",
                "quantity": 200,
                "avg_cost": 184.25,
                "market_value": 37200.00,
                "unrealized_pnl": 250.00,
                "unrealized_pnl_percent": 0.68,
                "strategy_id": "strategy_123"
            }
        ],
        "total_value": 37200.00,
        "total_pnl": 250.00,
        "cash_balance": 12800.00
    }
}
```

*Documentation continues with additional sections...*