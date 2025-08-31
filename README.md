# Financial Time Series Analysis Platform

**Professional quantitative finance platform integrating information theory, machine learning, and neural networks for advanced financial analysis and automated trading.**

[![Platform Status](https://img.shields.io/badge/Status-Production%20Ready-green.svg)](https://github.com/yourorg/financial-platform)
[![Version](https://img.shields.io/badge/Version-1.1.0-blue.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Platform Overview

A comprehensive quantitative trading platform designed for institutional traders, quantitative researchers, and algorithmic trading firms. The platform combines cutting-edge information-theoretic analysis with traditional machine learning and modern neural networks.

### ✅ Current Status: **FULLY OPERATIONAL**
All 6 phases of development are complete, resulting in a production-ready platform with:
- **Information Theory Analysis** (IDTxl integration)
- **Machine Learning & Neural Networks** (GPU accelerated)  
- **Live Trading Integration** (IBKR + IQFeed)
- **Professional Risk Management**
- **Modern React Frontend**
- **Enterprise Deployment Ready**

## 🚀 Quick Start

### Prerequisites
- **Docker Desktop** (required)
- **Python 3.11+** (for backend development)
- **Node.js 18+** (for frontend development)

### 1. Clone & Setup
```bash
git clone <repository-url>
cd Fina_platform
```

### 2. Start All Services
```bash
# One command starts everything
python scripts/dev.py start
```

### 3. Access Platform
- **Frontend Application**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### 4. Login
- **Demo Account**: `demo` / `demo`
- **IBKR Trading**: Configured via `.env` file
- **IQFeed Data**: Configured via `.env` file

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend (React/TypeScript)                  │
├─────────────────────────────────────────────────────────────────┤
│                      API Gateway (FastAPI)                      │
├──────────────┬──────────────┬──────────────┬──────────────────┤
│   Analysis   │   Strategy   │   Trading    │      Data        │
│   Services   │   Services   │   Services   │    Services      │
├──────────────┼──────────────┼──────────────┼──────────────────┤
│ • IDTxl      │ • Builder    │ • Order Mgr  │ • Yahoo Finance  │
│ • ML Models  │ • Backtesting│ • IBKR API   │ • IQFeed        │
│ • Neural Net │ • Risk Mgmt  │ • Positions  │ • Market Data   │
└──────────────┴──────────────┴──────────────┴──────────────────┘
```

## 🔧 Core Technologies

### Backend Stack
- **FastAPI** - High-performance async API framework
- **IDTxl** - Information-theoretic analysis toolkit
- **Machine Learning** - scikit-learn, XGBoost, GPU acceleration
- **Neural Networks** - TensorFlow, PyTorch with CUDA support
- **Trading APIs** - Interactive Brokers, IQFeed professional data
- **Infrastructure** - PostgreSQL, Redis, Docker containers

### Frontend Stack  
- **React 18** with TypeScript
- **Tailwind CSS** - Modern responsive design
- **Recharts** - Financial data visualization
- **WebSocket** - Real-time market data streaming

## 🎯 Key Features

### Analysis Capabilities
- **Information Theory**: Transfer entropy, mutual information, causal analysis
- **Machine Learning**: Random Forest, XGBoost, SVM with hyperparameter optimization  
- **Neural Networks**: LSTM, GRU, CNN, Transformer architectures
- **GPU Acceleration**: 3-5x performance improvement on compatible hardware

### Trading Infrastructure
- **Live Trading**: Interactive Brokers Client Portal API integration
- **Market Data**: Professional IQFeed real-time and historical data
- **Risk Management**: Position limits, VaR calculations, circuit breakers
- **Order Management**: Smart routing, execution algorithms, fill tracking

### Platform Features
- **Strategy Development**: Complete backtesting framework with realistic costs
- **Portfolio Management**: Real-time monitoring and rebalancing
- **AI Advisory**: PhD-level guidance throughout analysis workflow
- **Professional UI**: Modern, responsive trading interface

## 📈 Performance Specifications

- **Analysis Speed**: IDTxl < 60s (CPU) / < 15s (GPU) for 1000 data points
- **ML Training**: < 5 min (CPU) / < 2 min (GPU) for 10K samples  
- **Trading Latency**: < 50ms order placement, < 30ms cancellation
- **Data Capacity**: 500+ symbols monitored, 1M+ trades per minute backtesting

## 🔌 Development

### Backend Development
```bash
# Start FastAPI development server
cd backend
python main.py

# API available at: http://localhost:8000
# Documentation: http://localhost:8000/docs
```

### Frontend Development
```bash
# Start React development server
cd frontend
npm install
npm run dev

# Frontend available at: http://localhost:5173
```

### Docker Commands
```bash
python scripts/dev.py start    # Start all services
python scripts/dev.py stop     # Stop services
python scripts/dev.py logs     # View logs
python scripts/dev.py status   # Check status
```

## 🔒 Security & Compliance

- **Authentication**: JWT-based with OAuth 2.0 for broker connections
- **Encryption**: SSL/TLS for data transmission, encrypted credential storage
- **Risk Controls**: Position limits, daily loss limits, concentration controls
- **Audit Trail**: Complete transaction logging and execution reports

## 📚 Documentation

- **[Complete Documentation](DOCUMENTATION.md)** - Comprehensive technical guide
- **[Contributing Guidelines](CONTRIBUTING.md)** - Development workflow and standards  
- **[Changelog](CHANGELOG.md)** - Version history and release notes
- **[API Reference](http://localhost:8000/docs)** - Interactive API documentation

## 🚀 Deployment

### Quick Development
```bash
# Start everything locally
python scripts/dev.py start
```

### Production Deployment
```bash
# Docker Compose
docker-compose -f docker-compose.production.yml up -d

# Kubernetes
kubectl apply -f k8s/production/
```

See [DOCUMENTATION.md](DOCUMENTATION.md) for complete deployment guides including Docker, Kubernetes, security configuration, and monitoring setup.

## 🛠️ External Services Configuration

### Required Accounts
- **Interactive Brokers**: Live trading execution
- **IQFeed**: Professional market data (DTN subscription required)
- **Google Cloud**: Optional GPU compute for large workloads

### API Keys
Configure in `.env` file:
```env
# Trading
IBKR_USERNAME=your_username
IBKR_PASSWORD=your_password
IQFEED_LOGIN=your_login  
IQFEED_PASSWORD=your_password

# Optional services
SERP_API_KEY=your_key
CLAUDE_API_KEY=your_key
```

## 🤝 Support & Contributing

- **Issues**: [Report bugs or request features](https://github.com/yourorg/financial-platform/issues)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
- **Documentation**: Complete technical docs in [DOCUMENTATION.md](DOCUMENTATION.md)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Project Status**: ✅ Production Ready | **Platform Type**: Enterprise Quantitative Trading | **Last Updated**: August 2025

*Built for professional quantitative researchers, algorithmic traders, and institutional investment teams.*