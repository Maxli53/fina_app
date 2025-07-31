# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a Financial Time Series Analysis Platform that integrates information-theoretic analysis (IDTxl), machine learning, and neural networks for quantitative financial research. The platform is designed for professional quantitative researchers, algorithmic traders, and institutional investment teams.

## Core Technologies
- **Frontend**: React 18+ with TypeScript, Tailwind CSS, Recharts, Lucide React
- **Backend (Implemented)**: Python FastAPI, IDTxl, scikit-learn/XGBoost, TensorFlow/PyTorch
- **Infrastructure**: Docker, Kubernetes, PostgreSQL, Redis
- **AI Integration**: Claude Opus 4 as PhD-level advisory system
- **External APIs**: SERP API for market sentiment analysis, IQFeed for professional market data, Google Cloud Platform

## Development Commands
The backend is **FULLY IMPLEMENTED AND OPERATIONAL** with complete IDTxl integration. Here are the available commands:

### Frontend Development
```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Run tests
npm test

# Run linting
npm run lint

# Type checking
npm run typecheck
```

### Backend Development (Python) - ✅ FULLY OPERATIONAL
```bash
# Activate GPU environment (WSL Ubuntu)
source ~/gpu-env/bin/activate

# Navigate to backend
cd /mnt/c/Users/maxli/PycharmProjects/PythonProject/Fina_platform/backend

# Start FastAPI server
python main.py

# Server runs on: http://localhost:8000
# API Documentation: http://localhost:8000/docs
# Health Check: http://localhost:8000/api/health

# Test endpoints:
# - Search symbols: http://localhost:8000/api/data/search?query=AAPL
# - Historical data: http://localhost:8000/api/data/historical/GOOGL?start_date=2024-01-01&end_date=2024-12-31
# - Market status: http://localhost:8000/api/data/market-status

# Run linting (when available)
ruff check .

# Format code (when available)  
ruff format .
```

### Local GPU Setup & Configuration
```bash
# Check GPU availability
nvidia-smi  # For NVIDIA GPUs
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# Install GPU-accelerated libraries
pip install tensorflow-gpu>=2.13.0
pip install torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install cuml-cu11 cupy-cuda11x  # Rapids CuML for GPU-accelerated ML
pip install rapids-ai  # Full Rapids ecosystem

# Test GPU acceleration
python -c "
import tensorflow as tf
import torch
print('TensorFlow GPU:', tf.config.list_physical_devices('GPU'))
print('PyTorch CUDA:', torch.cuda.is_available())
print('PyTorch GPU Count:', torch.cuda.device_count())
"

# IDTxl GPU optimization (when available)
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export TF_FORCE_GPU_ALLOW_GROWTH=true  # Dynamic GPU memory allocation
```

### Docker Commands
```bash
# Using the organized development script
python scripts/dev.py start    # Start all services
python scripts/dev.py stop     # Stop all services
python scripts/dev.py logs     # View logs
python scripts/dev.py status   # Check status

# Direct docker-compose commands (from config directory)
cd config
docker-compose build
docker-compose up -d
docker-compose logs -f
docker-compose down
```

### Google Cloud Platform Commands
```bash
# Set up environment variables
export GOOGLE_APPLICATION_CREDENTIALS="./gcp-service-account.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Test GCP connection
python tests/integration/test_gcp_compute.py

# Create GPU instance for ML training
from backend.app.services.gcp_compute import create_small_gpu_instance
instance = create_small_gpu_instance("ml-training-job")

# Submit workload to cloud
from backend.app.services.ml_workload_manager import run_idtxl_analysis
workload_id = await run_idtxl_analysis(data_size_gb=2.0)
```

### SERP API Commands
```bash
# Install SERP API client
pip install google-search-results

# Test SERP API connection
python -c "from serpapi import GoogleSearch; print('SERP API ready')"

# Basic usage example
from serpapi import GoogleSearch
params = {
  "q": "TSLA stock news",
  "api_key": "your_serpapi_key",
  "engine": "google",
  "tbm": "nws"
}
search = GoogleSearch(params)
results = search.get_dict()
```

### IQFeed API Commands
```bash
# Install IQFeed Python client (pyiqfeed or iqfeed-python)
pip install pyiqfeed

# Start IQConnect service (Windows)
# Download and install IQConnect from DTN IQFeed website
# Start IQConnect.exe with your credentials

# Test IQFeed connection
python -c "
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('127.0.0.1', 5009))
    print('IQFeed Level 1 connection: SUCCESS')
    s.close()
except:
    print('IQFeed connection: FAILED - Start IQConnect service')
"

# Basic usage example
from pyiqfeed import QuoteConn, HistoryConn

# Real-time quotes
quote_conn = QuoteConn()
quote_conn.connect()
quote_conn.watch('AAPL')  # Subscribe to AAPL quotes

# Historical data
hist_conn = HistoryConn()
hist_conn.connect()
data = hist_conn.request_daily_data('AAPL', num_days=100)
```

## Project Architecture

### Platform Components
1. **Data Source Connection** (Section 1)
   - IBKR Gateway, Alpha Vantage, Yahoo Finance integration
   - Connection health monitoring and management
   - Data quality assessment

2. **Data Selection & Configuration** (Section 2)
   - Symbol selection with intelligent search
   - Timeframe configuration
   - Data preprocessing pipeline

3. **Analysis Configuration** (Section 3)
   - IDTxl information-theoretic analysis
   - ML models (Random Forest, XGBoost, SVM)
   - Neural networks (LSTM, GRU, CNN, Transformer)
   - Cross-method integration framework

4. **Trading Strategy Development** (Section 4)
   - Strategy design framework
   - Backtesting engine
   - Risk management system

5. **Results & Deployment** (Section 5)
   - Report generation
   - Live trading integration
   - Performance monitoring

### Directory Structure (CURRENT - ORGANIZED)
```
Fina_platform/
├── backend/                 # FastAPI application - FULLY OPERATIONAL
│   ├── app/
│   │   ├── api/            # API endpoints (health, data, analysis)
│   │   ├── models/         # Pydantic models (data, analysis)
│   │   └── services/       # Business logic
│   │       ├── analysis/   # IDTxl service (WORKING)
│   │       ├── data/       # Yahoo Finance service (WORKING)
│   │       ├── gcp_compute.py
│   │       └── ml_workload_manager.py
│   ├── Dockerfile          # Backend container
│   ├── main.py            # FastAPI app entry point
│   └── requirements.txt    # Python dependencies
├── frontend/               # React application (ready for development)
│   ├── src/               # React components and services
│   ├── public/            # Static assets
│   ├── Dockerfile         # Frontend container
│   └── package.json       # Node.js dependencies
├── config/                 # Configuration files
│   ├── docker-compose.yml # Multi-service configuration
│   └── gcp-service-account.json # GCP credentials
├── database/
│   └── init/              # Database initialization scripts
├── scripts/               # Development and utility scripts
│   ├── dev.py            # Development utility script
│   └── fix_idtxl_numpy.py # IDTxl compatibility fixes
├── tests/                 # Test suites
│   ├── backend/          # Backend unit tests
│   │   ├── test_api.py
│   │   ├── test_complete_idtxl.py
│   │   └── test_timezone_fix.py
│   └── integration/      # Integration tests
│       ├── test_docker_setup.py
│       └── test_gcp_compute.py
├── docs/                  # Documentation
│   ├── 01_platform_foundation.md
│   └── 02_analysis_engine.md
├── data/                  # Data and results
│   └── results/          # Analysis results
├── CLAUDE.md             # AI assistant instructions
└── README.md             # Project documentation
```

## Key Implementation Notes

### State Management Pattern
The platform uses React hooks with useReducer for complex state and custom hooks for reusable logic. Global state is shared via Context API.

### API Integration
- Claude Opus 4: Used as AI advisor with specific financial expertise context
- IBKR Gateway: TCP socket connection for market data
- IQFeed API: Professional-grade real-time and historical market data
- SERP API: Real-time market sentiment and news analysis
- Backend services: RESTful APIs with WebSocket for real-time updates

### Performance Targets
- IDTxl analysis: < 60 seconds for 1000 data points (CPU) / < 15 seconds (GPU)
- ML training: < 5 minutes for 10,000 samples (CPU) / < 2 minutes (GPU)
- Neural network training: < 30 minutes for 50,000 sequences (CPU) / < 10 minutes (GPU)
- Real-time signal generation: < 100ms latency
- Dashboard updates: < 200ms response time

### Local GPU Acceleration
- **Primary Strategy**: Utilize local GPU for all computational workloads
- **GPU Libraries**: CUDA, cuDNN, TensorFlow-GPU, PyTorch, CuML/Rapids
- **Memory Management**: Automatic GPU memory optimization and batch sizing
- **Multi-GPU Support**: Distributed training across multiple local GPUs
- **Fallback**: Cloud processing when local GPU resources are insufficient
- **Performance Boost**: 3-5x speedup for ML/neural network training

### Google Cloud Platform Integration
- **Compute Engine**: Automated GPU/CPU instance management
- **Workload Manager**: Intelligent job scheduling and distribution
- **Cost Optimization**: Preemptible instances for 80% savings
- **Instance Types**:
  - Small GPU (T4): $0.11/hour for light ML
  - Large GPU (V100): $1.10/hour for deep learning
  - CPU instances: $0.04-$0.11/hour for processing
- **Auto-scaling**: Dynamic resource allocation based on demand

### Security Considerations
- OAuth 2.0 authentication
- JWT token management
- API rate limiting
- Data encryption at rest and in transit
- Secure API key management for SERP API and Google Cloud
- IQFeed credential protection and secure socket connections
- Environment variable protection for sensitive credentials

## Development Workflow

### Phase 1: Core Analysis Engine ✅ COMPLETED
- ✅ Implement IDTxl analysis backend (FULLY OPERATIONAL)
- ✅ Deploy ML training infrastructure (Framework ready)
- ✅ Develop neural network training system (Framework ready)
- ✅ Create cross-method integration framework (API structure complete)

### Phase 2: Strategy Development Framework
- Build strategy design interface
- Implement backtesting engine
- Deploy risk management system
- Create performance analytics

### Phase 3: Live Trading Integration
- Integrate with trading platforms
- Implement execution systems
- Deploy monitoring infrastructure
- Create alerting framework

### Phase 4: Advanced Features
- Implement advanced analytics
- Deploy machine learning operations
- Create automated optimization
- Develop mobile capabilities

## Testing Strategy
- Unit tests for all components
- Integration tests for API endpoints
- End-to-end tests for critical workflows
- Performance testing for analysis engines
- Security testing for authentication/authorization

## Important Reminders
- Always validate financial data quality before analysis
- Implement proper error handling for market data connections
- Ensure all trading strategies include risk management
- Monitor API rate limits for external services
- Keep AI agent context updated with current analysis state