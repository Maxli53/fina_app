# Deployment Guide

This guide covers deployment options from development to production environments for the Financial Time Series Analysis Platform.

## Table of Contents
1. [Development Deployment](#development-deployment)
2. [Docker Deployment](#docker-deployment)
3. [Production Deployment](#production-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Security Hardening](#security-hardening)
6. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Development Deployment

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL 15+
- Redis 7+
- NVIDIA GPU (optional)

### Step 1: Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd Fina_platform

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install backend dependencies
cd backend
pip install -r requirements.txt

# Install frontend dependencies
cd ../frontend
npm install
```

### Step 2: Configure Environment

Create `.env` file in project root:
```env
# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/finplatform
DB_PASSWORD=your_secure_password

# Redis
REDIS_URL=redis://localhost:6379/0

# Trading Credentials
IBKR_USERNAME=your_username
IBKR_PASSWORD=your_password
IBKR_ACCOUNT_ID=your_account
IQFEED_LOGIN=your_login
IQFEED_PASSWORD=your_password
IQFEED_PRODUCT_ID=FINANCIAL_TIME_SERIES_PLATFORM

# API Keys
SERPAPI_KEY=your_serpapi_key
GOOGLE_APPLICATION_CREDENTIALS=./gcp-service-account.json
GOOGLE_CLOUD_PROJECT=your_project_id

# Security
JWT_SECRET_KEY=your_jwt_secret_key
ALLOWED_ORIGINS=http://localhost:3000
```

### Step 3: Start Services

```bash
# Start PostgreSQL
sudo systemctl start postgresql

# Start Redis
redis-server

# Start backend
cd backend
python main.py

# Start frontend (in new terminal)
cd frontend
npm run dev
```

### Step 4: Initialize Database

```bash
# Run migrations
cd backend
alembic upgrade head

# Create initial admin user
python scripts/create_admin.py
```

---

## Docker Deployment

### Prerequisites
- Docker Engine 24+
- Docker Compose 2.20+

### Step 1: Build Images

```bash
# Using development script
python scripts/dev.py build

# Or manually
docker-compose -f config/docker-compose.yml build
```

### Step 2: Configure Docker Environment

Create `config/.env.docker`:
```env
# Docker-specific configuration
DATABASE_URL=postgresql://postgres:password@postgres:5432/finplatform
REDIS_URL=redis://redis:6379/0

# External services (same as development)
IBKR_USERNAME=your_username
IBKR_PASSWORD=your_password
# ... other credentials
```

### Step 3: Start Services

```bash
# Start all services
python scripts/dev.py start

# Or manually
docker-compose -f config/docker-compose.yml up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend
```

### Step 4: Docker Commands

```bash
# Stop services
python scripts/dev.py stop

# Clean up
python scripts/dev.py clean

# Access container shell
python scripts/dev.py shell backend

# Run tests
python scripts/dev.py test
```

---

## Production Deployment

### Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Load Balancer │────▶│   Web Servers   │────▶│   App Servers   │
│     (NGINX)     │     │   (Frontend)    │     │   (Backend)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                          │
                        ┌─────────────────────────────────┼─────────┐
                        │                                 │         │
                  ┌─────▼─────┐     ┌─────────┐    ┌────▼────┐   │
                  │ PostgreSQL │     │  Redis  │    │  GPU    │   │
                  │  Primary   │     │ Cluster │    │ Workers │   │
                  └─────┬─────┘     └─────────┘    └─────────┘   │
                        │                                         │
                  ┌─────▼─────┐                                  │
                  │ PostgreSQL │                                  │
                  │  Replica   │                                  │
                  └───────────┘                                  │
```

### Step 1: Server Requirements

**Web Servers (2x)**
- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB SSD
- Network: 1Gbps

**App Servers (3x)**
- CPU: 8 cores
- RAM: 32GB
- Storage: 100GB SSD
- GPU: NVIDIA T4 (optional)

**Database Server**
- CPU: 16 cores
- RAM: 64GB
- Storage: 500GB NVMe SSD
- Backup: Daily snapshots

**Redis Cluster**
- CPU: 4 cores
- RAM: 16GB
- Storage: 50GB SSD

### Step 2: System Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y \
    nginx \
    postgresql-15 \
    redis-server \
    docker.io \
    docker-compose \
    certbot \
    python3-certbot-nginx \
    fail2ban \
    ufw

# Configure firewall
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 5432/tcp  # PostgreSQL (restrict to app servers)
sudo ufw allow 6379/tcp  # Redis (restrict to app servers)
sudo ufw enable

# Install NVIDIA drivers (for GPU servers)
sudo apt install nvidia-driver-525
sudo nvidia-smi
```

### Step 3: NGINX Configuration

Create `/etc/nginx/sites-available/finplatform`:
```nginx
upstream backend {
    least_conn;
    server app1.internal:8000;
    server app2.internal:8000;
    server app3.internal:8000;
}

upstream frontend {
    server web1.internal:3000;
    server web2.internal:3000;
}

server {
    listen 80;
    server_name api.finplatform.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.finplatform.com;

    ssl_certificate /etc/letsencrypt/live/api.finplatform.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.finplatform.com/privkey.pem;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # API endpoints
    location /api {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}

server {
    listen 443 ssl http2;
    server_name finplatform.com www.finplatform.com;

    ssl_certificate /etc/letsencrypt/live/finplatform.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/finplatform.com/privkey.pem;
    
    # Frontend
    location / {
        proxy_pass http://frontend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Step 4: Database Setup

```bash
# PostgreSQL configuration
sudo -u postgres psql

CREATE DATABASE finplatform;
CREATE USER finuser WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE finplatform TO finuser;

# Enable replication (on primary)
ALTER SYSTEM SET wal_level = replica;
ALTER SYSTEM SET max_wal_senders = 3;
ALTER SYSTEM SET wal_keep_segments = 64;

# Configure replica
pg_basebackup -h primary.internal -D /var/lib/postgresql/15/main -U replicator -v -P -W

# Performance tuning
ALTER SYSTEM SET shared_buffers = '16GB';
ALTER SYSTEM SET effective_cache_size = '48GB';
ALTER SYSTEM SET maintenance_work_mem = '2GB';
ALTER SYSTEM SET work_mem = '128MB';
```

### Step 5: Application Deployment

```bash
# Deploy backend
cd /opt/finplatform
git clone <repository-url> .
cd backend

# Create production config
cp .env.example .env.production
# Edit with production values

# Build and run with Docker
docker build -t finplatform-backend .
docker run -d \
    --name backend \
    --env-file .env.production \
    -p 8000:8000 \
    --restart unless-stopped \
    finplatform-backend

# Deploy frontend
cd ../frontend
npm run build
docker build -t finplatform-frontend .
docker run -d \
    --name frontend \
    -p 3000:3000 \
    --restart unless-stopped \
    finplatform-frontend
```

### Step 6: SSL Certificates

```bash
# Generate certificates
sudo certbot --nginx -d finplatform.com -d www.finplatform.com
sudo certbot --nginx -d api.finplatform.com

# Auto-renewal
sudo certbot renew --dry-run
```

---

## Cloud Deployment

### Google Cloud Platform

#### Step 1: Infrastructure as Code

Create `terraform/main.tf`:
```hcl
provider "google" {
  project = var.project_id
  region  = var.region
}

# VPC Network
resource "google_compute_network" "finplatform" {
  name                    = "finplatform-network"
  auto_create_subnetworks = false
}

# Subnet
resource "google_compute_subnetwork" "finplatform" {
  name          = "finplatform-subnet"
  network       = google_compute_network.finplatform.id
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
}

# GKE Cluster
resource "google_container_cluster" "finplatform" {
  name     = "finplatform-cluster"
  location = var.region
  
  node_pool {
    name       = "default-pool"
    node_count = 3
    
    node_config {
      machine_type = "n1-standard-4"
      
      oauth_scopes = [
        "https://www.googleapis.com/auth/cloud-platform"
      ]
    }
  }
  
  # GPU node pool
  node_pool {
    name       = "gpu-pool"
    node_count = 1
    
    node_config {
      machine_type = "n1-standard-8"
      
      guest_accelerator {
        type  = "nvidia-tesla-t4"
        count = 1
      }
    }
  }
}

# Cloud SQL
resource "google_sql_database_instance" "postgres" {
  name             = "finplatform-postgres"
  database_version = "POSTGRES_15"
  region           = var.region
  
  settings {
    tier = "db-n1-standard-4"
    
    database_flags {
      name  = "max_connections"
      value = "200"
    }
    
    backup_configuration {
      enabled    = true
      start_time = "03:00"
    }
  }
}

# Redis
resource "google_redis_instance" "cache" {
  name           = "finplatform-redis"
  tier           = "STANDARD_HA"
  memory_size_gb = 5
  region         = var.region
}
```

#### Step 2: Kubernetes Deployment

Create `k8s/deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: gcr.io/project-id/finplatform-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: backend-service
spec:
  selector:
    app: backend
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### Step 3: CI/CD Pipeline

Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Run tests
      run: |
        cd backend
        pip install -r requirements.txt
        pytest

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Google Cloud
      uses: google-github-actions/setup-gcloud@v1
      with:
        service_account_key: ${{ secrets.GCP_SA_KEY }}
        project_id: ${{ secrets.GCP_PROJECT_ID }}
    
    - name: Build and push Docker image
      run: |
        cd backend
        docker build -t gcr.io/${{ secrets.GCP_PROJECT_ID }}/finplatform-backend:${{ github.sha }} .
        docker push gcr.io/${{ secrets.GCP_PROJECT_ID }}/finplatform-backend:${{ github.sha }}
    
    - name: Deploy to GKE
      run: |
        gcloud container clusters get-credentials finplatform-cluster --region us-central1
        kubectl set image deployment/backend backend=gcr.io/${{ secrets.GCP_PROJECT_ID }}/finplatform-backend:${{ github.sha }}
        kubectl rollout status deployment/backend
```

---

## Security Hardening

### 1. Application Security

```python
# backend/app/security.py
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta

security = HTTPBearer()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### 2. Network Security

```bash
# Fail2ban configuration
sudo nano /etc/fail2ban/jail.local

[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true

[nginx-http-auth]
enabled = true

[nginx-limit-req]
enabled = true
```

### 3. Database Security

```sql
-- Row-level security
ALTER TABLE strategies ENABLE ROW LEVEL SECURITY;

CREATE POLICY strategy_policy ON strategies
    FOR ALL
    USING (user_id = current_user_id());

-- Audit logging
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    action VARCHAR(50),
    table_name VARCHAR(50),
    record_id INTEGER,
    old_values JSONB,
    new_values JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 4. Secret Management

```bash
# Use environment variables
export DATABASE_URL=$(vault kv get -field=url secret/database)
export JWT_SECRET=$(vault kv get -field=secret secret/jwt)

# Or use Google Secret Manager
gcloud secrets create db-password --data-file=password.txt
gcloud secrets add-iam-policy-binding db-password \
    --member=serviceAccount:finplatform@project.iam.gserviceaccount.com \
    --role=roles/secretmanager.secretAccessor
```

---

## Monitoring & Maintenance

### 1. Monitoring Stack

```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana:latest
    volumes:
      - grafana_data:/var/lib/grafana
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=secure_password
  
  alertmanager:
    image: prom/alertmanager:latest
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
    ports:
      - "9093:9093"
```

### 2. Application Metrics

```python
# backend/app/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
request_count = Counter('app_requests_total', 'Total requests', ['method', 'endpoint'])
request_duration = Histogram('app_request_duration_seconds', 'Request duration')
active_orders = Gauge('app_active_orders', 'Number of active orders')
portfolio_value = Gauge('app_portfolio_value', 'Current portfolio value')

# Middleware
@app.middleware("http")
async def add_metrics(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    request_count.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()
    request_duration.observe(duration)
    
    return response
```

### 3. Logging

```python
# backend/app/logging_config.py
import logging
from pythonjsonlogger import jsonlogger

# Configure JSON logging
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Log important events
logger.info("order_placed", extra={
    "order_id": order.id,
    "symbol": order.symbol,
    "quantity": order.quantity,
    "user_id": user.id
})
```

### 4. Backup Strategy

```bash
#!/bin/bash
# backup.sh

# Database backup
pg_dump $DATABASE_URL | gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz

# Upload to cloud storage
gsutil cp backup_*.sql.gz gs://finplatform-backups/

# Cleanup old backups (keep 30 days)
find . -name "backup_*.sql.gz" -mtime +30 -delete
```

### 5. Maintenance Checklist

**Daily**
- [ ] Check system health dashboard
- [ ] Review error logs
- [ ] Monitor disk space
- [ ] Verify backup completion

**Weekly**
- [ ] Review performance metrics
- [ ] Check security alerts
- [ ] Update dependencies
- [ ] Test disaster recovery

**Monthly**
- [ ] Security audit
- [ ] Performance optimization
- [ ] Capacity planning
- [ ] Cost analysis

---

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```bash
   # Find memory-hungry processes
   ps aux --sort=-%mem | head
   
   # Clear Redis cache
   redis-cli FLUSHDB
   
   # Restart services
   docker-compose restart backend
   ```

2. **Database Connection Issues**
   ```bash
   # Check connections
   sudo -u postgres psql -c "SELECT count(*) FROM pg_stat_activity;"
   
   # Kill idle connections
   SELECT pg_terminate_backend(pid) 
   FROM pg_stat_activity 
   WHERE state = 'idle' AND state_change < now() - interval '1 hour';
   ```

3. **API Performance Issues**
   ```bash
   # Enable slow query logging
   ALTER SYSTEM SET log_min_duration_statement = 1000;
   
   # Check slow endpoints
   grep -E "duration: [0-9]{4,}" /var/log/postgresql/postgresql.log
   ```

---

## Rollback Procedures

### Application Rollback
```bash
# Kubernetes rollback
kubectl rollout undo deployment/backend

# Docker rollback
docker stop backend
docker run -d --name backend finplatform-backend:previous-version
```

### Database Rollback
```bash
# Restore from backup
gunzip < backup_20250801_120000.sql.gz | psql $DATABASE_URL

# Point-in-time recovery
pg_restore --time="2025-08-01 11:00:00" backup.dump
```

---

**Last Updated**: August 2025  
**Version**: 1.0.0