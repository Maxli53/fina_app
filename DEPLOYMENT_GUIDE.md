# Production Deployment Guide

## Overview

This guide covers the complete deployment process for the Financial Time Series Analysis Platform, from local development to production deployment with high availability, monitoring, and security.

## Prerequisites

### Required Tools
- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.24+ (for K8s deployment)
- Helm 3.0+ (for K8s deployment)
- Git
- Python 3.11+
- Node.js 18+
- NVIDIA Docker runtime (for GPU support)

### Required Accounts
- Cloud provider account (AWS/GCP/Azure)
- Container registry access
- Domain name and SSL certificates
- IBKR trading account
- IQFeed data subscription
- Monitoring service accounts

## Deployment Options

### Option 1: Docker Compose (Recommended for Small/Medium Scale)

#### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/yourorg/financial-platform.git
cd financial-platform

# Create production environment file
cp .env.example .env.production

# Edit .env.production with your credentials
vim .env.production
```

Required environment variables:
```env
# Database
DB_PASSWORD=<strong_password>
DATABASE_URL=postgresql://postgres:<password>@postgres:5432/financial_platform

# Redis
REDIS_URL=redis://redis:6379/0

# Security
SECRET_KEY=<generate_with_openssl_rand_-hex_32>
JWT_SECRET_KEY=<generate_with_openssl_rand_-hex_32>

# Trading Credentials
IBKR_USERNAME=<your_ibkr_username>
IBKR_PASSWORD=<your_ibkr_password>
IQFEED_LOGIN=<your_iqfeed_login>
IQFEED_PASSWORD=<your_iqfeed_password>

# API Keys
ALPHA_VANTAGE_API_KEY=<your_key>
SERP_API_KEY=<your_key>

# Monitoring
GRAFANA_PASSWORD=<admin_password>

# Trading Mode
TRADING_MODE=live  # or 'paper' for testing
```

#### 2. Build and Deploy

```bash
# Build all images
docker-compose -f docker-compose.production.yml build

# Start all services
docker-compose -f docker-compose.production.yml up -d

# Check service health
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f
```

#### 3. Initialize Database

```bash
# Run database migrations
docker-compose -f docker-compose.production.yml exec backend alembic upgrade head

# Create initial admin user
docker-compose -f docker-compose.production.yml exec backend python scripts/create_admin.py
```

#### 4. Verify Deployment

```bash
# Check API health
curl http://localhost/api/health

# Check frontend
open http://localhost

# Access monitoring
open http://localhost:3000  # Grafana
open http://localhost:9090  # Prometheus
```

### Option 2: Kubernetes Deployment (Recommended for Large Scale)

#### 1. Prepare Kubernetes Cluster

```bash
# Create namespace
kubectl apply -f k8s/production/namespace.yaml

# Create secrets
kubectl create secret generic database-credentials \
  --from-literal=url='postgresql://postgres:password@postgres:5432/financial_platform' \
  -n financial-platform

kubectl create secret generic trading-credentials \
  --from-literal=ibkr-username='your_username' \
  --from-literal=ibkr-password='your_password' \
  --from-literal=iqfeed-login='your_login' \
  --from-literal=iqfeed-password='your_password' \
  -n financial-platform

# Create config maps
kubectl create configmap backend-config \
  --from-file=config/production.yaml \
  -n financial-platform
```

#### 2. Deploy Core Services

```bash
# Deploy PostgreSQL
kubectl apply -f k8s/production/postgres-statefulset.yaml

# Deploy Redis
kubectl apply -f k8s/production/redis-statefulset.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n financial-platform --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n financial-platform --timeout=300s
```

#### 3. Deploy Application Services

```bash
# Deploy backend API
kubectl apply -f k8s/production/backend-deployment.yaml

# Deploy analysis workers
kubectl apply -f k8s/production/analysis-worker-deployment.yaml

# Deploy trading engine
kubectl apply -f k8s/production/trading-engine-deployment.yaml

# Deploy frontend
kubectl apply -f k8s/production/frontend-deployment.yaml

# Deploy ingress
kubectl apply -f k8s/production/ingress.yaml
```

#### 4. Deploy Monitoring Stack

```bash
# Add Prometheus Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus and Grafana
helm install monitoring prometheus-community/kube-prometheus-stack \
  -n financial-platform \
  -f k8s/production/monitoring-values.yaml

# Deploy custom dashboards
kubectl apply -f k8s/production/grafana-dashboards.yaml
```

## Post-Deployment Configuration

### 1. SSL/TLS Setup

For Docker Compose:
```bash
# Install certbot
apt-get install certbot

# Generate certificates
certbot certonly --standalone -d yourdomain.com -d api.yourdomain.com

# Update nginx configuration
cp /etc/letsencrypt/live/yourdomain.com/* ./nginx/ssl/
```

For Kubernetes:
```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Create certificate issuer
kubectl apply -f k8s/production/cert-issuer.yaml
```

### 2. Configure Monitoring Alerts

```bash
# Access Grafana
# Default: admin/admin (change immediately)
open https://monitoring.yourdomain.com

# Import dashboards:
# - System Overview Dashboard
# - Trading Performance Dashboard
# - Risk Monitoring Dashboard
# - Infrastructure Dashboard

# Configure alert channels:
# - Email
# - Slack
# - PagerDuty
```

### 3. Set Up Backups

```bash
# Create backup script
cat > /opt/backups/backup.sh << 'EOF'
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/postgres"

# Backup PostgreSQL
docker exec postgres pg_dump -U postgres financial_platform | gzip > $BACKUP_DIR/backup_$TIMESTAMP.sql.gz

# Keep only last 7 days
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +7 -delete

# Upload to S3 (optional)
aws s3 cp $BACKUP_DIR/backup_$TIMESTAMP.sql.gz s3://your-backup-bucket/postgres/
EOF

# Schedule with cron
echo "0 2 * * * /opt/backups/backup.sh" | crontab -
```

### 4. Configure Auto-scaling

For Kubernetes:
```yaml
# Apply HPA for backend
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
  namespace: financial-platform
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
EOF
```

## Security Hardening

### 1. Network Security

```bash
# Configure firewall rules
ufw allow 22/tcp  # SSH
ufw allow 80/tcp  # HTTP
ufw allow 443/tcp # HTTPS
ufw enable

# For Kubernetes, apply network policies
kubectl apply -f k8s/production/network-policies.yaml
```

### 2. Secrets Management

```bash
# Rotate secrets regularly
# Use external secret management (AWS Secrets Manager, HashiCorp Vault)

# Example with Vault
helm install vault hashicorp/vault -n financial-platform
kubectl exec -it vault-0 -n financial-platform -- vault operator init
```

### 3. Security Scanning

```bash
# Scan Docker images
trivy image ghcr.io/yourorg/financial-platform/backend:latest

# Scan Kubernetes cluster
kubectl apply -f https://raw.githubusercontent.com/aquasecurity/trivy-operator/main/deploy/static/trivy-operator.yaml
```

## Performance Tuning

### 1. Database Optimization

```sql
-- Create indexes
CREATE INDEX idx_orders_symbol_timestamp ON orders(symbol, created_at);
CREATE INDEX idx_positions_account_symbol ON positions(account_id, symbol);
CREATE INDEX idx_analysis_results_created ON analysis_results(created_at);

-- Configure connection pooling
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
```

### 2. Redis Optimization

```bash
# Edit redis.conf
maxmemory 4gb
maxmemory-policy allkeys-lru
save ""  # Disable persistence for cache-only use
```

### 3. Application Optimization

```python
# Configure Gunicorn workers
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'uvicorn.workers.UvicornWorker'
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
```

## Monitoring and Maintenance

### 1. Health Checks

```bash
# Create health check script
cat > /opt/monitoring/health_check.sh << 'EOF'
#!/bin/bash

# Check all services
services=("backend" "frontend" "postgres" "redis" "trading-engine")

for service in "${services[@]}"; do
    if ! docker-compose -f /app/docker-compose.production.yml ps | grep -q "$service.*Up"; then
        echo "ALERT: $service is down!"
        # Send alert
    fi
done
EOF

# Run every 5 minutes
echo "*/5 * * * * /opt/monitoring/health_check.sh" | crontab -
```

### 2. Log Management

```bash
# Configure log rotation
cat > /etc/logrotate.d/financial-platform << EOF
/var/log/financial-platform/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 appuser appuser
}
EOF
```

### 3. Performance Monitoring

Access Grafana dashboards:
- **System Overview**: Overall health metrics
- **Trading Performance**: P&L, order execution, positions
- **Risk Monitoring**: VaR, exposure, circuit breakers
- **Infrastructure**: CPU, memory, disk, network

Key metrics to monitor:
- API response time < 200ms
- Order execution latency < 100ms
- Analysis queue depth < 100
- Database connection pool < 80%
- Error rate < 0.1%

## Troubleshooting

### Common Issues

1. **Service Won't Start**
```bash
# Check logs
docker-compose -f docker-compose.production.yml logs <service_name>

# Check resource availability
df -h  # Disk space
free -m  # Memory
```

2. **Database Connection Issues**
```bash
# Test connection
docker exec -it postgres psql -U postgres -d financial_platform -c "SELECT 1"

# Check connection pool
docker exec -it backend python -c "from app.database import engine; print(engine.pool.status())"
```

3. **Market Data Not Updating**
```bash
# Check IQFeed connection
docker exec -it trading-engine python scripts/test_iqfeed.py

# Check Redis for cached data
docker exec -it redis redis-cli GET "market_data:AAPL:last_update"
```

4. **High Memory Usage**
```bash
# Identify memory consumers
docker stats --no-stream

# Restart service with memory limit
docker-compose -f docker-compose.production.yml up -d --scale backend=3
```

## Rollback Procedures

### Docker Compose Rollback
```bash
# Tag current version
docker tag backend:latest backend:rollback

# Deploy previous version
docker-compose -f docker-compose.production.yml down
docker tag backend:v1.2.3 backend:latest
docker-compose -f docker-compose.production.yml up -d
```

### Kubernetes Rollback
```bash
# View rollout history
kubectl rollout history deployment/backend-api -n financial-platform

# Rollback to previous version
kubectl rollout undo deployment/backend-api -n financial-platform

# Rollback to specific revision
kubectl rollout undo deployment/backend-api --to-revision=3 -n financial-platform
```

## Support and Maintenance

### Regular Maintenance Tasks
- **Daily**: Check system health, review alerts
- **Weekly**: Review performance metrics, update dependencies
- **Monthly**: Security patches, capacity planning
- **Quarterly**: Disaster recovery testing, architecture review

### Support Contacts
- **Critical Issues**: oncall@yourcompany.com
- **Non-Critical**: support@yourcompany.com
- **Security**: security@yourcompany.com

### Documentation
- API Documentation: https://api.yourdomain.com/docs
- User Guide: https://docs.yourdomain.com
- Architecture: See SYSTEM_ARCHITECTURE.md

## Conclusion

Following this guide ensures a secure, scalable, and maintainable deployment of the Financial Time Series Analysis Platform. Regular monitoring and maintenance are crucial for optimal performance and reliability in production trading environments.