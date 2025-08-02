# Production Deployment Guide

## Overview

This guide covers deploying the Financial Time Series Analysis Platform to production environments, including Docker, Kubernetes, and cloud platforms.

## Prerequisites

- Docker 20.10+
- Kubernetes 1.25+ (for K8s deployment)
- Helm 3.10+ (for K8s deployment)
- PostgreSQL 14+
- Redis 7+
- NVIDIA GPU drivers (for GPU acceleration)
- Valid API keys (IQFeed, IBKR, SERP API)

## Environment Configuration

### Production Environment Variables

Create `.env.production`:

```bash
# Application
NODE_ENV=production
ENVIRONMENT=production

# Database
DATABASE_URL=postgresql://user:password@postgres:5432/finplatform
REDIS_URL=redis://redis:6379

# API Keys
IBKR_USERNAME=your_ibkr_username
IBKR_PASSWORD=your_ibkr_password
IBKR_ACCOUNT_ID=your_account_id
IQFEED_PRODUCT_ID=your_product_id
IQFEED_LOGIN=your_login
IQFEED_PASSWORD=your_password
SERPAPI_KEY=your_serpapi_key

# Security
JWT_SECRET=your_jwt_secret_key_here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=1440

# Services
BACKEND_URL=https://api.yourdomain.com
FRONTEND_URL=https://app.yourdomain.com
WEBSOCKET_URL=wss://ws.yourdomain.com

# Performance
WORKERS=4
MAX_CONNECTIONS=1000
POOL_SIZE=20

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
TF_FORCE_GPU_ALLOW_GROWTH=true

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
SENTRY_DSN=your_sentry_dsn

# Google Cloud Platform
GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-service-account.json
GOOGLE_CLOUD_PROJECT=your-project-id
```

## Docker Deployment

### Production Docker Compose

`docker-compose.production.yml`:

```yaml
version: '3.8'

services:
  # Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.production
    ports:
      - "80:80"
      - "443:443"
    environment:
      - REACT_APP_API_URL=${BACKEND_URL}
      - REACT_APP_WS_URL=${WEBSOCKET_URL}
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - backend
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Backend API
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.production
    ports:
      - "8000:8000"
      - "8765:8765"  # WebSocket
    env_file:
      - .env.production
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./gcp-service-account.json:/app/gcp-service-account.json:ro
    depends_on:
      - postgres
      - redis
    restart: always
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # PostgreSQL Database
  postgres:
    image: postgres:14-alpine
    environment:
      POSTGRES_DB: finplatform
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_INITDB_ARGS: "--encoding=UTF8 --locale=en_US.utf8"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init:/docker-entrypoint-initdb.d:ro
    ports:
      - "5432:5432"
    restart: always
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER} -d finplatform"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis Cache
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: always
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.prod.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - frontend
      - backend
    restart: always

  # Prometheus Monitoring
  prometheus:
    image: prom/prometheus:latest
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    restart: always

  # Grafana Dashboard
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    depends_on:
      - prometheus
    restart: always

  # Loki Log Aggregation
  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    command: -config.file=/etc/loki/local-config.yaml
    volumes:
      - ./monitoring/loki-config.yaml:/etc/loki/local-config.yaml:ro
      - loki_data:/loki
    restart: always

  # Promtail Log Collector
  promtail:
    image: grafana/promtail:latest
    volumes:
      - ./logs:/var/log/app:ro
      - ./monitoring/promtail-config.yaml:/etc/promtail/config.yml:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
    command: -config.file=/etc/promtail/config.yml
    depends_on:
      - loki
    restart: always

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
  loki_data:
```

### Building for Production

```bash
# Build all services
docker-compose -f docker-compose.production.yml build

# Start services
docker-compose -f docker-compose.production.yml up -d

# View logs
docker-compose -f docker-compose.production.yml logs -f

# Scale backend workers
docker-compose -f docker-compose.production.yml up -d --scale backend=3
```

## Kubernetes Deployment

### Namespace and ConfigMap

`k8s/namespace.yaml`:
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: finplatform
```

`k8s/configmap.yaml`:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: finplatform-config
  namespace: finplatform
data:
  ENVIRONMENT: "production"
  BACKEND_URL: "https://api.finplatform.com"
  FRONTEND_URL: "https://app.finplatform.com"
  WEBSOCKET_URL: "wss://ws.finplatform.com"
  WORKERS: "4"
  MAX_CONNECTIONS: "1000"
```

### Backend Deployment

`k8s/backend-deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
  namespace: finplatform
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
        image: finplatform/backend:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 8765
          name: websocket
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: finplatform-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: finplatform-secrets
              key: redis-url
        envFrom:
        - configMapRef:
            name: finplatform-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
            nvidia.com/gpu: 1
          limits:
            memory: "2Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /api/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /api/health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        volumeMounts:
        - name: gcp-credentials
          mountPath: /app/gcp-service-account.json
          subPath: gcp-service-account.json
          readOnly: true
      volumes:
      - name: gcp-credentials
        secret:
          secretName: gcp-credentials
---
apiVersion: v1
kind: Service
metadata:
  name: backend
  namespace: finplatform
spec:
  selector:
    app: backend
  ports:
  - name: http
    port: 8000
    targetPort: 8000
  - name: websocket
    port: 8765
    targetPort: 8765
  type: ClusterIP
```

### Frontend Deployment

`k8s/frontend-deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
  namespace: finplatform
spec:
  replicas: 2
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
      - name: frontend
        image: finplatform/frontend:latest
        ports:
        - containerPort: 80
        env:
        - name: REACT_APP_API_URL
          value: "https://api.finplatform.com"
        - name: REACT_APP_WS_URL
          value: "wss://ws.finplatform.com"
        resources:
          requests:
            memory: "256Mi"
            cpu: "200m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 10
          periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: frontend
  namespace: finplatform
spec:
  selector:
    app: frontend
  ports:
  - port: 80
    targetPort: 80
  type: ClusterIP
```

### Ingress Configuration

`k8s/ingress.yaml`:
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: finplatform-ingress
  namespace: finplatform
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/websocket-services: "backend"
spec:
  tls:
  - hosts:
    - app.finplatform.com
    - api.finplatform.com
    - ws.finplatform.com
    secretName: finplatform-tls
  rules:
  - host: app.finplatform.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend
            port:
              number: 80
  - host: api.finplatform.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend
            port:
              number: 8000
  - host: ws.finplatform.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend
            port:
              number: 8765
```

### Helm Chart

`helm/finplatform/Chart.yaml`:
```yaml
apiVersion: v2
name: finplatform
description: Financial Time Series Analysis Platform
type: application
version: 1.0.0
appVersion: "1.0"
dependencies:
  - name: postgresql
    version: 12.1.0
    repository: https://charts.bitnami.com/bitnami
  - name: redis
    version: 17.0.0
    repository: https://charts.bitnami.com/bitnami
```

`helm/finplatform/values.yaml`:
```yaml
# Global settings
global:
  storageClass: "fast-ssd"
  
# Backend configuration
backend:
  replicaCount: 3
  image:
    repository: finplatform/backend
    tag: latest
    pullPolicy: Always
  
  resources:
    requests:
      memory: "512Mi"
      cpu: "500m"
      nvidia.com/gpu: 1
    limits:
      memory: "2Gi"
      cpu: "2000m"
      nvidia.com/gpu: 1
  
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: 80

# Frontend configuration
frontend:
  replicaCount: 2
  image:
    repository: finplatform/frontend
    tag: latest
    pullPolicy: Always
  
  resources:
    requests:
      memory: "256Mi"
      cpu: "200m"
    limits:
      memory: "512Mi"
      cpu: "500m"

# Database configuration
postgresql:
  auth:
    database: finplatform
    username: finplatform
    existingSecret: finplatform-db-secret
  
  primary:
    persistence:
      enabled: true
      size: 50Gi
      storageClass: fast-ssd
    
    resources:
      requests:
        memory: "1Gi"
        cpu: "1000m"
      limits:
        memory: "4Gi"
        cpu: "4000m"

# Redis configuration
redis:
  auth:
    enabled: true
    existingSecret: finplatform-redis-secret
  
  master:
    persistence:
      enabled: true
      size: 10Gi
      storageClass: fast-ssd
    
    resources:
      requests:
        memory: "256Mi"
        cpu: "250m"
      limits:
        memory: "1Gi"
        cpu: "1000m"

# Ingress configuration
ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: app.finplatform.com
      paths:
        - path: /
          service: frontend
    - host: api.finplatform.com
      paths:
        - path: /
          service: backend
    - host: ws.finplatform.com
      paths:
        - path: /
          service: backend-ws
  tls:
    - secretName: finplatform-tls
      hosts:
        - app.finplatform.com
        - api.finplatform.com
        - ws.finplatform.com
```

### Deployment Commands

```bash
# Create namespace
kubectl create namespace finplatform

# Create secrets
kubectl create secret generic finplatform-secrets \
  --from-literal=database-url=$DATABASE_URL \
  --from-literal=redis-url=$REDIS_URL \
  --from-literal=jwt-secret=$JWT_SECRET \
  -n finplatform

# Create GCP credentials secret
kubectl create secret generic gcp-credentials \
  --from-file=gcp-service-account.json \
  -n finplatform

# Deploy with Helm
helm install finplatform ./helm/finplatform \
  --namespace finplatform \
  --values ./helm/finplatform/values.production.yaml

# Upgrade deployment
helm upgrade finplatform ./helm/finplatform \
  --namespace finplatform \
  --values ./helm/finplatform/values.production.yaml

# Check status
kubectl get all -n finplatform
```

## Cloud Platform Deployment

### Google Cloud Platform

#### GKE Cluster Setup

```bash
# Create GKE cluster with GPU nodes
gcloud container clusters create finplatform-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 10 \
  --enable-autorepair \
  --enable-autoupgrade

# Install NVIDIA GPU drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# Get credentials
gcloud container clusters get-credentials finplatform-cluster --zone us-central1-a
```

#### Cloud SQL Setup

```bash
# Create Cloud SQL instance
gcloud sql instances create finplatform-db \
  --database-version=POSTGRES_14 \
  --tier=db-custom-4-16384 \
  --region=us-central1 \
  --network=default \
  --backup-start-time=03:00 \
  --enable-point-in-time-recovery

# Create database and user
gcloud sql databases create finplatform --instance=finplatform-db
gcloud sql users create finplatform --instance=finplatform-db --password=$DB_PASSWORD
```

#### Memorystore Redis

```bash
# Create Redis instance
gcloud redis instances create finplatform-cache \
  --size=5 \
  --region=us-central1 \
  --redis-version=redis_7_0 \
  --enable-auth \
  --auth-string=$REDIS_PASSWORD
```

### AWS Deployment

#### EKS Cluster

```bash
# Create EKS cluster
eksctl create cluster \
  --name finplatform \
  --version 1.27 \
  --region us-east-1 \
  --nodegroup-name gpu-nodes \
  --node-type p3.2xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 10 \
  --managed
```

#### RDS PostgreSQL

```bash
# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier finplatform-db \
  --db-instance-class db.r6g.xlarge \
  --engine postgres \
  --engine-version 14.8 \
  --master-username finplatform \
  --master-user-password $DB_PASSWORD \
  --allocated-storage 100 \
  --storage-encrypted \
  --backup-retention-period 7 \
  --multi-az
```

## Security Hardening

### SSL/TLS Configuration

`nginx/nginx.prod.conf`:
```nginx
server {
    listen 443 ssl http2;
    server_name api.finplatform.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    location / {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /ws {
        proxy_pass http://backend:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Network Policies

`k8s/network-policy.yaml`:
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: backend-network-policy
  namespace: finplatform
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    - podSelector:
        matchLabels:
          app: nginx
    ports:
    - protocol: TCP
      port: 8000
    - protocol: TCP
      port: 8765
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

## Monitoring and Alerting

### Prometheus Configuration

`monitoring/prometheus.yml`:
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

rule_files:
  - "alerts.yml"

scrape_configs:
  - job_name: 'backend-api'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
  
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
  
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
  
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

### Alert Rules

`monitoring/alerts.yml`:
```yaml
groups:
  - name: platform_alerts
    rules:
      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 90% (current value: {{ $value }})"
      
      - alert: BackendDown
        expr: up{job="backend-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Backend API is down"
          description: "Backend API has been down for more than 1 minute"
      
      - alert: DatabaseConnectionFailure
        expr: pg_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failed"
          description: "Cannot connect to PostgreSQL database"
      
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate"
          description: "Error rate is above 5% (current value: {{ $value }})"
```

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup.sh

# Variables
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/postgres"
DB_NAME="finplatform"

# Create backup
pg_dump -h postgres -U $DB_USER -d $DB_NAME | gzip > $BACKUP_DIR/backup_$DATE.sql.gz

# Upload to cloud storage
gsutil cp $BACKUP_DIR/backup_$DATE.sql.gz gs://finplatform-backups/postgres/

# Clean old backups (keep last 30 days)
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +30 -delete
```

### Automated Backup CronJob

`k8s/backup-cronjob.yaml`:
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
  namespace: finplatform
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: postgres-backup
            image: postgres:14-alpine
            command:
            - /bin/sh
            - -c
            - |
              DATE=$(date +%Y%m%d_%H%M%S)
              pg_dump -h postgres -U $POSTGRES_USER -d finplatform | gzip > /backup/backup_$DATE.sql.gz
              # Upload to cloud storage
            env:
            - name: POSTGRES_USER
              valueFrom:
                secretKeyRef:
                  name: postgres-secret
                  key: username
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: postgres-secret
                  key: password
            volumeMounts:
            - name: backup
              mountPath: /backup
          volumes:
          - name: backup
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
```

## Performance Optimization

### Resource Allocation

```yaml
# Optimized resource requests/limits
backend:
  resources:
    requests:
      memory: "1Gi"      # Increased for production
      cpu: "1000m"       # 1 full CPU
      nvidia.com/gpu: 1  # GPU for ML/NN
    limits:
      memory: "4Gi"      # Allow burst
      cpu: "4000m"       # 4 CPUs max
      nvidia.com/gpu: 1

frontend:
  resources:
    requests:
      memory: "512Mi"
      cpu: "500m"
    limits:
      memory: "1Gi"
      cpu: "1000m"
```

### Caching Strategy

```yaml
# Redis configuration for production
redis:
  maxmemory: 4gb
  maxmemory-policy: allkeys-lru
  save: "900 1 300 10 60 10000"
  appendonly: yes
  appendfsync: everysec
```

### Database Optimization

```sql
-- Indexes for performance
CREATE INDEX idx_trades_symbol_timestamp ON trades(symbol, timestamp);
CREATE INDEX idx_positions_user_symbol ON positions(user_id, symbol);
CREATE INDEX idx_orders_status_created ON orders(status, created_at);

-- Partitioning for time-series data
CREATE TABLE trades_2024_01 PARTITION OF trades
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Connection pooling
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
```

## Deployment Checklist

### Pre-deployment
- [ ] All tests passing
- [ ] Security scan completed
- [ ] Environment variables configured
- [ ] SSL certificates obtained
- [ ] Database migrations ready
- [ ] Backup strategy tested

### Deployment
- [ ] Deploy database migrations
- [ ] Deploy backend services
- [ ] Deploy frontend services
- [ ] Configure load balancer
- [ ] Update DNS records
- [ ] Enable monitoring

### Post-deployment
- [ ] Verify all services healthy
- [ ] Test critical user flows
- [ ] Check monitoring dashboards
- [ ] Verify backup jobs
- [ ] Load testing
- [ ] Update documentation

## Rollback Procedure

```bash
# Kubernetes rollback
kubectl rollout undo deployment/backend -n finplatform
kubectl rollout undo deployment/frontend -n finplatform

# Helm rollback
helm rollback finplatform 1 -n finplatform

# Database rollback
psql -h postgres -U finplatform -d finplatform < backup_20240125.sql
```

## Maintenance Mode

```bash
# Enable maintenance mode
kubectl patch ingress finplatform-ingress -n finplatform \
  --type='json' -p='[{"op": "add", "path": "/metadata/annotations/nginx.ingress.kubernetes.io~1configuration-snippet", "value": "return 503;"}]'

# Disable maintenance mode
kubectl patch ingress finplatform-ingress -n finplatform \
  --type='json' -p='[{"op": "remove", "path": "/metadata/annotations/nginx.ingress.kubernetes.io~1configuration-snippet"}]'
```