# Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Financial Platform to production environments using Kubernetes, Terraform, and automated deployment scripts.

## Architecture Overview

```
Production Infrastructure
├── Cloud Provider (GCP/AWS)
│   ├── Kubernetes Cluster (GKE/EKS)
│   ├── Managed Database (Cloud SQL/RDS)
│   ├── Cache Layer (Memorystore/ElastiCache)
│   └── Load Balancers
├── Containerization
│   ├── Backend Services
│   ├── Frontend Application
│   └── Worker Processes
├── Monitoring & Observability
│   ├── Prometheus & Grafana
│   ├── Logging (ELK Stack)
│   └── Distributed Tracing
└── Security & Compliance
    ├── SSL/TLS Certificates
    ├── Network Policies
    └── Secret Management
```

## Prerequisites

### Required Tools

```bash
# Check required tools
./scripts/check_prerequisites.sh

# Required versions:
- Docker >= 20.10
- Kubernetes >= 1.25
- Terraform >= 1.0
- Helm >= 3.0
- gcloud CLI (for GCP)
- aws CLI (for AWS)
```

### Access Requirements

1. **Cloud Provider Access**
   - Project/Account admin permissions
   - Billing enabled
   - Required APIs enabled

2. **Domain & DNS**
   - Domain name registered
   - DNS management access
   - SSL certificate ready

3. **External Services**
   - API keys configured
   - Database credentials
   - Monitoring services

## Infrastructure Setup

### 1. Terraform Configuration

**Initialize Terraform:**
```bash
cd infrastructure/terraform
terraform init
```

**Configure Variables:**
```hcl
# terraform.tfvars
project_id = "finplatform-prod"
region     = "us-central1"
zone       = "us-central1-a"
environment = "production"

# Cluster configuration
cluster_config = {
  node_count     = 3
  machine_type   = "n1-standard-4"
  disk_size_gb   = 100
  enable_gpu     = true
}

# Database configuration
database_config = {
  tier              = "db-custom-4-16384"
  disk_size_gb      = 100
  backup_enabled    = true
  high_availability = true
}
```

**Deploy Infrastructure:**
```bash
# Plan deployment
terraform plan -out=tfplan

# Apply changes
terraform apply tfplan

# Output important values
terraform output -json > infrastructure_outputs.json
```

### 2. Kubernetes Cluster Setup

**Configure kubectl:**
```bash
# GCP
gcloud container clusters get-credentials finplatform-cluster-production \
  --zone us-central1-a --project finplatform-prod

# AWS
aws eks update-kubeconfig --name finplatform-cluster-production \
  --region us-east-1
```

**Create Namespaces:**
```bash
kubectl create namespace finplatform
kubectl create namespace monitoring
kubectl create namespace ingress
```

**Install Ingress Controller:**
```bash
# Install NGINX Ingress
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install nginx-ingress ingress-nginx/ingress-nginx \
  --namespace ingress \
  --set controller.service.type=LoadBalancer \
  --set controller.metrics.enabled=true
```

**Install Cert Manager:**
```bash
# Install cert-manager for SSL
helm repo add jetstack https://charts.jetstack.io
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager --create-namespace \
  --set installCRDs=true
```

## Application Deployment

### 1. Build and Push Images

**Automated Build:**
```bash
# Run deployment script
python scripts/deploy_production.py --environment production

# Or manually build
docker build -t gcr.io/finplatform-prod/backend:v2.0.0 -f backend/Dockerfile .
docker build -t gcr.io/finplatform-prod/frontend:v2.0.0 -f frontend/Dockerfile .
docker build -t gcr.io/finplatform-prod/worker:v2.0.0 -f backend/Dockerfile.worker .

# Push images
docker push gcr.io/finplatform-prod/backend:v2.0.0
docker push gcr.io/finplatform-prod/frontend:v2.0.0
docker push gcr.io/finplatform-prod/worker:v2.0.0
```

### 2. Configure Secrets

**Create Secrets:**
```bash
# Database credentials
kubectl create secret generic database-credentials \
  --from-literal=host=10.0.0.5 \
  --from-literal=username=finplatform \
  --from-literal=password='${DB_PASSWORD}' \
  -n finplatform

# API keys
kubectl create secret generic api-keys \
  --from-literal=openai='${OPENAI_API_KEY}' \
  --from-literal=serpapi='${SERPAPI_KEY}' \
  --from-literal=iqfeed='${IQFEED_CREDENTIALS}' \
  -n finplatform

# Application secrets
kubectl create secret generic app-secrets \
  --from-literal=jwt-secret='${JWT_SECRET}' \
  --from-literal=encryption-key='${ENCRYPTION_KEY}' \
  -n finplatform
```

### 3. Deploy Applications

**Deploy Backend:**
```bash
kubectl apply -f infrastructure/k8s/production/backend-deployment.yaml
```

**Deploy Frontend:**
```yaml
# frontend-deployment.yaml
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
        image: gcr.io/finplatform-prod/frontend:v2.0.0
        ports:
        - containerPort: 80
        env:
        - name: REACT_APP_API_URL
          value: "https://api.finplatform.io"
        - name: REACT_APP_WS_URL
          value: "wss://api.finplatform.io/ws"
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 1000m
            memory: 2Gi
```

**Deploy Workers:**
```yaml
# worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: worker
  namespace: finplatform
spec:
  replicas: 2
  selector:
    matchLabels:
      app: worker
  template:
    metadata:
      labels:
        app: worker
    spec:
      nodeSelector:
        gpu: "true"  # For GPU workloads
      containers:
      - name: worker
        image: gcr.io/finplatform-prod/worker:v2.0.0
        resources:
          requests:
            cpu: 2000m
            memory: 4Gi
            nvidia.com/gpu: 1
          limits:
            cpu: 4000m
            memory: 8Gi
            nvidia.com/gpu: 1
```

### 4. Configure Ingress

**Production Ingress:**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: finplatform-ingress
  namespace: finplatform
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
spec:
  tls:
  - hosts:
    - finplatform.io
    - api.finplatform.io
    secretName: finplatform-tls
  rules:
  - host: finplatform.io
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend
            port:
              number: 80
  - host: api.finplatform.io
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend
            port:
              number: 8000
```

## Database Migration

### 1. Backup Existing Data

```bash
# Create backup
pg_dump -h old-host -U username -d finplatform > backup.sql

# Verify backup
pg_restore --list backup.sql | head -20
```

### 2. Run Migrations

```bash
# Create migration job
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migration
  namespace: finplatform
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: migrate
        image: gcr.io/finplatform-prod/backend:v2.0.0
        command: ["alembic", "upgrade", "head"]
        envFrom:
        - secretRef:
            name: database-credentials
EOF

# Monitor migration
kubectl logs -f job/db-migration -n finplatform
```

## Monitoring Setup

### 1. Deploy Prometheus & Grafana

```bash
# Add Prometheus Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts

# Install kube-prometheus-stack
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=100Gi \
  --set grafana.adminPassword='${GRAFANA_PASSWORD}'
```

### 2. Configure Dashboards

**Import Dashboards:**
```bash
# Apply custom dashboards
kubectl apply -f infrastructure/k8s/monitoring/dashboards/

# Available dashboards:
- Platform Overview
- API Performance
- Trading Metrics
- System Resources
- Business KPIs
```

### 3. Setup Alerts

**Alert Rules:**
```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: finplatform-alerts
  namespace: monitoring
spec:
  groups:
  - name: platform
    rules:
    - alert: HighErrorRate
      expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
      for: 5m
      annotations:
        summary: "High error rate detected"
        
    - alert: HighLatency
      expr: histogram_quantile(0.95, http_request_duration_seconds_bucket) > 1
      for: 5m
      annotations:
        summary: "High API latency detected"
        
    - alert: PodCrashLooping
      expr: rate(kube_pod_container_status_restarts_total[1h]) > 5
      for: 5m
      annotations:
        summary: "Pod is crash looping"
```

## Security Hardening

### 1. Network Policies

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
    - namespaceSelector:
        matchLabels:
          name: ingress
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
    - protocol: TCP
      port: 443   # HTTPS
```

### 2. Pod Security Policies

```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: restricted
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

### 3. Secret Rotation

```bash
# Rotate database password
./scripts/rotate_secrets.sh database-password

# Rotate API keys
./scripts/rotate_secrets.sh api-keys

# Update JWT secret
kubectl create secret generic app-secrets \
  --from-literal=jwt-secret='${NEW_JWT_SECRET}' \
  --dry-run=client -o yaml | kubectl apply -f -
```

## Performance Optimization

### 1. Horizontal Pod Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
  namespace: finplatform
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend
  minReplicas: 3
  maxReplicas: 20
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
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
```

### 2. Cluster Autoscaling

```bash
# Enable cluster autoscaler
gcloud container clusters update finplatform-cluster-production \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 50 \
  --zone us-central1-a
```

### 3. CDN Configuration

```bash
# Configure Cloud CDN
gcloud compute backend-buckets create finplatform-static \
  --gcs-bucket-name=finplatform-static-assets

gcloud compute url-maps create finplatform-cdn \
  --default-backend-bucket=finplatform-static
```

## Deployment Checklist

### Pre-Deployment

- [ ] All tests passing
- [ ] Security scan completed
- [ ] Database backup created
- [ ] Rollback plan documented
- [ ] Team notified
- [ ] Monitoring alerts configured

### Deployment Steps

- [ ] Deploy infrastructure changes
- [ ] Run database migrations
- [ ] Deploy backend services
- [ ] Deploy frontend application
- [ ] Deploy worker processes
- [ ] Update DNS records
- [ ] Verify SSL certificates
- [ ] Test all endpoints

### Post-Deployment

- [ ] Verify application health
- [ ] Check monitoring dashboards
- [ ] Run smoke tests
- [ ] Monitor error rates
- [ ] Check performance metrics
- [ ] Document any issues

## Rollback Procedures

### Quick Rollback

```bash
# Rollback deployment
kubectl rollout undo deployment/backend -n finplatform
kubectl rollout undo deployment/frontend -n finplatform

# Check rollback status
kubectl rollout status deployment/backend -n finplatform
```

### Full Rollback

```bash
# Run full rollback script
./scripts/rollback_deployment.sh --version v1.9.0

# Restore database if needed
pg_restore -h production-host -U username -d finplatform backup.sql
```

## Troubleshooting

### Common Issues

**1. Pod Not Starting**
```bash
# Check pod status
kubectl describe pod <pod-name> -n finplatform

# Check logs
kubectl logs <pod-name> -n finplatform --previous
```

**2. Database Connection Issues**
```bash
# Test connection from pod
kubectl exec -it <backend-pod> -n finplatform -- \
  psql -h $DATABASE_HOST -U $DATABASE_USER -d $DATABASE_NAME
```

**3. High Memory Usage**
```bash
# Check resource usage
kubectl top pods -n finplatform

# Increase limits if needed
kubectl edit deployment backend -n finplatform
```

### Debug Mode

```bash
# Enable debug logging
kubectl set env deployment/backend LOG_LEVEL=DEBUG -n finplatform

# Port forward for debugging
kubectl port-forward svc/backend 8000:8000 -n finplatform
```

## Maintenance

### Regular Tasks

**Daily:**
- Check application health
- Review error logs
- Monitor resource usage

**Weekly:**
- Review security alerts
- Check backup status
- Update dependencies

**Monthly:**
- Security patching
- Performance review
- Cost optimization

### Backup Strategy

```bash
# Automated daily backups
0 2 * * * /scripts/backup_production.sh

# Backup components:
- Database (full + incremental)
- Application state
- Configuration files
- User uploads
```

## Disaster Recovery

### RTO/RPO Targets

- **RTO (Recovery Time Objective)**: 2 hours
- **RPO (Recovery Point Objective)**: 1 hour

### DR Procedures

1. **Database Recovery**
   ```bash
   # Restore from latest backup
   ./scripts/restore_database.sh --timestamp 2024-01-15T14:00:00
   ```

2. **Application Recovery**
   ```bash
   # Deploy to DR region
   ./scripts/deploy_dr.sh --region us-west1
   ```

3. **Data Verification**
   ```bash
   # Run integrity checks
   ./scripts/verify_recovery.sh
   ```

## Cost Optimization

### Resource Optimization

```bash
# Analyze resource usage
./scripts/analyze_resources.sh

# Recommendations:
- Use preemptible nodes for workers
- Enable cluster autoscaling
- Optimize container sizes
- Use regional persistent disks
```

### Cost Monitoring

```bash
# Set up budget alerts
gcloud billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="FinPlatform Production" \
  --budget-amount=5000 \
  --threshold-rule=percent=90
```

## Support

### Monitoring URLs

- Grafana: https://monitoring.finplatform.io
- Prometheus: https://prometheus.finplatform.io
- Logs: https://logs.finplatform.io

### Emergency Contacts

- On-call Engineer: +1-xxx-xxx-xxxx
- Platform Team: platform@finplatform.io
- Security Team: security@finplatform.io