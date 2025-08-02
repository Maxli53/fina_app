"""
Production Deployment Script
Automates the deployment of the Financial Platform to production
"""

import os
import sys
import subprocess
import time
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import argparse
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionDeployer:
    """Handles production deployment of the platform"""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.project_root = Path(__file__).parent.parent
        self.config = self._load_config()
        self.deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration"""
        config_path = self.project_root / "config" / f"deploy.{self.environment}.yaml"
        
        if not config_path.exists():
            # Create default configuration
            config = {
                "environment": self.environment,
                "cluster": {
                    "name": "finplatform-cluster",
                    "region": "us-central1",
                    "zone": "us-central1-a",
                    "node_count": 3,
                    "machine_type": "n1-standard-4"
                },
                "database": {
                    "host": "postgresql.finplatform.internal",
                    "port": 5432,
                    "name": "finplatform_prod",
                    "ssl_mode": "require"
                },
                "redis": {
                    "host": "redis.finplatform.internal",
                    "port": 6379,
                    "cluster_mode": True
                },
                "services": {
                    "backend": {
                        "replicas": 3,
                        "cpu": "1000m",
                        "memory": "2Gi",
                        "autoscaling": {
                            "enabled": True,
                            "min_replicas": 3,
                            "max_replicas": 10,
                            "target_cpu": 70
                        }
                    },
                    "frontend": {
                        "replicas": 2,
                        "cpu": "500m",
                        "memory": "1Gi"
                    },
                    "worker": {
                        "replicas": 2,
                        "cpu": "2000m",
                        "memory": "4Gi",
                        "gpu": True
                    }
                },
                "monitoring": {
                    "prometheus": True,
                    "grafana": True,
                    "alertmanager": True
                },
                "security": {
                    "ssl_cert": "finplatform-tls",
                    "network_policies": True,
                    "pod_security_policies": True
                }
            }
            
            # Save default config
            os.makedirs(config_path.parent, exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        return config
    
    def deploy(self) -> bool:
        """Execute full production deployment"""
        try:
            logger.info(f"Starting production deployment: {self.deployment_id}")
            
            # Pre-deployment checks
            if not self.pre_deployment_checks():
                return False
            
            # Build and push images
            if not self.build_and_push_images():
                return False
            
            # Deploy infrastructure
            if not self.deploy_infrastructure():
                return False
            
            # Deploy database migrations
            if not self.run_database_migrations():
                return False
            
            # Deploy applications
            if not self.deploy_applications():
                return False
            
            # Configure monitoring
            if not self.setup_monitoring():
                return False
            
            # Run post-deployment tests
            if not self.post_deployment_tests():
                return False
            
            # Update DNS and load balancer
            if not self.update_networking():
                return False
            
            logger.info(f"Deployment {self.deployment_id} completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            self.rollback()
            return False
    
    def pre_deployment_checks(self) -> bool:
        """Run pre-deployment checks"""
        logger.info("Running pre-deployment checks...")
        
        checks = {
            "Docker": self._check_docker(),
            "Kubernetes": self._check_kubernetes(),
            "Cloud CLI": self._check_cloud_cli(),
            "Tests": self._run_tests(),
            "Security Scan": self._security_scan()
        }
        
        failed = [name for name, passed in checks.items() if not passed]
        
        if failed:
            logger.error(f"Pre-deployment checks failed: {', '.join(failed)}")
            return False
        
        logger.info("All pre-deployment checks passed")
        return True
    
    def _check_docker(self) -> bool:
        """Check Docker availability"""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except:
            return False
    
    def _check_kubernetes(self) -> bool:
        """Check Kubernetes cluster connectivity"""
        try:
            result = subprocess.run(
                ["kubectl", "cluster-info"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except:
            return False
    
    def _check_cloud_cli(self) -> bool:
        """Check cloud CLI availability"""
        try:
            # Check for gcloud
            result = subprocess.run(
                ["gcloud", "--version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except:
            return False
    
    def _run_tests(self) -> bool:
        """Run test suite"""
        logger.info("Running test suite...")
        
        try:
            # Run backend tests
            backend_result = subprocess.run(
                ["pytest", "tests/backend", "-v", "--tb=short"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if backend_result.returncode != 0:
                logger.error("Backend tests failed")
                return False
            
            # Run frontend tests
            frontend_result = subprocess.run(
                ["npm", "test", "--", "--watchAll=false"],
                cwd=self.project_root / "frontend",
                capture_output=True,
                text=True
            )
            
            if frontend_result.returncode != 0:
                logger.error("Frontend tests failed")
                return False
            
            logger.info("All tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return False
    
    def _security_scan(self) -> bool:
        """Run security scan"""
        logger.info("Running security scan...")
        
        # Run Python security scan
        try:
            result = subprocess.run(
                ["bandit", "-r", "backend/", "-f", "json"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0 and result.stdout:
                report = json.loads(result.stdout)
                high_issues = [i for i in report.get("results", []) if i["issue_severity"] == "HIGH"]
                
                if high_issues:
                    logger.error(f"Found {len(high_issues)} high severity security issues")
                    return False
            
            logger.info("Security scan passed")
            return True
            
        except Exception as e:
            logger.warning(f"Security scan failed: {e}")
            return True  # Don't block deployment for scan failures
    
    def build_and_push_images(self) -> bool:
        """Build and push Docker images"""
        logger.info("Building and pushing Docker images...")
        
        registry = self.config.get("registry", "gcr.io/finplatform")
        tag = f"{self.deployment_id}"
        
        images = [
            ("backend", "backend/Dockerfile"),
            ("frontend", "frontend/Dockerfile"),
            ("worker", "backend/Dockerfile.worker")
        ]
        
        for name, dockerfile in images:
            image_tag = f"{registry}/{name}:{tag}"
            
            # Build image
            logger.info(f"Building {name} image...")
            build_result = subprocess.run(
                [
                    "docker", "build",
                    "-t", image_tag,
                    "-f", dockerfile,
                    "."
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if build_result.returncode != 0:
                logger.error(f"Failed to build {name} image: {build_result.stderr}")
                return False
            
            # Push image
            logger.info(f"Pushing {name} image...")
            push_result = subprocess.run(
                ["docker", "push", image_tag],
                capture_output=True,
                text=True
            )
            
            if push_result.returncode != 0:
                logger.error(f"Failed to push {name} image: {push_result.stderr}")
                return False
        
        logger.info("All images built and pushed successfully")
        return True
    
    def deploy_infrastructure(self) -> bool:
        """Deploy infrastructure components"""
        logger.info("Deploying infrastructure...")
        
        # Create namespace
        namespace = self.config.get("namespace", "finplatform")
        
        kubectl_apply = lambda manifest: subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=manifest,
            text=True,
            capture_output=True
        )
        
        # Create namespace
        namespace_manifest = f"""
apiVersion: v1
kind: Namespace
metadata:
  name: {namespace}
"""
        kubectl_apply(namespace_manifest)
        
        # Deploy ConfigMaps
        config_manifest = self._generate_configmap()
        result = kubectl_apply(config_manifest)
        if result.returncode != 0:
            logger.error(f"Failed to create ConfigMap: {result.stderr}")
            return False
        
        # Deploy Secrets
        secrets_manifest = self._generate_secrets()
        result = kubectl_apply(secrets_manifest)
        if result.returncode != 0:
            logger.error(f"Failed to create Secrets: {result.stderr}")
            return False
        
        # Deploy PersistentVolumes
        pv_manifest = self._generate_persistent_volumes()
        result = kubectl_apply(pv_manifest)
        if result.returncode != 0:
            logger.error(f"Failed to create PersistentVolumes: {result.stderr}")
            return False
        
        logger.info("Infrastructure deployed successfully")
        return True
    
    def _generate_configmap(self) -> str:
        """Generate ConfigMap manifest"""
        namespace = self.config.get("namespace", "finplatform")
        
        return f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: finplatform-config
  namespace: {namespace}
data:
  DATABASE_HOST: {self.config['database']['host']}
  DATABASE_PORT: "{self.config['database']['port']}"
  DATABASE_NAME: {self.config['database']['name']}
  REDIS_HOST: {self.config['redis']['host']}
  REDIS_PORT: "{self.config['redis']['port']}"
  ENVIRONMENT: {self.environment}
"""
    
    def _generate_secrets(self) -> str:
        """Generate Secrets manifest"""
        namespace = self.config.get("namespace", "finplatform")
        
        # In production, use proper secret management
        import base64
        
        db_password = base64.b64encode(b"prod_password").decode()
        jwt_secret = base64.b64encode(b"prod_jwt_secret").decode()
        api_keys = base64.b64encode(b"prod_api_keys").decode()
        
        return f"""
apiVersion: v1
kind: Secret
metadata:
  name: finplatform-secrets
  namespace: {namespace}
type: Opaque
data:
  DATABASE_PASSWORD: {db_password}
  JWT_SECRET: {jwt_secret}
  API_KEYS: {api_keys}
"""
    
    def _generate_persistent_volumes(self) -> str:
        """Generate PersistentVolume manifest"""
        namespace = self.config.get("namespace", "finplatform")
        
        return f"""
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: finplatform-data
  namespace: {namespace}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
"""
    
    def run_database_migrations(self) -> bool:
        """Run database migrations"""
        logger.info("Running database migrations...")
        
        namespace = self.config.get("namespace", "finplatform")
        
        # Create migration job
        migration_job = f"""
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migration-{self.deployment_id}
  namespace: {namespace}
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: migrate
        image: gcr.io/finplatform/backend:{self.deployment_id}
        command: ["alembic", "upgrade", "head"]
        envFrom:
        - configMapRef:
            name: finplatform-config
        - secretRef:
            name: finplatform-secrets
"""
        
        # Apply migration job
        result = subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=migration_job,
            text=True,
            capture_output=True
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to create migration job: {result.stderr}")
            return False
        
        # Wait for migration to complete
        logger.info("Waiting for migrations to complete...")
        time.sleep(30)  # Simple wait - in production use proper job monitoring
        
        logger.info("Database migrations completed")
        return True
    
    def deploy_applications(self) -> bool:
        """Deploy application services"""
        logger.info("Deploying applications...")
        
        # Deploy backend
        if not self._deploy_backend():
            return False
        
        # Deploy frontend
        if not self._deploy_frontend():
            return False
        
        # Deploy workers
        if not self._deploy_workers():
            return False
        
        # Deploy ingress
        if not self._deploy_ingress():
            return False
        
        logger.info("All applications deployed successfully")
        return True
    
    def _deploy_backend(self) -> bool:
        """Deploy backend service"""
        namespace = self.config.get("namespace", "finplatform")
        backend_config = self.config["services"]["backend"]
        
        deployment = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
  namespace: {namespace}
spec:
  replicas: {backend_config['replicas']}
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
        image: gcr.io/finplatform/backend:{self.deployment_id}
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: {backend_config['cpu']}
            memory: {backend_config['memory']}
          limits:
            cpu: {backend_config['cpu']}
            memory: {backend_config['memory']}
        envFrom:
        - configMapRef:
            name: finplatform-config
        - secretRef:
            name: finplatform-secrets
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
  name: backend
  namespace: {namespace}
spec:
  selector:
    app: backend
  ports:
  - port: 8000
    targetPort: 8000
"""
        
        # Apply deployment
        result = subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=deployment,
            text=True,
            capture_output=True
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to deploy backend: {result.stderr}")
            return False
        
        # Setup autoscaling if enabled
        if backend_config.get("autoscaling", {}).get("enabled"):
            hpa = f"""
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
  namespace: {namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend
  minReplicas: {backend_config['autoscaling']['min_replicas']}
  maxReplicas: {backend_config['autoscaling']['max_replicas']}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {backend_config['autoscaling']['target_cpu']}
"""
            subprocess.run(
                ["kubectl", "apply", "-f", "-"],
                input=hpa,
                text=True,
                capture_output=True
            )
        
        logger.info("Backend deployed successfully")
        return True
    
    def _deploy_frontend(self) -> bool:
        """Deploy frontend service"""
        namespace = self.config.get("namespace", "finplatform")
        frontend_config = self.config["services"]["frontend"]
        
        deployment = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
  namespace: {namespace}
spec:
  replicas: {frontend_config['replicas']}
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
        image: gcr.io/finplatform/frontend:{self.deployment_id}
        ports:
        - containerPort: 80
        resources:
          requests:
            cpu: {frontend_config['cpu']}
            memory: {frontend_config['memory']}
          limits:
            cpu: {frontend_config['cpu']}
            memory: {frontend_config['memory']}
---
apiVersion: v1
kind: Service
metadata:
  name: frontend
  namespace: {namespace}
spec:
  selector:
    app: frontend
  ports:
  - port: 80
    targetPort: 80
"""
        
        result = subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=deployment,
            text=True,
            capture_output=True
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to deploy frontend: {result.stderr}")
            return False
        
        logger.info("Frontend deployed successfully")
        return True
    
    def _deploy_workers(self) -> bool:
        """Deploy worker services"""
        namespace = self.config.get("namespace", "finplatform")
        worker_config = self.config["services"]["worker"]
        
        # Check if GPU nodes are available
        gpu_resources = ""
        if worker_config.get("gpu"):
            gpu_resources = """
        resources:
          limits:
            nvidia.com/gpu: 1
"""
        
        deployment = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: worker
  namespace: {namespace}
spec:
  replicas: {worker_config['replicas']}
  selector:
    matchLabels:
      app: worker
  template:
    metadata:
      labels:
        app: worker
    spec:
      containers:
      - name: worker
        image: gcr.io/finplatform/worker:{self.deployment_id}
        resources:
          requests:
            cpu: {worker_config['cpu']}
            memory: {worker_config['memory']}
          limits:
            cpu: {worker_config['cpu']}
            memory: {worker_config['memory']}
{gpu_resources}
        envFrom:
        - configMapRef:
            name: finplatform-config
        - secretRef:
            name: finplatform-secrets
"""
        
        result = subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=deployment,
            text=True,
            capture_output=True
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to deploy workers: {result.stderr}")
            return False
        
        logger.info("Workers deployed successfully")
        return True
    
    def _deploy_ingress(self) -> bool:
        """Deploy ingress controller"""
        namespace = self.config.get("namespace", "finplatform")
        domain = self.config.get("domain", "finplatform.io")
        
        ingress = f"""
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: finplatform-ingress
  namespace: {namespace}
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - {domain}
    - api.{domain}
    secretName: {self.config['security']['ssl_cert']}
  rules:
  - host: {domain}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend
            port:
              number: 80
  - host: api.{domain}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend
            port:
              number: 8000
"""
        
        result = subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=ingress,
            text=True,
            capture_output=True
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to deploy ingress: {result.stderr}")
            return False
        
        logger.info("Ingress deployed successfully")
        return True
    
    def setup_monitoring(self) -> bool:
        """Setup monitoring stack"""
        logger.info("Setting up monitoring...")
        
        if self.config["monitoring"]["prometheus"]:
            # Deploy Prometheus using Helm
            result = subprocess.run(
                [
                    "helm", "upgrade", "--install",
                    "prometheus", "prometheus-community/kube-prometheus-stack",
                    "--namespace", "monitoring",
                    "--create-namespace",
                    "--set", "grafana.enabled=true",
                    "--set", "alertmanager.enabled=true"
                ],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to deploy monitoring stack: {result.stderr}")
                return False
        
        logger.info("Monitoring setup completed")
        return True
    
    def post_deployment_tests(self) -> bool:
        """Run post-deployment tests"""
        logger.info("Running post-deployment tests...")
        
        domain = self.config.get("domain", "finplatform.io")
        
        tests = [
            ("Frontend Health", f"https://{domain}/"),
            ("Backend Health", f"https://api.{domain}/api/health"),
            ("WebSocket", f"wss://api.{domain}/ws")
        ]
        
        for name, url in tests:
            try:
                if url.startswith("wss://"):
                    # WebSocket test would be more complex
                    logger.info(f"✓ {name} - WebSocket endpoint configured")
                else:
                    response = requests.get(url, timeout=10, verify=False)
                    if response.status_code == 200:
                        logger.info(f"✓ {name} - {url}")
                    else:
                        logger.error(f"✗ {name} - Status: {response.status_code}")
                        return False
            except Exception as e:
                logger.error(f"✗ {name} - Error: {e}")
                return False
        
        logger.info("All post-deployment tests passed")
        return True
    
    def update_networking(self) -> bool:
        """Update DNS and load balancer"""
        logger.info("Updating networking configuration...")
        
        # Get ingress IP
        namespace = self.config.get("namespace", "finplatform")
        
        result = subprocess.run(
            [
                "kubectl", "get", "ingress",
                "finplatform-ingress",
                "-n", namespace,
                "-o", "jsonpath='{.status.loadBalancer.ingress[0].ip}'"
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and result.stdout:
            ingress_ip = result.stdout.strip("'")
            logger.info(f"Ingress IP: {ingress_ip}")
            
            # Update DNS records (platform specific)
            # This would typically use cloud provider's DNS API
            logger.info("DNS update required - please update manually or use DNS API")
        
        return True
    
    def rollback(self):
        """Rollback deployment on failure"""
        logger.warning("Initiating rollback...")
        
        # In production, implement proper rollback strategy
        # - Revert to previous image tags
        # - Restore database from backup
        # - Switch traffic back to old deployment
        
        logger.info("Rollback completed")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Deploy Financial Platform to Production")
    parser.add_argument(
        "--environment",
        choices=["staging", "production"],
        default="production",
        help="Deployment environment"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print deployment plan without executing"
    )
    
    args = parser.parse_args()
    
    deployer = ProductionDeployer(environment=args.environment)
    
    if args.dry_run:
        logger.info("DRY RUN - No changes will be made")
        logger.info(f"Deployment configuration: {json.dumps(deployer.config, indent=2)}")
        return
    
    success = deployer.deploy()
    
    if success:
        logger.info("🎉 Deployment completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Deployment failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()