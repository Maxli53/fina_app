# Main Terraform configuration for Financial Platform infrastructure

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
  
  backend "gcs" {
    bucket = "finplatform-terraform-state"
    prefix = "prod/state"
  }
}

# Provider configurations
provider "google" {
  project = var.project_id
  region  = var.region
}

provider "kubernetes" {
  host                   = google_container_cluster.primary.endpoint
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.primary.master_auth.0.cluster_ca_certificate)
}

provider "helm" {
  kubernetes {
    host                   = google_container_cluster.primary.endpoint
    token                  = data.google_client_config.default.access_token
    cluster_ca_certificate = base64decode(google_container_cluster.primary.master_auth.0.cluster_ca_certificate)
  }
}

# Data sources
data "google_client_config" "default" {}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

# Networking
resource "google_compute_network" "vpc" {
  name                    = "finplatform-vpc-${var.environment}"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "subnet" {
  name          = "finplatform-subnet-${var.environment}"
  ip_cidr_range = "10.0.0.0/16"
  region        = var.region
  network       = google_compute_network.vpc.id
  
  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.1.0.0/16"
  }
  
  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.2.0.0/16"
  }
}

# GKE Cluster
resource "google_container_cluster" "primary" {
  name     = "finplatform-cluster-${var.environment}"
  location = var.zone
  
  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1
  
  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name
  
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }
  
  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }
  
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  
  addons_config {
    horizontal_pod_autoscaling {
      disabled = false
    }
    
    http_load_balancing {
      disabled = false
    }
    
    network_policy_config {
      disabled = false
    }
  }
  
  cluster_autoscaling {
    enabled = true
    resource_limits {
      resource_type = "cpu"
      minimum       = 4
      maximum       = 100
    }
    resource_limits {
      resource_type = "memory"
      minimum       = 16
      maximum       = 400
    }
  }
}

# Node Pools
resource "google_container_node_pool" "primary_nodes" {
  name       = "primary-node-pool"
  location   = var.zone
  cluster    = google_container_cluster.primary.name
  node_count = 3
  
  autoscaling {
    min_node_count = 3
    max_node_count = 10
  }
  
  node_config {
    preemptible  = false
    machine_type = "n1-standard-4"
    
    service_account = google_service_account.kubernetes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    labels = {
      env  = var.environment
      pool = "primary"
    }
    
    tags = ["finplatform", "primary"]
    
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

resource "google_container_node_pool" "gpu_nodes" {
  name       = "gpu-node-pool"
  location   = var.zone
  cluster    = google_container_cluster.primary.name
  node_count = 1
  
  autoscaling {
    min_node_count = 0
    max_node_count = 3
  }
  
  node_config {
    preemptible  = true
    machine_type = "n1-standard-4"
    
    guest_accelerator {
      type  = "nvidia-tesla-t4"
      count = 1
    }
    
    service_account = google_service_account.kubernetes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    labels = {
      env  = var.environment
      pool = "gpu"
    }
    
    tags = ["finplatform", "gpu"]
    
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    
    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }
  }
  
  management {
    auto_repair  = true
    auto_upgrade = false # GPU drivers might need manual updates
  }
}

# Service Account
resource "google_service_account" "kubernetes" {
  account_id   = "finplatform-k8s-${var.environment}"
  display_name = "FinPlatform Kubernetes Service Account"
}

resource "google_project_iam_member" "kubernetes_roles" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/storage.objectViewer"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.kubernetes.email}"
}

# Cloud SQL (PostgreSQL)
resource "google_sql_database_instance" "postgres" {
  name             = "finplatform-db-${var.environment}"
  database_version = "POSTGRES_14"
  region           = var.region
  
  settings {
    tier = "db-custom-4-16384"
    
    disk_size         = 100
    disk_type         = "PD_SSD"
    disk_autoresize   = true
    
    backup_configuration {
      enabled                        = true
      start_time                     = "02:00"
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
      
      backup_retention_settings {
        retained_backups = 30
        retention_unit   = "COUNT"
      }
    }
    
    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.vpc.id
      require_ssl     = true
    }
    
    database_flags {
      name  = "max_connections"
      value = "200"
    }
    
    insights_config {
      query_insights_enabled  = true
      query_string_length     = 1024
      record_application_tags = true
      record_client_address   = true
    }
  }
}

resource "google_sql_database" "finplatform" {
  name     = "finplatform"
  instance = google_sql_database_instance.postgres.name
}

resource "google_sql_user" "app_user" {
  name     = "finplatform"
  instance = google_sql_database_instance.postgres.name
  password = random_password.db_password.result
}

resource "random_password" "db_password" {
  length  = 32
  special = true
}

# Redis (Memorystore)
resource "google_redis_instance" "cache" {
  name           = "finplatform-redis-${var.environment}"
  tier           = "STANDARD_HA"
  memory_size_gb = 5
  region         = var.region
  
  location_id             = var.zone
  alternative_location_id = "${var.region}-b"
  
  authorized_network = google_compute_network.vpc.id
  connect_mode       = "PRIVATE_SERVICE_ACCESS"
  
  redis_version = "REDIS_6_X"
  display_name  = "FinPlatform Redis Cache"
  
  redis_configs = {
    maxmemory-policy = "allkeys-lru"
  }
}

# Cloud Storage
resource "google_storage_bucket" "data" {
  name          = "finplatform-data-${var.environment}-${var.project_id}"
  location      = var.region
  force_destroy = false
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
  
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }
}

resource "google_storage_bucket" "backups" {
  name          = "finplatform-backups-${var.environment}-${var.project_id}"
  location      = var.region
  force_destroy = false
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type = "Delete"
    }
  }
}

# Cloud CDN
resource "google_compute_backend_bucket" "static_assets" {
  name        = "finplatform-static-${var.environment}"
  bucket_name = google_storage_bucket.static.name
  enable_cdn  = true
  
  cdn_policy {
    cache_mode = "CACHE_ALL_STATIC"
    default_ttl = 3600
    max_ttl     = 86400
  }
}

resource "google_storage_bucket" "static" {
  name          = "finplatform-static-${var.environment}-${var.project_id}"
  location      = var.region
  force_destroy = true
  
  uniform_bucket_level_access = true
  
  website {
    main_page_suffix = "index.html"
    not_found_page   = "404.html"
  }
  
  cors {
    origin          = ["https://finplatform.io"]
    method          = ["GET", "HEAD"]
    response_header = ["*"]
    max_age_seconds = 3600
  }
}

# Load Balancer
resource "google_compute_global_address" "default" {
  name = "finplatform-lb-ip-${var.environment}"
}

# SSL Certificate
resource "google_compute_managed_ssl_certificate" "default" {
  name = "finplatform-ssl-${var.environment}"
  
  managed {
    domains = ["finplatform.io", "*.finplatform.io"]
  }
}

# Firewall Rules
resource "google_compute_firewall" "allow_health_checks" {
  name    = "finplatform-allow-health-checks-${var.environment}"
  network = google_compute_network.vpc.name
  
  allow {
    protocol = "tcp"
    ports    = ["80", "443", "8080"]
  }
  
  source_ranges = [
    "35.191.0.0/16",
    "130.211.0.0/22"
  ]
  
  target_tags = ["finplatform"]
}

# Monitoring
resource "google_monitoring_notification_channel" "email" {
  display_name = "FinPlatform Email Alerts"
  type         = "email"
  
  labels = {
    email_address = "alerts@finplatform.io"
  }
}

resource "google_monitoring_alert_policy" "high_cpu" {
  display_name = "High CPU Usage"
  combiner     = "OR"
  
  conditions {
    display_name = "CPU usage above 80%"
    
    condition_threshold {
      filter          = "resource.type = \"k8s_container\" AND metric.type = \"kubernetes.io/container/cpu/core_usage_time\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.8
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }
  
  notification_channels = [google_monitoring_notification_channel.email.id]
}

# Outputs
output "cluster_endpoint" {
  value       = google_container_cluster.primary.endpoint
  description = "GKE cluster endpoint"
}

output "load_balancer_ip" {
  value       = google_compute_global_address.default.address
  description = "Load balancer IP address"
}

output "database_connection" {
  value       = google_sql_database_instance.postgres.connection_name
  description = "Cloud SQL connection name"
  sensitive   = true
}

output "redis_host" {
  value       = google_redis_instance.cache.host
  description = "Redis instance host"
}

output "static_bucket" {
  value       = google_storage_bucket.static.url
  description = "Static assets bucket URL"
}