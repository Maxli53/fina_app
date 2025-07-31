"""
Google Cloud Compute Engine management for ML training workloads.
Handles GPU/CPU instance creation, scaling, and cost optimization.
"""

import os
import time
from typing import Dict, List, Optional, Any
from google.cloud import compute_v1
from google.oauth2 import service_account
from dotenv import load_dotenv
import logging

load_dotenv()

class GCPComputeManager:
    """Manages Google Cloud Compute Engine instances for ML training."""
    
    def __init__(self):
        """Initialize the GCP Compute Manager."""
        self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        self.credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        # Initialize clients
        self.instances_client = compute_v1.InstancesClient()
        self.instance_templates_client = compute_v1.InstanceTemplatesClient()
        self.zones_client = compute_v1.ZonesClient()
        
        # Default configurations
        self.default_zone = "us-central1-a"  # GPU-enabled zone
        self.default_region = "us-central1"
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_ml_instance(
        self,
        instance_name: str,
        machine_type: str = "n1-standard-4",
        gpu_type: Optional[str] = "nvidia-tesla-t4",
        gpu_count: int = 1,
        disk_size_gb: int = 100,
        preemptible: bool = True,
        zone: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a Compute Engine instance optimized for ML training.
        
        Args:
            instance_name: Name of the instance
            machine_type: GCE machine type (e.g., 'n1-standard-4', 'n1-highmem-8')
            gpu_type: GPU type ('nvidia-tesla-t4', 'nvidia-tesla-v100', etc.)
            gpu_count: Number of GPUs to attach
            disk_size_gb: Boot disk size in GB
            preemptible: Use preemptible instance for cost savings
            zone: GCP zone (defaults to us-central1-a)
        
        Returns:
            Dict containing instance creation details
        """
        zone = zone or self.default_zone
        
        # Instance configuration
        instance_config = {
            "name": instance_name,
            "machine_type": f"zones/{zone}/machineTypes/{machine_type}",
            "scheduling": {
                "preemptible": preemptible
            },
            "disks": [
                {
                    "boot": True,
                    "auto_delete": True,
                    "initialize_params": {
                        "source_image": "projects/ml-images/global/images/family/tf-latest-gpu",
                        "disk_size_gb": str(disk_size_gb),
                        "disk_type": f"projects/{self.project_id}/zones/{zone}/diskTypes/pd-standard"
                    }
                }
            ],
            "network_interfaces": [
                {
                    "network": "global/networks/default",
                    "access_configs": [
                        {
                            "type": "ONE_TO_ONE_NAT",
                            "name": "External NAT"
                        }
                    ]
                }
            ],
            "metadata": {
                "items": [
                    {
                        "key": "startup-script",
                        "value": self._get_startup_script()
                    },
                    {
                        "key": "install-nvidia-driver",
                        "value": "True"
                    }
                ]
            },
            "tags": {
                "items": ["ml-training", "financial-analysis"]
            }
        }
        
        # Add GPU configuration if specified
        if gpu_type and gpu_count > 0:
            instance_config["guest_accelerators"] = [
                {
                    "accelerator_type": f"projects/{self.project_id}/zones/{zone}/acceleratorTypes/{gpu_type}",
                    "accelerator_count": gpu_count
                }
            ]
            # GPU instances require specific maintenance policy
            instance_config["scheduling"]["on_host_maintenance"] = "TERMINATE"
        
        try:
            # Create the instance
            operation = self.instances_client.insert(
                project=self.project_id,
                zone=zone,
                instance_resource=instance_config
            )
            
            self.logger.info(f"Creating instance {instance_name} in zone {zone}")
            
            # Wait for operation to complete
            self._wait_for_operation(operation, zone)
            
            # Get instance details
            instance = self.instances_client.get(
                project=self.project_id,
                zone=zone,
                instance=instance_name
            )
            
            return {
                "name": instance_name,
                "zone": zone,
                "machine_type": machine_type,
                "gpu_type": gpu_type,
                "gpu_count": gpu_count,
                "preemptible": preemptible,
                "status": instance.status,
                "external_ip": self._get_external_ip(instance),
                "cost_estimate_per_hour": self._estimate_cost(machine_type, gpu_type, gpu_count, preemptible)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create instance {instance_name}: {str(e)}")
            raise
    
    def delete_instance(self, instance_name: str, zone: Optional[str] = None) -> bool:
        """Delete a Compute Engine instance."""
        zone = zone or self.default_zone
        
        try:
            operation = self.instances_client.delete(
                project=self.project_id,
                zone=zone,
                instance=instance_name
            )
            
            self.logger.info(f"Deleting instance {instance_name}")
            self._wait_for_operation(operation, zone)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete instance {instance_name}: {str(e)}")
            return False
    
    def list_instances(self, zone: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all instances in a zone."""
        zone = zone or self.default_zone
        
        try:
            instances = self.instances_client.list(
                project=self.project_id,
                zone=zone
            )
            
            instance_list = []
            for instance in instances:
                instance_list.append({
                    "name": instance.name,
                    "status": instance.status,
                    "machine_type": instance.machine_type.split('/')[-1],
                    "external_ip": self._get_external_ip(instance),
                    "preemptible": instance.scheduling.preemptible if instance.scheduling else False,
                    "creation_timestamp": instance.creation_timestamp
                })
            
            return instance_list
            
        except Exception as e:
            self.logger.error(f"Failed to list instances: {str(e)}")
            return []
    
    def start_instance(self, instance_name: str, zone: Optional[str] = None) -> bool:
        """Start a stopped instance."""
        zone = zone or self.default_zone
        
        try:
            operation = self.instances_client.start(
                project=self.project_id,
                zone=zone,
                instance=instance_name
            )
            
            self.logger.info(f"Starting instance {instance_name}")
            self._wait_for_operation(operation, zone)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start instance {instance_name}: {str(e)}")
            return False
    
    def stop_instance(self, instance_name: str, zone: Optional[str] = None) -> bool:
        """Stop a running instance."""
        zone = zone or self.default_zone
        
        try:
            operation = self.instances_client.stop(
                project=self.project_id,
                zone=zone,
                instance=instance_name
            )
            
            self.logger.info(f"Stopping instance {instance_name}")
            self._wait_for_operation(operation, zone)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop instance {instance_name}: {str(e)}")
            return False
    
    def get_available_gpu_types(self, zone: Optional[str] = None) -> List[str]:
        """Get list of available GPU types in a zone."""
        zone = zone or self.default_zone
        
        try:
            accelerator_types = self.instances_client.aggregated_list_accelerator_types(
                project=self.project_id
            )
            
            gpu_types = []
            for zone_name, accelerators in accelerator_types:
                if zone in zone_name and accelerators.accelerator_types:
                    for accelerator in accelerators.accelerator_types:
                        gpu_types.append(accelerator.name)
            
            return gpu_types
            
        except Exception as e:
            self.logger.error(f"Failed to get GPU types: {str(e)}")
            return ["nvidia-tesla-t4", "nvidia-tesla-v100"]  # Default fallback
    
    def _get_startup_script(self) -> str:
        """Get startup script for ML instances."""
        return """#!/bin/bash
        
        # Update system
        apt-get update
        
        # Install Python dependencies
        pip3 install --upgrade pip
        pip3 install tensorflow torch scikit-learn pandas numpy jupyter
        
        # Install financial analysis packages
        pip3 install yfinance idtxl
        
        # Create working directory
        mkdir -p /opt/ml-training
        chown -R $USER:$USER /opt/ml-training
        
        # Start Jupyter notebook service
        nohup jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root &
        
        echo "ML instance setup complete"
        """
    
    def _wait_for_operation(self, operation, zone: str, timeout: int = 300):
        """Wait for a GCE operation to complete."""
        start_time = time.time()
        
        while operation.status != "DONE":
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Operation timed out after {timeout} seconds")
            
            time.sleep(5)
            operation = self.instances_client.get_operation(
                project=self.project_id,
                zone=zone,
                operation=operation.name
            )
    
    def _get_external_ip(self, instance) -> Optional[str]:
        """Extract external IP from instance."""
        try:
            if instance.network_interfaces:
                access_configs = instance.network_interfaces[0].access_configs
                if access_configs:
                    return access_configs[0].nat_ip
        except (AttributeError, IndexError):
            pass
        return None
    
    def _estimate_cost(self, machine_type: str, gpu_type: Optional[str], 
                      gpu_count: int, preemptible: bool) -> float:
        """Estimate hourly cost for instance configuration."""
        # Simplified cost estimation (actual costs may vary)
        base_costs = {
            "n1-standard-4": 0.19,
            "n1-standard-8": 0.38,
            "n1-highmem-8": 0.54,
            "n2-standard-4": 0.20,
            "n2-highmem-8": 0.56
        }
        
        gpu_costs = {
            "nvidia-tesla-t4": 0.35,
            "nvidia-tesla-v100": 2.48,
            "nvidia-tesla-k80": 0.45
        }
        
        base_cost = base_costs.get(machine_type, 0.20)
        gpu_cost = gpu_costs.get(gpu_type, 0) * gpu_count if gpu_type else 0
        
        total_cost = base_cost + gpu_cost
        
        if preemptible:
            total_cost *= 0.2  # Preemptible instances are ~80% cheaper
        
        return round(total_cost, 2)


# Convenience functions for common ML configurations
def create_small_gpu_instance(name: str, preemptible: bool = True) -> Dict[str, Any]:
    """Create a small GPU instance for light ML training."""
    manager = GCPComputeManager()
    return manager.create_ml_instance(
        instance_name=name,
        machine_type="n1-standard-4",
        gpu_type="nvidia-tesla-t4",
        gpu_count=1,
        preemptible=preemptible
    )

def create_large_gpu_instance(name: str, preemptible: bool = True) -> Dict[str, Any]:
    """Create a large GPU instance for intensive ML training."""
    manager = GCPComputeManager()
    return manager.create_ml_instance(
        instance_name=name,
        machine_type="n1-highmem-8",
        gpu_type="nvidia-tesla-v100",
        gpu_count=2,
        preemptible=preemptible
    )

def create_cpu_only_instance(name: str, preemptible: bool = True) -> Dict[str, Any]:
    """Create a CPU-only instance for data processing."""
    manager = GCPComputeManager()
    return manager.create_ml_instance(
        instance_name=name,
        machine_type="n1-highmem-8",
        gpu_type=None,
        gpu_count=0,
        preemptible=preemptible
    )