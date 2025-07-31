"""
ML Workload Manager for distributing computational tasks to Google Cloud instances.
Handles workload scheduling, monitoring, and cost optimization.
"""

import os
import time
import json
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from gcp_compute import GCPComputeManager
import logging

class WorkloadType(Enum):
    """Types of ML workloads."""
    IDTXL_ANALYSIS = "idtxl_analysis"
    ML_TRAINING = "ml_training"
    NEURAL_NETWORK = "neural_network"
    DATA_PROCESSING = "data_processing"

class WorkloadStatus(Enum):
    """Workload execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class WorkloadConfig:
    """Configuration for ML workloads."""
    workload_id: str
    workload_type: WorkloadType
    data_size_gb: float
    estimated_runtime_hours: float
    cpu_cores: int
    memory_gb: int
    gpu_required: bool = False
    gpu_type: Optional[str] = None
    gpu_count: int = 0
    priority: int = 1  # 1 = low, 5 = high
    max_cost_per_hour: float = 10.0
    preemptible_allowed: bool = True

@dataclass
class WorkloadResult:
    """Result of workload execution."""
    workload_id: str
    status: WorkloadStatus
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    instance_name: Optional[str] = None
    cost_usd: float = 0.0
    output_files: List[str] = None
    error_message: Optional[str] = None

class MLWorkloadManager:
    """Manages ML workloads across Google Cloud Compute instances."""
    
    def __init__(self):
        """Initialize the ML Workload Manager."""
        self.compute_manager = GCPComputeManager()
        self.active_workloads: Dict[str, WorkloadResult] = {}
        self.workload_queue: List[WorkloadConfig] = []
        self.instance_assignments: Dict[str, str] = {}  # instance_name -> workload_id
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def submit_workload(self, config: WorkloadConfig) -> str:
        """Submit a workload for execution."""
        self.logger.info(f"Submitting workload {config.workload_id}")
        
        # Add to results tracking
        self.active_workloads[config.workload_id] = WorkloadResult(
            workload_id=config.workload_id,
            status=WorkloadStatus.PENDING
        )
        
        # Add to queue (sort by priority)
        self.workload_queue.append(config)
        self.workload_queue.sort(key=lambda x: x.priority, reverse=True)
        
        return config.workload_id
    
    def get_optimal_instance_config(self, config: WorkloadConfig) -> Dict[str, Any]:
        """Determine optimal instance configuration for workload."""
        
        # Map workload requirements to instance specs
        if config.workload_type == WorkloadType.IDTXL_ANALYSIS:
            # IDTxl is CPU-intensive with high memory requirements
            return {
                "machine_type": "n1-highmem-8" if config.memory_gb > 30 else "n1-standard-4",
                "gpu_type": None,
                "gpu_count": 0,
                "disk_size_gb": max(100, int(config.data_size_gb * 3))
            }
        
        elif config.workload_type == WorkloadType.NEURAL_NETWORK:
            # Neural networks benefit from GPU acceleration
            gpu_type = config.gpu_type or "nvidia-tesla-t4"
            if config.estimated_runtime_hours > 4:
                gpu_type = "nvidia-tesla-v100"  # More powerful for long training
            
            return {
                "machine_type": "n1-standard-8",
                "gpu_type": gpu_type,
                "gpu_count": max(1, config.gpu_count),
                "disk_size_gb": max(100, int(config.data_size_gb * 2))
            }
        
        elif config.workload_type == WorkloadType.ML_TRAINING:
            # Traditional ML can use CPU or GPU depending on data size
            if config.data_size_gb > 10 or config.gpu_required:
                return {
                    "machine_type": "n1-highmem-4",
                    "gpu_type": "nvidia-tesla-t4",
                    "gpu_count": 1,
                    "disk_size_gb": max(100, int(config.data_size_gb * 2))
                }
            else:
                return {
                    "machine_type": "n1-highmem-4",
                    "gpu_type": None,
                    "gpu_count": 0,
                    "disk_size_gb": max(50, int(config.data_size_gb * 1.5))
                }
        
        else:  # DATA_PROCESSING
            return {
                "machine_type": "n1-highmem-4",
                "gpu_type": None,
                "gpu_count": 0,
                "disk_size_gb": max(100, int(config.data_size_gb * 2))
            }
    
    async def execute_workload(self, config: WorkloadConfig) -> WorkloadResult:
        """Execute a workload on Google Cloud."""
        workload_id = config.workload_id
        
        try:
            self.logger.info(f"Starting execution of workload {workload_id}")
            
            # Update status
            self.active_workloads[workload_id].status = WorkloadStatus.QUEUED
            
            # Get optimal instance configuration
            instance_config = self.get_optimal_instance_config(config)
            
            # Create instance name
            instance_name = f"ml-{config.workload_type.value}-{workload_id[:8]}"
            
            # Create instance
            instance_details = self.compute_manager.create_ml_instance(
                instance_name=instance_name,
                preemptible=config.preemptible_allowed,
                **instance_config
            )
            
            self.logger.info(f"Created instance {instance_name} for workload {workload_id}")
            
            # Update tracking
            self.active_workloads[workload_id].status = WorkloadStatus.RUNNING
            self.active_workloads[workload_id].start_time = time.strftime("%Y-%m-%d %H:%M:%S")
            self.active_workloads[workload_id].instance_name = instance_name
            self.instance_assignments[instance_name] = workload_id
            
            # Wait for instance to be ready and execute workload
            await self._wait_for_instance_ready(instance_name)
            await self._execute_workload_on_instance(config, instance_name)
            
            # Workload completed successfully
            self.active_workloads[workload_id].status = WorkloadStatus.COMPLETED
            self.active_workloads[workload_id].end_time = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Calculate cost
            runtime_hours = self._calculate_runtime_hours(workload_id)
            self.active_workloads[workload_id].cost_usd = (
                instance_details["cost_estimate_per_hour"] * runtime_hours
            )
            
            # Clean up instance
            self.compute_manager.delete_instance(instance_name)
            del self.instance_assignments[instance_name]
            
            self.logger.info(f"Workload {workload_id} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Workload {workload_id} failed: {str(e)}")
            self.active_workloads[workload_id].status = WorkloadStatus.FAILED
            self.active_workloads[workload_id].error_message = str(e)
            
            # Clean up failed instance
            if workload_id in self.active_workloads and self.active_workloads[workload_id].instance_name:
                instance_name = self.active_workloads[workload_id].instance_name
                self.compute_manager.delete_instance(instance_name)
                if instance_name in self.instance_assignments:
                    del self.instance_assignments[instance_name]
        
        return self.active_workloads[workload_id]
    
    async def process_workload_queue(self):
        """Process pending workloads in the queue."""
        while self.workload_queue:
            config = self.workload_queue.pop(0)
            
            # Check if we can afford this workload
            estimated_cost = self._estimate_workload_cost(config)
            if estimated_cost > config.max_cost_per_hour * config.estimated_runtime_hours:
                self.logger.warning(f"Workload {config.workload_id} exceeds cost limit")
                self.active_workloads[config.workload_id].status = WorkloadStatus.FAILED
                self.active_workloads[config.workload_id].error_message = "Exceeds cost limit"
                continue
            
            # Execute workload
            await self.execute_workload(config)
    
    def get_workload_status(self, workload_id: str) -> Optional[WorkloadResult]:
        """Get the status of a workload."""
        return self.active_workloads.get(workload_id)
    
    def list_active_workloads(self) -> List[WorkloadResult]:
        """List all active workloads."""
        return list(self.active_workloads.values())
    
    def cancel_workload(self, workload_id: str) -> bool:
        """Cancel a workload."""
        if workload_id not in self.active_workloads:
            return False
        
        result = self.active_workloads[workload_id]
        
        if result.status in [WorkloadStatus.PENDING, WorkloadStatus.QUEUED]:
            # Remove from queue
            self.workload_queue = [w for w in self.workload_queue if w.workload_id != workload_id]
            result.status = WorkloadStatus.CANCELLED
            return True
        
        elif result.status == WorkloadStatus.RUNNING and result.instance_name:
            # Stop running instance
            self.compute_manager.delete_instance(result.instance_name)
            if result.instance_name in self.instance_assignments:
                del self.instance_assignments[result.instance_name]
            result.status = WorkloadStatus.CANCELLED
            return True
        
        return False
    
    async def _wait_for_instance_ready(self, instance_name: str, timeout: int = 300):
        """Wait for instance to be ready for workload execution."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            instances = self.compute_manager.list_instances()
            instance = next((i for i in instances if i["name"] == instance_name), None)
            
            if instance and instance["status"] == "RUNNING":
                # Additional wait for startup script to complete
                await asyncio.sleep(60)
                return
            
            await asyncio.sleep(10)
        
        raise TimeoutError(f"Instance {instance_name} did not become ready within {timeout} seconds")
    
    async def _execute_workload_on_instance(self, config: WorkloadConfig, instance_name: str):
        """Execute the actual workload on the instance."""
        # This would contain the logic to:
        # 1. Transfer data to the instance
        # 2. Run the ML/analysis code
        # 3. Monitor progress
        # 4. Retrieve results
        
        # For now, simulate workload execution
        self.logger.info(f"Executing {config.workload_type.value} on {instance_name}")
        
        # Simulate workload time
        simulation_time = min(config.estimated_runtime_hours * 10, 300)  # Max 5 min simulation
        await asyncio.sleep(simulation_time)
        
        self.logger.info(f"Workload execution completed on {instance_name}")
    
    def _calculate_runtime_hours(self, workload_id: str) -> float:
        """Calculate actual runtime hours for a workload."""
        result = self.active_workloads[workload_id]
        if not result.start_time or not result.end_time:
            return 0.0
        
        start = time.strptime(result.start_time, "%Y-%m-%d %H:%M:%S")
        end = time.strptime(result.end_time, "%Y-%m-%d %H:%M:%S")
        
        start_seconds = time.mktime(start)
        end_seconds = time.mktime(end)
        
        return (end_seconds - start_seconds) / 3600.0
    
    def _estimate_workload_cost(self, config: WorkloadConfig) -> float:
        """Estimate cost per hour for a workload configuration."""
        instance_config = self.get_optimal_instance_config(config)
        
        # Use compute manager's cost estimation
        return self.compute_manager._estimate_cost(
            instance_config["machine_type"],
            instance_config["gpu_type"],
            instance_config["gpu_count"],
            config.preemptible_allowed
        )


# Convenience functions for common financial analysis workloads
async def run_idtxl_analysis(data_size_gb: float, priority: int = 3) -> str:
    """Submit IDTxl information-theoretic analysis workload."""
    manager = MLWorkloadManager()
    
    config = WorkloadConfig(
        workload_id=f"idtxl-{time.time():.0f}",
        workload_type=WorkloadType.IDTXL_ANALYSIS,
        data_size_gb=data_size_gb,
        estimated_runtime_hours=max(1.0, data_size_gb * 0.5),
        cpu_cores=8,
        memory_gb=32,
        priority=priority,
        preemptible_allowed=True
    )
    
    return manager.submit_workload(config)

async def run_ml_training(data_size_gb: float, use_gpu: bool = True, priority: int = 3) -> str:
    """Submit ML model training workload."""
    manager = MLWorkloadManager()
    
    config = WorkloadConfig(
        workload_id=f"ml-train-{time.time():.0f}",
        workload_type=WorkloadType.ML_TRAINING,
        data_size_gb=data_size_gb,
        estimated_runtime_hours=max(0.5, data_size_gb * 0.3),
        cpu_cores=4,
        memory_gb=16,
        gpu_required=use_gpu,
        gpu_type="nvidia-tesla-t4" if use_gpu else None,
        gpu_count=1 if use_gpu else 0,
        priority=priority,
        preemptible_allowed=True
    )
    
    return manager.submit_workload(config)

async def run_neural_network_training(data_size_gb: float, priority: int = 4) -> str:
    """Submit neural network training workload."""
    manager = MLWorkloadManager()
    
    config = WorkloadConfig(
        workload_id=f"nn-train-{time.time():.0f}",
        workload_type=WorkloadType.NEURAL_NETWORK,
        data_size_gb=data_size_gb,
        estimated_runtime_hours=max(2.0, data_size_gb * 0.8),
        cpu_cores=8,
        memory_gb=32,
        gpu_required=True,
        gpu_type="nvidia-tesla-v100",
        gpu_count=2,
        priority=priority,
        preemptible_allowed=True
    )
    
    return manager.submit_workload(config)