"""
Test script for Google Cloud Compute Engine ML workload management.
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'app', 'services'))

from gcp_compute import GCPComputeManager, create_small_gpu_instance
from ml_workload_manager import MLWorkloadManager, WorkloadConfig, WorkloadType, run_idtxl_analysis

async def test_compute_manager():
    """Test basic Compute Engine operations."""
    print("Testing GCP Compute Manager...")
    
    manager = GCPComputeManager()
    
    # Test 1: List available GPU types
    print("\n1. Available GPU types:")
    gpu_types = manager.get_available_gpu_types()
    for gpu in gpu_types[:5]:  # Show first 5
        print(f"   - {gpu}")
    
    # Test 2: List current instances
    print("\n2. Current instances:")
    instances = manager.list_instances()
    if instances:
        for instance in instances:
            print(f"   - {instance['name']}: {instance['status']}")
    else:
        print("   No instances found")
    
    # Test 3: Cost estimation
    print("\n3. Cost estimates (per hour):")
    configs = [
        ("Small GPU", "n1-standard-4", "nvidia-tesla-t4", 1, True),
        ("Large GPU", "n1-highmem-8", "nvidia-tesla-v100", 2, True),
        ("CPU Only", "n1-highmem-4", None, 0, True)
    ]
    
    for name, machine_type, gpu_type, gpu_count, preemptible in configs:
        cost = manager._estimate_cost(machine_type, gpu_type, gpu_count, preemptible)
        print(f"   - {name}: ${cost:.2f}/hour")
    
    print("\nCompute Manager test completed!")

async def test_workload_manager():
    """Test ML Workload Manager."""
    print("\nTesting ML Workload Manager...")
    
    manager = MLWorkloadManager()
    
    # Test 1: Create sample workload configurations
    print("\n1. Creating workload configurations:")
    
    workload_configs = [
        WorkloadConfig(
            workload_id="test-idtxl-001",
            workload_type=WorkloadType.IDTXL_ANALYSIS,
            data_size_gb=2.0,
            estimated_runtime_hours=1.0,
            cpu_cores=8,
            memory_gb=32,
            priority=3
        ),
        WorkloadConfig(
            workload_id="test-ml-001",
            workload_type=WorkloadType.ML_TRAINING,
            data_size_gb=5.0,
            estimated_runtime_hours=2.0,
            cpu_cores=4,
            memory_gb=16,
            gpu_required=True,
            gpu_type="nvidia-tesla-t4",
            gpu_count=1,
            priority=4
        )
    ]
    
    # Test 2: Submit workloads
    print("\n2. Submitting workloads:")
    for config in workload_configs:
        workload_id = manager.submit_workload(config)
        print(f"   - Submitted: {workload_id}")
        
        # Test optimal instance configuration
        instance_config = manager.get_optimal_instance_config(config)
        cost_estimate = manager._estimate_workload_cost(config)
        
        print(f"     Config: {instance_config}")
        print(f"     Est. cost: ${cost_estimate:.2f}/hour")
    
    # Test 3: Check workload status
    print("\n3. Workload status:")
    for workload in manager.list_active_workloads():
        print(f"   - {workload.workload_id}: {workload.status.value}")
    
    print("\nWorkload Manager test completed!")

async def test_instance_creation_simulation():
    """Simulate instance creation without actually creating instances."""
    print("\nSimulating instance creation...")
    
    manager = GCPComputeManager()
    
    # Test configurations
    test_configs = [
        {
            "name": "test-small-gpu",
            "machine_type": "n1-standard-4",
            "gpu_type": "nvidia-tesla-t4",
            "gpu_count": 1,
            "preemptible": True
        },
        {
            "name": "test-large-cpu",
            "machine_type": "n1-highmem-8",
            "gpu_type": None,
            "gpu_count": 0,
            "preemptible": True
        }
    ]
    
    print("\nInstance configurations that would be created:")
    for i, config in enumerate(test_configs, 1):
        cost = manager._estimate_cost(
            config["machine_type"], 
            config["gpu_type"], 
            config["gpu_count"], 
            config["preemptible"]
        )
        
        print(f"\n{i}. {config['name']}")
        print(f"   Machine Type: {config['machine_type']}")
        print(f"   GPU: {config['gpu_type']} x{config['gpu_count']}" if config['gpu_type'] else "   GPU: None")
        print(f"   Preemptible: {config['preemptible']}")
        print(f"   Est. Cost: ${cost:.2f}/hour")

async def test_convenience_functions():
    """Test convenience functions for workload submission."""
    print("\nTesting convenience functions...")
    
    # Test workload submission functions
    workload_functions = [
        ("IDTxl Analysis", run_idtxl_analysis, (1.5, 3)),
    ]
    
    print("\nWorkload submission tests:")
    for name, func, args in workload_functions:
        try:
            workload_id = await func(*args)
            print(f"   OK {name}: {workload_id}")
        except Exception as e:
            print(f"   ERROR {name}: {str(e)}")

def test_environment_setup():
    """Test environment setup and credentials."""
    print("Testing environment setup...")
    
    # Check environment variables
    required_vars = [
        'GOOGLE_APPLICATION_CREDENTIALS',
        'GOOGLE_CLOUD_PROJECT',
        'SERPAPI_KEY'
    ]
    
    print("\nEnvironment variables:")
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if 'KEY' in var or 'CREDENTIALS' in var:
                display_value = f"{value[:10]}...{value[-10:]}" if len(value) > 20 else "***"
            else:
                display_value = value
            print(f"   OK {var}: {display_value}")
        else:
            print(f"   MISSING {var}: Not set")
    
    # Test credentials file
    creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if creds_path and os.path.exists(creds_path):
        print(f"   OK Credentials file exists: {creds_path}")
    else:
        print(f"   MISSING Credentials file not found: {creds_path}")

async def main():
    """Run all tests."""
    print("=" * 60)
    print("Google Cloud Compute Engine ML Platform Tests")
    print("=" * 60)
    
    # Test 1: Environment setup
    test_environment_setup()
    
    # Test 2: Compute Manager
    try:
        await test_compute_manager()
    except Exception as e:
        print(f"Compute Manager test failed: {str(e)}")
    
    # Test 3: Workload Manager
    try:
        await test_workload_manager()
    except Exception as e:
        print(f"Workload Manager test failed: {str(e)}")
    
    # Test 4: Instance creation simulation
    try:
        await test_instance_creation_simulation()
    except Exception as e:
        print(f"Instance simulation test failed: {str(e)}")
    
    # Test 5: Convenience functions
    try:
        await test_convenience_functions()
    except Exception as e:
        print(f"Convenience functions test failed: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Run actual instance creation tests in a safe environment")
    print("2. Implement data transfer mechanisms")
    print("3. Add monitoring and logging")
    print("4. Set up billing alerts")
    print("5. Create workload templates for financial analysis")

if __name__ == "__main__":
    asyncio.run(main())