"""
Load testing framework for validating performance under production loads
"""

import asyncio
import aiohttp
import time
import statistics
from typing import Dict, List, Any, Tuple
from datetime import datetime
import json
import random
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import websockets
import numpy as np
from locust import HttpUser, task, between, events
import logging

logger = logging.getLogger(__name__)


@dataclass
class LoadTestResult:
    """Load test result metrics"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration: float
    min_response_time: float
    max_response_time: float
    mean_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    error_rate: float
    
    def __str__(self):
        return f"""
Load Test Results:
==================
Total Requests: {self.total_requests}
Successful: {self.successful_requests}
Failed: {self.failed_requests}
Error Rate: {self.error_rate:.2%}
Duration: {self.total_duration:.2f}s
RPS: {self.requests_per_second:.2f}

Response Times:
Min: {self.min_response_time*1000:.2f}ms
Max: {self.max_response_time*1000:.2f}ms
Mean: {self.mean_response_time*1000:.2f}ms
Median: {self.median_response_time*1000:.2f}ms
P95: {self.p95_response_time*1000:.2f}ms
P99: {self.p99_response_time*1000:.2f}ms
"""


class LoadTester:
    """Comprehensive load testing framework"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[Dict[str, Any]] = []
        
    async def run_load_test(
        self,
        endpoint: str,
        method: str = "GET",
        concurrent_users: int = 100,
        requests_per_user: int = 100,
        payload: Dict[str, Any] = None,
        headers: Dict[str, str] = None
    ) -> LoadTestResult:
        """Run load test on specific endpoint"""
        print(f"\nRunning load test on {endpoint}")
        print(f"Concurrent users: {concurrent_users}")
        print(f"Requests per user: {requests_per_user}")
        print(f"Total requests: {concurrent_users * requests_per_user}")
        
        self.results = []
        start_time = time.time()
        
        # Create user tasks
        tasks = []
        for user_id in range(concurrent_users):
            task = self._simulate_user(
                user_id,
                endpoint,
                method,
                requests_per_user,
                payload,
                headers
            )
            tasks.append(task)
        
        # Execute all users concurrently
        await asyncio.gather(*tasks)
        
        total_duration = time.time() - start_time
        
        # Calculate metrics
        return self._calculate_metrics(total_duration)
    
    async def _simulate_user(
        self,
        user_id: int,
        endpoint: str,
        method: str,
        num_requests: int,
        payload: Dict[str, Any] = None,
        headers: Dict[str, str] = None
    ):
        """Simulate a single user making requests"""
        async with aiohttp.ClientSession() as session:
            for request_num in range(num_requests):
                # Add some randomness to simulate real user behavior
                await asyncio.sleep(random.uniform(0.1, 0.5))
                
                start_time = time.time()
                success = False
                status_code = 0
                error_message = None
                
                try:
                    url = f"{self.base_url}{endpoint}"
                    
                    if method == "GET":
                        async with session.get(url, headers=headers) as response:
                            await response.read()
                            status_code = response.status
                            success = 200 <= status_code < 300
                    
                    elif method == "POST":
                        async with session.post(
                            url, 
                            json=payload, 
                            headers=headers
                        ) as response:
                            await response.read()
                            status_code = response.status
                            success = 200 <= status_code < 300
                    
                except Exception as e:
                    error_message = str(e)
                    success = False
                
                duration = time.time() - start_time
                
                self.results.append({
                    "user_id": user_id,
                    "request_num": request_num,
                    "success": success,
                    "status_code": status_code,
                    "duration": duration,
                    "error": error_message,
                    "timestamp": datetime.now()
                })
    
    def _calculate_metrics(self, total_duration: float) -> LoadTestResult:
        """Calculate performance metrics from results"""
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]
        
        if not successful:
            return LoadTestResult(
                total_requests=len(self.results),
                successful_requests=0,
                failed_requests=len(failed),
                total_duration=total_duration,
                min_response_time=0,
                max_response_time=0,
                mean_response_time=0,
                median_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                requests_per_second=0,
                error_rate=1.0
            )
        
        response_times = [r["duration"] for r in successful]
        response_times.sort()
        
        return LoadTestResult(
            total_requests=len(self.results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            total_duration=total_duration,
            min_response_time=min(response_times),
            max_response_time=max(response_times),
            mean_response_time=statistics.mean(response_times),
            median_response_time=statistics.median(response_times),
            p95_response_time=np.percentile(response_times, 95),
            p99_response_time=np.percentile(response_times, 99),
            requests_per_second=len(self.results) / total_duration,
            error_rate=len(failed) / len(self.results)
        )


class WebSocketLoadTester:
    """Load testing for WebSocket connections"""
    
    def __init__(self, ws_url: str = "ws://localhost:8765"):
        self.ws_url = ws_url
        self.results: List[Dict[str, Any]] = []
        
    async def run_websocket_load_test(
        self,
        num_connections: int = 100,
        messages_per_connection: int = 100,
        message_interval: float = 0.1
    ) -> LoadTestResult:
        """Test WebSocket performance with multiple concurrent connections"""
        print(f"\nRunning WebSocket load test")
        print(f"Connections: {num_connections}")
        print(f"Messages per connection: {messages_per_connection}")
        
        self.results = []
        start_time = time.time()
        
        # Create WebSocket connections
        tasks = []
        for conn_id in range(num_connections):
            task = self._simulate_websocket_client(
                conn_id,
                messages_per_connection,
                message_interval
            )
            tasks.append(task)
        
        # Run all connections concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
        
        total_duration = time.time() - start_time
        
        return self._calculate_metrics(total_duration)
    
    async def _simulate_websocket_client(
        self,
        client_id: int,
        num_messages: int,
        interval: float
    ):
        """Simulate a single WebSocket client"""
        try:
            async with websockets.connect(self.ws_url) as websocket:
                # Wait for initial connection message
                await websocket.recv()
                
                for msg_num in range(num_messages):
                    start_time = time.time()
                    
                    # Send message
                    message = {
                        "type": "subscribe",
                        "symbols": ["AAPL", "MSFT", "GOOGL"]
                    }
                    
                    await websocket.send(json.dumps(message))
                    
                    # Wait for response
                    response = await websocket.recv()
                    
                    duration = time.time() - start_time
                    
                    self.results.append({
                        "client_id": client_id,
                        "message_num": msg_num,
                        "success": True,
                        "duration": duration,
                        "timestamp": datetime.now()
                    })
                    
                    await asyncio.sleep(interval)
                    
        except Exception as e:
            self.results.append({
                "client_id": client_id,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now()
            })
    
    def _calculate_metrics(self, total_duration: float) -> LoadTestResult:
        """Calculate WebSocket performance metrics"""
        successful = [r for r in self.results if r.get("success", False)]
        failed = [r for r in self.results if not r.get("success", True)]
        
        if not successful:
            return LoadTestResult(
                total_requests=len(self.results),
                successful_requests=0,
                failed_requests=len(failed),
                total_duration=total_duration,
                min_response_time=0,
                max_response_time=0,
                mean_response_time=0,
                median_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                requests_per_second=0,
                error_rate=1.0
            )
        
        response_times = [r["duration"] for r in successful if "duration" in r]
        response_times.sort()
        
        return LoadTestResult(
            total_requests=len(self.results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            total_duration=total_duration,
            min_response_time=min(response_times),
            max_response_time=max(response_times),
            mean_response_time=statistics.mean(response_times),
            median_response_time=statistics.median(response_times),
            p95_response_time=np.percentile(response_times, 95),
            p99_response_time=np.percentile(response_times, 99),
            requests_per_second=len(self.results) / total_duration,
            error_rate=len(failed) / len(self.results)
        )


# Locust test for distributed load testing
class PlatformUser(HttpUser):
    """Simulated platform user for Locust testing"""
    wait_time = between(1, 3)
    
    def on_start(self):
        """Login and get token"""
        response = self.client.post("/api/auth/login", json={
            "username": "testuser",
            "password": "testpass"
        })
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.headers = {}
    
    @task(3)
    def get_quote(self):
        """Get market quote"""
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        symbol = random.choice(symbols)
        self.client.get(f"/api/data/quote/{symbol}", headers=self.headers)
    
    @task(2)
    def get_historical_data(self):
        """Get historical data"""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        symbol = random.choice(symbols)
        self.client.get(
            f"/api/data/historical/{symbol}?start_date=2024-01-01&end_date=2024-01-31",
            headers=self.headers
        )
    
    @task(1)
    def run_analysis(self):
        """Run analysis"""
        self.client.post(
            "/api/analysis/ml",
            json={
                "symbols": ["AAPL"],
                "model_type": "random_forest",
                "features": ["rsi", "macd", "volume_ratio"],
                "target": "next_day_return"
            },
            headers=self.headers
        )
    
    @task(1)
    def get_positions(self):
        """Get trading positions"""
        self.client.get("/api/trading/positions", headers=self.headers)
    
    @task(1)
    def system_health(self):
        """Check system health"""
        self.client.get("/api/health", headers=self.headers)


async def run_comprehensive_load_tests():
    """Run comprehensive load tests on all critical endpoints"""
    tester = LoadTester()
    ws_tester = WebSocketLoadTester()
    
    print("=" * 60)
    print("COMPREHENSIVE LOAD TESTING")
    print("=" * 60)
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Light Load - Market Data",
            "endpoint": "/api/data/quote/AAPL",
            "concurrent_users": 10,
            "requests_per_user": 100
        },
        {
            "name": "Medium Load - Historical Data",
            "endpoint": "/api/data/historical/AAPL?start_date=2024-01-01&end_date=2024-01-31",
            "concurrent_users": 50,
            "requests_per_user": 50
        },
        {
            "name": "Heavy Load - Mixed Endpoints",
            "endpoint": "/api/health",
            "concurrent_users": 100,
            "requests_per_user": 100
        },
        {
            "name": "Spike Test - Sudden Load",
            "endpoint": "/api/data/quote/MSFT",
            "concurrent_users": 200,
            "requests_per_user": 10
        }
    ]
    
    # Run HTTP load tests
    for scenario in test_scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'='*60}")
        
        result = await tester.run_load_test(
            endpoint=scenario["endpoint"],
            concurrent_users=scenario["concurrent_users"],
            requests_per_user=scenario["requests_per_user"]
        )
        
        print(result)
        
        # Check if performance meets requirements
        if result.p95_response_time > 0.5:  # 500ms
            print("⚠️  WARNING: P95 response time exceeds 500ms target")
        
        if result.error_rate > 0.01:  # 1%
            print("⚠️  WARNING: Error rate exceeds 1% threshold")
        
        # Cool down between tests
        await asyncio.sleep(5)
    
    # WebSocket load test
    print(f"\n{'='*60}")
    print("WebSocket Load Test")
    print(f"{'='*60}")
    
    ws_result = await ws_tester.run_websocket_load_test(
        num_connections=50,
        messages_per_connection=100,
        message_interval=0.1
    )
    
    print(ws_result)
    
    # Analysis workload test
    print(f"\n{'='*60}")
    print("Analysis Workload Test")
    print(f"{'='*60}")
    
    analysis_result = await tester.run_load_test(
        endpoint="/api/analysis/idtxl",
        method="POST",
        concurrent_users=10,
        requests_per_user=5,
        payload={
            "symbols": ["AAPL", "MSFT"],
            "analysis_type": "transfer_entropy",
            "parameters": {"max_lag": 5}
        }
    )
    
    print(analysis_result)


if __name__ == "__main__":
    # Run load tests
    asyncio.run(run_comprehensive_load_tests())
    
    # For distributed testing with Locust:
    # locust -f test_load_testing.py --host=http://localhost:8000