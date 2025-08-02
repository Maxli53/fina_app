"""
Comprehensive System Integration Tests
Tests the entire system from end-to-end ensuring all components work together
"""

import asyncio
import pytest
import aiohttp
from datetime import datetime, timedelta
import json
import websockets
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class SystemIntegrationTest:
    """
    Tests complete system workflows from market data to trading execution
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws") + "/ws"
        self.session = None
        self.auth_token = None
        
    async def setup(self):
        """Initialize test environment"""
        self.session = aiohttp.ClientSession()
        await self.authenticate()
        
    async def teardown(self):
        """Cleanup test environment"""
        if self.session:
            await self.session.close()
            
    async def authenticate(self):
        """Get authentication token"""
        async with self.session.post(
            f"{self.base_url}/api/auth/login",
            json={"username": "demo", "password": "demo"}
        ) as response:
            assert response.status == 200
            data = await response.json()
            self.auth_token = data["access_token"]
            self.session.headers.update({"Authorization": f"Bearer {self.auth_token}"})
    
    @pytest.mark.asyncio
    async def test_complete_trading_cycle(self):
        """
        Test 1: Complete trading cycle from analysis to execution
        """
        logger.info("Starting complete trading cycle test")
        
        # Step 1: Check system health
        health_status = await self.check_system_health()
        assert all(health_status.values()), f"System health check failed: {health_status}"
        
        # Step 2: Fetch market data
        symbol = "AAPL"
        market_data = await self.fetch_market_data(symbol)
        assert market_data["symbol"] == symbol
        assert "price" in market_data
        
        # Step 3: Run IDTxl analysis
        analysis_result = await self.run_analysis(["AAPL", "MSFT", "GOOGL"])
        assert analysis_result["status"] == "completed"
        assert "transfer_entropy" in analysis_result
        
        # Step 4: Generate trading signal
        signal = await self.generate_signal(analysis_result)
        assert signal["symbol"] in ["AAPL", "MSFT", "GOOGL"]
        assert signal["action"] in ["buy", "sell", "hold"]
        
        # Step 5: Validate risk
        risk_check = await self.check_risk(signal)
        assert risk_check["approved"] == True
        
        # Step 6: Place order
        if signal["action"] != "hold":
            order = await self.place_order(signal)
            assert order["status"] in ["pending", "filled"]
            assert order["symbol"] == signal["symbol"]
            
            # Step 7: Verify position update
            await asyncio.sleep(2)  # Wait for position update
            position = await self.get_position(signal["symbol"])
            assert position is not None
            
        logger.info("Complete trading cycle test passed")
    
    @pytest.mark.asyncio
    async def test_websocket_data_flow(self):
        """
        Test 2: WebSocket real-time data flow
        """
        logger.info("Starting WebSocket data flow test")
        
        received_messages = []
        
        async with websockets.connect(
            self.ws_url,
            extra_headers={"Authorization": f"Bearer {self.auth_token}"}
        ) as websocket:
            # Subscribe to symbols
            await websocket.send(json.dumps({
                "action": "subscribe",
                "symbols": ["AAPL", "MSFT"]
            }))
            
            # Collect messages for 5 seconds
            start_time = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start_time < 5:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    received_messages.append(json.loads(message))
                except asyncio.TimeoutError:
                    continue
        
        # Verify we received market data updates
        assert len(received_messages) > 0
        market_updates = [m for m in received_messages if m.get("type") == "market_data"]
        assert len(market_updates) > 0
        
        logger.info(f"WebSocket test passed - received {len(market_updates)} updates")
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """
        Test 3: System behavior under concurrent load
        """
        logger.info("Starting concurrent operations test")
        
        # Run multiple operations concurrently
        tasks = [
            self.fetch_market_data("AAPL"),
            self.fetch_market_data("MSFT"),
            self.fetch_market_data("GOOGL"),
            self.run_analysis(["AAPL", "MSFT"]),
            self.run_analysis(["GOOGL", "AMZN"]),
            self.get_portfolio_summary(),
            self.get_open_orders()
        ]
        
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = asyncio.get_event_loop().time() - start_time
        
        # Check all operations completed successfully
        failures = [r for r in results if isinstance(r, Exception)]
        assert len(failures) == 0, f"Concurrent operations failed: {failures}"
        
        # Verify performance
        assert duration < 10, f"Concurrent operations took too long: {duration}s"
        
        logger.info(f"Concurrent operations test passed - completed in {duration:.2f}s")
    
    @pytest.mark.asyncio
    async def test_failure_recovery(self):
        """
        Test 4: System recovery from failures
        """
        logger.info("Starting failure recovery test")
        
        # Test 1: Invalid order handling
        invalid_order = {
            "symbol": "INVALID",
            "quantity": -100,  # Invalid quantity
            "side": "buy",
            "order_type": "market"
        }
        
        response = await self.session.post(
            f"{self.base_url}/api/trading/orders",
            json=invalid_order
        )
        assert response.status == 400
        error_data = await response.json()
        assert "error" in error_data
        
        # Test 2: Recovery after invalid request
        valid_order = {
            "symbol": "AAPL",
            "quantity": 100,
            "side": "buy",
            "order_type": "market"
        }
        
        response = await self.session.post(
            f"{self.base_url}/api/trading/orders",
            json=valid_order
        )
        assert response.status in [200, 201]
        
        # Test 3: System health after errors
        health_status = await self.check_system_health()
        assert all(health_status.values()), "System unhealthy after error handling"
        
        logger.info("Failure recovery test passed")
    
    @pytest.mark.asyncio
    async def test_data_consistency(self):
        """
        Test 5: Data consistency across services
        """
        logger.info("Starting data consistency test")
        
        # Get portfolio from different endpoints
        portfolio_summary = await self.get_portfolio_summary()
        positions = await self.get_all_positions()
        
        # Calculate total from positions
        total_from_positions = sum(p["market_value"] for p in positions)
        
        # Compare with portfolio summary
        portfolio_total = portfolio_summary["total_value"] - portfolio_summary["cash_balance"]
        
        # Allow small difference for real-time price changes
        assert abs(total_from_positions - portfolio_total) < 100, \
            f"Portfolio inconsistency: {total_from_positions} vs {portfolio_total}"
        
        # Check order history consistency
        orders = await self.get_order_history()
        open_orders = await self.get_open_orders()
        
        # Open orders should be subset of all orders
        open_order_ids = {o["id"] for o in open_orders}
        all_order_ids = {o["id"] for o in orders if o["status"] == "open"}
        assert open_order_ids == all_order_ids, "Order data inconsistency"
        
        logger.info("Data consistency test passed")
    
    # Helper methods
    async def check_system_health(self) -> Dict[str, bool]:
        """Check health of all system components"""
        endpoints = [
            "/api/health",
            "/api/data/health",
            "/api/analysis/health",
            "/api/trading/health"
        ]
        
        health_status = {}
        for endpoint in endpoints:
            try:
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    health_status[endpoint] = response.status == 200
            except Exception as e:
                health_status[endpoint] = False
                logger.error(f"Health check failed for {endpoint}: {e}")
        
        return health_status
    
    async def fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch market data for a symbol"""
        async with self.session.get(
            f"{self.base_url}/api/data/quote/{symbol}"
        ) as response:
            assert response.status == 200
            return await response.json()
    
    async def run_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Run IDTxl analysis"""
        config = {
            "symbols": symbols,
            "analysis_type": "transfer_entropy",
            "max_lag": 5,
            "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
            "end_date": datetime.now().isoformat()
        }
        
        async with self.session.post(
            f"{self.base_url}/api/analysis/idtxl",
            json=config
        ) as response:
            assert response.status in [200, 201]
            task_data = await response.json()
            
        # Wait for analysis to complete
        task_id = task_data["task_id"]
        for _ in range(60):  # Max 60 seconds
            async with self.session.get(
                f"{self.base_url}/api/analysis/task/{task_id}"
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result["status"] == "completed":
                        return result
            await asyncio.sleep(1)
        
        raise TimeoutError("Analysis did not complete in time")
    
    async def generate_signal(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal from analysis"""
        # Mock signal generation based on analysis
        te_values = analysis_result.get("transfer_entropy", {})
        if te_values:
            # Find strongest information flow
            max_te = max(te_values.items(), key=lambda x: x[1]["value"])
            target_symbol = max_te[0].split("->")[1]
            
            return {
                "symbol": target_symbol,
                "action": "buy" if max_te[1]["value"] > 0.1 else "hold",
                "confidence": max_te[1]["value"],
                "quantity": 100
            }
        
        return {"symbol": "AAPL", "action": "hold", "confidence": 0}
    
    async def check_risk(self, signal: Dict[str, Any]) -> Dict[str, bool]:
        """Check risk for trading signal"""
        async with self.session.post(
            f"{self.base_url}/api/trading/risk/check",
            json=signal
        ) as response:
            if response.status == 200:
                return await response.json()
            return {"approved": False, "reason": "Risk check failed"}
    
    async def place_order(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Place trading order"""
        order = {
            "symbol": signal["symbol"],
            "quantity": signal.get("quantity", 100),
            "side": signal["action"],
            "order_type": "market"
        }
        
        async with self.session.post(
            f"{self.base_url}/api/trading/orders",
            json=order
        ) as response:
            assert response.status in [200, 201]
            return await response.json()
    
    async def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get position for symbol"""
        async with self.session.get(
            f"{self.base_url}/api/trading/positions/{symbol}"
        ) as response:
            if response.status == 200:
                return await response.json()
            return None
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        async with self.session.get(
            f"{self.base_url}/api/trading/portfolio/summary"
        ) as response:
            assert response.status == 200
            return await response.json()
    
    async def get_all_positions(self) -> List[Dict[str, Any]]:
        """Get all positions"""
        async with self.session.get(
            f"{self.base_url}/api/trading/positions"
        ) as response:
            assert response.status == 200
            return await response.json()
    
    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get open orders"""
        async with self.session.get(
            f"{self.base_url}/api/trading/orders/open"
        ) as response:
            assert response.status == 200
            return await response.json()
    
    async def get_order_history(self) -> List[Dict[str, Any]]:
        """Get order history"""
        async with self.session.get(
            f"{self.base_url}/api/trading/orders"
        ) as response:
            assert response.status == 200
            return await response.json()


# Test runner
async def run_all_tests():
    """Run all integration tests"""
    test = SystemIntegrationTest()
    
    try:
        await test.setup()
        
        # Run all tests
        await test.test_complete_trading_cycle()
        await test.test_websocket_data_flow()
        await test.test_concurrent_operations()
        await test.test_failure_recovery()
        await test.test_data_consistency()
        
        logger.info("All integration tests passed!")
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        raise
    finally:
        await test.teardown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_all_tests())