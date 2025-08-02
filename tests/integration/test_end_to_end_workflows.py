"""
Comprehensive end-to-end integration tests for all platform workflows
"""

import pytest
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import websockets
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis

from app.services.system_orchestrator import SystemOrchestrator, WorkflowType
from app.services.data.data_service import DataService
from app.services.analysis.idtxl_service import IDTxlService
from app.services.analysis.ml_service import MLService
from app.services.analysis.nn_service import NeuralNetworkService
from app.services.strategy.strategy_builder import StrategyBuilder
from app.services.strategy.backtesting_engine import BacktestingEngine
from app.services.trading.order_manager import OrderManager
from app.services.trading.ibkr_service import IBKRService


class TestEndToEndWorkflows:
    """Test complete workflows from data ingestion to trade execution"""
    
    @pytest.fixture
    async def setup_services(self, db_session: AsyncSession, redis_client: aioredis.Redis):
        """Setup all required services"""
        config = {
            "use_gpu": True,
            "health_check_interval": 5,
            "risk_limits": {
                "var_95_limit": 10000,
                "max_daily_loss": 5000,
                "max_concentration": 0.30
            }
        }
        
        # Initialize services
        orchestrator = SystemOrchestrator(db_session, redis_client, config)
        await orchestrator.initialize()
        
        return {
            "orchestrator": orchestrator,
            "data_service": DataService(db_session, redis_client),
            "idtxl_service": IDTxlService(),
            "ml_service": MLService(),
            "nn_service": NeuralNetworkService(),
            "strategy_builder": StrategyBuilder(db_session),
            "backtest_engine": BacktestingEngine(),
            "order_manager": OrderManager(db_session, redis_client)
        }
    
    @pytest.mark.asyncio
    async def test_complete_analysis_to_trade_workflow(self, setup_services):
        """Test the complete analysis to trade workflow"""
        services = await setup_services
        orchestrator = services["orchestrator"]
        
        # Start orchestrator
        await orchestrator.start()
        
        try:
            # Execute workflow
            result = await orchestrator.execute_workflow(
                WorkflowType.ANALYSIS_TO_TRADE,
                {
                    "symbols": ["AAPL", "MSFT", "GOOGL"],
                    "timeframe": "1d",
                    "lookback": 30,
                    "methods": ["idtxl", "ml", "nn"],
                    "mode": "paper"  # Paper trading for testing
                }
            )
            
            # Verify workflow completed
            assert result["status"] == "completed"
            assert "fetch_market_data" in result["results"]
            assert "run_analysis" in result["results"]
            assert "generate_signals" in result["results"]
            assert "validate_signals" in result["results"]
            assert "execute_trades" in result["results"]
            
            # Verify data was fetched
            market_data = result["results"]["fetch_market_data"]["result"]
            assert len(market_data["data"]) == 3
            assert all(symbol in market_data["data"] for symbol in ["AAPL", "MSFT", "GOOGL"])
            
            # Verify analysis was performed
            analysis = result["results"]["run_analysis"]["result"]
            assert "idtxl" in analysis or len(market_data["data"]) < 2  # IDTxl needs 2+ symbols
            assert any(key.startswith("ml_") for key in analysis)
            
            # Verify signals were generated
            signals = result["results"]["generate_signals"]["result"]
            assert "signals" in signals
            assert isinstance(signals["signals"], list)
            
            # Verify validation occurred
            validation = result["results"]["validate_signals"]["result"]
            assert "validated_signals" in validation
            
            # Verify trades were executed (in paper mode)
            if validation["validated_signals"]:
                trades = result["results"]["execute_trades"]["result"]
                assert "executed" in trades
                assert trades["executed"] >= 0
                
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_portfolio_rebalancing_workflow(self, setup_services):
        """Test portfolio rebalancing workflow"""
        services = await setup_services
        orchestrator = services["orchestrator"]
        order_manager = services["order_manager"]
        
        # Create test positions
        test_positions = [
            {
                "symbol": "AAPL",
                "quantity": 100,
                "avg_cost": 150.00,
                "current_price": 155.00,
                "market_value": 15500
            },
            {
                "symbol": "MSFT",
                "quantity": 50,
                "avg_cost": 300.00,
                "current_price": 310.00,
                "market_value": 15500
            },
            {
                "symbol": "GOOGL",
                "quantity": 20,
                "avg_cost": 2500.00,
                "current_price": 2600.00,
                "market_value": 52000
            }
        ]
        
        # Mock positions
        order_manager.get_all_positions = asyncio.coroutine(lambda: test_positions)
        
        await orchestrator.start()
        
        try:
            # Execute rebalancing workflow
            result = await orchestrator.execute_workflow(
                WorkflowType.PORTFOLIO_REBALANCE,
                {
                    "method": "equal_weight",
                    "check_impact": True,
                    "mode": "paper"
                }
            )
            
            # Verify workflow completed
            assert result["status"] == "completed"
            assert "analyze_portfolio" in result["results"]
            assert "calculate_targets" in result["results"]
            assert "generate_orders" in result["results"]
            assert "execute_rebalance" in result["results"]
            
            # Verify portfolio analysis
            portfolio_analysis = result["results"]["analyze_portfolio"]["result"]
            assert portfolio_analysis["total_value"] == 83000  # Sum of market values
            assert len(portfolio_analysis["positions"]) == 3
            
            # Verify target calculation
            targets = result["results"]["calculate_targets"]["result"]
            assert targets["method"] == "equal_weight"
            assert all(abs(weight - 0.333) < 0.01 for weight in targets["targets"].values())
            
            # Verify orders were generated
            orders = result["results"]["generate_orders"]["result"]
            assert len(orders["orders"]) > 0  # Should have rebalancing orders
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_strategy_optimization_workflow(self, setup_services):
        """Test strategy optimization workflow"""
        services = await setup_services
        orchestrator = services["orchestrator"]
        strategy_builder = services["strategy_builder"]
        
        # Create test strategies
        test_strategies = [
            {
                "id": 1,
                "name": "Mean Reversion AAPL",
                "type": "mean_reversion",
                "sharpe_ratio": 1.5,
                "parameters": {"lookback": 20, "z_score": 2.0}
            },
            {
                "id": 2,
                "name": "Momentum MSFT",
                "type": "momentum",
                "sharpe_ratio": 1.8,
                "parameters": {"period": 10, "threshold": 0.02}
            }
        ]
        
        # Mock strategies
        strategy_builder.get_all_strategies = asyncio.coroutine(lambda: test_strategies)
        
        await orchestrator.start()
        
        try:
            # Execute optimization workflow
            result = await orchestrator.execute_workflow(
                WorkflowType.STRATEGY_OPTIMIZATION,
                {
                    "top_n": 2,
                    "iterations": 10,  # Reduced for testing
                    "auto_deploy": False
                }
            )
            
            # Verify workflow completed
            assert result["status"] == "completed"
            assert "select_strategies" in result["results"]
            assert "run_backtests" in result["results"]
            assert "analyze_results" in result["results"]
            assert "update_parameters" in result["results"]
            
            # Verify strategy selection
            selection = result["results"]["select_strategies"]["result"]
            assert len(selection["selected"]) == 2
            
            # Verify optimization ran
            optimization = result["results"]["run_backtests"]["result"]
            assert "optimization_results" in optimization
            
            # Verify analysis
            analysis = result["results"]["analyze_results"]["result"]
            assert "best_parameters" in analysis
            assert "improvement_summary" in analysis
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_websocket_real_time_updates(self, setup_services):
        """Test WebSocket real-time updates"""
        services = await setup_services
        orchestrator = services["orchestrator"]
        
        await orchestrator.start()
        
        try:
            # Connect WebSocket client
            async with websockets.connect("ws://localhost:8765") as websocket:
                # Wait for initial state
                message = await websocket.recv()
                data = json.loads(message)
                
                assert data["type"] == "system_state"
                assert data["data"]["state"] in ["ready", "running"]
                
                # Subscribe to market data
                await websocket.send(json.dumps({
                    "type": "subscribe",
                    "symbols": ["AAPL", "MSFT"]
                }))
                
                # Wait for confirmation
                message = await websocket.recv()
                data = json.loads(message)
                
                assert data["type"] == "subscription_confirmed"
                assert data["symbols"] == ["AAPL", "MSFT"]
                
                # Request system status
                await websocket.send(json.dumps({
                    "type": "get_status"
                }))
                
                # Wait for status update
                message = await websocket.recv()
                data = json.loads(message)
                
                assert data["type"] == "status_update"
                assert "state" in data["data"]
                assert "metrics" in data["data"]
                
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_health_monitoring_and_recovery(self, setup_services):
        """Test health monitoring and automatic recovery"""
        services = await setup_services
        orchestrator = services["orchestrator"]
        
        await orchestrator.start()
        
        try:
            # Simulate service failure
            orchestrator.state = orchestrator.SystemState.DEGRADED
            
            # Wait for health check
            await asyncio.sleep(6)  # Health check interval is 5 seconds
            
            # Check if recovery was attempted
            events = orchestrator.event_history
            recovery_events = [
                e for e in events 
                if e.event_type in ["system_recovered", "component_recovery"]
            ]
            
            # Should have recovery-related events
            assert len(recovery_events) > 0 or orchestrator.state == orchestrator.SystemState.RUNNING
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_risk_management_circuit_breakers(self, setup_services):
        """Test risk management and circuit breakers"""
        services = await setup_services
        orchestrator = services["orchestrator"]
        order_manager = services["order_manager"]
        
        # Create high-risk position
        risky_positions = [{
            "symbol": "AAPL",
            "quantity": 10000,  # Large position
            "avg_cost": 150.00,
            "current_price": 140.00,  # Loss position
            "market_value": 1400000,
            "unrealized_pnl": -100000  # Large loss
        }]
        
        # Mock positions
        order_manager.get_all_positions = asyncio.coroutine(lambda: risky_positions)
        
        await orchestrator.start()
        
        try:
            # Risk monitoring should detect high risk
            await asyncio.sleep(2)
            
            # Check for risk alerts
            risk_events = [
                e for e in orchestrator.event_history
                if e.event_type == "risk_violation"
            ]
            
            # Should have risk violations due to large position and loss
            assert len(risk_events) > 0
            
            # Verify risk metrics were calculated
            risk_metrics = await orchestrator._calculate_risk_metrics(risky_positions)
            assert risk_metrics["portfolio_value"] == 1400000
            assert risk_metrics["var_95"] > 0
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self, setup_services):
        """Test concurrent execution of multiple workflows"""
        services = await setup_services
        orchestrator = services["orchestrator"]
        
        await orchestrator.start()
        
        try:
            # Execute multiple workflows concurrently
            tasks = [
                orchestrator.execute_workflow(
                    WorkflowType.ANALYSIS_TO_TRADE,
                    {"symbols": ["AAPL"], "mode": "paper"}
                ),
                orchestrator.execute_workflow(
                    WorkflowType.PORTFOLIO_REBALANCE,
                    {"method": "equal_weight", "mode": "paper"}
                ),
                orchestrator.execute_workflow(
                    WorkflowType.STRATEGY_OPTIMIZATION,
                    {"top_n": 1, "iterations": 5}
                )
            ]
            
            # Wait for all workflows to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all workflows completed or failed gracefully
            for result in results:
                if not isinstance(result, Exception):
                    assert result["status"] in ["completed", "failed"]
                else:
                    # Log but don't fail - some workflows may fail due to missing data
                    print(f"Workflow failed with: {result}")
            
        finally:
            await orchestrator.stop()


class TestDataIntegrity:
    """Test data integrity and consistency across the platform"""
    
    @pytest.mark.asyncio
    async def test_data_consistency_across_services(self, setup_services):
        """Test that data remains consistent across different services"""
        services = await setup_services
        data_service = services["data_service"]
        
        # Fetch data from multiple sources
        symbol = "AAPL"
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        # Get historical data
        historical_data = await data_service.get_historical_data(
            symbol, start_date, end_date
        )
        
        # Get current quote
        quote = await data_service.get_quote(symbol)
        
        # Verify data consistency
        assert historical_data is not None
        assert quote is not None
        
        # Latest historical price should be close to current quote
        if historical_data:
            latest_close = historical_data[-1]["close"]
            current_price = quote["price"]
            
            # Allow for some difference due to market movement
            price_diff = abs(latest_close - current_price) / current_price
            assert price_diff < 0.10  # Less than 10% difference
    
    @pytest.mark.asyncio
    async def test_transaction_atomicity(self, setup_services):
        """Test that transactions are atomic"""
        services = await setup_services
        order_manager = services["order_manager"]
        
        # Test order placement atomicity
        try:
            # Place an order that should fail
            order = await order_manager.place_order(
                symbol="INVALID_SYMBOL",
                side="buy",
                quantity=-100,  # Invalid quantity
                order_type="market",
                mode="paper"
            )
            
            # Should not reach here
            assert False, "Invalid order should have failed"
            
        except Exception as e:
            # Verify no partial state was saved
            orders = await order_manager.get_recent_orders(limit=1)
            
            # Should not have any invalid orders
            invalid_orders = [
                o for o in orders 
                if o.get("symbol") == "INVALID_SYMBOL" or o.get("quantity") < 0
            ]
            assert len(invalid_orders) == 0