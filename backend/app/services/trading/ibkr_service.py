"""Interactive Brokers Client Portal API service"""

import aiohttp
import asyncio
import logging
import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import ssl
import certifi

from app.models.trading import (
    IBKRClientPortalConfig, BrokerConnection, ConnectionStatus,
    OrderRequest, Order, OrderStatus, OrderSide, OrderType,
    Position, PortfolioSnapshot, Fill, ExecutionReport,
    BrokerType, TimeInForce
)

logger = logging.getLogger(__name__)


class IBKRService:
    """Service for Interactive Brokers Client Portal API integration"""
    
    def __init__(self):
        self.config: Optional[IBKRClientPortalConfig] = None
        self.connection = BrokerConnection(
            broker_type=BrokerType.IBKR_CP,
            status=ConnectionStatus.DISCONNECTED
        )
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.auth_token: Optional[str] = None
        self.last_auth_time: Optional[datetime] = None
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.order_map: Dict[str, str] = {}  # Internal ID -> IBKR ID
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.portfolio_snapshot: Optional[PortfolioSnapshot] = None
        
        # WebSocket for streaming
        self.ws_connection = None
        self.running = False
        
    async def initialize(self, config: Optional[IBKRClientPortalConfig] = None) -> bool:
        """Initialize IBKR connection"""
        try:
            # Load config from environment if not provided
            if config is None:
                config = IBKRClientPortalConfig(
                    username=os.getenv("IBKR_USERNAME", ""),
                    account_id=os.getenv("IBKR_ACCOUNT_ID", "")
                )
            
            self.config = config
            
            # Create SSL context for self-signed certificates
            ssl_context = ssl.create_default_context()
            if not self.config.verify_ssl:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            
            # Create session
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
            )
            
            # Authenticate
            success = await self._authenticate()
            if not success:
                return False
            
            # Get account info
            await self._get_account_info()
            
            self.connection.status = ConnectionStatus.CONNECTED
            self.connection.connected_at = datetime.utcnow()
            self.running = True
            
            # Start background tasks
            asyncio.create_task(self._keep_alive_loop())
            asyncio.create_task(self._position_update_loop())
            
            logger.info("IBKR service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize IBKR: {str(e)}")
            self.connection.status = ConnectionStatus.ERROR
            self.connection.error_message = str(e)
            return False
    
    async def _authenticate(self) -> bool:
        """Authenticate with IBKR Client Portal"""
        try:
            # Check if already authenticated
            status_url = f"{self.config.gateway_url}/iserver/auth/status"
            async with self.session.post(status_url) as response:
                data = await response.json()
                
                if data.get("authenticated", False):
                    self.connection.is_authenticated = True
                    self.last_auth_time = datetime.utcnow()
                    logger.info("Already authenticated with IBKR")
                    return True
            
            # Initiate authentication
            init_url = f"{self.config.gateway_url}/iserver/auth/ssodh/init"
            async with self.session.post(
                init_url,
                json={
                    "username": self.config.username,
                    "password": os.getenv("IBKR_PASSWORD", "")
                }
            ) as response:
                if response.status != 200:
                    logger.error(f"Authentication init failed: {response.status}")
                    return False
            
            # Wait for authentication to complete
            max_attempts = 30
            for attempt in range(max_attempts):
                await asyncio.sleep(2)
                
                async with self.session.post(status_url) as response:
                    data = await response.json()
                    
                    if data.get("authenticated", False):
                        self.connection.is_authenticated = True
                        self.last_auth_time = datetime.utcnow()
                        logger.info("Successfully authenticated with IBKR")
                        return True
                    
                    if data.get("competing", False):
                        logger.warning("Competing session detected")
                        # Handle competing session
                        continue
            
            logger.error("Authentication timeout")
            return False
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False
    
    async def _get_account_info(self) -> bool:
        """Get account information"""
        try:
            # Get accounts
            accounts_url = f"{self.config.gateway_url}/portfolio/accounts"
            async with self.session.get(accounts_url) as response:
                accounts = await response.json()
                
                if accounts:
                    # Use first account if not specified
                    if not self.config.account_id and len(accounts) > 0:
                        self.config.account_id = accounts[0]["accountId"]
                    
                    self.connection.account_id = self.config.account_id
                    
                    # Get account details
                    for account in accounts:
                        if account["accountId"] == self.config.account_id:
                            self.connection.account_type = account.get("type", "")
                            self.connection.base_currency = account.get("currency", "USD")
                            break
            
            # Get portfolio summary
            await self._update_portfolio_snapshot()
            
            logger.info(f"Account info loaded: {self.connection.account_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to get account info: {str(e)}")
            return False
    
    async def place_order(self, request: OrderRequest) -> Optional[Order]:
        """Place an order through IBKR"""
        try:
            # Get contract ID if not provided
            if not request.conid:
                conid = await self._get_contract_id(request.symbol)
                if not conid:
                    logger.error(f"Failed to get contract ID for {request.symbol}")
                    return None
                request.conid = conid
            
            # Build IBKR order
            ibkr_order = self._build_ibkr_order(request)
            
            # Place order
            order_url = f"{self.config.gateway_url}/iserver/account/{self.config.account_id}/orders"
            
            async with self.session.post(
                order_url,
                json={"orders": [ibkr_order]}
            ) as response:
                result = await response.json()
                
                # Handle order confirmation
                if isinstance(result, list) and len(result) > 0:
                    order_data = result[0]
                    
                    # Check if confirmation required
                    if "id" in order_data and order_data.get("order_status") == "PreSubmitted":
                        # Confirm order
                        confirm_url = f"{order_url}/{order_data['id']}/confirm"
                        async with self.session.post(confirm_url) as confirm_response:
                            confirm_result = await confirm_response.json()
                            
                            if confirm_result.get("order_status") == "Submitted":
                                order_data = confirm_result
                    
                    # Create order object
                    order = Order(
                        symbol=request.symbol,
                        quantity=request.quantity,
                        side=request.side,
                        order_type=request.order_type,
                        status=OrderStatus.SUBMITTED,
                        limit_price=request.limit_price,
                        stop_price=request.stop_price,
                        strategy_id=request.strategy_id,
                        submitted_at=datetime.utcnow(),
                        broker_order_id=str(order_data.get("order_id", "")),
                        conid=request.conid
                    )
                    
                    # Track order
                    self.active_orders[order.order_id] = order
                    if order.broker_order_id:
                        self.order_map[order.order_id] = order.broker_order_id
                    
                    logger.info(f"Order placed: {order.order_id} -> {order.broker_order_id}")
                    return order
                else:
                    logger.error(f"Order placement failed: {result}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to place order: {str(e)}")
            return None
    
    def _build_ibkr_order(self, request: OrderRequest) -> Dict[str, Any]:
        """Build IBKR order format"""
        # Map order types
        order_type_map = {
            OrderType.MARKET: "MKT",
            OrderType.LIMIT: "LMT",
            OrderType.STOP: "STP",
            OrderType.STOP_LIMIT: "STP LMT",
            OrderType.TRAILING_STOP: "TRAIL",
            OrderType.MOC: "MOC",
            OrderType.LOC: "LOC"
        }
        
        # Map time in force
        tif_map = {
            TimeInForce.DAY: "DAY",
            TimeInForce.GTC: "GTC",
            TimeInForce.IOC: "IOC",
            TimeInForce.FOK: "FOK",
            TimeInForce.GTD: "GTD",
            TimeInForce.OPG: "OPG",
            TimeInForce.CLO: "CLO"
        }
        
        # Build order
        ibkr_order = {
            "conid": request.conid,
            "orderType": order_type_map.get(request.order_type, "MKT"),
            "side": "BUY" if request.side in [OrderSide.BUY, OrderSide.COVER] else "SELL",
            "quantity": request.quantity,
            "tif": tif_map.get(request.time_in_force, "DAY"),
            "useAdaptive": True  # Use IBKR smart routing
        }
        
        # Add price fields
        if request.limit_price:
            ibkr_order["price"] = request.limit_price
        if request.stop_price:
            ibkr_order["auxPrice"] = request.stop_price
        
        # Add execution algorithm
        if request.execution_algorithm:
            algo_map = {
                "vwap": {"strategyType": "Vwap", "strategyParameters": {}},
                "twap": {"strategyType": "Twap", "strategyParameters": {}},
                "adaptive": {"strategyType": "Adaptive", "strategyParameters": {"adaptivePriority": "Normal"}}
            }
            
            if request.execution_algorithm.value in algo_map:
                ibkr_order.update(algo_map[request.execution_algorithm.value])
        
        return ibkr_order
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            if order_id not in self.active_orders:
                logger.warning(f"Order {order_id} not found")
                return False
            
            order = self.active_orders[order_id]
            
            if not order.broker_order_id:
                logger.error(f"No broker order ID for {order_id}")
                return False
            
            # Cancel order
            cancel_url = f"{self.config.gateway_url}/iserver/account/{self.config.account_id}/order/{order.broker_order_id}"
            
            async with self.session.delete(cancel_url) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if result.get("order_status") == "Cancelled":
                        order.status = OrderStatus.CANCELLED
                        order.cancelled_at = datetime.utcnow()
                        logger.info(f"Order cancelled: {order_id}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel order: {str(e)}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get current order status"""
        try:
            if order_id not in self.active_orders:
                return None
            
            order = self.active_orders[order_id]
            
            if order.broker_order_id:
                # Get order status from IBKR
                status_url = f"{self.config.gateway_url}/iserver/account/order/status/{order.broker_order_id}"
                
                async with self.session.get(status_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Update order status
                        status_map = {
                            "PendingSubmit": OrderStatus.PENDING,
                            "PreSubmitted": OrderStatus.PENDING,
                            "Submitted": OrderStatus.SUBMITTED,
                            "Filled": OrderStatus.FILLED,
                            "Cancelled": OrderStatus.CANCELLED,
                            "Inactive": OrderStatus.EXPIRED
                        }
                        
                        ibkr_status = data.get("order_status", "")
                        if ibkr_status in status_map:
                            order.status = status_map[ibkr_status]
                        
                        # Update fill information
                        if data.get("filled_quantity"):
                            order.filled_quantity = float(data["filled_quantity"])
                        if data.get("avg_fill_price"):
                            order.avg_fill_price = float(data["avg_fill_price"])
                        if data.get("commission"):
                            order.commission = float(data["commission"])
                        
                        order.last_update = datetime.utcnow()
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to get order status: {str(e)}")
            return None
    
    async def get_positions(self) -> List[Position]:
        """Get current positions"""
        try:
            positions_url = f"{self.config.gateway_url}/portfolio/{self.config.account_id}/positions/0"
            
            async with self.session.get(positions_url) as response:
                if response.status == 200:
                    positions_data = await response.json()
                    
                    positions = []
                    for pos_data in positions_data:
                        position = Position(
                            symbol=pos_data.get("contractDesc", ""),
                            quantity=float(pos_data.get("position", 0)),
                            avg_cost=float(pos_data.get("avgCost", 0)),
                            current_price=float(pos_data.get("mktPrice", 0)),
                            market_value=float(pos_data.get("mktValue", 0)),
                            unrealized_pnl=float(pos_data.get("unrealizedPnl", 0)),
                            realized_pnl=float(pos_data.get("realizedPnl", 0)),
                            total_pnl=float(pos_data.get("unrealizedPnl", 0)) + float(pos_data.get("realizedPnl", 0)),
                            pnl_percent=float(pos_data.get("unrealizedPnl", 0)) / float(pos_data.get("avgCost", 1)) if pos_data.get("avgCost") else 0,
                            side="long" if float(pos_data.get("position", 0)) > 0 else "short",
                            opened_at=datetime.utcnow(),  # IBKR doesn't provide this
                            conid=pos_data.get("conid"),
                            account_id=self.config.account_id
                        )
                        positions.append(position)
                        
                        # Update tracking
                        self.positions[position.symbol] = position
                    
                    return positions
                    
        except Exception as e:
            logger.error(f"Failed to get positions: {str(e)}")
            return []
    
    async def _update_portfolio_snapshot(self) -> Optional[PortfolioSnapshot]:
        """Update portfolio snapshot"""
        try:
            # Get account summary
            summary_url = f"{self.config.gateway_url}/portfolio/{self.config.account_id}/summary"
            
            async with self.session.get(summary_url) as response:
                if response.status == 200:
                    summary = await response.json()
                    
                    # Get positions
                    positions = await self.get_positions()
                    
                    # Create snapshot
                    self.portfolio_snapshot = PortfolioSnapshot(
                        account_id=self.config.account_id,
                        total_value=float(summary.get("netLiquidation", {}).get("value", 0)),
                        cash_balance=float(summary.get("totalCashValue", {}).get("value", 0)),
                        securities_value=float(summary.get("grossPositionValue", {}).get("value", 0)),
                        daily_pnl=float(summary.get("dailyPnL", {}).get("value", 0)),
                        unrealized_pnl=float(summary.get("unrealizedPnL", {}).get("value", 0)),
                        realized_pnl=float(summary.get("realizedPnL", {}).get("value", 0)),
                        positions=positions,
                        position_count=len(positions),
                        buying_power=float(summary.get("buyingPower", {}).get("value", 0)),
                        margin_used=float(summary.get("marginUsed", {}).get("value", 0)) if summary.get("marginUsed") else 0,
                        margin_available=float(summary.get("availableFunds", {}).get("value", 0)) if summary.get("availableFunds") else 0
                    )
                    
                    return self.portfolio_snapshot
                    
        except Exception as e:
            logger.error(f"Failed to update portfolio snapshot: {str(e)}")
            return None
    
    async def _get_contract_id(self, symbol: str) -> Optional[int]:
        """Get IBKR contract ID for a symbol"""
        try:
            search_url = f"{self.config.gateway_url}/iserver/secdef/search"
            
            async with self.session.post(
                search_url,
                json={"symbol": symbol, "name": False, "secType": "STK"}
            ) as response:
                if response.status == 200:
                    results = await response.json()
                    
                    if results and len(results) > 0:
                        # Use first result
                        return results[0].get("conid")
                        
        except Exception as e:
            logger.error(f"Failed to get contract ID for {symbol}: {str(e)}")
            
        return None
    
    async def _keep_alive_loop(self):
        """Keep connection alive"""
        while self.running:
            try:
                # Send tickle to keep session alive
                tickle_url = f"{self.config.gateway_url}/tickle"
                async with self.session.post(tickle_url) as response:
                    data = await response.json()
                    
                    if data.get("iserver", {}).get("authStatus", {}).get("authenticated", False):
                        self.connection.last_heartbeat = datetime.utcnow()
                    else:
                        # Re-authenticate if needed
                        logger.warning("Session expired, re-authenticating")
                        await self._authenticate()
                
                await asyncio.sleep(self.config.keep_alive_interval)
                
            except Exception as e:
                logger.error(f"Keep-alive error: {str(e)}")
                await asyncio.sleep(30)
    
    async def _position_update_loop(self):
        """Update positions periodically"""
        while self.running:
            try:
                await self._update_portfolio_snapshot()
                
                # Update active orders
                for order_id in list(self.active_orders.keys()):
                    order = await self.get_order_status(order_id)
                    
                    if order and order.is_complete():
                        # Remove from active orders
                        del self.active_orders[order_id]
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Position update error: {str(e)}")
                await asyncio.sleep(10)
    
    async def get_execution_report(self, order_id: str) -> Optional[ExecutionReport]:
        """Get execution quality report for an order"""
        try:
            if order_id not in self.active_orders and order_id not in self.order_map:
                return None
            
            order = self.active_orders.get(order_id)
            if not order or not order.is_complete():
                return None
            
            # Get fills for the order
            fills_url = f"{self.config.gateway_url}/iserver/account/orders/{order.broker_order_id}/fills"
            
            async with self.session.get(fills_url) as response:
                if response.status == 200:
                    fills_data = await response.json()
                    
                    # Calculate execution metrics
                    total_quantity = sum(f.get("quantity", 0) for f in fills_data)
                    avg_price = sum(f.get("price", 0) * f.get("quantity", 0) for f in fills_data) / total_quantity if total_quantity > 0 else 0
                    
                    # Get market data at order time
                    # This would need real market data integration
                    arrival_price = order.limit_price or avg_price
                    
                    return ExecutionReport(
                        order_id=order_id,
                        symbol=order.symbol,
                        side=order.side,
                        intended_quantity=order.quantity,
                        filled_quantity=order.filled_quantity,
                        fill_rate=order.filled_quantity / order.quantity if order.quantity > 0 else 0,
                        arrival_price=arrival_price,
                        avg_execution_price=avg_price,
                        benchmark_price=avg_price,  # Would need VWAP calculation
                        implementation_shortfall=(avg_price - arrival_price) * order.filled_quantity,
                        market_impact=0.0,  # Would need calculation
                        timing_cost=0.0,  # Would need calculation
                        spread_cost=(avg_price - arrival_price) * 0.5,  # Simplified
                        total_cost=order.commission,
                        execution_time_seconds=(order.filled_at - order.submitted_at).total_seconds() if order.filled_at and order.submitted_at else 0,
                        number_of_fills=len(fills_data),
                        venues_used=list(set(f.get("exchange", "UNKNOWN") for f in fills_data)),
                        execution_score=85.0,  # Would need calculation
                        price_improvement=0.0,  # Would need calculation
                        speed_score=90.0  # Would need calculation
                    )
                    
        except Exception as e:
            logger.error(f"Failed to get execution report: {str(e)}")
            return None
    
    async def disconnect(self):
        """Disconnect from IBKR"""
        self.running = False
        
        if self.session:
            await self.session.close()
        
        self.connection.status = ConnectionStatus.DISCONNECTED
        self.connection.connected_at = None
        
        logger.info("IBKR service disconnected")
    
    def get_connection_status(self) -> BrokerConnection:
        """Get current connection status"""
        return self.connection