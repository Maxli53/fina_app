"""Order execution and management system"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import uuid

from app.models.trading import (
    OrderRequest, Order, OrderStatus, OrderSide, OrderType,
    Fill, ExecutionReport, Alert, CircuitBreakerStatus,
    BrokerType, ExecutionAlgorithm, TimeInForce
)
from app.models.strategy import RiskManagementConfig
from app.services.trading.ibkr_service import IBKRService
from app.services.trading.iqfeed_service import IQFeedService
from app.services.strategy.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class OrderManager:
    """Centralized order execution and management system"""
    
    def __init__(self):
        # Services
        self.ibkr_service = IBKRService()
        self.iqfeed_service = IQFeedService()
        self.risk_manager = RiskManager()
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.active_orders: Dict[str, Order] = {}
        self.fills: Dict[str, List[Fill]] = defaultdict(list)
        
        # Strategy tracking
        self.strategy_orders: Dict[str, List[str]] = defaultdict(list)
        self.strategy_positions: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreakerStatus] = {}
        self._init_circuit_breakers()
        
        # Event handlers
        self.order_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.fill_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.alert_handlers: List[Callable] = []
        
        # Execution state
        self.is_trading_enabled = True
        self.max_orders_per_minute = 100
        self.order_rate_limiter = []
        
        # Monitoring
        self.order_stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "rejected_orders": 0,
            "cancelled_orders": 0,
            "total_volume": 0.0,
            "total_commission": 0.0
        }
        
    def _init_circuit_breakers(self):
        """Initialize default circuit breakers"""
        # Daily loss circuit breaker
        self.circuit_breakers["daily_loss"] = CircuitBreakerStatus(
            breaker_id="daily_loss",
            name="Daily Loss Limit",
            status="active",
            threshold=0.05,  # 5% daily loss
            current_value=0.0,
            threshold_type="loss",
            actions=["halt_trading", "alert_only"]
        )
        
        # Order rejection rate circuit breaker
        self.circuit_breakers["rejection_rate"] = CircuitBreakerStatus(
            breaker_id="rejection_rate",
            name="Order Rejection Rate",
            status="active",
            threshold=0.2,  # 20% rejection rate
            current_value=0.0,
            threshold_type="error_rate",
            actions=["reduce_size", "alert_only"]
        )
        
        # Position concentration circuit breaker
        self.circuit_breakers["concentration"] = CircuitBreakerStatus(
            breaker_id="concentration",
            name="Position Concentration",
            status="active",
            threshold=0.3,  # 30% in single position
            current_value=0.0,
            threshold_type="concentration",
            actions=["halt_trading", "alert_only"]
        )
    
    async def initialize(self, broker_config=None, data_config=None) -> bool:
        """Initialize order manager with broker and data connections"""
        try:
            # Initialize broker connection
            broker_success = await self.ibkr_service.initialize(broker_config)
            if not broker_success:
                logger.error("Failed to initialize broker connection")
                return False
            
            # Initialize data feed
            data_success = await self.iqfeed_service.initialize(data_config)
            if not data_success:
                logger.warning("Failed to initialize IQFeed, continuing with broker data")
            
            # Start monitoring tasks
            asyncio.create_task(self._order_monitoring_loop())
            asyncio.create_task(self._circuit_breaker_monitoring_loop())
            
            logger.info("Order manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize order manager: {str(e)}")
            return False
    
    async def place_order(
        self, 
        request: OrderRequest, 
        risk_config: Optional[RiskManagementConfig] = None
    ) -> Optional[Order]:
        """Place an order with risk checks and execution"""
        try:
            # Check if trading is enabled
            if not self.is_trading_enabled:
                logger.warning("Trading is disabled")
                await self._create_alert(
                    "trading_disabled",
                    "critical",
                    "Trading Disabled",
                    f"Order rejected for {request.symbol} - trading is disabled"
                )
                return None
            
            # Check circuit breakers
            if not self._check_circuit_breakers(request):
                return None
            
            # Check rate limits
            if not self._check_rate_limits():
                logger.warning("Order rate limit exceeded")
                return None
            
            # Risk validation
            if risk_config:
                portfolio = await self._get_portfolio_snapshot()
                validation = self.risk_manager.validate_trade(
                    request.strategy_id,
                    request.symbol,
                    request.quantity * (request.limit_price or await self._get_current_price(request.symbol)),
                    portfolio,
                    risk_config
                )
                
                if not validation["approved"]:
                    logger.warning(f"Order rejected by risk manager: {validation['rejections']}")
                    await self._create_alert(
                        "risk_rejection",
                        "warning",
                        "Order Rejected by Risk Manager",
                        f"Order for {request.symbol} rejected: {', '.join(validation['rejections'])}"
                    )
                    return None
                
                # Apply suggested size if different
                if validation["suggested_size"] != request.quantity:
                    logger.info(f"Adjusting order size from {request.quantity} to {validation['suggested_size']}")
                    request.quantity = validation["suggested_size"]
            
            # Apply execution algorithm if specified
            if request.execution_algorithm:
                request = await self._apply_execution_algorithm(request)
            
            # Place order through broker
            order = await self.ibkr_service.place_order(request)
            
            if order:
                # Track order
                self.orders[order.order_id] = order
                self.active_orders[order.order_id] = order
                self.strategy_orders[request.strategy_id].append(order.order_id)
                
                # Update statistics
                self.order_stats["total_orders"] += 1
                self.order_rate_limiter.append(datetime.utcnow())
                
                # Notify handlers
                await self._notify_order_handlers(order)
                
                logger.info(f"Order placed successfully: {order.order_id}")
                return order
            else:
                self.order_stats["rejected_orders"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Failed to place order: {str(e)}")
            self.order_stats["rejected_orders"] += 1
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            if order_id not in self.active_orders:
                logger.warning(f"Order {order_id} not found or not active")
                return False
            
            # Cancel through broker
            success = await self.ibkr_service.cancel_order(order_id)
            
            if success:
                order = self.active_orders[order_id]
                order.status = OrderStatus.CANCELLED
                order.cancelled_at = datetime.utcnow()
                
                # Remove from active orders
                del self.active_orders[order_id]
                
                # Update statistics
                self.order_stats["cancelled_orders"] += 1
                
                # Notify handlers
                await self._notify_order_handlers(order)
                
                logger.info(f"Order cancelled: {order_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel order: {str(e)}")
            return False
    
    async def modify_order(
        self, 
        order_id: str, 
        new_quantity: Optional[float] = None,
        new_limit_price: Optional[float] = None,
        new_stop_price: Optional[float] = None
    ) -> bool:
        """Modify an existing order"""
        try:
            if order_id not in self.active_orders:
                logger.warning(f"Order {order_id} not found or not active")
                return False
            
            order = self.active_orders[order_id]
            
            # Cancel and replace (most brokers don't support true modify)
            success = await self.cancel_order(order_id)
            if not success:
                return False
            
            # Create new order request
            new_request = OrderRequest(
                symbol=order.symbol,
                quantity=new_quantity or order.quantity,
                side=order.side,
                order_type=order.order_type,
                limit_price=new_limit_price or order.limit_price,
                stop_price=new_stop_price or order.stop_price,
                strategy_id=order.strategy_id,
                parent_order_id=order_id,
                notes=f"Modified from order {order_id}"
            )
            
            # Place new order
            new_order = await self.place_order(new_request)
            
            return new_order is not None
            
        except Exception as e:
            logger.error(f"Failed to modify order: {str(e)}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get current order status"""
        try:
            # Get latest status from broker
            order = await self.ibkr_service.get_order_status(order_id)
            
            if order:
                # Update local tracking
                self.orders[order_id] = order
                
                # Update active orders
                if order.is_complete():
                    if order_id in self.active_orders:
                        del self.active_orders[order_id]
                    
                    # Update statistics
                    if order.status == OrderStatus.FILLED:
                        self.order_stats["filled_orders"] += 1
                        self.order_stats["total_volume"] += order.filled_quantity
                        self.order_stats["total_commission"] += order.commission
                
                return order
            
            # Return cached version if broker doesn't have it
            return self.orders.get(order_id)
            
        except Exception as e:
            logger.error(f"Failed to get order status: {str(e)}")
            return self.orders.get(order_id)
    
    async def get_execution_report(self, order_id: str) -> Optional[ExecutionReport]:
        """Get execution quality report"""
        try:
            return await self.ibkr_service.get_execution_report(order_id)
        except Exception as e:
            logger.error(f"Failed to get execution report: {str(e)}")
            return None
    
    def _check_circuit_breakers(self, request: OrderRequest) -> bool:
        """Check if any circuit breakers would prevent order"""
        for breaker_id, breaker in self.circuit_breakers.items():
            if breaker.is_tripped():
                logger.warning(f"Circuit breaker {breaker_id} is tripped")
                
                if "halt_trading" in breaker.actions:
                    asyncio.create_task(self._create_alert(
                        "circuit_breaker_tripped",
                        "critical",
                        f"Circuit Breaker: {breaker.name}",
                        f"Trading halted due to {breaker.name} breach"
                    ))
                    return False
                
                elif "reduce_size" in breaker.actions:
                    # Reduce order size by 50%
                    request.quantity *= 0.5
                    logger.info(f"Order size reduced due to circuit breaker {breaker_id}")
        
        return True
    
    def _check_rate_limits(self) -> bool:
        """Check order rate limits"""
        # Clean old entries
        cutoff_time = datetime.utcnow() - timedelta(minutes=1)
        self.order_rate_limiter = [
            t for t in self.order_rate_limiter if t > cutoff_time
        ]
        
        # Check limit
        if len(self.order_rate_limiter) >= self.max_orders_per_minute:
            logger.warning(f"Order rate limit exceeded: {len(self.order_rate_limiter)} orders in last minute")
            return False
        
        return True
    
    async def _apply_execution_algorithm(self, request: OrderRequest) -> OrderRequest:
        """Apply execution algorithm to order request"""
        if request.execution_algorithm == ExecutionAlgorithm.TWAP:
            # Time-weighted average price - split order over time
            # This is simplified - real TWAP would create child orders
            request.algorithm_params = {
                "duration_minutes": 30,
                "interval_seconds": 60
            }
        
        elif request.execution_algorithm == ExecutionAlgorithm.VWAP:
            # Volume-weighted average price
            request.algorithm_params = {
                "participation_rate": 0.1,  # 10% of volume
                "max_duration_minutes": 60
            }
        
        elif request.execution_algorithm == ExecutionAlgorithm.ICEBERG:
            # Show only part of order
            request.algorithm_params = {
                "display_size": min(request.quantity * 0.1, 100),  # Show 10% or 100 shares
                "variance": 0.2  # 20% variance in display size
            }
        
        return request
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol"""
        try:
            # Try IQFeed first
            quote = await self.iqfeed_service.get_latest_quote(symbol)
            if quote:
                return quote.last
            
            # Fallback to a default or last known price
            return 100.0  # Placeholder
            
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {str(e)}")
            return 100.0
    
    async def _get_portfolio_snapshot(self) -> Dict[str, Any]:
        """Get current portfolio snapshot"""
        snapshot = self.ibkr_service.portfolio_snapshot
        
        if snapshot:
            return {
                "total_value": snapshot.total_value,
                "positions": {p.symbol: p.market_value for p in snapshot.positions},
                "daily_pnl": snapshot.daily_pnl,
                "portfolio_volatility": 0.15  # Placeholder
            }
        
        return {
            "total_value": 100000,
            "positions": {},
            "daily_pnl": 0,
            "portfolio_volatility": 0.15
        }
    
    async def _order_monitoring_loop(self):
        """Monitor active orders and update status"""
        while True:
            try:
                for order_id in list(self.active_orders.keys()):
                    order = await self.get_order_status(order_id)
                    
                    if order and order.is_complete():
                        # Process completed order
                        if order.status == OrderStatus.FILLED:
                            await self._process_filled_order(order)
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Order monitoring error: {str(e)}")
                await asyncio.sleep(5)
    
    async def _circuit_breaker_monitoring_loop(self):
        """Monitor and update circuit breakers"""
        while True:
            try:
                # Update daily loss circuit breaker
                portfolio = await self._get_portfolio_snapshot()
                daily_pnl = portfolio.get("daily_pnl", 0)
                total_value = portfolio.get("total_value", 100000)
                
                if total_value > 0:
                    daily_return = daily_pnl / total_value
                    self.circuit_breakers["daily_loss"].current_value = abs(min(0, daily_return))
                    
                    if abs(daily_return) >= self.circuit_breakers["daily_loss"].threshold:
                        if self.circuit_breakers["daily_loss"].status != "tripped":
                            self.circuit_breakers["daily_loss"].status = "tripped"
                            self.circuit_breakers["daily_loss"].tripped_at = datetime.utcnow()
                            self.is_trading_enabled = False
                            
                            await self._create_alert(
                                "daily_loss_limit",
                                "critical",
                                "Daily Loss Limit Reached",
                                f"Trading halted: Daily loss of {abs(daily_return):.2%} exceeds limit"
                            )
                
                # Update rejection rate circuit breaker
                if self.order_stats["total_orders"] > 10:
                    rejection_rate = self.order_stats["rejected_orders"] / self.order_stats["total_orders"]
                    self.circuit_breakers["rejection_rate"].current_value = rejection_rate
                    
                    if rejection_rate >= self.circuit_breakers["rejection_rate"].threshold:
                        if self.circuit_breakers["rejection_rate"].status != "tripped":
                            self.circuit_breakers["rejection_rate"].status = "tripped"
                            self.circuit_breakers["rejection_rate"].tripped_at = datetime.utcnow()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Circuit breaker monitoring error: {str(e)}")
                await asyncio.sleep(30)
    
    async def _process_filled_order(self, order: Order):
        """Process a filled order"""
        try:
            # Update strategy positions
            strategy_id = order.strategy_id
            symbol = order.symbol
            
            if symbol not in self.strategy_positions[strategy_id]:
                self.strategy_positions[strategy_id][symbol] = 0
            
            if order.side in [OrderSide.BUY, OrderSide.COVER]:
                self.strategy_positions[strategy_id][symbol] += order.filled_quantity
            else:
                self.strategy_positions[strategy_id][symbol] -= order.filled_quantity
            
            # Create fill record
            fill = Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                quantity=order.filled_quantity,
                price=order.avg_fill_price or 0,
                side=order.side,
                commission=order.commission,
                executed_at=order.filled_at or datetime.utcnow()
            )
            
            self.fills[order.order_id].append(fill)
            
            # Notify handlers
            await self._notify_fill_handlers(fill)
            
        except Exception as e:
            logger.error(f"Failed to process filled order: {str(e)}")
    
    async def _create_alert(
        self, 
        alert_type: str, 
        severity: str, 
        title: str, 
        message: str,
        **kwargs
    ):
        """Create and dispatch alert"""
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            details=kwargs
        )
        
        # Notify alert handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {str(e)}")
        
        logger.warning(f"Alert: {title} - {message}")
    
    async def _notify_order_handlers(self, order: Order):
        """Notify order event handlers"""
        handlers = self.order_handlers.get(order.strategy_id, [])
        handlers.extend(self.order_handlers.get("*", []))  # Global handlers
        
        for handler in handlers:
            try:
                await handler(order)
            except Exception as e:
                logger.error(f"Order handler error: {str(e)}")
    
    async def _notify_fill_handlers(self, fill: Fill):
        """Notify fill event handlers"""
        order = self.orders.get(fill.order_id)
        if not order:
            return
        
        handlers = self.fill_handlers.get(order.strategy_id, [])
        handlers.extend(self.fill_handlers.get("*", []))  # Global handlers
        
        for handler in handlers:
            try:
                await handler(fill)
            except Exception as e:
                logger.error(f"Fill handler error: {str(e)}")
    
    def register_order_handler(self, strategy_id: str, handler: Callable):
        """Register handler for order events"""
        self.order_handlers[strategy_id].append(handler)
    
    def register_fill_handler(self, strategy_id: str, handler: Callable):
        """Register handler for fill events"""
        self.fill_handlers[strategy_id].append(handler)
    
    def register_alert_handler(self, handler: Callable):
        """Register handler for alerts"""
        self.alert_handlers.append(handler)
    
    def enable_trading(self):
        """Enable trading"""
        self.is_trading_enabled = True
        logger.info("Trading enabled")
    
    def disable_trading(self):
        """Disable trading"""
        self.is_trading_enabled = False
        logger.warning("Trading disabled")
    
    def reset_circuit_breaker(self, breaker_id: str):
        """Reset a circuit breaker"""
        if breaker_id in self.circuit_breakers:
            self.circuit_breakers[breaker_id].status = "active"
            self.circuit_breakers[breaker_id].current_value = 0
            self.circuit_breakers[breaker_id].reset_at = datetime.utcnow()
            logger.info(f"Circuit breaker {breaker_id} reset")
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """Get order execution statistics"""
        return self.order_stats.copy()
    
    def get_circuit_breaker_status(self) -> Dict[str, CircuitBreakerStatus]:
        """Get current circuit breaker status"""
        return self.circuit_breakers.copy()
    
    async def shutdown(self):
        """Shutdown order manager"""
        # Cancel all active orders
        for order_id in list(self.active_orders.keys()):
            await self.cancel_order(order_id)
        
        # Disconnect services
        await self.ibkr_service.disconnect()
        await self.iqfeed_service.disconnect()
        
        logger.info("Order manager shutdown complete")