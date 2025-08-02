"""Trading API endpoints"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from app.models.trading import (
    OrderRequest, Order, Position, PortfolioSnapshot,
    BrokerConnection, DataProviderConnection, ExecutionReport,
    TradingSession, Alert, CircuitBreakerStatus,
    IBKRClientPortalConfig, IQFeedConfig, MarketDataSnapshot
)
from app.models.strategy import RiskManagementConfig
from app.services.trading.order_manager import OrderManager
from app.services.trading.ibkr_service import IBKRService
from app.services.trading.iqfeed_service import IQFeedService

logger = logging.getLogger(__name__)

router = APIRouter()

# Global instances (in production, use dependency injection)
order_manager = OrderManager()
ibkr_service = IBKRService()
iqfeed_service = IQFeedService()
active_session: Optional[TradingSession] = None


@router.post("/session/start")
async def start_trading_session(
    broker_config: Optional[IBKRClientPortalConfig] = None,
    data_config: Optional[IQFeedConfig] = None,
    strategies: List[str] = [],
    mode: str = "paper"
) -> TradingSession:
    """Start a new trading session"""
    global active_session
    
    try:
        # End existing session if any
        if active_session and active_session.is_active:
            await end_trading_session()
        
        # Initialize services
        success = await order_manager.initialize(broker_config, data_config)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to initialize trading services")
        
        # Get initial portfolio value
        portfolio = ibkr_service.portfolio_snapshot
        starting_balance = portfolio.total_value if portfolio else 100000.0
        
        # Create session
        active_session = TradingSession(
            start_time=datetime.utcnow(),
            broker_config=broker_config.dict() if broker_config else None,
            data_provider_config=data_config,
            active_strategies=strategies,
            mode=mode,
            starting_balance=starting_balance,
            current_balance=starting_balance
        )
        
        logger.info(f"Trading session started: {active_session.session_id}")
        return active_session
        
    except Exception as e:
        logger.error(f"Failed to start trading session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/end")
async def end_trading_session() -> Dict[str, Any]:
    """End the current trading session"""
    global active_session
    
    if not active_session:
        raise HTTPException(status_code=404, detail="No active trading session")
    
    try:
        # Get final portfolio value
        portfolio = ibkr_service.portfolio_snapshot
        if portfolio:
            active_session.current_balance = portfolio.total_value
            active_session.session_pnl = portfolio.total_value - active_session.starting_balance
            active_session.session_return = active_session.session_pnl / active_session.starting_balance
        
        # Update session stats
        stats = order_manager.get_order_statistics()
        active_session.orders_placed = stats["total_orders"]
        active_session.orders_filled = stats["filled_orders"]
        active_session.total_volume = stats["total_volume"]
        active_session.total_commission = stats["total_commission"]
        
        # Mark session as ended
        active_session.end_time = datetime.utcnow()
        active_session.is_active = False
        
        # Shutdown services
        await order_manager.shutdown()
        
        session_summary = {
            "session_id": active_session.session_id,
            "duration_minutes": (active_session.end_time - active_session.start_time).total_seconds() / 60,
            "total_pnl": active_session.session_pnl,
            "total_return": active_session.session_return,
            "orders_placed": active_session.orders_placed,
            "orders_filled": active_session.orders_filled,
            "total_volume": active_session.total_volume,
            "total_commission": active_session.total_commission
        }
        
        active_session = None
        
        logger.info("Trading session ended")
        return session_summary
        
    except Exception as e:
        logger.error(f"Failed to end trading session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/status")
async def get_session_status() -> Optional[TradingSession]:
    """Get current trading session status"""
    if not active_session:
        raise HTTPException(status_code=404, detail="No active trading session")
    
    # Update current balance
    portfolio = ibkr_service.portfolio_snapshot
    if portfolio:
        active_session.current_balance = portfolio.total_value
        active_session.session_pnl = portfolio.total_value - active_session.starting_balance
        active_session.session_return = active_session.session_pnl / active_session.starting_balance
    
    return active_session


@router.post("/orders/place")
async def place_order(
    request: OrderRequest,
    risk_config: Optional[RiskManagementConfig] = None
) -> Order:
    """Place a new order"""
    if not active_session or not active_session.is_active:
        raise HTTPException(status_code=400, detail="No active trading session")
    
    try:
        order = await order_manager.place_order(request, risk_config)
        
        if not order:
            raise HTTPException(status_code=400, detail="Order rejected")
        
        return order
        
    except Exception as e:
        logger.error(f"Failed to place order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/orders/{order_id}")
async def cancel_order(order_id: str) -> Dict[str, Any]:
    """Cancel an order"""
    if not active_session or not active_session.is_active:
        raise HTTPException(status_code=400, detail="No active trading session")
    
    try:
        success = await order_manager.cancel_order(order_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Order not found or cannot be cancelled")
        
        return {"status": "success", "order_id": order_id}
        
    except Exception as e:
        logger.error(f"Failed to cancel order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/orders/{order_id}")
async def modify_order(
    order_id: str,
    new_quantity: Optional[float] = None,
    new_limit_price: Optional[float] = None,
    new_stop_price: Optional[float] = None
) -> Dict[str, Any]:
    """Modify an existing order"""
    if not active_session or not active_session.is_active:
        raise HTTPException(status_code=400, detail="No active trading session")
    
    try:
        success = await order_manager.modify_order(
            order_id, new_quantity, new_limit_price, new_stop_price
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to modify order")
        
        return {"status": "success", "order_id": order_id}
        
    except Exception as e:
        logger.error(f"Failed to modify order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orders/{order_id}")
async def get_order_status(order_id: str) -> Order:
    """Get order status"""
    try:
        order = await order_manager.get_order_status(order_id)
        
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        
        return order
        
    except Exception as e:
        logger.error(f"Failed to get order status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orders")
async def get_orders(
    strategy_id: Optional[str] = None,
    active_only: bool = False
) -> List[Order]:
    """Get orders"""
    try:
        if active_only:
            orders = list(order_manager.active_orders.values())
        else:
            orders = list(order_manager.orders.values())
        
        # Filter by strategy if specified
        if strategy_id:
            orders = [o for o in orders if o.strategy_id == strategy_id]
        
        return orders
        
    except Exception as e:
        logger.error(f"Failed to get orders: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orders/{order_id}/execution")
async def get_execution_report(order_id: str) -> ExecutionReport:
    """Get execution quality report for an order"""
    try:
        report = await order_manager.get_execution_report(order_id)
        
        if not report:
            raise HTTPException(status_code=404, detail="Execution report not found")
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to get execution report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions")
async def get_positions() -> List[Position]:
    """Get current positions"""
    try:
        return await ibkr_service.get_positions()
        
    except Exception as e:
        logger.error(f"Failed to get positions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio")
async def get_portfolio() -> PortfolioSnapshot:
    """Get portfolio snapshot"""
    try:
        snapshot = ibkr_service.portfolio_snapshot
        
        if not snapshot:
            # Get fresh snapshot
            snapshot = await ibkr_service._update_portfolio_snapshot()
        
        if not snapshot:
            raise HTTPException(status_code=404, detail="Portfolio data not available")
        
        return snapshot
        
    except Exception as e:
        logger.error(f"Failed to get portfolio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market-data/{symbol}")
async def get_market_data(symbol: str) -> MarketDataSnapshot:
    """Get real-time market data for a symbol"""
    try:
        # Try IQFeed first
        quote = await iqfeed_service.get_latest_quote(symbol)
        
        if quote:
            return quote
        
        raise HTTPException(status_code=404, detail=f"No market data available for {symbol}")
        
    except Exception as e:
        logger.error(f"Failed to get market data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/market-data/subscribe")
async def subscribe_market_data(symbols: List[str]) -> Dict[str, Any]:
    """Subscribe to real-time market data"""
    try:
        success_count = 0
        failed_symbols = []
        
        for symbol in symbols:
            success = await iqfeed_service.watch_symbol(symbol)
            if success:
                success_count += 1
            else:
                failed_symbols.append(symbol)
        
        return {
            "subscribed": success_count,
            "failed": failed_symbols,
            "total": len(symbols)
        }
        
    except Exception as e:
        logger.error(f"Failed to subscribe to market data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/market-data/subscribe/{symbol}")
async def unsubscribe_market_data(symbol: str) -> Dict[str, Any]:
    """Unsubscribe from market data"""
    try:
        success = await iqfeed_service.unwatch_symbol(symbol)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to unsubscribe")
        
        return {"status": "success", "symbol": symbol}
        
    except Exception as e:
        logger.error(f"Failed to unsubscribe from market data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/connections")
async def get_connections() -> Dict[str, Any]:
    """Get connection status for all services"""
    return {
        "broker": ibkr_service.get_connection_status(),
        "data_provider": iqfeed_service.get_connection_status()
    }


@router.get("/circuit-breakers")
async def get_circuit_breakers() -> Dict[str, CircuitBreakerStatus]:
    """Get circuit breaker status"""
    return order_manager.get_circuit_breaker_status()


@router.post("/circuit-breakers/{breaker_id}/reset")
async def reset_circuit_breaker(breaker_id: str) -> Dict[str, Any]:
    """Reset a circuit breaker"""
    try:
        order_manager.reset_circuit_breaker(breaker_id)
        return {"status": "success", "breaker_id": breaker_id}
        
    except Exception as e:
        logger.error(f"Failed to reset circuit breaker: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trading/enable")
async def enable_trading() -> Dict[str, Any]:
    """Enable trading"""
    order_manager.enable_trading()
    return {"status": "success", "trading_enabled": True}


@router.post("/trading/disable")
async def disable_trading() -> Dict[str, Any]:
    """Disable trading"""
    order_manager.disable_trading()
    return {"status": "success", "trading_enabled": False}


@router.get("/statistics")
async def get_trading_statistics() -> Dict[str, Any]:
    """Get trading statistics"""
    stats = order_manager.get_order_statistics()
    
    # Add session statistics if available
    if active_session:
        stats["session"] = {
            "session_id": active_session.session_id,
            "start_time": active_session.start_time.isoformat(),
            "duration_minutes": (datetime.utcnow() - active_session.start_time).total_seconds() / 60,
            "mode": active_session.mode,
            "active_strategies": active_session.active_strategies
        }
    
    return stats


@router.get("/alerts")
async def get_alerts(
    since: Optional[datetime] = None,
    severity: Optional[str] = None
) -> List[Alert]:
    """Get trading alerts"""
    # This would need implementation to store and retrieve alerts
    return []


@router.websocket("/ws/market-data")
async def market_data_websocket(websocket):
    """WebSocket endpoint for streaming market data"""
    await websocket.accept()
    
    try:
        # Handler for streaming data
        async def stream_handler(data):
            await websocket.send_json(data.dict())
        
        # Subscribe to symbols requested by client
        while True:
            message = await websocket.receive_json()
            
            if message.get("action") == "subscribe":
                symbols = message.get("symbols", [])
                for symbol in symbols:
                    await iqfeed_service.watch_symbol(symbol, stream_handler)
            
            elif message.get("action") == "unsubscribe":
                symbols = message.get("symbols", [])
                for symbol in symbols:
                    await iqfeed_service.unwatch_symbol(symbol)
                    
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()