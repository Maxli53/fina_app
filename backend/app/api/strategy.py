from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import uuid
import logging

from app.models.strategy import (
    Strategy, StrategyConfig, BacktestConfig, BacktestResult,
    StrategyStatus, StrategyOptimizationConfig, StrategyOptimizationResult
)
from app.services.strategy.strategy_builder import StrategyBuilderService
from app.services.strategy.backtesting_engine import BacktestingEngine
from app.services.strategy.risk_manager import RiskManager

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory storage (will be replaced with database)
strategies: Dict[str, Strategy] = {}
backtest_tasks: Dict[str, Dict[str, Any]] = {}
optimization_tasks: Dict[str, Dict[str, Any]] = {}

# Initialize services
strategy_builder = StrategyBuilderService()
backtesting_engine = BacktestingEngine()
risk_manager = RiskManager()


@router.post("/create")
async def create_strategy(config: StrategyConfig) -> Strategy:
    """Create a new trading strategy"""
    try:
        strategy_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        # Validate strategy configuration
        validation_result = await strategy_builder.validate_strategy(config)
        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy configuration: {validation_result['errors']}"
            )
        
        # Create strategy
        strategy = Strategy(
            id=strategy_id,
            config=config,
            created_at=now,
            updated_at=now
        )
        
        strategies[strategy_id] = strategy
        
        logger.info(f"Created strategy: {strategy_id} - {config.name}")
        return strategy
        
    except Exception as e:
        logger.error(f"Failed to create strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_strategies(
    strategy_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100)
) -> List[Strategy]:
    """List all strategies with optional filtering"""
    filtered_strategies = list(strategies.values())
    
    if strategy_type:
        filtered_strategies = [s for s in filtered_strategies if s.config.strategy_type == strategy_type]
    
    if status:
        filtered_strategies = [s for s in filtered_strategies if s.live_status and s.live_status.status == status]
    
    return filtered_strategies[:limit]


@router.get("/{strategy_id}")
async def get_strategy(strategy_id: str) -> Strategy:
    """Get a specific strategy by ID"""
    if strategy_id not in strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    return strategies[strategy_id]


@router.put("/{strategy_id}")
async def update_strategy(strategy_id: str, config: StrategyConfig) -> Strategy:
    """Update an existing strategy"""
    if strategy_id not in strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    try:
        # Validate updated configuration
        validation_result = await strategy_builder.validate_strategy(config)
        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy configuration: {validation_result['errors']}"
            )
        
        # Update strategy
        strategy = strategies[strategy_id]
        strategy.config = config
        strategy.updated_at = datetime.utcnow()
        strategy.version += 1
        
        # Clear previous backtest results if configuration changed significantly
        if strategy.backtest_results:
            # Check if key parameters changed
            needs_rebacktest = await strategy_builder.check_rebacktest_needed(
                strategy.backtest_results, config
            )
            if needs_rebacktest:
                strategy.backtest_results = None
        
        logger.info(f"Updated strategy: {strategy_id}")
        return strategy
        
    except Exception as e:
        logger.error(f"Failed to update strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{strategy_id}")
async def delete_strategy(strategy_id: str) -> Dict[str, str]:
    """Delete a strategy"""
    if strategy_id not in strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    # Stop strategy if it's running
    strategy = strategies[strategy_id]
    if strategy.live_status and strategy.live_status.status == "active":
        await stop_strategy(strategy_id)
    
    del strategies[strategy_id]
    logger.info(f"Deleted strategy: {strategy_id}")
    
    return {"message": "Strategy deleted successfully"}


@router.post("/{strategy_id}/backtest")
async def start_backtest(
    strategy_id: str,
    config: BacktestConfig,
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Start backtesting a strategy"""
    if strategy_id not in strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    task_id = str(uuid.uuid4())
    
    # Initialize backtest status
    backtest_tasks[task_id] = {
        "strategy_id": strategy_id,
        "status": "pending",
        "created_at": datetime.utcnow(),
        "config": config
    }
    
    # Start background backtesting
    background_tasks.add_task(run_backtest, task_id, strategy_id, config)
    
    return {"task_id": task_id, "status": "started"}


@router.get("/backtest/{task_id}/status")
async def get_backtest_status(task_id: str) -> Dict[str, Any]:
    """Get backtesting status"""
    if task_id not in backtest_tasks:
        raise HTTPException(status_code=404, detail="Backtest task not found")
    
    return backtest_tasks[task_id]


@router.get("/backtest/{task_id}/results")
async def get_backtest_results(task_id: str) -> BacktestResult:
    """Get backtesting results"""
    if task_id not in backtest_tasks:
        raise HTTPException(status_code=404, detail="Backtest task not found")
    
    task = backtest_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Backtest not completed. Status: {task['status']}"
        )
    
    return task["results"]


@router.post("/{strategy_id}/optimize")
async def start_optimization(
    strategy_id: str,
    config: StrategyOptimizationConfig,
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Start strategy parameter optimization"""
    if strategy_id not in strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    task_id = str(uuid.uuid4())
    
    # Initialize optimization status
    optimization_tasks[task_id] = {
        "strategy_id": strategy_id,
        "status": "pending",
        "created_at": datetime.utcnow(),
        "config": config
    }
    
    # Start background optimization
    background_tasks.add_task(run_optimization, task_id, strategy_id, config)
    
    return {"task_id": task_id, "status": "started"}


@router.get("/optimization/{task_id}/status")
async def get_optimization_status(task_id: str) -> Dict[str, Any]:
    """Get optimization status"""
    if task_id not in optimization_tasks:
        raise HTTPException(status_code=404, detail="Optimization task not found")
    
    return optimization_tasks[task_id]


@router.get("/optimization/{task_id}/results")
async def get_optimization_results(task_id: str) -> StrategyOptimizationResult:
    """Get optimization results"""
    if task_id not in optimization_tasks:
        raise HTTPException(status_code=404, detail="Optimization task not found")
    
    task = optimization_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Optimization not completed. Status: {task['status']}"
        )
    
    return task["results"]


@router.post("/{strategy_id}/start")
async def start_strategy(strategy_id: str) -> Dict[str, str]:
    """Start live trading for a strategy"""
    if strategy_id not in strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    strategy = strategies[strategy_id]
    
    # Validate strategy is ready for live trading
    validation_result = await strategy_builder.validate_for_live_trading(strategy)
    if not validation_result["ready"]:
        raise HTTPException(
            status_code=400,
            detail=f"Strategy not ready for live trading: {validation_result['issues']}"
        )
    
    try:
        # Initialize live trading status
        strategy.live_status = StrategyStatus(
            strategy_id=strategy_id,
            name=strategy.config.name,
            status="active",
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            current_positions={},
            current_pnl=0.0,
            current_drawdown=0.0,
            total_trades_today=0,
            total_pnl_today=0.0,
            risk_metrics={},
            latest_signals={},
            signal_strength={}
        )
        
        logger.info(f"Started live trading for strategy: {strategy_id}")
        return {"message": "Strategy started successfully", "status": "active"}
        
    except Exception as e:
        logger.error(f"Failed to start strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{strategy_id}/stop")
async def stop_strategy(strategy_id: str) -> Dict[str, str]:
    """Stop live trading for a strategy"""
    if strategy_id not in strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    strategy = strategies[strategy_id]
    
    if not strategy.live_status:
        raise HTTPException(status_code=400, detail="Strategy is not running")
    
    try:
        # Update status
        strategy.live_status.status = "stopped"
        strategy.live_status.last_updated = datetime.utcnow()
        
        # Close all positions (in a real implementation)
        # await position_manager.close_all_positions(strategy_id)
        
        logger.info(f"Stopped strategy: {strategy_id}")
        return {"message": "Strategy stopped successfully", "status": "stopped"}
        
    except Exception as e:
        logger.error(f"Failed to stop strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{strategy_id}/status")
async def get_strategy_status(strategy_id: str) -> StrategyStatus:
    """Get current status of a live strategy"""
    if strategy_id not in strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    strategy = strategies[strategy_id]
    
    if not strategy.live_status:
        raise HTTPException(status_code=400, detail="Strategy is not running")
    
    return strategy.live_status


@router.get("/{strategy_id}/performance")
async def get_strategy_performance(
    strategy_id: str,
    days: int = Query(30, ge=1, le=365)
) -> Dict[str, Any]:
    """Get strategy performance metrics"""
    if strategy_id not in strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    # In a real implementation, this would fetch performance data from database
    # For now, return mock data
    return {
        "strategy_id": strategy_id,
        "period_days": days,
        "total_return": 0.05,
        "sharpe_ratio": 1.2,
        "max_drawdown": -0.03,
        "volatility": 0.15,
        "trades": 45,
        "win_rate": 0.58
    }


# Background task functions
async def run_backtest(task_id: str, strategy_id: str, config: BacktestConfig):
    """Background task to run strategy backtesting"""
    try:
        backtest_tasks[task_id]["status"] = "running"
        backtest_tasks[task_id]["started_at"] = datetime.utcnow()
        
        strategy = strategies[strategy_id]
        
        # Run backtesting
        result = await backtesting_engine.run_backtest(strategy.config, config)
        
        # Store results
        strategy.backtest_results = result
        backtest_tasks[task_id]["status"] = "completed"
        backtest_tasks[task_id]["completed_at"] = datetime.utcnow()
        backtest_tasks[task_id]["results"] = result
        
        logger.info(f"Completed backtest for strategy: {strategy_id}")
        
    except Exception as e:
        backtest_tasks[task_id]["status"] = "failed"
        backtest_tasks[task_id]["error"] = str(e)
        backtest_tasks[task_id]["completed_at"] = datetime.utcnow()
        logger.error(f"Backtest failed for strategy {strategy_id}: {str(e)}")


async def run_optimization(task_id: str, strategy_id: str, config: StrategyOptimizationConfig):
    """Background task to run strategy optimization"""
    try:
        optimization_tasks[task_id]["status"] = "running"
        optimization_tasks[task_id]["started_at"] = datetime.utcnow()
        
        strategy = strategies[strategy_id]
        
        # Run optimization
        result = await strategy_builder.optimize_strategy(strategy.config, config)
        
        # Store results
        optimization_tasks[task_id]["status"] = "completed"
        optimization_tasks[task_id]["completed_at"] = datetime.utcnow()
        optimization_tasks[task_id]["results"] = result
        
        logger.info(f"Completed optimization for strategy: {strategy_id}")
        
    except Exception as e:
        optimization_tasks[task_id]["status"] = "failed"
        optimization_tasks[task_id]["error"] = str(e)
        optimization_tasks[task_id]["completed_at"] = datetime.utcnow()
        logger.error(f"Optimization failed for strategy {strategy_id}: {str(e)}")


@router.get("/")
async def strategy_dashboard() -> Dict[str, Any]:
    """Get strategy dashboard overview"""
    total_strategies = len(strategies)
    active_strategies = len([s for s in strategies.values() if s.live_status and s.live_status.status == "active"])
    
    return {
        "total_strategies": total_strategies,
        "active_strategies": active_strategies,
        "backtest_tasks": len(backtest_tasks),
        "optimization_tasks": len(optimization_tasks),
        "recent_strategies": list(strategies.values())[-5:] if strategies else []
    }