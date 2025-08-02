"""
System management and orchestration API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from typing import Dict, Any, List
import asyncio
import logging

from app.services.system_orchestrator import (
    SystemOrchestrator,
    WorkflowType,
    SystemState
)
from app.services.system_health import SystemHealthMonitor
from app.models.system import (
    WorkflowRequest,
    SystemStatusResponse,
    HealthCheckResponse
)
from app.dependencies import get_db_session, get_redis_client, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/system", tags=["system"])

# Global orchestrator instance
orchestrator = None


@router.on_event("startup")
async def startup_event():
    """Initialize system orchestrator on startup"""
    global orchestrator
    # This will be initialized with proper config from main app
    logger.info("System API router started")


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(
    current_user: dict = Depends(get_current_user)
) -> SystemStatusResponse:
    """Get current system status"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System orchestrator not initialized")
    
    status = await orchestrator._get_system_status()
    return SystemStatusResponse(**status)


@router.get("/health", response_model=HealthCheckResponse)
async def get_system_health(
    current_user: dict = Depends(get_current_user)
) -> HealthCheckResponse:
    """Get comprehensive system health check"""
    db = await get_db_session()
    redis = await get_redis_client()
    
    health_monitor = SystemHealthMonitor(db, redis, {})
    health_status = await health_monitor.check_all_systems()
    
    return HealthCheckResponse(
        status=health_status["overall"].status.value,
        services={
            service: {
                "status": result.status.value,
                "latency_ms": result.latency_ms,
                "details": result.details
            }
            for service, result in health_status.items()
        }
    )


@router.post("/workflow/{workflow_type}")
async def execute_workflow(
    workflow_type: WorkflowType,
    request: WorkflowRequest,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Execute a system workflow"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System orchestrator not initialized")
    
    if orchestrator.state != SystemState.RUNNING:
        raise HTTPException(
            status_code=503, 
            detail=f"System not ready. Current state: {orchestrator.state}"
        )
    
    try:
        result = await orchestrator.execute_workflow(
            workflow_type=workflow_type,
            params=request.parameters
        )
        return result
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/control/{action}")
async def system_control(
    action: str,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, str]:
    """Control system operations (start, stop, restart)"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System orchestrator not initialized")
    
    if action == "start":
        await orchestrator.start()
        return {"status": "started", "message": "System started successfully"}
    
    elif action == "stop":
        await orchestrator.stop()
        return {"status": "stopped", "message": "System stopped successfully"}
    
    elif action == "restart":
        await orchestrator.stop()
        await asyncio.sleep(2)
        await orchestrator.initialize()
        await orchestrator.start()
        return {"status": "restarted", "message": "System restarted successfully"}
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time system updates"""
    await websocket.accept()
    
    if not orchestrator:
        await websocket.send_json({
            "type": "error",
            "message": "System orchestrator not initialized"
        })
        await websocket.close()
        return
    
    # Add client to orchestrator
    orchestrator.websocket_clients.append(websocket)
    
    try:
        # Send initial state
        await websocket.send_json({
            "type": "system_state",
            "data": {
                "state": orchestrator.state.value,
                "active_workflows": list(orchestrator.active_workflows.keys()),
                "metrics": dict(orchestrator.metrics)
            }
        })
        
        # Handle incoming messages
        while True:
            data = await websocket.receive_json()
            await orchestrator._handle_websocket_message(websocket, data)
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in orchestrator.websocket_clients:
            orchestrator.websocket_clients.remove(websocket)


@router.get("/events")
async def get_system_events(
    limit: int = 50,
    severity: str = None,
    current_user: dict = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get recent system events"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System orchestrator not initialized")
    
    events = orchestrator.event_history[-limit:]
    
    if severity:
        events = [e for e in events if e.severity == severity]
    
    return [
        {
            "timestamp": e.timestamp.isoformat(),
            "event_type": e.event_type,
            "source": e.source,
            "severity": e.severity,
            "data": e.data
        }
        for e in events
    ]


@router.get("/metrics")
async def get_system_metrics(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get system performance metrics"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System orchestrator not initialized")
    
    return {
        "state": orchestrator.state.value,
        "metrics": dict(orchestrator.metrics),
        "active_workflows": len(orchestrator.active_workflows),
        "connected_clients": len(orchestrator.websocket_clients),
        "event_count": len(orchestrator.event_history)
    }


def set_orchestrator(orch: SystemOrchestrator):
    """Set the global orchestrator instance"""
    global orchestrator
    orchestrator = orch