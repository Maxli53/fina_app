from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Dict, Any

router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Financial Analysis Platform API"
    }


@router.get("/status")
async def detailed_status() -> Dict[str, Any]:
    """Detailed status check for all services"""
    services_status = {
        "api": "operational",
        "database": "not_configured",
        "idtxl_service": "not_configured",
        "ml_service": "not_configured",
        "data_sources": {
            "yahoo_finance": "ready",
            "ibkr": "not_configured",
            "alpha_vantage": "not_configured"
        }
    }
    
    overall_status = "operational" if all(
        status != "error" for status in 
        [services_status["api"], services_status["database"]]
    ) else "degraded"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "services": services_status,
        "version": "0.1.0"
    }