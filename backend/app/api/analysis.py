from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import uuid
from datetime import datetime, timedelta

from app.models.analysis import (
    IDTxlConfig, MLConfig, NeuralNetworkConfig,
    AnalysisResult, AnalysisStatus
)
from app.services.analysis.idtxl_service import IDTxlService
from app.services.data.yahoo_finance import YahooFinanceService

router = APIRouter()

# In-memory storage for analysis tasks (will be replaced with Redis)
analysis_tasks: Dict[str, AnalysisStatus] = {}

# Initialize services
idtxl_service = IDTxlService()
yf_service = YahooFinanceService()


class IDTxlAnalysisRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Time series data")
    config: IDTxlConfig = Field(..., description="IDTxl analysis configuration")


class MLAnalysisRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Time series data")
    config: MLConfig = Field(..., description="ML model configuration")


class NNAnalysisRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Time series data")
    config: NeuralNetworkConfig = Field(..., description="Neural network configuration")


async def run_idtxl_analysis(task_id: str, data: Dict[str, Any], config: IDTxlConfig):
    """Background task for IDTxl analysis"""
    try:
        analysis_tasks[task_id].status = "running"
        analysis_tasks[task_id].started_at = datetime.utcnow()
        
        # Fetch actual time series data
        symbols = config.variables
        start_date = datetime.now() - timedelta(days=365)  # Last year of data
        end_date = datetime.now()
        
        # Fetch data for all symbols
        time_series_data = {}
        for symbol in symbols:
            ts_data = await yf_service.fetch_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )
            time_series_data[symbol] = ts_data
        
        # Run IDTxl analysis
        idtxl_result = await idtxl_service.analyze(time_series_data, config)
        
        # Convert result to dict
        result = {
            "transfer_entropy": idtxl_result.transfer_entropy,
            "mutual_information": idtxl_result.mutual_information,
            "significant_connections": [
                {
                    "type": conn["type"],
                    "source": conn["source"],
                    "target": conn["target"],
                    "strength": float(conn["strength"]),
                    "lag": conn.get("lag", 0)
                }
                for conn in idtxl_result.significant_connections
            ],
            "processing_time": idtxl_result.processing_time
        }
        
        analysis_tasks[task_id].status = "completed"
        analysis_tasks[task_id].completed_at = datetime.utcnow()
        analysis_tasks[task_id].result = result
        
    except Exception as e:
        analysis_tasks[task_id].status = "failed"
        analysis_tasks[task_id].error = str(e)
        analysis_tasks[task_id].completed_at = datetime.utcnow()


@router.post("/idtxl/start")
async def start_idtxl_analysis(
    request: IDTxlAnalysisRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Start IDTxl information-theoretic analysis"""
    task_id = str(uuid.uuid4())
    
    # Initialize task status
    analysis_tasks[task_id] = AnalysisStatus(
        task_id=task_id,
        analysis_type="idtxl",
        status="pending",
        created_at=datetime.utcnow()
    )
    
    # Start background task
    background_tasks.add_task(run_idtxl_analysis, task_id, request.data, request.config)
    
    return {"task_id": task_id, "status": "started"}


@router.get("/status/{task_id}")
async def get_analysis_status(task_id: str) -> AnalysisStatus:
    """Get status of an analysis task"""
    if task_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return analysis_tasks[task_id]


@router.get("/results/{task_id}")
async def get_analysis_results(task_id: str) -> AnalysisResult:
    """Get results of a completed analysis"""
    if task_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = analysis_tasks[task_id]
    if task.status != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Analysis not completed. Current status: {task.status}"
        )
    
    return AnalysisResult(
        task_id=task_id,
        analysis_type=task.analysis_type,
        results=task.result,
        metadata={
            "created_at": task.created_at.isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "duration_seconds": (
                (task.completed_at - task.created_at).total_seconds() 
                if task.completed_at else None
            )
        }
    )


@router.post("/ml/start")
async def start_ml_analysis(
    request: MLAnalysisRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Start machine learning analysis"""
    # TODO: Implement ML analysis
    return {"task_id": str(uuid.uuid4()), "status": "not_implemented"}


@router.post("/nn/start")
async def start_nn_analysis(
    request: NNAnalysisRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Start neural network analysis"""
    # TODO: Implement NN analysis
    return {"task_id": str(uuid.uuid4()), "status": "not_implemented"}


@router.get("/tasks")
async def list_analysis_tasks(
    analysis_type: Optional[str] = None,
    status: Optional[str] = None
) -> List[AnalysisStatus]:
    """List all analysis tasks with optional filtering"""
    tasks = list(analysis_tasks.values())
    
    if analysis_type:
        tasks = [t for t in tasks if t.analysis_type == analysis_type]
    
    if status:
        tasks = [t for t in tasks if t.status == status]
    
    return tasks