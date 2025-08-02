from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import uuid
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from app.models.analysis import (
    IDTxlConfig, MLConfig, NeuralNetworkConfig,
    AnalysisResult, AnalysisStatus
)
from app.services.analysis.idtxl_service import IDTxlService
from app.services.analysis.ml_service import MLService
from app.services.analysis.nn_service import NeuralNetworkService
from app.services.data.yahoo_finance import YahooFinanceService

router = APIRouter()

# In-memory storage for analysis tasks (will be replaced with Redis)
analysis_tasks: Dict[str, AnalysisStatus] = {}

# Initialize services
idtxl_service = IDTxlService()
ml_service = MLService()
nn_service = NeuralNetworkService()
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


async def run_ml_analysis(task_id: str, data: Dict[str, Any], config: MLConfig):
    """Background task for ML analysis"""
    try:
        analysis_tasks[task_id].status = "running"
        analysis_tasks[task_id].started_at = datetime.utcnow()
        
        # Fetch time series data for specified symbols or use provided data
        if 'symbols' in data:
            symbols = data['symbols']
            start_date = datetime.now() - timedelta(days=data.get('days', 365))
            end_date = datetime.now()
            
            # Fetch price data
            price_data = {}
            for symbol in symbols:
                ts_data = await yf_service.fetch_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval="1d"
                )
                
                # Convert to DataFrame
                df = pd.DataFrame([
                    {
                        'timestamp': d.timestamp,
                        'open': d.open,
                        'high': d.high,
                        'low': d.low,
                        'close': d.close,
                        'volume': d.volume
                    }
                    for d in ts_data.data
                ])
                df.set_index('timestamp', inplace=True)
                price_data[symbol] = df
            
            # Use primary symbol for features and targets
            primary_symbol = symbols[0]
            main_data = price_data[primary_symbol]
            
            # Generate features
            features = ml_service.generate_features(main_data)
            
            # Generate targets based on config
            if config.target.value == "direction":
                targets = (main_data['close'].pct_change(config.prediction_horizon).shift(-config.prediction_horizon) > 0).astype(int)
            elif config.target.value == "returns":
                targets = main_data['close'].pct_change(config.prediction_horizon).shift(-config.prediction_horizon)
            else:  # volatility
                targets = main_data['close'].pct_change().rolling(config.prediction_horizon).std().shift(-config.prediction_horizon)
            
            # Remove NaN values
            combined_data = pd.concat([features, targets], axis=1).dropna()
            features = combined_data.iloc[:, :-1]
            targets = combined_data.iloc[:, -1]
        
        else:
            # Use provided preprocessed data
            features = pd.DataFrame(data['features'])
            targets = pd.Series(data['targets'])
        
        # Train ML model
        ml_result = await ml_service.train_model(config, features, targets)
        
        # Convert result to serializable format
        result = {
            "model_type": ml_result.model_type,
            "target": ml_result.target,
            "prediction_horizon": ml_result.prediction_horizon,
            "features": ml_result.features,
            "validation_results": ml_result.validation_results,
            "feature_importance": ml_result.feature_importance,
            "final_metrics": ml_result.final_metrics,
            "training_data_info": ml_result.training_data_info
        }
        
        analysis_tasks[task_id].status = "completed"
        analysis_tasks[task_id].completed_at = datetime.utcnow()
        analysis_tasks[task_id].result = result
        
    except Exception as e:
        analysis_tasks[task_id].status = "failed"
        analysis_tasks[task_id].error = str(e)
        analysis_tasks[task_id].completed_at = datetime.utcnow()


async def run_nn_analysis(task_id: str, data: Dict[str, Any], config: NeuralNetworkConfig):
    """Background task for neural network analysis"""
    try:
        analysis_tasks[task_id].status = "running"
        analysis_tasks[task_id].started_at = datetime.utcnow()
        
        # Fetch and prepare sequential data
        if 'symbols' in data:
            symbols = data['symbols']
            start_date = datetime.now() - timedelta(days=data.get('days', 730))  # 2 years for NN
            end_date = datetime.now()
            
            # Fetch price data for primary symbol
            primary_symbol = symbols[0]
            ts_data = await yf_service.fetch_historical_data(
                symbol=primary_symbol,
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'timestamp': d.timestamp,
                    'open': d.open,
                    'high': d.high,
                    'low': d.low,
                    'close': d.close,
                    'volume': d.volume
                }
                for d in ts_data.data
            ])
            df.set_index('timestamp', inplace=True)
            
            # Generate features for sequences
            features = ml_service.generate_features(df)
            
            # Prepare sequences
            sequence_length = data.get('sequence_length', 60)
            target_column = 'close'
            sequences, targets = nn_service.prepare_sequences(
                features, target_column, sequence_length, config.prediction_horizon
            )
            
        else:
            # Use provided sequence data
            sequences = np.array(data['sequences'])
            targets = np.array(data['targets'])
        
        # Split data for validation
        split_idx = int(len(sequences) * 0.8)
        train_sequences = sequences[:split_idx]
        train_targets = targets[:split_idx]
        val_sequences = sequences[split_idx:]
        val_targets = targets[split_idx:]
        
        # Train neural network
        nn_result = await nn_service.train_network(
            config, 
            train_sequences, 
            train_targets,
            validation_data=(val_sequences, val_targets)
        )
        
        # Convert result to serializable format
        result = {
            "architecture": nn_result.architecture,
            "layers": nn_result.layers,
            "training_config": nn_result.training_config,
            "training_history": nn_result.training_history,
            "final_metrics": nn_result.final_metrics,
            "training_info": nn_result.training_info
        }
        
        analysis_tasks[task_id].status = "completed"
        analysis_tasks[task_id].completed_at = datetime.utcnow()
        analysis_tasks[task_id].result = result
        
    except Exception as e:
        analysis_tasks[task_id].status = "failed"
        analysis_tasks[task_id].error = str(e)
        analysis_tasks[task_id].completed_at = datetime.utcnow()


@router.post("/ml/start")
async def start_ml_analysis(
    request: MLAnalysisRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Start machine learning analysis"""
    task_id = str(uuid.uuid4())
    
    # Initialize task status
    analysis_tasks[task_id] = AnalysisStatus(
        task_id=task_id,
        analysis_type="ml",
        status="pending",
        created_at=datetime.utcnow()
    )
    
    # Start background task
    background_tasks.add_task(run_ml_analysis, task_id, request.data, request.config)
    
    return {"task_id": task_id, "status": "started"}


@router.post("/nn/start")
async def start_nn_analysis(
    request: NNAnalysisRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Start neural network analysis"""
    task_id = str(uuid.uuid4())
    
    # Initialize task status
    analysis_tasks[task_id] = AnalysisStatus(
        task_id=task_id,
        analysis_type="nn",
        status="pending",
        created_at=datetime.utcnow()
    )
    
    # Start background task
    background_tasks.add_task(run_nn_analysis, task_id, request.data, request.config)
    
    return {"task_id": task_id, "status": "started"}


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