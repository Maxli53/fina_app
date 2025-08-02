"""
API endpoints for Custom Indicators Builder
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import pandas as pd
import json
import logging

from app.services.indicators.custom_indicators import (
    CustomIndicatorBuilder,
    IndicatorType,
    IndicatorParameter,
    DataField
)
from app.services.auth import get_current_user
from app.models.auth import User
from app.services.data.data_service import DataService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/indicators", tags=["Custom Indicators"])

# Initialize indicator builder
indicator_builder = CustomIndicatorBuilder()
data_service = None


def get_data_service():
    """Get data service instance"""
    global data_service
    if data_service is None:
        data_service = DataService(None, None)
    return data_service


# Request/Response models
class ParameterDefinition(BaseModel):
    """Parameter definition for indicator"""
    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (int/float/bool)")
    default_value: Any = Field(..., description="Default value")
    min_value: Optional[float] = Field(None, description="Minimum value")
    max_value: Optional[float] = Field(None, description="Maximum value")
    description: str = Field("", description="Parameter description")


class CreateIndicatorRequest(BaseModel):
    """Request to create custom indicator"""
    name: str = Field(..., description="Indicator name")
    type: str = Field(..., description="Indicator type")
    description: str = Field(..., description="Indicator description")
    formula: str = Field(..., description="Python formula/expression")
    parameters: List[ParameterDefinition] = Field(..., description="Parameters")
    dependencies: Optional[List[str]] = Field(None, description="Other indicators this depends on")


class CalculateIndicatorRequest(BaseModel):
    """Request to calculate indicator values"""
    indicator_name: str = Field(..., description="Indicator to calculate")
    symbol: str = Field(..., description="Symbol to analyze")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    parameters: Optional[Dict[str, Any]] = Field({}, description="Indicator parameters")


class BacktestIndicatorRequest(BaseModel):
    """Request to backtest indicator"""
    indicator_name: str = Field(..., description="Indicator to backtest")
    symbol: str = Field(..., description="Symbol to test on")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(10000, description="Initial capital")
    parameters: Optional[Dict[str, Any]] = Field({}, description="Indicator parameters")


class OptimizeIndicatorRequest(BaseModel):
    """Request to optimize indicator parameters"""
    indicator_name: str = Field(..., description="Indicator to optimize")
    symbol: str = Field(..., description="Symbol to optimize on")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    param_ranges: Dict[str, List[float]] = Field(..., description="Parameter ranges [min, max]")
    objective: str = Field("sharpe_ratio", description="Optimization objective")


class CombineIndicatorsRequest(BaseModel):
    """Request to combine multiple indicators"""
    name: str = Field(..., description="Combined indicator name")
    indicators: List[Dict[str, Any]] = Field(..., description="Indicators to combine")
    combination_method: str = Field("weighted_average", description="How to combine")
    weights: Optional[List[float]] = Field(None, description="Weights for combination")


# API Endpoints
@router.get("/library")
async def get_indicator_library(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get all available indicators"""
    
    indicators = []
    for name, definition in indicator_builder.indicators.items():
        indicators.append({
            "name": name,
            "type": definition.type.value,
            "description": definition.description,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type.__name__,
                    "default": p.default_value,
                    "min": p.min_value,
                    "max": p.max_value,
                    "description": p.description
                }
                for p in definition.parameters
            ],
            "author": definition.author,
            "created_at": definition.created_at.isoformat() if definition.created_at else None
        })
    
    return {
        "status": "success",
        "indicators": indicators,
        "total": len(indicators),
        "categories": {
            "trend": len([i for i in indicators if i["type"] == "trend"]),
            "momentum": len([i for i in indicators if i["type"] == "momentum"]),
            "volatility": len([i for i in indicators if i["type"] == "volatility"]),
            "volume": len([i for i in indicators if i["type"] == "volume"]),
            "custom": len([i for i in indicators if i["type"] == "custom"])
        }
    }


@router.post("/create")
async def create_custom_indicator(
    request: CreateIndicatorRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Create a new custom indicator"""
    try:
        # Convert parameter definitions
        parameters = []
        for param in request.parameters:
            param_type = {
                "int": int,
                "float": float,
                "bool": bool,
                "str": str
            }.get(param.type, float)
            
            parameters.append(IndicatorParameter(
                name=param.name,
                type=param_type,
                default_value=param.default_value,
                min_value=param.min_value,
                max_value=param.max_value,
                description=param.description
            ))
        
        # Create indicator
        indicator_def = indicator_builder.create_indicator(
            name=request.name,
            indicator_type=IndicatorType(request.type),
            description=request.description,
            formula=request.formula,
            parameters=parameters,
            dependencies=request.dependencies,
            author=current_user.username
        )
        
        return {
            "status": "success",
            "indicator": {
                "name": indicator_def.name,
                "type": indicator_def.type.value,
                "description": indicator_def.description,
                "formula": indicator_def.formula,
                "parameters": [
                    {
                        "name": p.name,
                        "type": p.type.__name__,
                        "default": p.default_value
                    }
                    for p in indicator_def.parameters
                ],
                "created_at": indicator_def.created_at
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating indicator: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/calculate")
async def calculate_indicator(
    request: CalculateIndicatorRequest,
    current_user: User = Depends(get_current_user),
    data_svc: DataService = Depends(get_data_service)
) -> Dict[str, Any]:
    """Calculate indicator values for a symbol"""
    try:
        # Fetch historical data
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        
        historical_data = await data_svc.get_historical_data(
            request.symbol,
            start_date,
            end_date
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Calculate indicator
        result = indicator_builder.calculate_indicator(
            request.indicator_name,
            df,
            **request.parameters
        )
        
        # Prepare response
        values_dict = result.values.to_dict()
        signals_dict = result.signals.to_dict() if result.signals is not None else None
        
        return {
            "status": "success",
            "indicator": request.indicator_name,
            "symbol": request.symbol,
            "values": [
                {"date": k.isoformat(), "value": v}
                for k, v in values_dict.items()
                if pd.notna(v)
            ],
            "signals": [
                {"date": k.isoformat(), "signal": v}
                for k, v in signals_dict.items()
                if pd.notna(v)
            ] if signals_dict else None,
            "metadata": result.metadata,
            "statistics": {
                "mean": float(result.values.mean()) if not result.values.empty else None,
                "std": float(result.values.std()) if not result.values.empty else None,
                "min": float(result.values.min()) if not result.values.empty else None,
                "max": float(result.values.max()) if not result.values.empty else None
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating indicator: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest")
async def backtest_indicator(
    request: BacktestIndicatorRequest,
    current_user: User = Depends(get_current_user),
    data_svc: DataService = Depends(get_data_service)
) -> Dict[str, Any]:
    """Backtest indicator performance"""
    try:
        # Fetch historical data
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        
        historical_data = await data_svc.get_historical_data(
            request.symbol,
            start_date,
            end_date
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Run backtest
        backtest_result = indicator_builder.backtest_indicator(
            request.indicator_name,
            df,
            initial_capital=request.initial_capital,
            **request.parameters
        )
        
        # Prepare response
        if "error" in backtest_result:
            return {
                "status": "error",
                "message": backtest_result["error"]
            }
        
        # Convert series to lists for JSON serialization
        cumulative_returns = backtest_result["cumulative_returns"].to_dict()
        
        return {
            "status": "success",
            "backtest": {
                "indicator": request.indicator_name,
                "symbol": request.symbol,
                "period": f"{request.start_date} to {request.end_date}",
                "initial_capital": request.initial_capital,
                "final_capital": backtest_result["final_capital"],
                "total_return": backtest_result["total_return"],
                "sharpe_ratio": backtest_result["sharpe_ratio"],
                "max_drawdown": backtest_result["max_drawdown"],
                "num_trades": backtest_result["num_trades"],
                "cumulative_returns": [
                    {"date": k.isoformat(), "value": v}
                    for k, v in cumulative_returns.items()
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Error backtesting indicator: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize")
async def optimize_indicator(
    request: OptimizeIndicatorRequest,
    current_user: User = Depends(get_current_user),
    data_svc: DataService = Depends(get_data_service),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> Dict[str, Any]:
    """Optimize indicator parameters"""
    
    # Start optimization in background
    optimization_id = f"opt_{request.indicator_name}_{datetime.now().timestamp()}"
    
    background_tasks.add_task(
        run_optimization_background,
        optimization_id,
        request,
        indicator_builder,
        data_svc
    )
    
    return {
        "status": "success",
        "message": "Optimization started",
        "optimization_id": optimization_id,
        "estimated_time": "1-5 minutes depending on parameter space"
    }


async def run_optimization_background(
    optimization_id: str,
    request: OptimizeIndicatorRequest,
    builder: CustomIndicatorBuilder,
    data_svc: DataService
):
    """Run optimization in background"""
    try:
        # Fetch historical data
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        
        historical_data = await data_svc.get_historical_data(
            request.symbol,
            start_date,
            end_date
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Convert param ranges to tuples
        param_ranges = {
            k: tuple(v) for k, v in request.param_ranges.items()
        }
        
        # Run optimization
        result = builder.optimize_parameters(
            request.indicator_name,
            df,
            param_ranges,
            request.objective
        )
        
        # Store result (implement storage)
        logger.info(f"Optimization {optimization_id} completed: {result['best_parameters']}")
        
    except Exception as e:
        logger.error(f"Optimization {optimization_id} failed: {e}")


@router.post("/combine")
async def combine_indicators(
    request: CombineIndicatorsRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Combine multiple indicators into one"""
    try:
        # Build combination formula
        if request.combination_method == "weighted_average":
            if not request.weights or len(request.weights) != len(request.indicators):
                weights = [1.0 / len(request.indicators)] * len(request.indicators)
            else:
                weights = request.weights
            
            # Create weighted average formula
            terms = []
            dependencies = []
            
            for i, (indicator, weight) in enumerate(zip(request.indicators, weights)):
                ind_name = indicator["name"]
                dependencies.append(ind_name)
                terms.append(f"{weight} * {ind_name}")
            
            formula = " + ".join(terms)
            
        elif request.combination_method == "voting":
            # Voting combination
            dependencies = [ind["name"] for ind in request.indicators]
            formula = f"np.sign(sum([{', '.join(dependencies)}]))"
            
        else:
            raise ValueError(f"Unknown combination method: {request.combination_method}")
        
        # Create combined indicator
        combined = indicator_builder.create_indicator(
            name=request.name,
            indicator_type=IndicatorType.CUSTOM,
            description=f"Combined indicator using {request.combination_method}",
            formula=formula,
            parameters=[],  # Inherit parameters from dependencies
            dependencies=dependencies,
            author=current_user.username
        )
        
        return {
            "status": "success",
            "combined_indicator": {
                "name": combined.name,
                "formula": combined.formula,
                "dependencies": combined.dependencies,
                "method": request.combination_method
            }
        }
        
    except Exception as e:
        logger.error(f"Error combining indicators: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{indicator_name}/code")
async def export_indicator_code(
    indicator_name: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Export indicator as Python code"""
    try:
        code = indicator_builder.export_indicator(indicator_name)
        
        return {
            "status": "success",
            "indicator_name": indicator_name,
            "code": code,
            "language": "python"
        }
        
    except Exception as e:
        logger.error(f"Error exporting indicator: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/upload")
async def upload_indicator(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Upload indicator from file"""
    try:
        # Read file content
        content = await file.read()
        
        # Save temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Load indicator
        indicator = indicator_builder.load_indicator(temp_path)
        
        return {
            "status": "success",
            "indicator": {
                "name": indicator.name,
                "type": indicator.type.value,
                "description": indicator.description
            }
        }
        
    except Exception as e:
        logger.error(f"Error uploading indicator: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{indicator_name}")
async def delete_indicator(
    indicator_name: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Delete custom indicator"""
    try:
        if indicator_name not in indicator_builder.indicators:
            raise HTTPException(status_code=404, detail="Indicator not found")
        
        # Check if it's a built-in indicator
        if indicator_builder.indicators[indicator_name].author == "system":
            raise HTTPException(status_code=403, detail="Cannot delete built-in indicators")
        
        # Check ownership
        if indicator_builder.indicators[indicator_name].author != current_user.username:
            raise HTTPException(status_code=403, detail="Can only delete your own indicators")
        
        # Delete indicator
        del indicator_builder.indicators[indicator_name]
        del indicator_builder.compiled_functions[indicator_name]
        
        return {
            "status": "success",
            "message": f"Indicator '{indicator_name}' deleted"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting indicator: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def get_indicator_templates(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get indicator templates for quick start"""
    
    templates = [
        {
            "name": "Custom Moving Average",
            "type": "trend",
            "description": "Weighted moving average with custom weights",
            "formula": "data.rolling(window=period).apply(lambda x: np.average(x, weights=np.linspace(0.5, 1.0, len(x))))",
            "parameters": [
                {
                    "name": "period",
                    "type": "int",
                    "default_value": 20,
                    "min_value": 5,
                    "max_value": 200,
                    "description": "Rolling window period"
                }
            ]
        },
        {
            "name": "Momentum Oscillator",
            "type": "momentum",
            "description": "Custom momentum oscillator",
            "formula": "(data - data.shift(period)) / data.shift(period) * 100",
            "parameters": [
                {
                    "name": "period",
                    "type": "int",
                    "default_value": 14,
                    "min_value": 5,
                    "max_value": 50,
                    "description": "Lookback period"
                }
            ]
        },
        {
            "name": "Volatility Bands",
            "type": "volatility",
            "description": "Dynamic volatility bands",
            "formula": "{'upper': data + multiplier * data.rolling(period).std(), 'middle': data, 'lower': data - multiplier * data.rolling(period).std()}",
            "parameters": [
                {
                    "name": "period",
                    "type": "int",
                    "default_value": 20,
                    "min_value": 10,
                    "max_value": 50,
                    "description": "Standard deviation period"
                },
                {
                    "name": "multiplier",
                    "type": "float",
                    "default_value": 2.0,
                    "min_value": 1.0,
                    "max_value": 3.0,
                    "description": "Band width multiplier"
                }
            ]
        },
        {
            "name": "Volume Flow Index",
            "type": "volume",
            "description": "Custom volume flow indicator",
            "formula": "((2 * data - high - low) / (high - low)) * volume",
            "parameters": [],
            "requires": ["high", "low", "volume"]
        }
    ]
    
    return {
        "status": "success",
        "templates": templates
    }


# Health check
@router.get("/health")
async def indicators_health_check() -> Dict[str, Any]:
    """Check indicators service health"""
    
    built_in_count = len([
        i for i in indicator_builder.indicators.values() 
        if i.author == "system"
    ])
    
    custom_count = len([
        i for i in indicator_builder.indicators.values() 
        if i.author != "system"
    ])
    
    return {
        "status": "healthy",
        "service": "Custom Indicators Builder",
        "indicators": {
            "built_in": built_in_count,
            "custom": custom_count,
            "total": len(indicator_builder.indicators)
        },
        "features": [
            "create_custom_indicators",
            "calculate_values",
            "backtest_performance",
            "optimize_parameters",
            "combine_indicators",
            "export_code",
            "templates"
        ]
    }