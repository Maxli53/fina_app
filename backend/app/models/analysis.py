from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from enum import Enum


class EstimatorType(str, Enum):
    KRASKOV = "kraskov"
    GAUSSIAN = "gaussian"
    SYMBOLIC = "symbolic"


class MLModelType(str, Enum):
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    SVM = "svm"
    LOGISTIC_REGRESSION = "logistic_regression"


class NNArchitecture(str, Enum):
    LSTM = "lstm"
    GRU = "gru"
    CNN = "cnn"
    TRANSFORMER = "transformer"


class IDTxlConfig(BaseModel):
    """Configuration for IDTxl analysis"""
    analysis_type: Literal["mutual_information", "transfer_entropy", "both"] = "both"
    max_lag: int = Field(5, ge=1, le=20)
    estimator: EstimatorType = EstimatorType.KRASKOV
    significance_level: float = Field(0.05, ge=0.001, le=0.1)
    permutations: int = Field(200, ge=50, le=1000)
    variables: List[str] = Field(..., min_items=2)
    
    # Estimator-specific settings
    k_neighbors: Optional[int] = Field(3, ge=1, le=10)  # For Kraskov
    noise_level: Optional[float] = Field(1e-8, ge=1e-10, le=1e-6)  # For Kraskov
    alphabet_size: Optional[int] = Field(2, ge=2, le=5)  # For Symbolic


class MLConfig(BaseModel):
    """Configuration for ML models"""
    model_type: MLModelType
    target_variable: Literal["direction", "returns", "volatility"] = "direction"
    prediction_horizon: int = Field(1, ge=1, le=20)
    features: List[str] = Field(..., min_items=1)
    validation_strategy: Literal["time_series_cv", "walk_forward", "purged_cv"] = "time_series_cv"
    test_size: float = Field(0.2, ge=0.1, le=0.5)
    
    # Model-specific hyperparameters
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class NeuralNetworkConfig(BaseModel):
    """Configuration for neural networks"""
    architecture: NNArchitecture
    layers: List[int] = Field(..., min_items=1, description="Units per layer")
    epochs: int = Field(100, ge=1, le=1000)
    batch_size: int = Field(32, ge=1, le=512)
    optimizer: Literal["adam", "sgd", "rmsprop"] = "adam"
    learning_rate: float = Field(0.001, ge=0.0001, le=0.1)
    dropout_rate: float = Field(0.2, ge=0, le=0.5)
    
    # Architecture-specific settings
    bidirectional: Optional[bool] = False  # For LSTM/GRU
    attention_heads: Optional[int] = Field(8, ge=4, le=16)  # For Transformer
    encoder_layers: Optional[int] = Field(4, ge=2, le=8)  # For Transformer


class AnalysisStatus(BaseModel):
    """Status of an analysis task"""
    task_id: str
    analysis_type: Literal["idtxl", "ml", "nn", "integrated"]
    status: Literal["pending", "running", "completed", "failed"]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: Optional[float] = Field(None, ge=0, le=1)
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AnalysisResult(BaseModel):
    """Result of a completed analysis"""
    task_id: str
    analysis_type: str
    results: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class IDTxlResult(BaseModel):
    """Results from IDTxl analysis"""
    transfer_entropy: Optional[Dict[str, Any]] = None
    mutual_information: Optional[Dict[str, Any]] = None
    causal_network: Optional[Dict[str, Any]] = None
    significant_connections: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time: float
    
    
class MLResult(BaseModel):
    """Results from ML model training"""
    model_type: MLModelType
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    feature_importance: Dict[str, float]
    confusion_matrix: List[List[int]]
    training_history: List[Dict[str, float]]
    best_hyperparameters: Dict[str, Any]


class NeuralNetworkResult(BaseModel):
    """Results from neural network training"""
    architecture: NNArchitecture
    final_loss: float
    final_accuracy: float
    training_history: Dict[str, List[float]]
    validation_metrics: Dict[str, float]
    model_summary: str