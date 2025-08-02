from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal, Union
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


class ArchitectureType(str, Enum):
    LSTM = "lstm"
    GRU = "gru"
    CNN = "cnn"
    TRANSFORMER = "transformer"


class PredictionTarget(str, Enum):
    DIRECTION = "direction"
    RETURNS = "returns"
    VOLATILITY = "volatility"


class ValidationStrategy(str, Enum):
    TIME_SERIES_CV = "time_series_cv"
    WALK_FORWARD = "walk_forward"
    PURGED_CV = "purged_cv"
    TRAIN_TEST_SPLIT = "train_test_split"


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
    target: PredictionTarget = PredictionTarget.DIRECTION
    prediction_horizon: int = Field(1, ge=1, le=20)
    features: Optional[List[str]] = None
    validation: ValidationStrategy = ValidationStrategy.TIME_SERIES_CV
    test_size: float = Field(0.2, ge=0.1, le=0.5)
    
    # Model-specific hyperparameters
    hyperparameters: Optional[Dict[str, Any]] = None


class NeuralNetworkConfig(BaseModel):
    """Configuration for neural networks"""
    architecture: ArchitectureType
    layers: List[int] = Field(..., min_items=1, description="Units per layer")
    dense_layers: List[int] = Field(default_factory=lambda: [64, 32])
    epochs: int = Field(100, ge=1, le=1000)
    batch_size: int = Field(32, ge=1, le=512)
    optimizer: Literal["adam", "sgd", "rmsprop"] = "adam"
    learning_rate: float = Field(0.001, ge=0.0001, le=0.1)
    dropout_rate: float = Field(0.2, ge=0, le=0.5)
    early_stopping_patience: int = Field(10, ge=5, le=50)
    batch_normalization: bool = False
    
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
    target: PredictionTarget
    prediction_horizon: int
    features: List[str]
    hyperparameters: Optional[Dict[str, Any]]
    validation_results: Dict[str, Any]
    feature_importance: Dict[str, float]
    final_metrics: Dict[str, float]
    model_artifact: Any = Field(exclude=True)  # Exclude from serialization
    training_data_info: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class NeuralNetworkResult(BaseModel):
    """Results from neural network training"""
    architecture: ArchitectureType
    layers: List[int]
    training_config: Dict[str, Any]
    training_history: Dict[str, Any]
    final_metrics: Dict[str, Optional[Dict[str, float]]]
    model_artifact: Any = Field(exclude=True)  # Exclude from serialization
    training_info: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True