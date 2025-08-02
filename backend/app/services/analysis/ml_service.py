import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# ML models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression

from app.models.analysis import MLConfig, MLResult, PredictionTarget, ValidationStrategy

logger = logging.getLogger(__name__)


class MLService:
    """Machine Learning service for financial time series analysis"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    async def train_model(
        self,
        config: MLConfig,
        features: pd.DataFrame,
        targets: pd.Series,
        metadata: Dict[str, Any] = None
    ) -> MLResult:
        """Train ML model with given configuration and data"""
        
        def _train():
            try:
                logger.info(f"Starting ML training with model: {config.model_type}")
                
                # Prepare data
                X, y = self._prepare_data(features, targets, config)
                
                # Create model
                model = self._create_model(config)
                
                # Validate model
                validation_results = self._validate_model(model, X, y, config)
                
                # Train final model on all data
                model.fit(X, y)
                
                # Generate feature importance
                feature_importance = self._get_feature_importance(model, features.columns)
                
                # Create predictions on training data for analysis
                predictions = model.predict(X)
                if hasattr(model, 'predict_proba') and config.target == PredictionTarget.DIRECTION:
                    prediction_probabilities = model.predict_proba(X)
                else:
                    prediction_probabilities = None
                
                # Calculate final metrics
                final_metrics = self._calculate_metrics(y, predictions, config.target)
                
                return MLResult(
                    model_type=config.model_type,
                    target=config.target,
                    prediction_horizon=config.prediction_horizon,
                    features=list(features.columns),
                    hyperparameters=config.hyperparameters,
                    validation_results=validation_results,
                    feature_importance=feature_importance,
                    final_metrics=final_metrics,
                    model_artifact=model,  # In production, save to disk/cloud
                    training_data_info={
                        "n_samples": len(X),
                        "n_features": len(features.columns),
                        "training_period": {
                            "start": features.index.min().isoformat() if hasattr(features.index, 'min') else None,
                            "end": features.index.max().isoformat() if hasattr(features.index, 'max') else None
                        }
                    },
                    metadata=metadata or {}
                )
                
            except Exception as e:
                logger.error(f"ML training failed: {str(e)}")
                raise
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _train)
    
    def _prepare_data(
        self, 
        features: pd.DataFrame, 
        targets: pd.Series, 
        config: MLConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for ML training"""
        
        # Remove any NaN values
        combined_data = pd.concat([features, targets], axis=1).dropna()
        X = combined_data.iloc[:, :-1]
        y = combined_data.iloc[:, -1]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode targets if classification
        if config.target == PredictionTarget.DIRECTION:
            # Convert returns to direction (1 for up, 0 for down)
            y_encoded = (y > 0).astype(int)
        else:
            y_encoded = y.values
            
        return X_scaled, y_encoded
    
    def _create_model(self, config: MLConfig):
        """Create ML model based on configuration"""
        
        params = config.hyperparameters or {}
        
        if config.model_type == "random_forest":
            if config.target == PredictionTarget.DIRECTION:
                return RandomForestClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', None),
                    min_samples_split=params.get('min_samples_split', 2),
                    max_features=params.get('max_features', 'sqrt'),
                    random_state=42,
                    n_jobs=-1
                )
            else:
                return RandomForestRegressor(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', None),
                    min_samples_split=params.get('min_samples_split', 2),
                    max_features=params.get('max_features', 'sqrt'),
                    random_state=42,
                    n_jobs=-1
                )
                
        elif config.model_type == "xgboost":
            if config.target == PredictionTarget.DIRECTION:
                return XGBClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', 6),
                    learning_rate=params.get('learning_rate', 0.1),
                    subsample=params.get('subsample', 1.0),
                    random_state=42,
                    n_jobs=-1
                )
            else:
                return XGBRegressor(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', 6),
                    learning_rate=params.get('learning_rate', 0.1),
                    subsample=params.get('subsample', 1.0),
                    random_state=42,
                    n_jobs=-1
                )
                
        elif config.model_type == "svm":
            if config.target == PredictionTarget.DIRECTION:
                return SVC(
                    C=params.get('C', 1.0),
                    kernel=params.get('kernel', 'rbf'),
                    gamma=params.get('gamma', 'scale'),
                    probability=True,  # Enable probability predictions
                    random_state=42
                )
            else:
                return SVR(
                    C=params.get('C', 1.0),
                    kernel=params.get('kernel', 'rbf'),
                    gamma=params.get('gamma', 'scale')
                )
                
        elif config.model_type == "logistic":
            return LogisticRegression(
                C=params.get('C', 1.0),
                max_iter=params.get('max_iter', 1000),
                random_state=42
            )
            
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
    
    def _validate_model(
        self, 
        model, 
        X: np.ndarray, 
        y: np.ndarray, 
        config: MLConfig
    ) -> Dict[str, Any]:
        """Validate model using time series cross-validation"""
        
        if config.validation == ValidationStrategy.TIME_SERIES_CV:
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                predictions = model.predict(X_val)
                
                metrics = self._calculate_metrics(y_val, predictions, config.target)
                scores.append(metrics)
            
            # Average scores across folds
            avg_metrics = {}
            for key in scores[0].keys():
                avg_metrics[key] = np.mean([score[key] for score in scores])
                avg_metrics[f"{key}_std"] = np.std([score[key] for score in scores])
            
            return {
                "method": "time_series_cv",
                "n_splits": 5,
                "metrics": avg_metrics,
                "individual_scores": scores
            }
            
        elif config.validation == ValidationStrategy.WALK_FORWARD:
            # Walk-forward validation
            window_size = int(len(X) * 0.7)  # 70% for initial training
            step_size = max(1, int(len(X) * 0.05))  # 5% step size
            
            scores = []
            for start in range(0, len(X) - window_size, step_size):
                train_end = start + window_size
                val_start = train_end
                val_end = min(val_start + step_size, len(X))
                
                if val_end <= val_start:
                    break
                
                X_train = X[start:train_end]
                y_train = y[start:train_end]
                X_val = X[val_start:val_end]
                y_val = y[val_start:val_end]
                
                model.fit(X_train, y_train)
                predictions = model.predict(X_val)
                
                metrics = self._calculate_metrics(y_val, predictions, config.target)
                scores.append(metrics)
            
            # Average scores
            avg_metrics = {}
            for key in scores[0].keys():
                avg_metrics[key] = np.mean([score[key] for score in scores])
                avg_metrics[f"{key}_std"] = np.std([score[key] for score in scores])
            
            return {
                "method": "walk_forward",
                "n_windows": len(scores),
                "metrics": avg_metrics,
                "individual_scores": scores
            }
            
        else:
            # Simple train/test split
            split_idx = int(len(X) * (1 - config.test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            metrics = self._calculate_metrics(y_test, predictions, config.target)
            
            return {
                "method": "train_test_split",
                "test_size": config.test_size,
                "metrics": metrics
            }
    
    def _calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        target: PredictionTarget
    ) -> Dict[str, float]:
        """Calculate appropriate metrics based on target type"""
        
        if target == PredictionTarget.DIRECTION:
            # Classification metrics
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
                "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
        else:
            # Regression metrics
            return {
                "mse": mean_squared_error(y_true, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "r2_score": r2_score(y_true, y_pred),
                "mae": np.mean(np.abs(y_true - y_pred))
            }
    
    def _get_feature_importance(
        self, 
        model, 
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Extract feature importance from trained model"""
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance_scores = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importance_scores = np.abs(model.coef_).flatten()
        else:
            # Models without intrinsic feature importance
            return {}
        
        # Normalize to sum to 1
        importance_scores = importance_scores / importance_scores.sum()
        
        return dict(zip(feature_names, importance_scores))
    
    async def predict(
        self, 
        model, 
        features: pd.DataFrame
    ) -> Dict[str, Any]:
        """Make predictions using trained model"""
        
        def _predict():
            # Scale features using the same scaler used during training
            X_scaled = self.scaler.transform(features)
            
            # Make predictions
            predictions = model.predict(X_scaled)
            
            # Get probabilities if available
            probabilities = None
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_scaled)
            
            return {
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist() if probabilities is not None else None,
                "feature_names": list(features.columns),
                "timestamp": datetime.now().isoformat()
            }
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _predict)
    
    async def optimize_hyperparameters(
        self,
        config: MLConfig,
        features: pd.DataFrame,
        targets: pd.Series,
        param_ranges: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using grid search"""
        
        def _optimize():
            logger.info(f"Starting hyperparameter optimization for {config.model_type}")
            
            # Prepare data
            X, y = self._prepare_data(features, targets, config)
            
            # Create parameter grid
            param_grid = list(ParameterGrid(param_ranges))
            
            best_score = -np.inf if config.target != PredictionTarget.DIRECTION else 0
            best_params = None
            best_model = None
            all_results = []
            
            for params in param_grid:
                # Update config with current parameters
                config_copy = MLConfig(
                    model_type=config.model_type,
                    target=config.target,
                    prediction_horizon=config.prediction_horizon,
                    hyperparameters=params,
                    validation=config.validation,
                    test_size=config.test_size
                )
                
                # Create and validate model
                model = self._create_model(config_copy)
                validation_results = self._validate_model(model, X, y, config_copy)
                
                # Get primary metric
                if config.target == PredictionTarget.DIRECTION:
                    score = validation_results['metrics']['accuracy']
                else:
                    score = -validation_results['metrics']['mse']  # Minimize MSE
                
                all_results.append({
                    "parameters": params,
                    "score": score,
                    "metrics": validation_results['metrics']
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = model
            
            return {
                "best_parameters": best_params,
                "best_score": best_score,
                "best_model": best_model,
                "all_results": sorted(all_results, key=lambda x: x['score'], reverse=True),
                "optimization_completed": datetime.now().isoformat()
            }
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _optimize)
    
    def generate_features(
        self, 
        price_data: pd.DataFrame,
        lookback_periods: List[int] = [5, 10, 20, 50]
    ) -> pd.DataFrame:
        """Generate technical features from price data"""
        
        features = pd.DataFrame(index=price_data.index)
        
        # Price-based features
        for period in lookback_periods:
            # Returns
            features[f'return_{period}d'] = price_data['close'].pct_change(period)
            
            # Moving averages
            features[f'sma_{period}'] = price_data['close'].rolling(period).mean()
            features[f'sma_ratio_{period}'] = price_data['close'] / features[f'sma_{period}']
            
            # Volatility
            features[f'volatility_{period}d'] = price_data['close'].pct_change().rolling(period).std()
            
            # RSI
            delta = price_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Volume features (if available)
        if 'volume' in price_data.columns:
            for period in lookback_periods:
                features[f'volume_sma_{period}'] = price_data['volume'].rolling(period).mean()
                features[f'volume_ratio_{period}'] = price_data['volume'] / features[f'volume_sma_{period}']
        
        # OHLC features (if available)
        if all(col in price_data.columns for col in ['open', 'high', 'low', 'close']):
            # Price ranges
            features['daily_range'] = (price_data['high'] - price_data['low']) / price_data['close']
            features['overnight_return'] = (price_data['open'] - price_data['close'].shift(1)) / price_data['close'].shift(1)
            features['intraday_return'] = (price_data['close'] - price_data['open']) / price_data['open']
        
        # Remove any infinite or NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        return features