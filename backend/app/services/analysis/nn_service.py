import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Iterator
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import json

# Deep learning frameworks
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, 
        Flatten, Input, MultiHeadAttention, LayerNormalization,
        GlobalAveragePooling1D, BatchNormalization
    )
    from tensorflow.keras.optimizers import Adam, RMSprop, SGD
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError, Accuracy
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from app.models.analysis import NeuralNetworkConfig, NeuralNetworkResult, ArchitectureType

logger = logging.getLogger(__name__)


class NeuralNetworkService:
    """Neural Network service for financial time series analysis"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)  # Limit for GPU memory
        self.framework = self._detect_framework()
        
        # Configure GPU if available
        if self.framework == 'tensorflow' and TENSORFLOW_AVAILABLE:
            self._configure_tensorflow_gpu()
        elif self.framework == 'pytorch' and PYTORCH_AVAILABLE:
            self._configure_pytorch_gpu()
    
    def _detect_framework(self) -> str:
        """Detect available deep learning framework"""
        if TENSORFLOW_AVAILABLE:
            return 'tensorflow'
        elif PYTORCH_AVAILABLE:
            return 'pytorch'
        else:
            raise RuntimeError("No deep learning framework available. Install TensorFlow or PyTorch.")
    
    def _configure_tensorflow_gpu(self):
        """Configure TensorFlow for GPU usage"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # Enable memory growth to avoid allocating all GPU memory
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"TensorFlow configured with {len(gpus)} GPU(s)")
            else:
                logger.info("TensorFlow running on CPU")
        except Exception as e:
            logger.warning(f"GPU configuration failed: {e}")
    
    def _configure_pytorch_gpu(self):
        """Configure PyTorch for GPU usage"""
        try:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                logger.info(f"PyTorch configured with {device_count} GPU(s)")
                self.device = torch.device('cuda')
            else:
                logger.info("PyTorch running on CPU")
                self.device = torch.device('cpu')
        except Exception as e:
            logger.warning(f"GPU configuration failed: {e}")
            self.device = torch.device('cpu')
    
    async def train_network(
        self,
        config: NeuralNetworkConfig,
        sequences: np.ndarray,
        targets: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        metadata: Dict[str, Any] = None
    ) -> NeuralNetworkResult:
        """Train neural network with given configuration"""
        
        def _train():
            try:
                logger.info(f"Starting NN training with architecture: {config.architecture}")
                
                if self.framework == 'tensorflow':
                    return self._train_tensorflow(config, sequences, targets, validation_data)
                else:
                    return self._train_pytorch(config, sequences, targets, validation_data)
                    
            except Exception as e:
                logger.error(f"Neural network training failed: {str(e)}")
                raise
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _train)
    
    def _train_tensorflow(
        self,
        config: NeuralNetworkConfig,
        sequences: np.ndarray,
        targets: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> NeuralNetworkResult:
        """Train neural network using TensorFlow/Keras"""
        
        # Build model
        model = self._build_tensorflow_model(config, sequences.shape)
        
        # Configure optimizer
        optimizer = self._get_tensorflow_optimizer(config)
        
        # Compile model
        if len(np.unique(targets)) <= 10:  # Classification
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        else:  # Regression
            loss = 'mse'
            metrics = ['mae']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=config.early_stopping_patience,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=config.early_stopping_patience // 2,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = model.fit(
            sequences, targets,
            validation_data=validation_data,
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Generate predictions
        train_predictions = model.predict(sequences)
        val_predictions = None
        if validation_data:
            val_predictions = model.predict(validation_data[0])
        
        # Calculate metrics
        train_metrics = self._calculate_nn_metrics(targets, train_predictions)
        val_metrics = None
        if validation_data:
            val_metrics = self._calculate_nn_metrics(validation_data[1], val_predictions)
        
        return NeuralNetworkResult(
            architecture=config.architecture,
            layers=config.layers,
            training_config={
                'epochs': config.epochs,
                'batch_size': config.batch_size,
                'optimizer': config.optimizer,
                'learning_rate': config.learning_rate
            },
            training_history={
                'loss': history.history['loss'],
                'val_loss': history.history.get('val_loss', []),
                'metrics': {k: v for k, v in history.history.items() if k not in ['loss', 'val_loss']}
            },
            final_metrics={
                'train': train_metrics,
                'validation': val_metrics
            },
            model_artifact=model,  # In production, save to disk/cloud
            training_info={
                'framework': 'tensorflow',
                'total_epochs': len(history.history['loss']),
                'input_shape': sequences.shape,
                'output_shape': targets.shape,
                'gpu_used': len(tf.config.experimental.list_physical_devices('GPU')) > 0
            },
            metadata=metadata or {}
        )
    
    def _build_tensorflow_model(
        self, 
        config: NeuralNetworkConfig, 
        input_shape: Tuple[int, ...]
    ) -> tf.keras.Model:
        """Build TensorFlow/Keras model based on configuration"""
        
        model = Sequential()
        
        if config.architecture == ArchitectureType.LSTM:
            # LSTM architecture
            for i, units in enumerate(config.layers):
                return_sequences = i < len(config.layers) - 1
                if i == 0:
                    model.add(LSTM(
                        units, 
                        return_sequences=return_sequences,
                        input_shape=input_shape[1:],
                        dropout=config.dropout_rate,
                        recurrent_dropout=config.dropout_rate
                    ))
                else:
                    model.add(LSTM(
                        units,
                        return_sequences=return_sequences,
                        dropout=config.dropout_rate,
                        recurrent_dropout=config.dropout_rate
                    ))
                
                if config.batch_normalization:
                    model.add(BatchNormalization())
        
        elif config.architecture == ArchitectureType.GRU:
            # GRU architecture
            for i, units in enumerate(config.layers):
                return_sequences = i < len(config.layers) - 1
                if i == 0:
                    model.add(GRU(
                        units,
                        return_sequences=return_sequences,
                        input_shape=input_shape[1:],
                        dropout=config.dropout_rate,
                        recurrent_dropout=config.dropout_rate
                    ))
                else:
                    model.add(GRU(
                        units,
                        return_sequences=return_sequences,
                        dropout=config.dropout_rate,
                        recurrent_dropout=config.dropout_rate
                    ))
                
                if config.batch_normalization:
                    model.add(BatchNormalization())
        
        elif config.architecture == ArchitectureType.CNN:
            # CNN architecture
            model.add(Conv1D(
                filters=config.layers[0],
                kernel_size=3,
                activation='relu',
                input_shape=input_shape[1:]
            ))
            model.add(MaxPooling1D(pool_size=2))
            
            for units in config.layers[1:]:
                model.add(Conv1D(filters=units, kernel_size=3, activation='relu'))
                model.add(MaxPooling1D(pool_size=2))
                model.add(Dropout(config.dropout_rate))
            
            model.add(GlobalAveragePooling1D())
        
        elif config.architecture == ArchitectureType.TRANSFORMER:
            # Simple Transformer architecture
            inputs = Input(shape=input_shape[1:])
            
            # Multi-head attention
            attention = MultiHeadAttention(
                num_heads=config.attention_heads,
                key_dim=config.layers[0]
            )(inputs, inputs)
            
            # Add & Norm
            attention = LayerNormalization()(inputs + attention)
            
            # Feed Forward
            ffn = Dense(config.layers[1], activation='relu')(attention)
            ffn = Dense(config.layers[0])(ffn)
            ffn = LayerNormalization()(attention + ffn)
            
            # Global pooling and output
            outputs = GlobalAveragePooling1D()(ffn)
            
            model = Model(inputs, outputs)
        
        # Add final dense layers
        for units in config.dense_layers:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(config.dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='linear'))  # Regression by default
        
        return model
    
    def _get_tensorflow_optimizer(self, config: NeuralNetworkConfig):
        """Get TensorFlow optimizer based on configuration"""
        
        if config.optimizer == 'adam':
            return Adam(learning_rate=config.learning_rate)
        elif config.optimizer == 'rmsprop':
            return RMSprop(learning_rate=config.learning_rate)
        elif config.optimizer == 'sgd':
            return SGD(learning_rate=config.learning_rate)
        else:
            return Adam(learning_rate=config.learning_rate)
    
    def _train_pytorch(
        self,
        config: NeuralNetworkConfig,
        sequences: np.ndarray,
        targets: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> NeuralNetworkResult:
        """Train neural network using PyTorch"""
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(sequences).to(self.device)
        y_tensor = torch.FloatTensor(targets).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        
        # Validation data
        val_loader = None
        if validation_data:
            val_X = torch.FloatTensor(validation_data[0]).to(self.device)
            val_y = torch.FloatTensor(validation_data[1]).to(self.device)
            val_dataset = TensorDataset(val_X, val_y)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
        
        # Build model
        model = self._build_pytorch_model(config, sequences.shape).to(self.device)
        
        # Setup optimizer and loss
        optimizer = self._get_pytorch_optimizer(model, config)
        criterion = nn.MSELoss()  # Default to regression
        
        # Training loop
        history = {'loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.epochs):
            # Training phase
            model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            history['loss'].append(avg_loss)
            
            # Validation phase
            if val_loader:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        loss = criterion(outputs.squeeze(), batch_y)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                history['val_loss'].append(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model state
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    model.load_state_dict(best_model_state)
                    break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{config.epochs}, Loss: {avg_loss:.6f}")
        
        # Generate predictions
        model.eval()
        with torch.no_grad():
            train_predictions = model(X_tensor).cpu().numpy()
            val_predictions = None
            if validation_data:
                val_predictions = model(val_X).cpu().numpy()
        
        # Calculate metrics
        train_metrics = self._calculate_nn_metrics(targets, train_predictions)
        val_metrics = None
        if validation_data:
            val_metrics = self._calculate_nn_metrics(validation_data[1], val_predictions)
        
        return NeuralNetworkResult(
            architecture=config.architecture,
            layers=config.layers,
            training_config={
                'epochs': config.epochs,
                'batch_size': config.batch_size,
                'optimizer': config.optimizer,
                'learning_rate': config.learning_rate
            },
            training_history=history,
            final_metrics={
                'train': train_metrics,
                'validation': val_metrics
            },
            model_artifact=model,  # In production, save to disk/cloud
            training_info={
                'framework': 'pytorch',
                'total_epochs': len(history['loss']),
                'input_shape': sequences.shape,
                'output_shape': targets.shape,
                'gpu_used': torch.cuda.is_available()
            },
            metadata={}
        )
    
    def _build_pytorch_model(
        self, 
        config: NeuralNetworkConfig, 
        input_shape: Tuple[int, ...]
    ) -> nn.Module:
        """Build PyTorch model based on configuration"""
        
        class FinancialNN(nn.Module):
            def __init__(self, config, input_size):
                super(FinancialNN, self).__init__()
                self.config = config
                
                if config.architecture == ArchitectureType.LSTM:
                    self.lstm_layers = nn.ModuleList()
                    input_dim = input_size
                    
                    for i, hidden_size in enumerate(config.layers):
                        self.lstm_layers.append(
                            nn.LSTM(input_dim, hidden_size, batch_first=True, dropout=config.dropout_rate)
                        )
                        input_dim = hidden_size
                    
                    self.fc_layers = nn.ModuleList()
                    for units in config.dense_layers:
                        self.fc_layers.append(nn.Linear(input_dim, units))
                        self.fc_layers.append(nn.ReLU())
                        self.fc_layers.append(nn.Dropout(config.dropout_rate))
                        input_dim = units
                    
                    self.output_layer = nn.Linear(input_dim, 1)
                
                elif config.architecture == ArchitectureType.GRU:
                    self.gru_layers = nn.ModuleList()
                    input_dim = input_size
                    
                    for hidden_size in config.layers:
                        self.gru_layers.append(
                            nn.GRU(input_dim, hidden_size, batch_first=True, dropout=config.dropout_rate)
                        )
                        input_dim = hidden_size
                    
                    self.fc_layers = nn.ModuleList()
                    for units in config.dense_layers:
                        self.fc_layers.append(nn.Linear(input_dim, units))
                        self.fc_layers.append(nn.ReLU())
                        self.fc_layers.append(nn.Dropout(config.dropout_rate))
                        input_dim = units
                    
                    self.output_layer = nn.Linear(input_dim, 1)
            
            def forward(self, x):
                if self.config.architecture == ArchitectureType.LSTM:
                    for lstm in self.lstm_layers:
                        x, _ = lstm(x)
                    
                    # Take last time step
                    x = x[:, -1, :]
                    
                    for layer in self.fc_layers:
                        x = layer(x)
                    
                    return self.output_layer(x)
                
                elif self.config.architecture == ArchitectureType.GRU:
                    for gru in self.gru_layers:
                        x, _ = gru(x)
                    
                    # Take last time step
                    x = x[:, -1, :]
                    
                    for layer in self.fc_layers:
                        x = layer(x)
                    
                    return self.output_layer(x)
        
        return FinancialNN(config, input_shape[-1])
    
    def _get_pytorch_optimizer(self, model, config: NeuralNetworkConfig):
        """Get PyTorch optimizer based on configuration"""
        
        if config.optimizer == 'adam':
            return optim.Adam(model.parameters(), lr=config.learning_rate)
        elif config.optimizer == 'rmsprop':
            return optim.RMSprop(model.parameters(), lr=config.learning_rate)
        elif config.optimizer == 'sgd':
            return optim.SGD(model.parameters(), lr=config.learning_rate)
        else:
            return optim.Adam(model.parameters(), lr=config.learning_rate)
    
    def _calculate_nn_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate neural network performance metrics"""
        
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        return {
            "mse": float(np.mean((y_true - y_pred) ** 2)),
            "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            "mae": float(np.mean(np.abs(y_true - y_pred))),
            "r2": float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        }
    
    def prepare_sequences(
        self,
        data: pd.DataFrame,
        target_column: str,
        sequence_length: int = 60,
        prediction_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for neural network training"""
        
        # Sort by time
        data_sorted = data.sort_index()
        
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(data_sorted) - prediction_horizon + 1):
            # Input sequence
            seq = data_sorted.iloc[i - sequence_length:i].values
            sequences.append(seq)
            
            # Target (future value)
            target = data_sorted.iloc[i + prediction_horizon - 1][target_column]
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    async def predict_sequence(
        self,
        model,
        input_sequence: np.ndarray,
        steps: int = 1
    ) -> Dict[str, Any]:
        """Make predictions using trained neural network"""
        
        def _predict():
            if self.framework == 'tensorflow':
                predictions = model.predict(input_sequence)
            else:  # PyTorch
                model.eval()
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(input_sequence).to(self.device)
                    predictions = model(input_tensor).cpu().numpy()
            
            return {
                "predictions": predictions.tolist(),
                "input_shape": input_sequence.shape,
                "timestamp": datetime.now().isoformat()
            }
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _predict)