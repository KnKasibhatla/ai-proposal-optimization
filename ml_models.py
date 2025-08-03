# ml_models.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class ProposalPredictionModel:
    """
    Base class for proposal prediction models
    Implements the supervised learning approach described in the paper
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.training_history = {}
        self.performance_metrics = {}
        
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """Train the model"""
        raise NotImplementedError
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        raise NotImplementedError
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        raise NotImplementedError
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        raise NotImplementedError
        
    def save_model(self, filepath: str):
        """Save model to disk"""
        raise NotImplementedError
        
    def load_model(self, filepath: str):
        """Load model from disk"""
        raise NotImplementedError

class DeepNeuralNetwork(ProposalPredictionModel):
    """
    Deep Neural Network for proposal prediction
    Implements the custom deep learning architecture from the paper
    """
    
    def __init__(self, input_dim: int, hidden_layers: List[int] = [256, 128, 64, 32]):
        super().__init__("DeepNeuralNetwork")
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """
        Build the neural network architecture optimized for proposal prediction
        """
        inputs = Input(shape=(self.input_dim,))
        x = inputs
        
        # Input normalization
        x = BatchNormalization()(x)
        
        # Hidden layers with dropout and batch normalization
        for i, units in enumerate(self.hidden_layers):
            x = Dense(units, activation='relu', name=f'hidden_{i+1}')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
        
        # Output layers
        # Binary classification (win/lose)
        win_probability = Dense(1, activation='sigmoid', name='win_probability')(x)
        
        # Regression for price prediction
        price_prediction = Dense(1, activation='linear', name='price_prediction')(x)
        
        model = Model(inputs=inputs, outputs=[win_probability, price_prediction])
        
        # Compile with multiple objectives
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'win_probability': 'binary_crossentropy',
                'price_prediction': 'mse'
            },
            loss_weights={
                'win_probability': 1.0,
                'price_prediction': 0.1
            },
            metrics={
                'win_probability': ['accuracy', tf.keras.metrics.AUC()],
                'price_prediction': ['mae']
            }
        )
        
        return model
    
    def train(self, X: np.ndarray, y_win: np.ndarray, y_price: np.ndarray, 
              validation_split: float = 0.2, epochs: int = 100, batch_size: int = 32):
        """
        Train the deep neural network
        """
        # Prepare callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-7)
        ]
        
        # Train the model
        history = self.model.fit(
            X, 
            {'win_probability': y_win, 'price_prediction': y_price},
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        self.training_history = history.history
        
        logger.info(f"Model {self.model_name} trained successfully")
        return history
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict both win probability and price
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        win_prob = predictions[0].flatten()
        price_pred = predictions[1].flatten()
        
        return win_prob, price_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict win probability
        """
        win_prob, _ = self.predict(X)
        return np.column_stack([1 - win_prob, win_prob])
    
    def evaluate(self, X: np.ndarray, y_win: np.ndarray, y_price: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        """
        win_prob, price_pred = self.predict(X)
        
        # Classification metrics
        win_pred_binary = (win_prob > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_win, win_pred_binary),
            'precision': precision_score(y_win, win_pred_binary),
            'recall': recall_score(y_win, win_pred_binary),
            'f1_score': f1_score(y_win, win_pred_binary),
            'auc_roc': roc_auc_score(y_win, win_prob),
            'price_mae': np.mean(np.abs(y_price - price_pred)),
            'price_rmse': np.sqrt(np.mean((y_price - price_pred) ** 2))
        }
        
        self.performance_metrics = metrics
        return metrics
    
    def save_model(self, filepath: str):
        """Save the TensorFlow model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        
        # Save additional metadata
        metadata = {
            'model_name': self.model_name,
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'is_trained': self.is_trained,
            'performance_metrics': self.performance_metrics,
            'training_history': self.training_history
        }
        
        metadata_path = filepath.replace('.h5', '_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the TensorFlow model"""
        self.model = tf.keras.models.load_model(filepath)
        
        # Load metadata
        metadata_path = filepath.replace('.h5', '_metadata.pkl')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.input_dim = metadata.get('input_dim', self.input_dim)
            self.hidden_layers = metadata.get('hidden_layers', self.hidden_layers)
            self.is_trained = metadata.get('is_trained', False)
            self.performance_metrics = metadata.get('performance_metrics', {})
            self.training_history = metadata.get('training_history', {})
        
        logger.info(f"Model loaded from {filepath}")

class EnsembleProposalModel(ProposalPredictionModel):
    """
    Ensemble model combining multiple algorithms for robust predictions
    """
    
    def __init__(self):
        super().__init__("EnsembleModel")
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
        }
        self.weights = {'random_forest': 0.3, 'gradient_boosting': 0.25, 'logistic_regression': 0.2, 'neural_network': 0.25}
        
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """Train all models in the ensemble"""
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            if name == 'gradient_boosting':
                # For regression model, we'll predict win probability as continuous
                model.fit(X_train, y_train.astype(float))
            else:
                model.fit(X_train, y_train)
            
            # Evaluate on validation set
            if name == 'gradient_boosting':
                val_pred = model.predict(X_val)
                val_pred_binary = (val_pred > 0.5).astype(int)
            else:
                val_pred_binary = model.predict(X_val)
            
            accuracy = accuracy_score(y_val, val_pred_binary)
            logger.info(f"{name} validation accuracy: {accuracy:.4f}")
        
        self.is_trained = True
        logger.info("Ensemble model training completed")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction using weighted voting"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        
        for name, model in self.models.items():
            if name == 'gradient_boosting':
                # Convert regression output to probability
                prob = model.predict(X)
                prob = np.clip(prob, 0, 1)  # Ensure probabilities are in [0,1]
                prob_matrix = np.column_stack([1 - prob, prob])
            else:
                if hasattr(model, 'predict_proba'):
                    prob_matrix = model.predict_proba(X)
                else:
                    pred = model.predict(X)
                    prob_matrix = np.column_stack([1 - pred, pred])
            
            weighted_prob = prob_matrix * self.weights[name]
            predictions.append(weighted_prob)
        
        # Combine predictions
        ensemble_prob = np.sum(predictions, axis=0)
        return ensemble_prob
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make binary predictions"""
        prob = self.predict_proba(X)
        return (prob[:, 1] > 0.5).astype(int)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble performance"""
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'auc_roc': roc_auc_score(y, y_prob)
        }
        
        self.performance_metrics = metrics
        return metrics
    
    def save_model(self, filepath: str):
        """Save ensemble model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'models': self.models,
            'weights': self.weights,
            'is_trained': self.is_trained,
            'performance_metrics': self.performance_metrics
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Ensemble model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load ensemble model"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.weights = model_data['weights']
        self.is_trained = model_data['is_trained']
        self.performance_metrics = model_data.get('performance_metrics', {})
        logger.info(f"Ensemble model loaded from {filepath}")

class ModelManager:
    """
    Manages multiple models and handles model selection, training, and deployment
    """
    
    def __init__(self):
        self.models = {}
        self.active_model = None
        self.model_performances = {}
        
    def register_model(self, model: ProposalPredictionModel, name: str):
        """Register a model with the manager"""
        self.models[name] = model
        logger.info(f"Model {name} registered")
    
    def train_model(self, model_name: str, X: np.ndarray, y: np.ndarray, **kwargs):
        """Train a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if isinstance(model, DeepNeuralNetwork):
            # Deep learning model expects separate targets
            y_win = y
            y_price = kwargs.get('y_price', np.zeros_like(y))
            model.train(X, y_win, y_price, **kwargs)
        else:
            model.train(X, y, **kwargs)
        
        logger.info(f"Model {model_name} trained successfully")
    
    def evaluate_models(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models"""
        results = {}
        
        for name, model in self.models.items():
            if model.is_trained:
                if isinstance(model, DeepNeuralNetwork):
                    y_price = kwargs.get('y_price', np.zeros_like(y))
                    metrics = model.evaluate(X, y, y_price)
                else:
                    metrics = model.evaluate(X, y)
                
                results[name] = metrics
                self.model_performances[name] = metrics
                
                logger.info(f"{name} - Accuracy: {metrics.get('accuracy', 0):.4f}, "
                          f"F1: {metrics.get('f1_score', 0):.4f}, "
                          f"AUC: {metrics.get('auc_roc', 0):.4f}")
        
        return results
    
    def select_best_model(self, metric: str = 'f1_score'):
        """Select the best performing model based on a specific metric"""
        if not self.model_performances:
            raise ValueError("No model performances available. Train and evaluate models first.")
        
        best_model_name = max(self.model_performances.keys(), 
                            key=lambda x: self.model_performances[x].get(metric, 0))
        
        self.active_model = best_model_name
        logger.info(f"Selected {best_model_name} as active model based on {metric}")
        
        return best_model_name
    
    def predict(self, X: np.ndarray, model_name: Optional[str] = None) -> np.ndarray:
        """Make predictions using active model or specified model"""
        model_to_use = model_name or self.active_model
        
        if model_to_use is None:
            raise ValueError("No active model set. Select a model first.")
        
        if model_to_use not in self.models:
            raise ValueError(f"Model {model_to_use} not found")
        
        model = self.models[model_to_use]
        if not model.is_trained:
            raise ValueError(f"Model {model_to_use} is not trained")
        
        return model.predict(X)
    
    def predict_proba(self, X: np.ndarray, model_name: Optional[str] = None) -> np.ndarray:
        """Predict probabilities using active model or specified model"""
        model_to_use = model_name or self.active_model
        
        if model_to_use is None:
            raise ValueError("No active model set. Select a model first.")
        
        model = self.models[model_to_use]
        if not model.is_trained:
            raise ValueError(f"Model {model_to_use} is not trained")
        
        return model.predict_proba(X)
    
    def save_models(self, directory: str):
        """Save all trained models"""
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            if model.is_trained:
                if isinstance(model, DeepNeuralNetwork):
                    filepath = os.path.join(directory, f"{name}.h5")
                else:
                    filepath = os.path.join(directory, f"{name}.pkl")
                
                model.save_model(filepath)
        
        # Save manager metadata
        metadata = {
            'active_model': self.active_model,
            'model_performances': self.model_performances
        }
        
        metadata_path = os.path.join(directory, 'model_manager_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"All models saved to {directory}")
    
    def load_models(self, directory: str):
        """Load all models from directory"""
        metadata_path = os.path.join(directory, 'model_manager_metadata.pkl')
        
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.active_model = metadata.get('active_model')
            self.model_performances = metadata.get('model_performances', {})
        
        # Load individual models
        for filename in os.listdir(directory):
            if filename.endswith('.pkl') and 'metadata' not in filename:
                model_name = filename.replace('.pkl', '')
                filepath = os.path.join(directory, filename)
                
                # Create and load ensemble model
                model = EnsembleProposalModel()
                model.load_model(filepath)
                self.models[model_name] = model
                
            elif filename.endswith('.h5'):
                model_name = filename.replace('.h5', '')
                filepath = os.path.join(directory, filename)
                
                # Create and load deep learning model
                model = DeepNeuralNetwork(input_dim=10)  # Will be updated when loaded
                model.load_model(filepath)
                self.models[model_name] = model
        
        logger.info(f"Models loaded from {directory}")

# Model training pipeline
class ModelTrainingPipeline:
    """
    Complete pipeline for training proposal prediction models
    """
    
    def __init__(self, feature_engineer, model_manager: ModelManager):
        self.feature_engineer = feature_engineer
        self.model_manager = model_manager
        
    def prepare_training_data(self, historical_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from historical bid data
        """
        # Engineer features for all bids
        feature_vectors = []
        labels = []
        
        for _, bid in historical_data.iterrows():
            try:
                # Get competition data for this project
                project_bids = historical_data[historical_data['project_id'] == bid['project_id']]
                competition_data = project_bids.to_dict('records')
                
                # Create feature vector
                feature_vector, feature_names = self.feature_engineer.create_feature_vector(
                    bid_data=bid.to_dict(),
                    project_data=bid.to_dict(),  # Simplified - in real case would join with project table
                    provider_data=bid.to_dict(),  # Simplified - in real case would join with provider table
                    competition_data=competition_data,
                    historical_data=historical_data
                )
                
                feature_vectors.append(feature_vector)
                labels.append(int(bid['is_winner']))
                
            except Exception as e:
                logger.warning(f"Error processing bid {bid.get('id', 'unknown')}: {e}")
                continue
        
        X = np.array(feature_vectors)
        y = np.array(labels)
        
        # Store feature names
        self.feature_names = feature_names
        
        logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray):
        """
        Train all registered models
        """
        # Train ensemble model
        ensemble_model = EnsembleProposalModel()
        self.model_manager.register_model(ensemble_model, 'ensemble')
        self.model_manager.train_model('ensemble', X, y)
        
        # Train deep learning model
        deep_model = DeepNeuralNetwork(input_dim=X.shape[1])
        self.model_manager.register_model(deep_model, 'deep_nn')
        
        # Create dummy price targets for multi-output training
        y_price = np.random.normal(50000, 20000, len(y))  # Placeholder
        self.model_manager.train_model('deep_nn', X, y, y_price=y_price)
        
        logger.info("All models trained successfully")
    
    def run_training_pipeline(self, historical_data: pd.DataFrame):
        """
        Run the complete training pipeline
        """
        logger.info("Starting model training pipeline...")
        
        # Prepare data
        X, y = self.prepare_training_data(historical_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit feature scalers
        self.feature_engineer.fit_scalers(pd.DataFrame(X_train))
        
        # Transform features
        X_train_scaled = self.feature_engineer.transform_features(pd.DataFrame(X_train)).values
        X_test_scaled = self.feature_engineer.transform_features(pd.DataFrame(X_test)).values
        
        # Train models
        self.train_all_models(X_train_scaled, y_train)
        
        # Evaluate models
        results = self.model_manager.evaluate_models(X_test_scaled, y_test)
        
        # Select best model
        best_model = self.model_manager.select_best_model()
        
        logger.info(f"Training pipeline completed. Best model: {best_model}")
        return results