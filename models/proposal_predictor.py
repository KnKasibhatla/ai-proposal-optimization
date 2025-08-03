"""
Proposal Predictor Model
Implements the core machine learning model for bid price prediction
Based on the methodology described in the research paper
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
import joblib
import logging
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ProposalPredictor:
    """
    Advanced proposal prediction model implementing the methodology
    from "Winning B2B Proposals: An AI-Powered Approach"
    """
    
    def __init__(self):
        """Initialize the proposal predictor"""
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.is_trained = False
        self.ensemble_weights = {}
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize the ensemble of prediction models"""
        
        # Random Forest for robust baseline predictions
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting for handling complex interactions
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
        
        # Neural Network for capturing non-linear patterns
        self.models['neural_network'] = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.01,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        
        # Deep Learning model for advanced feature learning
        self.models['deep_learning'] = None  # Will be built dynamically
        
        # Initialize scalers
        self.scalers['standard'] = StandardScaler()
        self.scalers['minmax'] = StandardScaler()  # Will use MinMaxScaler in practice
        
        # Initialize encoders
        self.encoders = {}
        
        logger.info("Proposal predictor models initialized")
    
    def preprocess_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features according to the paper's feature engineering approach
        
        Args:
            data: Raw input data
            
        Returns:
            Preprocessed feature matrix
        """
        try:
            df = data.copy()
            
            # Provider and Contract Attributes
            df = self._engineer_provider_features(df)
            
            # Procurement Context Features
            df = self._engineer_procurement_features(df)
            
            # Temporal Features
            df = self._engineer_temporal_features(df)
            
            # Competitive Features
            df = self._engineer_competitive_features(df)
            
            # Non-price Attributes
            df = self._engineer_quality_features(df)
            
            # Handle categorical variables
            df = self._encode_categorical_features(df)
            
            # Scale numerical features
            df = self._scale_numerical_features(df)
            
            logger.info(f"Features preprocessed: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Feature preprocessing error: {str(e)}")
            raise
    
    def _engineer_provider_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer provider-specific features"""
        
        # Provider reputation score (based on historical win rate)
        if 'provider_id' in df.columns:
            provider_stats = df.groupby('provider_id').agg({
                'win_loss': lambda x: (x == 'win').mean(),
                'quality_score': 'mean',
                'delivery_time': 'mean'
            }).add_suffix('_provider_avg')
            
            df = df.merge(provider_stats, left_on='provider_id', right_index=True, how='left')
        
        # Provider experience (number of previous bids)
        if 'provider_id' in df.columns:
            provider_experience = df.groupby('provider_id').size().rename('provider_experience')
            df = df.merge(provider_experience, left_on='provider_id', right_index=True, how='left')
        
        # Provider specialization score
        if 'project_category' in df.columns and 'provider_id' in df.columns:
            category_specialization = df.groupby(['provider_id', 'project_category']).size().groupby('provider_id').max()
            df = df.merge(category_specialization.rename('provider_specialization'), 
                         left_on='provider_id', right_index=True, how='left')
        
        return df
    
    def _engineer_procurement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer procurement context features"""
        
        # Competitive intensity
        if 'num_bids' in df.columns:
            df['competitive_intensity'] = df['num_bids'] / df['num_bids'].max()
        
        # Price positioning
        if 'price' in df.columns and 'winning_price' in df.columns:
            df['price_ratio_to_winner'] = df['price'] / df['winning_price']
            df['price_competitiveness'] = 1 / (1 + np.abs(df['price'] - df['winning_price']))
        
        # Complexity-price ratio
        if 'complexity' in df.columns and 'price' in df.columns:
            df['complexity_price_ratio'] = df['complexity'] / (df['price'] + 1)
        
        return df
    
    def _engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer temporal features"""
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            df['quarter'] = df['date'].dt.quarter
            
            # Seasonal trends
            df['is_q4'] = (df['quarter'] == 4).astype(int)
            df['is_year_end'] = (df['month'] == 12).astype(int)
        
        # Delivery time features
        if 'delivery_time' in df.columns:
            df['delivery_urgency'] = 1 / (df['delivery_time'] + 1)
            df['delivery_category'] = pd.cut(df['delivery_time'], bins=[0, 7, 30, 90, float('inf')], 
                                           labels=['urgent', 'fast', 'normal', 'extended'])
        
        return df
    
    def _engineer_competitive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer competitive dynamics features"""
        
        # Market position features
        if 'price' in df.columns:
            # Price percentile within each bid
            df['price_percentile'] = df.groupby('bid_id')['price'].rank(pct=True)
            
            # Relative price position
            df['price_vs_median'] = df['price'] / df.groupby('bid_id')['price'].transform('median')
            df['price_vs_mean'] = df['price'] / df.groupby('bid_id')['price'].transform('mean')
        
        # Provider market share
        if 'provider_id' in df.columns:
            total_bids = df.shape[0]
            provider_share = df['provider_id'].value_counts() / total_bids
            df['provider_market_share'] = df['provider_id'].map(provider_share)
        
        return df
    
    def _engineer_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer quality and non-price attribute features"""
        
        # Quality-price ratio
        if 'quality_score' in df.columns and 'price' in df.columns:
            df['quality_price_ratio'] = df['quality_score'] / (df['price'] + 1)
        
        # Composite value score
        if all(col in df.columns for col in ['quality_score', 'delivery_time', 'price']):
            # Higher quality, faster delivery, lower price = higher value
            df['value_score'] = (df['quality_score'] / df['quality_score'].max()) + \
                               (1 - df['delivery_time'] / df['delivery_time'].max()) + \
                               (1 - df['price'] / df['price'].max())
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        
        categorical_cols = ['project_category', 'provider_id', 'client_id']
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories
                    known_categories = set(self.encoders[col].classes_)
                    df[col] = df[col].astype(str)
                    df[col] = df[col].apply(lambda x: x if x in known_categories else 'unknown')
                    
                    if 'unknown' not in known_categories:
                        # Add 'unknown' category
                        self.encoders[col].classes_ = np.append(self.encoders[col].classes_, 'unknown')
                    
                    df[f'{col}_encoded'] = self.encoders[col].transform(df[col])
        
        return df
    
    def _scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features"""
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target variables and IDs
        exclude_cols = ['bid_id', 'winning_price', 'price']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        if numerical_cols:
            if not hasattr(self.scalers['standard'], 'mean_'):
                self.scalers['standard'].fit(df[numerical_cols])
            
            df[numerical_cols] = self.scalers['standard'].transform(df[numerical_cols])
        
        return df
    
    def build_deep_learning_model(self, input_shape: int) -> tf.keras.Model:
        """Build deep learning model for proposal prediction"""
        
        model = keras.Sequential([
            keras.layers.Dense(256, activation='relu', input_shape=(input_shape,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.1),
            
            keras.layers.Dense(1, activation='linear')  # Regression output
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train(self, data: pd.DataFrame, target_column: str = 'winning_price') -> Dict[str, Any]:
        """
        Train the ensemble of prediction models
        
        Args:
            data: Training data
            target_column: Target variable column name
            
        Returns:
            Training results and metrics
        """
        try:
            logger.info("Starting model training...")
            
            # Preprocess features
            features_df = self.preprocess_features(data)
            
            # Prepare target variable
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            y = data[target_column].values
            
            # Remove non-feature columns
            feature_cols = [col for col in features_df.columns 
                          if col not in ['bid_id', 'date', 'win_loss', target_column]]
            X = features_df[feature_cols].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            results = {}
            
            # Train traditional ML models
            for name, model in self.models.items():
                if name == 'deep_learning':
                    continue
                
                logger.info(f"Training {name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                
                metrics = self._calculate_metrics(y_test, y_pred)
                results[name] = metrics
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(feature_cols, model.feature_importances_))
                
                logger.info(f"{name} - MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
            
            # Train deep learning model
            logger.info("Training deep learning model...")
            self.models['deep_learning'] = self.build_deep_learning_model(X_train.shape[1])
            
            # Early stopping callback
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=20, restore_best_weights=True
            )
            
            # Train deep model
            history = self.models['deep_learning'].fit(
                X_train, y_train,
                epochs=200,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate deep model
            y_pred_dl = self.models['deep_learning'].predict(X_test).flatten()
            results['deep_learning'] = self._calculate_metrics(y_test, y_pred_dl)
            
            # Calculate ensemble weights based on performance
            self._calculate_ensemble_weights(results)
            
            # Store feature columns for prediction
            self.feature_columns = feature_cols
            self.is_trained = True
            
            logger.info("Model training completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Model training error: {str(e)}")
            raise
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction using the trained ensemble
        
        Args:
            input_data: Input features for prediction
            
        Returns:
            Prediction results with confidence intervals
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            # Convert input to DataFrame
            df = pd.DataFrame([input_data])
            
            # Preprocess features
            features_df = self.preprocess_features(df)
            X = features_df[self.feature_columns].values
            
            # Get predictions from each model
            predictions = {}
            
            for name, model in self.models.items():
                if name == 'deep_learning':
                    pred = model.predict(X, verbose=0).flatten()[0]
                else:
                    pred = model.predict(X)[0]
                
                predictions[name] = pred
            
            # Ensemble prediction
            ensemble_pred = sum(
                pred * self.ensemble_weights.get(name, 0.25) 
                for name, pred in predictions.items()
            )
            
            # Calculate confidence interval
            pred_std = np.std(list(predictions.values()))
            confidence_interval = {
                'lower': ensemble_pred - 1.96 * pred_std,
                'upper': ensemble_pred + 1.96 * pred_std
            }
            
            # Calculate win probability based on competitive positioning
            win_probability = self._calculate_win_probability(input_data, ensemble_pred)
            
            # Get feature importance for this prediction
            feature_importance = self._get_prediction_feature_importance(features_df.iloc[0])
            
            result = {
                'price': ensemble_pred,
                'win_probability': win_probability,
                'confidence_interval': confidence_interval,
                'individual_predictions': predictions,
                'feature_importance': feature_importance
            }
            
            logger.info(f"Prediction completed: ${ensemble_pred:.2f} (win prob: {win_probability:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def _calculate_ensemble_weights(self, results: Dict[str, Dict[str, float]]):
        """Calculate ensemble weights based on model performance"""
        # Use inverse of MAE as weight (better models get higher weights)
        total_inverse_mae = sum(1 / result['mae'] for result in results.values())
        
        for name, result in results.items():
            self.ensemble_weights[name] = (1 / result['mae']) / total_inverse_mae
        
        logger.info(f"Ensemble weights: {self.ensemble_weights}")
    
    def _calculate_win_probability(self, input_data: Dict[str, Any], predicted_price: float) -> float:
        """Calculate probability of winning based on competitive factors"""
        try:
            # Base probability from quality and delivery time
            quality_factor = input_data.get('quality_score', 5) / 10.0
            delivery_factor = max(0, 1 - (input_data.get('delivery_time', 30) / 90.0))
            
            # Competitive factor based on number of bidders
            num_competitors = input_data.get('num_bids', 5)
            competition_factor = 1 / (1 + num_competitors * 0.1)
            
            # Price competitiveness (assuming market average around predicted price)
            price_factor = max(0, 1 - (predicted_price / (predicted_price * 1.2)))
            
            # Combine factors
            win_prob = (quality_factor * 0.3 + 
                       delivery_factor * 0.2 + 
                       competition_factor * 0.2 + 
                       price_factor * 0.3)
            
            return min(0.95, max(0.05, win_prob))
            
        except Exception as e:
            logger.error(f"Win probability calculation error: {str(e)}")
            return 0.5
    
    def _get_prediction_feature_importance(self, features: pd.Series) -> Dict[str, float]:
        """Get feature importance for current prediction"""
        try:
            # Use Random Forest feature importance as baseline
            rf_importance = self.feature_importance.get('random_forest', {})
            
            # Weight by feature values (normalized)
            weighted_importance = {}
            total_weight = 0
            
            for feature, importance in rf_importance.items():
                if feature in features.index:
                    weight = abs(features[feature]) * importance
                    weighted_importance[feature] = weight
                    total_weight += weight
            
            # Normalize
            if total_weight > 0:
                weighted_importance = {
                    k: v / total_weight for k, v in weighted_importance.items()
                }
            
            # Return top 10 features
            sorted_importance = sorted(weighted_importance.items(), 
                                     key=lambda x: x[1], reverse=True)[:10]
            
            return dict(sorted_importance)
            
        except Exception as e:
            logger.error(f"Feature importance calculation error: {str(e)}")
            return {}
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        try:
            model_data = {
                'models': {name: model for name, model in self.models.items() if name != 'deep_learning'},
                'scalers': self.scalers,
                'encoders': self.encoders,
                'feature_importance': self.feature_importance,
                'ensemble_weights': self.ensemble_weights,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained
            }
            
            # Save traditional models
            joblib.dump(model_data, f"{filepath}_traditional.pkl")
            
            # Save deep learning model separately
            if self.models['deep_learning'] is not None:
                self.models['deep_learning'].save(f"{filepath}_deep_model.h5")
            
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Model save error: {str(e)}")
            raise
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        try:
            # Load traditional models
            model_data = joblib.load(f"{filepath}_traditional.pkl")
            
            self.models.update(model_data['models'])
            self.scalers = model_data['scalers']
            self.encoders = model_data['encoders']
            self.feature_importance = model_data['feature_importance']
            self.ensemble_weights = model_data['ensemble_weights']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            
            # Load deep learning model
            try:
                self.models['deep_learning'] = keras.models.load_model(f"{filepath}_deep_model.h5")
            except Exception as e:
                logger.warning(f"Could not load deep learning model: {str(e)}")
                self.models['deep_learning'] = None
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Model load error: {str(e)}")
            raise
    
    def initialize_with_data(self, data: pd.DataFrame):
        """Initialize model with historical data"""
        try:
            if 'winning_price' in data.columns:
                self.train(data, 'winning_price')
                logger.info("Model initialized with historical data")
            else:
                logger.warning("No winning_price column found for training")
        except Exception as e:
            logger.error(f"Model initialization error: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model"""
        return {
            'is_trained': self.is_trained,
            'models': list(self.models.keys()),
            'feature_columns': getattr(self, 'feature_columns', []),
            'ensemble_weights': self.ensemble_weights,
            'num_features': len(getattr(self, 'feature_columns', []))
        }