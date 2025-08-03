"""
Feature Engineer
Advanced feature engineering for proposal optimization
Implements the feature engineering methodology from the research paper
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Advanced feature engineering for competitive bidding optimization
    Implements the methodology described in the research paper
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        self.scalers = {}
        self.encoders = {}
        self.feature_stats = {}
        self.interaction_features = []
        self.selected_features = []
        
        logger.info("Feature engineer initialized")
    
    def create_features(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create comprehensive feature set for proposal prediction
        
        Args:
            input_data: Raw input data dictionary
            
        Returns:
            Engineered features dictionary
        """
        try:
            # Convert to DataFrame for easier processing
            df = pd.DataFrame([input_data])
            
            # Apply all feature engineering steps
            features_df = self._engineer_provider_features(df)
            features_df = self._engineer_procurement_features(features_df)
            features_df = self._engineer_temporal_features(features_df)
            features_df = self._engineer_competitive_features(features_df)
            features_df = self._engineer_quality_features(features_df)
            features_df = self._engineer_interaction_features(features_df)
            features_df = self._engineer_advanced_features(features_df)
            
            # Convert back to dictionary
            features = features_df.iloc[0].to_dict()
            
            # Remove any NaN values
            features = {k: v for k, v in features.items() if not pd.isna(v)}
            
            logger.info(f"Feature engineering completed: {len(features)} features created")
            return features
            
        except Exception as e:
            logger.error(f"Feature engineering error: {str(e)}")
            raise
    
    def _engineer_provider_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer provider-specific features"""
        
        # Provider reputation and experience
        df['provider_experience_normalized'] = df.get('provider_experience', 10) / 50.0
        
        # Provider specialization score
        df['provider_specialization_score'] = df.get('provider_specialization', 5) / 10.0
        
        # Historical performance metrics
        df['provider_win_rate'] = df.get('provider_win_rate', 0.5)
        df['provider_avg_quality'] = df.get('provider_avg_quality', 5) / 10.0
        df['provider_avg_delivery'] = df.get('provider_avg_delivery', 30) / 90.0
        
        # Provider market presence
        df['provider_market_share'] = df.get('provider_market_share', 0.1)
        df['provider_bid_frequency'] = df.get('provider_bid_frequency', 10) / 100.0
        
        # Provider competitive positioning
        df['provider_price_competitiveness'] = df.get('provider_price_competitiveness', 0.5)
        df['provider_differentiation_score'] = (
            df['provider_avg_quality'] * 0.4 +
            (1 - df['provider_avg_delivery']) * 0.3 +
            df['provider_price_competitiveness'] * 0.3
        )
        
        # Provider reliability indicators
        df['provider_consistency_score'] = df.get('provider_consistency', 0.7)
        df['provider_reputation_index'] = (
            df['provider_win_rate'] * 0.3 +
            df['provider_avg_quality'] * 0.3 +
            df['provider_consistency_score'] * 0.2 +
            df['provider_market_share'] * 10 * 0.2  # Scale market share
        )
        
        return df
    
    def _engineer_procurement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer procurement context features"""
        
        # Basic procurement parameters
        df['num_bids'] = df.get('num_bids', 5)
        df['complexity'] = df.get('complexity', 5)
        df['quality_score'] = df.get('quality_score', 5)
        df['delivery_time'] = df.get('delivery_time', 30)
        
        # Competitive intensity features
        df['competitive_intensity'] = np.log1p(df['num_bids'])
        df['competition_density'] = df['num_bids'] / (df['num_bids'].max() + 1)
        
        # Complexity-related features
        df['complexity_normalized'] = df['complexity'] / 10.0
        df['complexity_category'] = pd.cut(
            df['complexity'], 
            bins=[0, 3, 6, 8, 10], 
            labels=['low', 'medium', 'high', 'very_high']
        ).astype(str)
        
        # Project scale indicators
        df['project_scale_factor'] = df['complexity'] * np.log1p(df['delivery_time'])
        df['urgency_factor'] = 1 / (df['delivery_time'] / 30 + 1)
        
        # Market context
        df['market_volatility'] = df.get('market_volatility', 1.0)
        df['economic_indicator'] = df.get('economic_indicator', 1.0)
        df['seasonal_factor'] = df.get('seasonal_factor', 1.0)
        
        # Client relationship factors
        df['client_relationship'] = df.get('client_relationship', 0.5)
        df['client_loyalty_score'] = df.get('client_loyalty', 0.5)
        df['past_client_performance'] = df.get('past_client_performance', 0.7)
        
        return df
    
    def _engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer temporal and time-based features"""
        
        # Current date features
        current_date = datetime.now()
        df['current_year'] = current_date.year
        df['current_month'] = current_date.month
        df['current_quarter'] = (current_date.month - 1) // 3 + 1
        df['current_day_of_week'] = current_date.weekday()
        df['current_day_of_year'] = current_date.timetuple().tm_yday
        
        # Seasonal patterns
        df['is_q4'] = (df['current_quarter'] == 4).astype(int)
        df['is_year_end'] = (df['current_month'] == 12).astype(int)
        df['is_weekend'] = (df['current_day_of_week'] >= 5).astype(int)
        df['is_month_end'] = (current_date.day > 25).astype(int)
        
        # Business cycle features
        df['seasonal_business_factor'] = np.sin(2 * np.pi * df['current_month'] / 12)
        df['quarterly_business_cycle'] = np.cos(2 * np.pi * df['current_quarter'] / 4)
        
        # Delivery time features
        df['delivery_urgency'] = 1 / (df['delivery_time'] + 1)
        df['delivery_risk'] = np.where(df['delivery_time'] < 7, 0.8, 
                                     np.where(df['delivery_time'] < 30, 0.3, 0.1))
        
        # Time pressure indicators
        df['time_pressure_score'] = (
            df['delivery_urgency'] * 0.6 +
            df['delivery_risk'] * 0.4
        )
        
        return df
    
    def _engineer_competitive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer competitive dynamics features"""
        
        # Competitive positioning
        df['competitive_strength'] = df.get('competitive_strength', 0.5)
        df['market_position_score'] = df.get('market_position_score', 0.5)
        
        # Competitor analysis features
        df['avg_competitor_quality'] = df.get('avg_competitor_quality', 5) / 10.0
        df['avg_competitor_price'] = df.get('avg_competitor_price', 1.0)
        df['competitor_threat_level'] = df.get('competitor_threat_level', 0.5)
        
        # Competitive advantage metrics
        df['quality_advantage'] = (df['quality_score'] / 10.0) - df['avg_competitor_quality']
        df['price_advantage'] = df['avg_competitor_price'] - 1.0  # Assuming our price is baseline
        
        # Market share and dominance
        df['market_dominance'] = df.get('market_dominance', 0.1)
        df['competitive_moat'] = df.get('competitive_moat', 0.3)
        
        # Strategic positioning
        df['differentiation_index'] = (
            abs(df['quality_advantage']) * 0.4 +
            abs(df['price_advantage']) * 0.3 +
            df['competitive_moat'] * 0.3
        )
        
        # Competition intensity effects
        df['competition_pressure'] = df['num_bids'] / (df['market_dominance'] * 10 + 1)
        df['competitive_response_factor'] = (
            df['competitor_threat_level'] * df['competition_pressure']
        )
        
        # Winner prediction features
        df['win_probability_base'] = (
            df['quality_score'] / 10.0 * 0.3 +
            df['delivery_urgency'] * 0.2 +
            df['competitive_strength'] * 0.5
        )
        
        return df
    
    def _engineer_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer quality and non-price attribute features"""
        
        # Quality metrics
        df['quality_normalized'] = df['quality_score'] / 10.0
        df['quality_category'] = pd.cut(
            df['quality_score'],
            bins=[0, 3, 6, 8, 10],
            labels=['poor', 'average', 'good', 'excellent']
        ).astype(str)
        
        # Quality-delivery combinations
        df['quality_delivery_ratio'] = df['quality_score'] / (df['delivery_time'] / 30 + 1)
        df['quality_speed_index'] = (
            df['quality_normalized'] * 0.6 +
            df['delivery_urgency'] * 0.4
        )
        
        # Value proposition features
        df['value_score'] = df.get('value_score', 50)
        df['value_per_complexity'] = df['value_score'] / (df['complexity'] + 1)
        df['quality_complexity_fit'] = (
            df['quality_score'] * df['complexity'] / 100  # Normalized
        )
        
        # Service level features
        df['service_reliability'] = df.get('service_reliability', 0.7)
        df['customer_satisfaction'] = df.get('customer_satisfaction', 0.7)
        df['innovation_score'] = df.get('innovation_score', 0.5)
        
        # Composite quality index
        df['comprehensive_quality_index'] = (
            df['quality_normalized'] * 0.3 +
            df['service_reliability'] * 0.2 +
            df['customer_satisfaction'] * 0.2 +
            df['innovation_score'] * 0.15 +
            df['value_per_complexity'] / 10 * 0.15  # Normalized
        )
        
        # Quality differentiation
        df['quality_premium_justification'] = np.where(
            df['quality_score'] > 7,
            (df['quality_score'] - 7) / 3,
            0
        )
        
        return df
    
    def _engineer_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer interaction features between key variables"""
        
        # Quality-Competition interactions
        df['quality_competition_interaction'] = (
            df['quality_normalized'] * (1 / (df['num_bids'] + 1))
        )
        
        # Price-Quality interactions
        df['price_quality_balance'] = df['quality_normalized']  # Simplified
        
        # Time-Complexity interactions
        df['complexity_time_pressure'] = (
            df['complexity_normalized'] * df['delivery_urgency']
        )
        
        # Market-Competition interactions
        df['market_competition_dynamics'] = (
            df['market_volatility'] * df['competitive_intensity']
        )
        
        # Provider-Market fit
        df['provider_market_fit'] = (
            df['provider_reputation_index'] * df['market_position_score']
        )
        
        # Risk-Reward interactions
        df['risk_adjusted_opportunity'] = (
            df['win_probability_base'] * (1 - df.get('risk_level', 0.5))
        )
        
        # Client-Provider relationship strength
        df['relationship_advantage'] = (
            df['client_relationship'] * df['past_client_performance']
        )
        
        # Strategic positioning interactions
        df['strategic_alignment'] = (
            df['differentiation_index'] * df['provider_market_fit']
        )
        
        return df
    
    def _engineer_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer advanced mathematical and statistical features"""
        
        # Polynomial features for key variables
        key_vars = ['quality_normalized', 'competitive_intensity', 'delivery_urgency']
        for var in key_vars:
            if var in df.columns:
                df[f'{var}_squared'] = df[var] ** 2
                df[f'{var}_cubed'] = df[var] ** 3
                df[f'{var}_sqrt'] = np.sqrt(np.abs(df[var]))
        
        # Logarithmic transformations
        log_vars = ['delivery_time', 'complexity', 'num_bids']
        for var in log_vars:
            if var in df.columns:
                df[f'{var}_log'] = np.log1p(df[var])
                df[f'{var}_log_normalized'] = df[f'{var}_log'] / df[f'{var}_log'].max()
        
        # Exponential features
        df['urgency_exponential'] = np.exp(-df['delivery_time'] / 30)
        df['quality_exponential'] = np.exp(df['quality_normalized'])
        
        # Trigonometric features for cyclical patterns
        df['seasonal_sin'] = np.sin(2 * np.pi * df['current_month'] / 12)
        df['seasonal_cos'] = np.cos(2 * np.pi * df['current_month'] / 12)
        df['weekly_pattern'] = np.sin(2 * np.pi * df['current_day_of_week'] / 7)
        
        # Statistical aggregation features
        feature_groups = {
            'quality_group': ['quality_normalized', 'service_reliability', 'customer_satisfaction'],
            'competition_group': ['competitive_intensity', 'competitor_threat_level', 'competition_pressure'],
            'time_group': ['delivery_urgency', 'time_pressure_score', 'urgency_factor']
        }
        
        for group_name, features in feature_groups.items():
            available_features = [f for f in features if f in df.columns]
            if available_features:
                df[f'{group_name}_mean'] = df[available_features].mean(axis=1)
                df[f'{group_name}_std'] = df[available_features].std(axis=1)
                df[f'{group_name}_max'] = df[available_features].max(axis=1)
                df[f'{group_name}_min'] = df[available_features].min(axis=1)
        
        # Distance and similarity features
        df['euclidean_distance_to_ideal'] = np.sqrt(
            (df['quality_normalized'] - 1) ** 2 +
            (df['delivery_urgency'] - 1) ** 2 +
            (df['competitive_strength'] - 1) ** 2
        )
        
        # Percentile-based features
        if len(df) > 1:  # Only if we have multiple records
            percentile_vars = ['quality_score', 'delivery_time', 'complexity']
            for var in percentile_vars:
                if var in df.columns:
                    df[f'{var}_percentile'] = df[var].rank(pct=True)
        
        return df
    
    def select_important_features(self, data: pd.DataFrame, target: str, 
                                k_best: int = 50) -> List[str]:
        """Select most important features using statistical methods"""
        try:
            # Prepare feature matrix
            feature_cols = [col for col in data.columns if col != target and 
                           data[col].dtype in ['int64', 'float64', 'int32', 'float32']]
            
            X = data[feature_cols].fillna(0)
            y = data[target].fillna(0)
            
            # Remove constant features
            constant_features = [col for col in X.columns if X[col].nunique() <= 1]
            X = X.drop(columns=constant_features)
            
            if len(X.columns) == 0:
                logger.warning("No valid features found for selection")
                return []
            
            # Feature selection using multiple methods
            
            # 1. Statistical significance (F-test)
            try:
                f_selector = SelectKBest(score_func=f_regression, k=min(k_best, len(X.columns)))
                f_selector.fit(X, y)
                f_scores = pd.Series(f_selector.scores_, index=X.columns).sort_values(ascending=False)
                f_selected = f_scores.head(k_best).index.tolist()
            except:
                f_selected = X.columns.tolist()[:k_best]
            
            # 2. Mutual information
            try:
                mi_scores = mutual_info_regression(X, y, random_state=42)
                mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
                mi_selected = mi_scores.head(k_best).index.tolist()
            except:
                mi_selected = X.columns.tolist()[:k_best]
            
            # Combine selections
            combined_features = list(set(f_selected + mi_selected))
            
            # Rank by average importance
            feature_importance = {}
            for feature in combined_features:
                f_rank = f_selected.index(feature) + 1 if feature in f_selected else len(f_selected) + 1
                mi_rank = mi_selected.index(feature) + 1 if feature in mi_selected else len(mi_selected) + 1
                feature_importance[feature] = (f_rank + mi_rank) / 2
            
            # Sort by importance and select top k
            selected_features = sorted(feature_importance.items(), key=lambda x: x[1])[:k_best]
            self.selected_features = [feature for feature, _ in selected_features]
            
            logger.info(f"Selected {len(self.selected_features)} important features")
            return self.selected_features
            
        except Exception as e:
            logger.error(f"Feature selection error: {str(e)}")
            return []
    
    def create_polynomial_features(self, data: pd.DataFrame, 
                                 features: List[str], degree: int = 2) -> pd.DataFrame:
        """Create polynomial interaction features"""
        try:
            if not features or len(features) < 2:
                return data
            
            # Select only the specified features
            feature_data = data[features].fillna(0)
            
            # Create polynomial features
            poly = PolynomialFeatures(degree=degree, include_bias=False, 
                                    interaction_only=True)
            poly_features = poly.fit_transform(feature_data)
            
            # Get feature names
            poly_feature_names = poly.get_feature_names_out(features)
            
            # Create DataFrame with polynomial features
            poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=data.index)
            
            # Remove original features to avoid duplication
            original_features = set(features)
            new_features = [col for col in poly_df.columns if col not in original_features]
            
            # Add only new interaction features to original data
            result_df = data.copy()
            for feature in new_features:
                result_df[f'poly_{feature}'] = poly_df[feature]
            
            logger.info(f"Created {len(new_features)} polynomial features")
            return result_df
            
        except Exception as e:
            logger.error(f"Polynomial feature creation error: {str(e)}")
            return data
    
    def engineer_domain_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer domain-specific features for B2B procurement"""
        
        # Procurement-specific indicators
        df['bid_competitiveness_index'] = (
            df['quality_normalized'] * 0.4 +
            df['delivery_urgency'] * 0.3 +
            (1 - df.get('risk_level', 0.5)) * 0.3
        )
        
        # Strategic fit indicators
        df['strategic_value_alignment'] = (
            df.get('innovation_score', 0.5) * 0.4 +
            df['relationship_advantage'] * 0.3 +
            df['provider_specialization_score'] * 0.3
        )
        
        # Risk assessment features
        df['execution_risk'] = (
            (1 - df['service_reliability']) * 0.4 +
            df['delivery_risk'] * 0.3 +
            (1 - df['provider_consistency_score']) * 0.3
        )
        
        # Economic value features
        df['total_value_proposition'] = (
            df['comprehensive_quality_index'] * 0.4 +
            df['strategic_value_alignment'] * 0.3 +
            (1 - df['execution_risk']) * 0.3
        )
        
        # Market dynamics features
        df['market_opportunity_score'] = (
            df['market_position_score'] * 0.5 +
            (1 - df['competition_pressure']) * 0.3 +
            df['seasonal_factor'] * 0.2
        )
        
        # Client-specific features
        df['client_fit_score'] = (
            df['client_relationship'] * 0.4 +
            df['past_client_performance'] * 0.3 +
            df.get('cultural_fit', 0.7) * 0.3
        )
        
        # Proposal strength indicators
        df['proposal_strength_index'] = (
            df['bid_competitiveness_index'] * 0.3 +
            df['total_value_proposition'] * 0.3 +
            df['client_fit_score'] * 0.2 +
            df['market_opportunity_score'] * 0.2
        )
        
        return df
    
    def normalize_features(self, data: pd.DataFrame, 
                          feature_stats: Optional[Dict] = None) -> pd.DataFrame:
        """Normalize features using stored statistics"""
        try:
            normalized_data = data.copy()
            
            # Get numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if feature_stats is None:
                # Calculate statistics from current data
                feature_stats = {}
                for col in numeric_cols:
                    feature_stats[col] = {
                        'mean': float(data[col].mean()),
                        'std': float(data[col].std()),
                        'min': float(data[col].min()),
                        'max': float(data[col].max())
                    }
                self.feature_stats = feature_stats
            
            # Normalize using z-score or min-max based on data distribution
            for col in numeric_cols:
                if col in feature_stats:
                    stats = feature_stats[col]
                    
                    # Use z-score normalization if std > 0
                    if stats['std'] > 0:
                        normalized_data[col] = (data[col] - stats['mean']) / stats['std']
                    # Use min-max normalization for constant or near-constant features
                    elif stats['max'] > stats['min']:
                        normalized_data[col] = (data[col] - stats['min']) / (stats['max'] - stats['min'])
                    else:
                        normalized_data[col] = 0  # Constant feature
            
            return normalized_data
            
        except Exception as e:
            logger.error(f"Feature normalization error: {str(e)}")
            return data
    
    def get_feature_importance_analysis(self, data: pd.DataFrame, 
                                      target: str) -> Dict[str, Any]:
        """Analyze feature importance and correlations"""
        try:
            analysis = {}
            
            # Select numeric features
            feature_cols = [col for col in data.columns if col != target and 
                           data[col].dtype in ['int64', 'float64', 'int32', 'float32']]
            
            if len(feature_cols) == 0:
                return {'error': 'No numeric features found'}
            
            X = data[feature_cols].fillna(0)
            y = data[target].fillna(0)
            
            # Correlation analysis
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            analysis['correlations'] = correlations.head(20).to_dict()
            
            # Feature statistics
            analysis['feature_stats'] = {
                'total_features': len(feature_cols),
                'high_correlation_features': (correlations > 0.3).sum(),
                'low_variance_features': (X.var() < 0.01).sum()
            }
            
            # Feature categories
            analysis['feature_categories'] = {
                'provider_features': [col for col in feature_cols if 'provider' in col],
                'competitive_features': [col for col in feature_cols if 'compet' in col],
                'quality_features': [col for col in feature_cols if 'quality' in col],
                'temporal_features': [col for col in feature_cols if any(time_word in col for time_word in ['time', 'season', 'month', 'day'])],
                'interaction_features': [col for col in feature_cols if '_' in col and any(op in col for op in ['interaction', 'ratio', 'index'])]
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Feature importance analysis error: {str(e)}")
            return {'error': str(e)}
    
    def save_feature_engineering_config(self, filepath: str):
        """Save feature engineering configuration"""
        try:
            config = {
                'feature_stats': self.feature_stats,
                'selected_features': self.selected_features,
                'interaction_features': self.interaction_features,
                'scalers': {name: scaler.__dict__ for name, scaler in self.scalers.items()},
                'encoders': {name: encoder.__dict__ for name, encoder in self.encoders.items()}
            }
            
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(config, f)
            
            logger.info(f"Feature engineering config saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Config save error: {str(e)}")
    
    def load_feature_engineering_config(self, filepath: str):
        """Load feature engineering configuration"""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                config = pickle.load(f)
            
            self.feature_stats = config.get('feature_stats', {})
            self.selected_features = config.get('selected_features', [])
            self.interaction_features = config.get('interaction_features', [])
            
            logger.info(f"Feature engineering config loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Config load error: {str(e)}")
    
    def validate_features(self, features: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate engineered features"""
        
        errors = []
        
        # Check for required feature types
        required_patterns = ['quality', 'competitive', 'temporal', 'provider']
        for pattern in required_patterns:
            if not any(pattern in key for key in features.keys()):
                errors.append(f"Missing {pattern} features")
        
        # Check for infinite or NaN values
        for key, value in features.items():
            if pd.isna(value) or np.isinf(value):
                errors.append(f"Invalid value in feature {key}: {value}")
        
        # Check feature ranges
        for key, value in features.items():
            if isinstance(value, (int, float)):
                if abs(value) > 1e6:  # Very large values might indicate errors
                    errors.append(f"Extremely large value in feature {key}: {value}")
        
        is_valid = len(errors) == 0
        return is_valid, errors