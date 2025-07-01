# feature_engineering.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class BidFeatures:
    """Data class to hold all engineered features for a bid"""
    # Provider attributes
    provider_reputation: float
    provider_success_rate: float
    provider_avg_delivery: float
    provider_quality_score: float
    
    # Procurement context
    price_relative_to_budget: float
    price_rank_among_bids: float
    num_competitors: int
    market_competitiveness: float
    
    # Temporal features
    time_to_deadline_hours: float
    submission_timing_percentile: float
    
    # Historical patterns
    provider_category_experience: float
    provider_complexity_match: float
    
    # Competitive dynamics
    price_vs_avg_competitor: float
    quality_vs_avg_competitor: float
    delivery_vs_avg_competitor: float

class FeatureEngineer:
    """
    Advanced feature engineering for proposal optimization
    Implements the features described in the research paper
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        
    def engineer_provider_features(self, provider_data: pd.DataFrame, 
                                 historical_bids: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer provider-specific features including reputation and track record
        """
        features = provider_data.copy()
        
        # Calculate provider success metrics
        provider_stats = historical_bids.groupby('provider_id').agg({
            'is_winner': ['sum', 'count', 'mean'],
            'price': ['mean', 'std'],
            'delivery_time': ['mean', 'std'],
            'quality_score': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        provider_stats.columns = [f"{col[0]}_{col[1]}" for col in provider_stats.columns]
        
        # Calculate advanced metrics
        provider_stats['win_rate'] = provider_stats['is_winner_mean']
        provider_stats['bid_frequency'] = provider_stats['is_winner_count']
        provider_stats['price_consistency'] = 1 / (1 + provider_stats['price_std'].fillna(1))
        provider_stats['delivery_reliability'] = 1 / (1 + provider_stats['delivery_time_std'].fillna(1))
        
        # Merge with provider data
        features = features.merge(provider_stats, left_on='id', right_index=True, how='left')
        
        # Fill NaN values for new providers
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].fillna(0)
        
        return features
    
    def engineer_competition_features(self, current_bid: Dict, 
                                    all_bids_for_project: List[Dict]) -> Dict:
        """
        Engineer features related to competitive dynamics
        """
        if len(all_bids_for_project) <= 1:
            return {
                'num_competitors': 0,
                'price_rank': 1,
                'price_percentile': 50,
                'price_vs_avg': 0,
                'quality_vs_avg': 0,
                'delivery_vs_avg': 0,
                'market_competitiveness': 0
            }
        
        prices = [bid['price'] for bid in all_bids_for_project]
        qualities = [bid['quality_score'] for bid in all_bids_for_project]
        deliveries = [bid['delivery_time'] for bid in all_bids_for_project]
        
        current_price = current_bid['price']
        current_quality = current_bid['quality_score']
        current_delivery = current_bid['delivery_time']
        
        # Competitive positioning
        price_rank = sorted(prices).index(current_price) + 1
        price_percentile = (price_rank / len(prices)) * 100
        
        # Relative performance metrics
        avg_price = np.mean(prices)
        avg_quality = np.mean(qualities)
        avg_delivery = np.mean(deliveries)
        
        # Market competitiveness (price variance as proxy)
        market_competitiveness = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
        
        return {
            'num_competitors': len(all_bids_for_project) - 1,
            'price_rank': price_rank,
            'price_percentile': price_percentile,
            'price_vs_avg': (current_price - avg_price) / avg_price if avg_price > 0 else 0,
            'quality_vs_avg': (current_quality - avg_quality) / avg_quality if avg_quality > 0 else 0,
            'delivery_vs_avg': (current_delivery - avg_delivery) / avg_delivery if avg_delivery > 0 else 0,
            'market_competitiveness': market_competitiveness
        }
    
    def engineer_temporal_features(self, bid_submission_time: datetime,
                                 project_deadline: datetime,
                                 all_submission_times: List[datetime]) -> Dict:
        """
        Engineer time-based features
        """
        # Time to deadline
        time_to_deadline = project_deadline - bid_submission_time
        hours_to_deadline = time_to_deadline.total_seconds() / 3600
        
        # Submission timing relative to others
        if len(all_submission_times) > 1:
            sorted_times = sorted(all_submission_times)
            submission_rank = sorted_times.index(bid_submission_time) + 1
            timing_percentile = (submission_rank / len(sorted_times)) * 100
        else:
            timing_percentile = 50
        
        # Early vs late submission indicator
        total_auction_duration = max(all_submission_times) - min(all_submission_times)
        relative_timing = (bid_submission_time - min(all_submission_times)) / total_auction_duration if total_auction_duration.total_seconds() > 0 else 0.5
        
        return {
            'hours_to_deadline': hours_to_deadline,
            'timing_percentile': timing_percentile,
            'relative_timing': relative_timing.total_seconds() if hasattr(relative_timing, 'total_seconds') else relative_timing,
            'is_early_bird': timing_percentile < 25,
            'is_last_minute': timing_percentile > 75
        }
    
    def engineer_historical_pattern_features(self, provider_id: str,
                                           project_category: str,
                                           project_complexity: int,
                                           historical_data: pd.DataFrame) -> Dict:
        """
        Engineer features based on historical bidding patterns
        """
        provider_history = historical_data[historical_data['provider_id'] == provider_id]
        
        if len(provider_history) == 0:
            return {
                'category_experience': 0,
                'complexity_match_score': 0,
                'avg_price_trend': 0,
                'win_rate_trend': 0
            }
        
        # Category experience
        category_bids = provider_history[provider_history['project_category'] == project_category]
        category_experience = len(category_bids) / len(provider_history) if len(provider_history) > 0 else 0
        
        # Complexity matching
        complexity_diffs = np.abs(provider_history['complexity'] - project_complexity)
        complexity_match_score = np.exp(-np.mean(complexity_diffs)) if len(complexity_diffs) > 0 else 0
        
        # Recent trends (last 10 bids)
        recent_history = provider_history.tail(10)
        if len(recent_history) >= 3:
            # Price trend
            recent_prices = recent_history['price'].values
            price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            
            # Win rate trend
            recent_wins = recent_history['is_winner'].values
            win_rate_trend = np.mean(recent_wins[-5:]) - np.mean(recent_wins[:5]) if len(recent_wins) >= 5 else 0
        else:
            price_trend = 0
            win_rate_trend = 0
        
        return {
            'category_experience': category_experience,
            'complexity_match_score': complexity_match_score,
            'avg_price_trend': price_trend,
            'win_rate_trend': win_rate_trend
        }
    
    def engineer_project_features(self, project_data: Dict) -> Dict:
        """
        Engineer project-specific features
        """
        budget_range = project_data.get('budget_max', 0) - project_data.get('budget_min', 0)
        budget_flexibility = budget_range / project_data.get('budget_max', 1) if project_data.get('budget_max', 0) > 0 else 0
        
        # Time pressure
        deadline = datetime.fromisoformat(project_data['deadline']) if isinstance(project_data['deadline'], str) else project_data['deadline']
        time_pressure = max(0, 1 - (deadline - datetime.utcnow()).days / 30)  # Normalized by 30 days
        
        return {
            'budget_flexibility': budget_flexibility,
            'time_pressure': time_pressure,
            'complexity_normalized': project_data['complexity'] / 10.0,
            'budget_midpoint': (project_data.get('budget_min', 0) + project_data.get('budget_max', 0)) / 2
        }
    
    def create_feature_vector(self, bid_data: Dict, project_data: Dict,
                            provider_data: Dict, competition_data: List[Dict],
                            historical_data: pd.DataFrame) -> np.ndarray:
        """
        Create complete feature vector for a bid
        """
        features = {}
        
        # Basic bid features
        features.update({
            'price': bid_data['price'],
            'delivery_time': bid_data['delivery_time'],
            'quality_score': bid_data['quality_score']
        })
        
        # Provider features
        features.update({
            'provider_reputation': provider_data.get('reputation_score', 0),
            'provider_success_rate': provider_data.get('historical_success_rate', 0),
            'provider_avg_delivery': provider_data.get('average_delivery_time', 0),
            'provider_quality': provider_data.get('quality_score', 0)
        })
        
        # Competition features
        competition_features = self.engineer_competition_features(bid_data, competition_data)
        features.update(competition_features)
        
        # Project features
        project_features = self.engineer_project_features(project_data)
        features.update(project_features)
        
        # Price relative to budget
        budget_max = project_data.get('budget_max', float('inf'))
        features['price_to_budget_ratio'] = bid_data['price'] / budget_max if budget_max > 0 else 0
        
        # Temporal features (if available)
        if 'submitted_at' in bid_data and 'deadline' in project_data:
            submission_time = datetime.fromisoformat(bid_data['submitted_at']) if isinstance(bid_data['submitted_at'], str) else bid_data['submitted_at']
            deadline = datetime.fromisoformat(project_data['deadline']) if isinstance(project_data['deadline'], str) else project_data['deadline']
            all_submission_times = [datetime.fromisoformat(comp['submitted_at']) if isinstance(comp['submitted_at'], str) else comp['submitted_at'] for comp in competition_data]
            
            temporal_features = self.engineer_temporal_features(submission_time, deadline, all_submission_times)
            features.update(temporal_features)
        
        # Historical pattern features
        if 'provider_id' in bid_data:
            historical_features = self.engineer_historical_pattern_features(
                bid_data['provider_id'],
                project_data['category'],
                project_data['complexity'],
                historical_data
            )
            features.update(historical_features)
        
        # Convert to numpy array
        feature_names = sorted(features.keys())
        feature_vector = np.array([features[name] for name in feature_names])
        
        return feature_vector, feature_names
    
    def fit_scalers(self, training_data: pd.DataFrame):
        """
        Fit scalers and encoders on training data
        """
        numeric_columns = training_data.select_dtypes(include=[np.number]).columns
        categorical_columns = training_data.select_dtypes(include=['object']).columns
        
        # Fit standard scaler for numeric features
        self.scalers['standard'] = StandardScaler()
        self.scalers['standard'].fit(training_data[numeric_columns])
        
        # Fit label encoders for categorical features
        for col in categorical_columns:
            self.encoders[col] = LabelEncoder()
            self.encoders[col].fit(training_data[col].astype(str))
    
    def transform_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted scalers and encoders
        """
        transformed_data = data.copy()
        
        # Transform numeric features
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if 'standard' in self.scalers and len(numeric_columns) > 0:
            transformed_data[numeric_columns] = self.scalers['standard'].transform(data[numeric_columns])
        
        # Transform categorical features
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col in self.encoders:
                transformed_data[col] = self.encoders[col].transform(data[col].astype(str))
        
        return transformed_data
    
    def calculate_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """
        Calculate and store feature importance scores
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_scores = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            else:
                logger.warning("Model does not have feature importance or coefficients")
                return {}
            
            importance_dict = dict(zip(feature_names, importance_scores))
            self.feature_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            return self.feature_importance
        
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}