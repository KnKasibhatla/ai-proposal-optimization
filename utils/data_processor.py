"""
Data Processor Utility
Handles data loading, processing, and preparation for AI models
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Data processing utility for proposal optimization platform
    Handles data loading, cleaning, validation, and preparation
    """
    
    def __init__(self):
        """Initialize data processor"""
        self.data_schema = self._define_data_schema()
        self.processed_data_cache = {}
        
    def _define_data_schema(self) -> Dict[str, Dict[str, Any]]:
        """Define expected data schema"""
        return {
            'bid_id': {'type': 'string', 'required': True},
            'price': {'type': 'float', 'required': True, 'min': 0},
            'provider_id': {'type': 'string', 'required': True},
            'win_loss': {'type': 'string', 'required': True, 'values': ['win', 'loss']},
            'quality_score': {'type': 'int', 'required': True, 'min': 1, 'max': 10},
            'delivery_time': {'type': 'int', 'required': True, 'min': 1},
            'date': {'type': 'datetime', 'required': True},
            'complexity': {'type': 'int', 'required': True, 'min': 1, 'max': 10},
            'project_category': {'type': 'string', 'required': True},
            'winning_provider_id': {'type': 'string', 'required': False},
            'winning_price': {'type': 'float', 'required': False, 'min': 0},
            'num_bids': {'type': 'int', 'required': False, 'min': 1},
            'client_id': {'type': 'string', 'required': False}
        }
    
    def load_initial_data(self) -> Optional[pd.DataFrame]:
        """Load initial dataset from various sources"""
        try:
            # Try to load from common data paths
            data_paths = [
                '../../data/raw/bidding_data.csv',
                '../../bidding_data.csv',
                './data/bidding_data.csv',
                './bidding_data.csv'
            ]
            
            for path in data_paths:
                if os.path.exists(path):
                    logger.info(f"Loading data from {path}")
                    data = pd.read_csv(path)
                    return self.process_raw_data(data)
            
            # Generate sample data if no file found
            logger.warning("No data file found, generating sample data")
            return self.generate_sample_data()
            
        except Exception as e:
            logger.error(f"Error loading initial data: {str(e)}")
            return None
    
    def process_raw_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process raw data with cleaning and validation"""
        try:
            logger.info(f"Processing raw data with {len(data)} rows")
            
            # Make a copy to avoid modifying original
            processed_data = data.copy()
            
            # Data cleaning
            processed_data = self._clean_data(processed_data)
            
            # Data validation
            processed_data = self._validate_data(processed_data)
            
            # Feature engineering
            processed_data = self._engineer_basic_features(processed_data)
            
            # Remove duplicates
            processed_data = processed_data.drop_duplicates(subset=['bid_id'])
            
            logger.info(f"Data processing completed: {len(processed_data)} rows remaining")
            return processed_data
            
        except Exception as e:
            logger.error(f"Data processing error: {str(e)}")
            raise
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data"""
        
        # Convert date column
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
        
        # Clean string columns
        string_columns = ['provider_id', 'project_category', 'client_id', 'win_loss']
        for col in string_columns:
            if col in data.columns:
                data[col] = data[col].astype(str).str.strip().str.lower()
        
        # Clean win_loss column
        if 'win_loss' in data.columns:
            data['win_loss'] = data['win_loss'].map({
                'win': 'win', 'won': 'win', 'w': 'win', '1': 'win', 'true': 'win',
                'loss': 'loss', 'lost': 'loss', 'lose': 'loss', 'l': 'loss', 
                '0': 'loss', 'false': 'loss'
            })
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Remove outliers
        data = self._remove_outliers(data)
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values appropriately"""
        
        # Fill missing winning_provider_id and winning_price for wins
        if 'win_loss' in data.columns:
            win_mask = data['win_loss'] == 'win'
            
            if 'winning_provider_id' in data.columns:
                data.loc[win_mask & data['winning_provider_id'].isna(), 'winning_provider_id'] = \
                    data.loc[win_mask & data['winning_provider_id'].isna(), 'provider_id']
            
            if 'winning_price' in data.columns:
                data.loc[win_mask & data['winning_price'].isna(), 'winning_price'] = \
                    data.loc[win_mask & data['winning_price'].isna(), 'price']
        
        # Fill missing num_bids with median
        if 'num_bids' in data.columns:
            data['num_bids'] = data['num_bids'].fillna(data['num_bids'].median())
        
        # Fill missing client_id
        if 'client_id' in data.columns:
            data['client_id'] = data['client_id'].fillna('unknown_client')
        
        return data
    
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove obvious outliers"""
        
        initial_count = len(data)
        
        # Remove extreme prices (using IQR method)
        if 'price' in data.columns:
            Q1 = data['price'].quantile(0.25)
            Q3 = data['price'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            data = data[(data['price'] >= lower_bound) & (data['price'] <= upper_bound)]
        
        # Remove extreme delivery times
        # Remove extreme delivery times
        if 'delivery_time' in data.columns:
            data = data[(data['delivery_time'] >= 1) & (data['delivery_time'] <= 365)]
        
        # Remove invalid quality scores
        if 'quality_score' in data.columns:
            data = data[(data['quality_score'] >= 1) & (data['quality_score'] <= 10)]
        
        # Remove invalid complexity scores
        if 'complexity' in data.columns:
            data = data[(data['complexity'] >= 1) & (data['complexity'] <= 10)]
        
        outliers_removed = initial_count - len(data)
        if outliers_removed > 0:
            logger.info(f"Removed {outliers_removed} outlier records")
        
        return data
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data against schema"""
        
        validation_errors = []
        
        for column, schema in self.data_schema.items():
            if schema['required'] and column not in data.columns:
                validation_errors.append(f"Required column '{column}' is missing")
                continue
            
            if column not in data.columns:
                continue
            
            # Type validation
            if schema['type'] == 'float':
                try:
                    data[column] = pd.to_numeric(data[column], errors='coerce')
                except:
                    validation_errors.append(f"Column '{column}' cannot be converted to float")
            
            elif schema['type'] == 'int':
                try:
                    data[column] = pd.to_numeric(data[column], errors='coerce').astype('Int64')
                except:
                    validation_errors.append(f"Column '{column}' cannot be converted to int")
            
            elif schema['type'] == 'datetime':
                try:
                    data[column] = pd.to_datetime(data[column], errors='coerce')
                except:
                    validation_errors.append(f"Column '{column}' cannot be converted to datetime")
            
            # Value validation
            if 'min' in schema:
                invalid_mask = data[column] < schema['min']
                if invalid_mask.any():
                    data.loc[invalid_mask, column] = schema['min']
            
            if 'max' in schema:
                invalid_mask = data[column] > schema['max']
                if invalid_mask.any():
                    data.loc[invalid_mask, column] = schema['max']
            
            if 'values' in schema:
                valid_values = schema['values']
                invalid_mask = ~data[column].isin(valid_values)
                if invalid_mask.any():
                    logger.warning(f"Found {invalid_mask.sum()} invalid values in '{column}'")
                    data = data[~invalid_mask]
        
        if validation_errors:
            logger.warning(f"Validation errors: {validation_errors}")
        
        return data
    
    def _engineer_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer basic features for analysis"""
        
        # Time-based features
        if 'date' in data.columns:
            data['year'] = data['date'].dt.year
            data['month'] = data['date'].dt.month
            data['quarter'] = data['date'].dt.quarter
            data['day_of_week'] = data['date'].dt.dayofweek
            data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        
        # Win rate by provider
        if 'provider_id' in data.columns and 'win_loss' in data.columns:
            provider_win_rates = data.groupby('provider_id')['win_loss'].apply(
                lambda x: (x == 'win').mean()
            )
            data['provider_win_rate'] = data['provider_id'].map(provider_win_rates)
        
        # Average price by category
        if 'project_category' in data.columns and 'price' in data.columns:
            category_avg_prices = data.groupby('project_category')['price'].mean()
            data['category_avg_price'] = data['project_category'].map(category_avg_prices)
            data['price_vs_category_avg'] = data['price'] / data['category_avg_price']
        
        # Competitive intensity
        if 'num_bids' in data.columns:
            data['competitive_intensity'] = np.log1p(data['num_bids'])
        
        # Value score (quality/price ratio)
        if 'quality_score' in data.columns and 'price' in data.columns:
            data['value_score'] = data['quality_score'] / (data['price'] / 1000 + 1)
        
        return data
    
    def generate_sample_data(self, n_samples: int = 2000) -> pd.DataFrame:
        """Generate realistic sample data for testing"""
        
        logger.info(f"Generating {n_samples} sample records")
        
        np.random.seed(42)
        
        # Base parameters
        providers = [f'provider_{i:03d}' for i in range(1, 21)]
        categories = ['software_development', 'consulting', 'manufacturing', 
                     'logistics', 'marketing', 'construction', 'design']
        clients = [f'client_{i:03d}' for i in range(1, 51)]
        
        # Generate data
        data = []
        
        for i in range(n_samples):
            # Basic info
            bid_id = f'bid_{i:06d}'
            provider_id = np.random.choice(providers)
            project_category = np.random.choice(categories)
            client_id = np.random.choice(clients)
            
            # Date (last 2 years)
            start_date = datetime.now() - timedelta(days=730)
            random_days = np.random.randint(0, 730)
            date = start_date + timedelta(days=random_days)
            
            # Complexity and quality (correlated)
            complexity = np.random.randint(1, 11)
            quality_score = max(1, min(10, int(np.random.normal(
                5 + complexity * 0.3, 1.5
            ))))
            
            # Delivery time (inversely correlated with quality)
            base_delivery = 30
            delivery_time = max(1, int(np.random.normal(
                base_delivery + (10 - quality_score) * 3, 10
            )))
            
            # Number of bids
            num_bids = max(2, int(np.random.poisson(5) + 2))
            
            # Base price (influenced by complexity and quality)
            base_price = 10000 + complexity * 5000 + quality_score * 2000
            price_noise = np.random.normal(1, 0.2)
            price = max(1000, base_price * price_noise)
            
            # Determine winner (quality and price influence)
            # Simulate competitive bidding
            competitors = []
            for j in range(num_bids):
                comp_quality = np.random.randint(1, 11)
                comp_price = max(1000, base_price * np.random.normal(1, 0.25))
                
                # Winning score (lower price and higher quality are better)
                winning_score = comp_quality / 10 + (base_price / comp_price) * 0.5
                competitors.append({
                    'provider': provider_id if j == 0 else f'comp_{j}',
                    'quality': comp_quality if j == 0 else comp_quality,
                    'price': price if j == 0 else comp_price,
                    'score': winning_score
                })
            
            # Find winner
            winner = max(competitors, key=lambda x: x['score'])
            win_loss = 'win' if winner['provider'] == provider_id else 'loss'
            winning_provider_id = winner['provider']
            winning_price = winner['price']
            
            record = {
                'bid_id': bid_id,
                'price': price,
                'provider_id': provider_id,
                'win_loss': win_loss,
                'quality_score': quality_score,
                'delivery_time': delivery_time,
                'date': date,
                'complexity': complexity,
                'project_category': project_category,
                'winning_provider_id': winning_provider_id,
                'winning_price': winning_price,
                'num_bids': num_bids,
                'client_id': client_id
            }
            
            data.append(record)
        
        df = pd.DataFrame(data)
        logger.info("Sample data generation completed")
        
        return self.process_raw_data(df)
    
    def load_training_data(self) -> pd.DataFrame:
        """Load and prepare data for model training"""
        try:
            # Try to load from cache first
            if 'training_data' in self.processed_data_cache:
                logger.info("Loading training data from cache")
                return self.processed_data_cache['training_data']
            
            # Load initial data
            data = self.load_initial_data()
            
            if data is None or len(data) == 0:
                raise ValueError("No training data available")
            
            # Additional processing for training
            training_data = self._prepare_training_data(data)
            
            # Cache the processed data
            self.processed_data_cache['training_data'] = training_data
            
            logger.info(f"Training data prepared: {len(training_data)} records")
            return training_data
            
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            raise
    
    def _prepare_training_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data specifically for model training"""
        
        # Remove records with missing target variable
        if 'winning_price' in data.columns:
            data = data.dropna(subset=['winning_price'])
        
        # Ensure minimum data quality
        required_columns = ['price', 'quality_score', 'delivery_time', 'complexity']
        data = data.dropna(subset=required_columns)
        
        # Split data chronologically for realistic training
        if 'date' in data.columns:
            data = data.sort_values('date')
        
        # Balance win/loss ratio if too skewed
        if 'win_loss' in data.columns:
            win_ratio = (data['win_loss'] == 'win').mean()
            if win_ratio < 0.1 or win_ratio > 0.9:
                logger.warning(f"Unbalanced win ratio: {win_ratio:.2f}")
                # Could implement balancing logic here
        
        return data
    
    def process_uploaded_file(self, file) -> List[Dict[str, Any]]:
        """Process uploaded CSV file"""
        try:
            # Read the uploaded file
            if file.filename.endswith('.csv'):
                data = pd.read_csv(file)
            elif file.filename.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(file)
            else:
                raise ValueError("Unsupported file format. Please upload CSV or Excel file.")
            
            # Process the data
            processed_data = self.process_raw_data(data)
            
            # Convert to list of dictionaries for database insertion
            records = processed_data.to_dict('records')
            
            # Convert numpy types to Python types for JSON serialization
            for record in records:
                for key, value in record.items():
                    if isinstance(value, (np.integer, np.int64)):
                        record[key] = int(value)
                    elif isinstance(value, (np.floating, np.float64)):
                        record[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        record[key] = value.tolist()
                    elif pd.isna(value):
                        record[key] = None
            
            logger.info(f"Processed uploaded file: {len(records)} records")
            return records
            
        except Exception as e:
            logger.error(f"File processing error: {str(e)}")
            raise
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data summary"""
        try:
            summary = {
                'basic_stats': {
                    'total_records': len(data),
                    'date_range': {
                        'start': data['date'].min().isoformat() if 'date' in data.columns else None,
                        'end': data['date'].max().isoformat() if 'date' in data.columns else None
                    },
                    'unique_providers': data['provider_id'].nunique() if 'provider_id' in data.columns else 0,
                    'unique_clients': data['client_id'].nunique() if 'client_id' in data.columns else 0,
                    'unique_categories': data['project_category'].nunique() if 'project_category' in data.columns else 0
                }
            }
            
            # Win/loss distribution
            if 'win_loss' in data.columns:
                win_loss_dist = data['win_loss'].value_counts().to_dict()
                summary['win_loss_distribution'] = win_loss_dist
                summary['overall_win_rate'] = win_loss_dist.get('win', 0) / len(data)
            
            # Price statistics
            if 'price' in data.columns:
                summary['price_stats'] = {
                    'mean': float(data['price'].mean()),
                    'median': float(data['price'].median()),
                    'std': float(data['price'].std()),
                    'min': float(data['price'].min()),
                    'max': float(data['price'].max())
                }
            
            # Quality score distribution
            if 'quality_score' in data.columns:
                summary['quality_distribution'] = data['quality_score'].value_counts().sort_index().to_dict()
            
            # Category analysis
            if 'project_category' in data.columns and 'win_loss' in data.columns:
                category_stats = data.groupby('project_category').agg({
                    'win_loss': lambda x: (x == 'win').mean(),
                    'price': 'mean'
                }).round(3)
                summary['category_performance'] = category_stats.to_dict('index')
            
            # Provider performance
            if 'provider_id' in data.columns and 'win_loss' in data.columns:
                provider_stats = data.groupby('provider_id').agg({
                    'win_loss': lambda x: (x == 'win').mean(),
                    'price': 'mean',
                    'bid_id': 'count'
                }).round(3)
                
                # Top 10 providers by bid count
                top_providers = provider_stats.sort_values('bid_id', ascending=False).head(10)
                summary['top_providers'] = top_providers.to_dict('index')
            
            # Temporal patterns
            if 'date' in data.columns:
                data['month'] = data['date'].dt.month
                monthly_stats = data.groupby('month').agg({
                    'bid_id': 'count',
                    'win_loss': lambda x: (x == 'win').mean() if 'win_loss' in data.columns else 0
                }).round(3)
                summary['monthly_patterns'] = monthly_stats.to_dict('index')
            
            # Data quality metrics
            summary['data_quality'] = {
                'missing_values': data.isnull().sum().to_dict(),
                'duplicate_bids': data['bid_id'].duplicated().sum() if 'bid_id' in data.columns else 0,
                'completeness_score': (1 - data.isnull().mean().mean())
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating data summary: {str(e)}")
            return {'error': str(e)}
    
    def export_processed_data(self, data: pd.DataFrame, filepath: str, format: str = 'csv'):
        """Export processed data to file"""
        try:
            if format.lower() == 'csv':
                data.to_csv(filepath, index=False)
            elif format.lower() in ['xlsx', 'excel']:
                data.to_excel(filepath, index=False)
            elif format.lower() == 'json':
                data.to_json(filepath, orient='records', date_format='iso')
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Export error: {str(e)}")
            raise
    
    def validate_prediction_input(self, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate input data for prediction"""
        
        errors = []
        
        # Required fields for prediction
        required_fields = ['quality_score', 'delivery_time', 'complexity', 'project_category']
        
        for field in required_fields:
            if field not in input_data:
                errors.append(f"Missing required field: {field}")
            elif input_data[field] is None:
                errors.append(f"Field {field} cannot be null")
        
        # Validate field values
        if 'quality_score' in input_data:
            if not (1 <= input_data['quality_score'] <= 10):
                errors.append("Quality score must be between 1 and 10")
        
        if 'delivery_time' in input_data:
            if input_data['delivery_time'] <= 0:
                errors.append("Delivery time must be positive")
        
        if 'complexity' in input_data:
            if not (1 <= input_data['complexity'] <= 10):
                errors.append("Complexity must be between 1 and 10")
        
        if 'num_bids' in input_data:
            if input_data['num_bids'] < 1:
                errors.append("Number of bids must be at least 1")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def prepare_features_for_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare and standardize features for prediction"""
        
        # Create a copy to avoid modifying original
        features = input_data.copy()
        
        # Set defaults for optional fields
        defaults = {
            'num_bids': 5,
            'provider_experience': 10,
            'market_share': 0.1,
            'client_relationship': 0.5,
            'past_performance': 0.7,
            'risk_level': 0.5,
            'market_volatility': 1.0,
            'economic_indicator': 1.0,
            'seasonal_factor': 1.0,
            'urgency': 0.5
        }
        
        for field, default_value in defaults.items():
            if field not in features:
                features[field] = default_value
        
        # Normalize certain fields
        if 'provider_experience' in features:
            features['provider_experience'] = min(50, max(1, features['provider_experience']))
        
        # Calculate derived features
        if 'quality_score' in features and 'delivery_time' in features:
            features['quality_delivery_ratio'] = features['quality_score'] / (features['delivery_time'] / 30 + 1)
        
        return features
    
    def get_feature_statistics(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of features for normalization"""
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        stats = {}
        for col in numeric_columns:
            stats[col] = {
                'mean': float(data[col].mean()),
                'std': float(data[col].std()),
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'median': float(data[col].median()),
                'q25': float(data[col].quantile(0.25)),
                'q75': float(data[col].quantile(0.75))
            }
        
        return stats
    
    def detect_data_drift(self, reference_data: pd.DataFrame, 
                         new_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect data drift between reference and new data"""
        
        drift_results = {
            'overall_drift_detected': False,
            'drift_score': 0.0,
            'feature_drift': {}
        }
        
        try:
            numeric_columns = reference_data.select_dtypes(include=[np.number]).columns
            
            total_drift_score = 0
            num_features = 0
            
            for col in numeric_columns:
                if col in new_data.columns:
                    # Calculate statistical distance (simplified)
                    ref_mean = reference_data[col].mean()
                    ref_std = reference_data[col].std()
                    new_mean = new_data[col].mean()
                    new_std = new_data[col].std()
                    
                    # Normalized difference in means
                    mean_drift = abs(new_mean - ref_mean) / (ref_std + 1e-8)
                    
                    # Difference in standard deviations
                    std_drift = abs(new_std - ref_std) / (ref_std + 1e-8)
                    
                    # Combined drift score
                    feature_drift_score = (mean_drift + std_drift) / 2
                    
                    drift_results['feature_drift'][col] = {
                        'drift_score': float(feature_drift_score),
                        'mean_change': float(new_mean - ref_mean),
                        'std_change': float(new_std - ref_std),
                        'drift_detected': feature_drift_score > 0.2
                    }
                    
                    total_drift_score += feature_drift_score
                    num_features += 1
            
            if num_features > 0:
                drift_results['drift_score'] = total_drift_score / num_features
                drift_results['overall_drift_detected'] = drift_results['drift_score'] > 0.15
            
        except Exception as e:
            logger.error(f"Data drift detection error: {str(e)}")
            drift_results['error'] = str(e)
        
        return drift_results