"""
Configuration module for AI Proposal Optimization Platform
Centralized configuration management
"""

import os
from datetime import timedelta
from typing import Dict, Any

class Config:
    """Base configuration class"""
    
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'ai-proposal-optimization-secret-key-2024'
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///../../data/proposals.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }
    
    # Security
    SESSION_COOKIE_SECURE = os.environ.get('FLASK_ENV') == 'production'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = '../../data/uploads'
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    
    # AI Model configuration
    MODEL_CONFIG = {
        'proposal_predictor': {
            'model_type': 'ensemble',
            'ensemble_models': ['random_forest', 'gradient_boosting', 'neural_network', 'deep_learning'],
            'feature_selection_k': 50,
            'cross_validation_folds': 5,
            'hyperparameter_tuning': True
        },
        'competitive_analyzer': {
            'game_theory_enabled': True,
            'nash_equilibrium_solver': 'iterative',
            'clustering_algorithm': 'kmeans',
            'scenario_count': 5
        },
        'reinforcement_agent': {
            'algorithm': 'dqn',
            'learning_rate': 0.0005,
            'epsilon_decay': 0.995,
            'memory_size': 10000,
            'batch_size': 32,
            'target_update_frequency': 1000
        }
    }
    
    # Feature engineering configuration
    FEATURE_CONFIG = {
        'polynomial_degree': 2,
        'interaction_features': True,
        'temporal_features': True,
        'advanced_features': True,
        'feature_selection': True,
        'normalization_method': 'z_score'
    }
    
    # Data processing configuration
    DATA_CONFIG = {
        'outlier_threshold': 3,  # Standard deviations
        'missing_value_threshold': 0.3,  # Maximum proportion of missing values
        'min_data_points': 100,  # Minimum data points for training
        'validation_split': 0.2,
        'test_split': 0.2,
        'time_series_split': True
    }
    
    # Logging configuration
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_handler': True,
        'console_handler': True,
        'max_bytes': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5
    }
    
    # Cache configuration
    CACHE_CONFIG = {
        'type': 'simple',
        'default_timeout': 300,  # 5 minutes
        'prediction_cache_timeout': 600,  # 10 minutes
        'model_cache_timeout': 3600  # 1 hour
    }
    
    # API configuration
    API_CONFIG = {
        'rate_limit': '100 per hour',
        'cors_origins': ['http://localhost:3000', 'http://localhost:5000'],
        'api_version': 'v1',
        'pagination_size': 50,
        'max_pagination_size': 1000
    }
    
    # Performance monitoring
    MONITORING_CONFIG = {
        'enabled': True,
        'metrics_endpoint': '/metrics',
        'health_check_endpoint': '/health',
        'performance_tracking': True,
        'error_tracking': True
    }

class DevelopmentConfig(Config):
    """Development configuration"""
    
    DEBUG = True
    TESTING = False
    
    # Use in-memory SQLite for development
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
    # Relaxed security for development
    SESSION_COOKIE_SECURE = False
    
    # Enhanced logging for development
    LOGGING_CONFIG = {
        **Config.LOGGING_CONFIG,
        'level': 'DEBUG'
    }
    
    # Faster training for development
    MODEL_CONFIG = {
        **Config.MODEL_CONFIG,
        'proposal_predictor': {
            **Config.MODEL_CONFIG['proposal_predictor'],
            'cross_validation_folds': 3,
            'hyperparameter_tuning': False
        },
        'reinforcement_agent': {
            **Config.MODEL_CONFIG['reinforcement_agent'],
            'memory_size': 1000
        }
    }

class TestingConfig(Config):
    """Testing configuration"""
    
    DEBUG = False
    TESTING = True
    
    # Use test database
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
    # Disable CSRF for testing
    WTF_CSRF_ENABLED = False
    
    # Minimal configuration for faster tests
    MODEL_CONFIG = {
        'proposal_predictor': {
            'model_type': 'simple',
            'ensemble_models': ['random_forest'],
            'feature_selection_k': 10,
            'cross_validation_folds': 2,
            'hyperparameter_tuning': False
        },
        'competitive_analyzer': {
            'game_theory_enabled': False,
            'clustering_algorithm': 'kmeans',
            'scenario_count': 2
        },
        'reinforcement_agent': {
            'algorithm': 'simple',
            'memory_size': 100
        }
    }

class ProductionConfig(Config):
    """Production configuration"""
    
    DEBUG = False
    TESTING = False
    
    # Use PostgreSQL in production
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'postgresql://user:password@localhost/proposal_optimization'
    
    # Enhanced security for production
    SESSION_COOKIE_SECURE = True
    
    # Production logging
    LOGGING_CONFIG = {
        **Config.LOGGING_CONFIG,
        'level': 'WARNING'
    }
    
    # Production performance settings
    MODEL_CONFIG = {
        **Config.MODEL_CONFIG,
        'proposal_predictor': {
            **Config.MODEL_CONFIG['proposal_predictor'],
            'cross_validation_folds': 10,
            'hyperparameter_tuning': True
        }
    }
    
    # Redis cache for production
    CACHE_CONFIG = {
        'type': 'redis',
        'redis_url': os.environ.get('REDIS_URL') or 'redis://localhost:6379',
        'default_timeout': 1800,  # 30 minutes
        'prediction_cache_timeout': 3600,  # 1 hour
        'model_cache_timeout': 86400  # 24 hours
    }

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config() -> Config:
    """Get configuration based on environment variable"""
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])

# Additional configuration utilities
class ModelPaths:
    """Model file paths configuration"""
    
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, '..', '..', 'data', 'models')
    
    PROPOSAL_PREDICTOR_PATH = os.path.join(MODELS_DIR, 'proposal_predictor')
    COMPETITIVE_ANALYZER_PATH = os.path.join(MODELS_DIR, 'competitive_analyzer')
    REINFORCEMENT_AGENT_PATH = os.path.join(MODELS_DIR, 'reinforcement_agent')
    FEATURE_ENGINEER_PATH = os.path.join(MODELS_DIR, 'feature_engineer.pkl')
    
    @classmethod
    def ensure_directories(cls):
        """Ensure model directories exist"""
        os.makedirs(cls.MODELS_DIR, exist_ok=True)

class DataPaths:
    """Data file paths configuration"""
    
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data')
    
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    UPLOADS_DIR = os.path.join(DATA_DIR, 'uploads')
    EXPORTS_DIR = os.path.join(DATA_DIR, 'exports')
    
    # Specific file paths
    BIDDING_DATA_PATH = os.path.join(RAW_DATA_DIR, 'bidding_data.csv')
    SAMPLE_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'sample_data.csv')
    TRAINING_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'training_data.csv')
    
    @classmethod
    def ensure_directories(cls):
        """Ensure data directories exist"""
        for directory in [cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, 
                         cls.UPLOADS_DIR, cls.EXPORTS_DIR]:
            os.makedirs(directory, exist_ok=True)

class FeatureConfig:
    """Feature engineering configuration"""
    
    # Provider features configuration
    PROVIDER_FEATURES = {
        'experience_weight': 0.3,
        'reputation_weight': 0.4,
        'specialization_weight': 0.3,
        'consistency_threshold': 0.7,
        'market_share_scaling': 10.0
    }
    
    # Competitive features configuration
    COMPETITIVE_FEATURES = {
        'threat_level_threshold': 0.7,
        'competition_intensity_scaling': 'log1p',
        'market_dominance_threshold': 0.2,
        'quality_advantage_weight': 0.4,
        'price_advantage_weight': 0.3
    }
    
    # Temporal features configuration
    TEMPORAL_FEATURES = {
        'seasonal_cycles': ['monthly', 'quarterly', 'weekly'],
        'business_cycle_periods': [12, 4, 52],  # months, quarters, weeks
        'urgency_threshold_days': 7,
        'delivery_risk_thresholds': [7, 30, 90]
    }
    
    # Quality features configuration
    QUALITY_FEATURES = {
        'quality_score_range': [1, 10],
        'quality_categories': ['poor', 'average', 'good', 'excellent'],
        'quality_thresholds': [3, 6, 8],
        'service_reliability_weight': 0.2,
        'customer_satisfaction_weight': 0.2,
        'innovation_weight': 0.15
    }
    
    # Advanced features configuration
    ADVANCED_FEATURES = {
        'polynomial_degree': 2,
        'enable_interactions': True,
        'enable_logarithmic': True,
        'enable_exponential': True,
        'enable_trigonometric': True,
        'outlier_detection_method': 'iqr',
        'outlier_threshold': 3.0
    }

class ValidationConfig:
    """Data validation configuration"""
    
    # Schema validation rules
    SCHEMA_RULES = {
        'required_columns': [
            'bid_id', 'price', 'provider_id', 'win_loss', 
            'quality_score', 'delivery_time', 'complexity'
        ],
        'optional_columns': [
            'date', 'project_category', 'winning_provider_id', 
            'winning_price', 'num_bids', 'client_id'
        ],
        'data_types': {
            'price': 'float',
            'quality_score': 'int',
            'delivery_time': 'int',
            'complexity': 'int',
            'num_bids': 'int'
        },
        'value_ranges': {
            'quality_score': [1, 10],
            'complexity': [1, 10],
            'delivery_time': [1, 365],
            'price': [0, float('inf')]
        },
        'categorical_values': {
            'win_loss': ['win', 'loss']
        }
    }
    
    # Data quality thresholds
    QUALITY_THRESHOLDS = {
        'missing_data_threshold': 0.3,
        'duplicate_threshold': 0.05,
        'outlier_threshold': 0.1,
        'min_records_training': 100,
        'min_records_validation': 50
    }

class SecurityConfig:
    """Security configuration"""
    
    # Input validation
    INPUT_VALIDATION = {
        'max_string_length': 255,
        'allowed_file_types': ['csv', 'xlsx', 'xls'],
        'max_upload_size': 16 * 1024 * 1024,  # 16MB
        'sanitize_inputs': True,
        'escape_html': True
    }
    
    # Rate limiting
    RATE_LIMITING = {
        'prediction_requests': '100 per hour',
        'upload_requests': '10 per hour',
        'training_requests': '5 per hour',
        'api_requests': '1000 per hour'
    }
    
    # Authentication (if implemented)
    AUTHENTICATION = {
        'session_timeout': 3600,  # 1 hour
        'max_login_attempts': 5,
        'lockout_duration': 300,  # 5 minutes
        'password_min_length': 8
    }

class PerformanceConfig:
    """Performance optimization configuration"""
    
    # Model training performance
    TRAINING_PERFORMANCE = {
        'batch_size': 32,
        'max_epochs': 200,
        'early_stopping_patience': 20,
        'learning_rate_decay': 0.1,
        'parallel_jobs': -1  # Use all available cores
    }
    
    # Prediction performance
    PREDICTION_PERFORMANCE = {
        'cache_predictions': True,
        'batch_predictions': True,
        'max_batch_size': 1000,
        'prediction_timeout': 30  # seconds
    }
    
    # Memory management
    MEMORY_MANAGEMENT = {
        'max_memory_usage': 0.8,  # 80% of available memory
        'garbage_collection_threshold': 0.9,
        'chunk_size_large_datasets': 10000
    }

class MonitoringConfig:
    """Monitoring and alerting configuration"""
    
    # Performance metrics
    PERFORMANCE_METRICS = {
        'response_time_threshold': 5.0,  # seconds
        'error_rate_threshold': 0.05,  # 5%
        'memory_usage_threshold': 0.9,  # 90%
        'cpu_usage_threshold': 0.8,  # 80%
    }
    
    # Model performance monitoring
    MODEL_MONITORING = {
        'accuracy_threshold': 0.7,
        'drift_detection_threshold': 0.15,
        'retraining_trigger_accuracy': 0.6,
        'monitoring_window_days': 30
    }
    
    # Alerting configuration
    ALERTING = {
        'email_alerts': False,  # Set to True and configure SMTP for email alerts
        'log_alerts': True,
        'alert_cooldown': 300,  # 5 minutes between similar alerts
        'critical_error_immediate_alert': True
    }

class ExperimentConfig:
    """A/B testing and experimentation configuration"""
    
    # A/B testing
    AB_TESTING = {
        'enabled': False,
        'default_treatment': 'control',
        'traffic_split': {'control': 0.5, 'treatment': 0.5},
        'minimum_sample_size': 1000,
        'statistical_significance': 0.05
    }
    
    # Feature flags
    FEATURE_FLAGS = {
        'advanced_rl_algorithms': False,
        'real_time_learning': False,
        'blockchain_integration': False,
        'advanced_game_theory': True,
        'automated_retraining': True
    }

# Environment-specific overrides
ENVIRONMENT_OVERRIDES = {
    'development': {
        'MODEL_CONFIG': {
            'proposal_predictor': {
                'cross_validation_folds': 3,
                'hyperparameter_tuning': False
            }
        },
        'FEATURE_FLAGS': {
            'real_time_learning': True  # Enable for development testing
        }
    },
    'production': {
        'PERFORMANCE_METRICS': {
            'response_time_threshold': 2.0,  # Stricter in production
            'error_rate_threshold': 0.01   # 1% in production
        },
        'ALERTING': {
            'email_alerts': True,
            'critical_error_immediate_alert': True
        }
    },
    'testing': {
        'MODEL_CONFIG': {
            'proposal_predictor': {
                'cross_validation_folds': 2
            }
        },
        'QUALITY_THRESHOLDS': {
            'min_records_training': 10,  # Lower threshold for testing
            'min_records_validation': 5
        }
    }
}

def get_environment_config() -> Dict[str, Any]:
    """Get environment-specific configuration overrides"""
    env = os.environ.get('FLASK_ENV', 'development')
    return ENVIRONMENT_OVERRIDES.get(env, {})

def apply_environment_overrides(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment-specific overrides to base configuration"""
    overrides = get_environment_config()
    
    def deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
        """Recursively update nested dictionaries"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                base_dict[key] = deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict
    
    return deep_update(base_config.copy(), overrides)

# Configuration validation
def validate_config(config_obj: Config) -> bool:
    """Validate configuration object"""
    try:
        # Check required attributes
        required_attrs = ['SECRET_KEY', 'SQLALCHEMY_DATABASE_URI', 'MODEL_CONFIG']
        for attr in required_attrs:
            if not hasattr(config_obj, attr):
                raise ValueError(f"Missing required configuration: {attr}")
        
        # Validate model configuration
        model_config = config_obj.MODEL_CONFIG
        required_models = ['proposal_predictor', 'competitive_analyzer', 'reinforcement_agent']
        for model in required_models:
            if model not in model_config:
                raise ValueError(f"Missing model configuration: {model}")
        
        # Validate database URI
        if not config_obj.SQLALCHEMY_DATABASE_URI:
            raise ValueError("Database URI cannot be empty")
        
        return True
        
    except Exception as e:
        print(f"Configuration validation error: {e}")
        return False

# Initialize configuration
def initialize_config():
    """Initialize configuration and create necessary directories"""
    try:
        # Ensure directories exist
        ModelPaths.ensure_directories()
        DataPaths.ensure_directories()
        
        # Create uploads directory with proper permissions
        uploads_dir = DataPaths.UPLOADS_DIR
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir, mode=0o755)
        
        # Create logs directory
        logs_dir = os.path.join(DataPaths.DATA_DIR, '..', 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        print("Configuration initialized successfully")
        return True
        
    except Exception as e:
        print(f"Configuration initialization error: {e}")
        return False

# Export commonly used configurations
__all__ = [
    'Config', 'DevelopmentConfig', 'TestingConfig', 'ProductionConfig',
    'get_config', 'ModelPaths', 'DataPaths', 'FeatureConfig',
    'ValidationConfig', 'SecurityConfig', 'PerformanceConfig',
    'MonitoringConfig', 'ExperimentConfig', 'initialize_config'
]