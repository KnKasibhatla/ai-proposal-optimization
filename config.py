# config.py
import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost/proposal_db"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # API Keys
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Machine Learning
    MODEL_PATH: str = "./models"
    RETRAIN_INTERVAL_HOURS: int = 24
    
    # Feature Engineering
    LOOKBACK_DAYS: int = 90
    MIN_HISTORICAL_BIDS: int = 10
    
    # Reinforcement Learning
    RL_EPISODES: int = 1000
    RL_LEARNING_RATE: float = 0.001
    RL_DISCOUNT_FACTOR: float = 0.95
    
    # Performance
    BATCH_SIZE: int = 32
    MAX_WORKERS: int = 4
    CACHE_TTL: int = 3600
    
    class Config:
        env_file = ".env"

settings = Settings()