
# models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class Client(Base):
    __tablename__ = "clients"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    industry = Column(String)
    company_size = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    projects = relationship("Project", back_populates="client")

class Provider(Base):
    __tablename__ = "providers"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    industry_expertise = Column(JSON)  # List of industries
    reputation_score = Column(Float, default=0.0)
    historical_success_rate = Column(Float, default=0.0)
    average_delivery_time = Column(Integer)  # in days
    quality_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    bids = relationship("Bid", back_populates="provider")

class Project(Base):
    __tablename__ = "projects"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id = Column(String, ForeignKey("clients.id"), nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text)
    category = Column(String, nullable=False)
    complexity = Column(Integer, nullable=False)  # 1-10 scale
    budget_min = Column(Float)
    budget_max = Column(Float)
    deadline = Column(DateTime)
    status = Column(String, default="open")  # open, closed, awarded
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    client = relationship("Client", back_populates="projects")
    bids = relationship("Bid", back_populates="project")

class Bid(Base):
    __tablename__ = "bids"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    provider_id = Column(String, ForeignKey("providers.id"), nullable=False)
    price = Column(Float, nullable=False)
    delivery_time = Column(Integer, nullable=False)  # in days
    quality_score = Column(Integer, nullable=False)  # 1-10 scale
    proposal_text = Column(Text)
    is_winner = Column(Boolean, default=False)
    ai_confidence_score = Column(Float)  # AI prediction confidence
    predicted_win_probability = Column(Float)
    submitted_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="bids")
    provider = relationship("Provider", back_populates="bids")

class PredictionModel(Base):
    __tablename__ = "prediction_models"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    model_type = Column(String, nullable=False)  # supervised, reinforcement, hybrid
    accuracy_score = Column(Float)
    is_active = Column(Boolean, default=False)
    hyperparameters = Column(JSON)
    training_data_size = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
class FeatureImportance(Base):
    __tablename__ = "feature_importance"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id = Column(String, ForeignKey("prediction_models.id"))
    feature_name = Column(String, nullable=False)
    importance_score = Column(Float, nullable=False)
    feature_type = Column(String)  # price, quality, timing, etc.

class AuctionSession(Base):
    __tablename__ = "auction_sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"))
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    num_participants = Column(Integer, default=0)
    winning_bid_id = Column(String, ForeignKey("bids.id"))
    ai_predictions = Column(JSON)  # Store AI predictions for analysis
    market_conditions = Column(JSON)  # Competition level, etc.

# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://user:password@localhost/proposal_optimization"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)

# config.py
import os
from typing import List

class Settings:
    PROJECT_NAME: str = "AI Proposal Optimization Platform"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/proposal_optimization")
    
    # Redis for caching and Celery
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # ML Model settings
    MODEL_TRAINING_INTERVAL: int = 24  # hours
    MIN_TRAINING_DATA_SIZE: int = 100
    
    # API settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

settings = Settings()