# main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os

# Import our modules
from models import *
from database import get_db, create_tables
from config import settings
from feature_engineering import FeatureEngineer
from ml_models import ModelManager, ModelTrainingPipeline
from reinforcement_learning import ReinforcementLearningTrainer, AdaptiveBiddingStrategy
from schemas import *  # Pydantic schemas (will create next)
from auth import AuthManager  # Authentication (will create next)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
feature_engineer = FeatureEngineer()
model_manager = ModelManager()
auth_manager = AuthManager()
security = HTTPBearer()

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database and load models"""
    create_tables()
    
    # Load existing models if available
    model_dir = "trained_models"
    if os.path.exists(model_dir):
        try:
            model_manager.load_models(model_dir)
            logger.info("Loaded existing models")
        except Exception as e:
            logger.warning(f"Could not load existing models: {e}")
    
    logger.info("Application startup complete")

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate JWT token and return current user"""
    return auth_manager.get_current_user(credentials.credentials)

# API Routes

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "AI Proposal Optimization Platform", "version": settings.VERSION}

@app.get(f"{settings.API_V1_STR}/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "models_loaded": len(model_manager.models),
        "active_model": model_manager.active_model
    }

# Project Management Endpoints

@app.post(f"{settings.API_V1_STR}/projects", response_model=ProjectResponse)
async def create_project(
    project: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Create a new project"""
    try:
        db_project = Project(
            client_id=current_user["id"],
            title=project.title,
            description=project.description,
            category=project.category,
            complexity=project.complexity,
            budget_min=project.budget_min,
            budget_max=project.budget_max,
            deadline=project.deadline
        )
        
        db.add(db_project)
        db.commit()
        db.refresh(db_project)
        
        logger.info(f"Created project {db_project.id} by user {current_user['id']}")
        return ProjectResponse.from_orm(db_project)
        
    except Exception as e:
        logger.error(f"Error creating project: {e}")
        raise HTTPException(status_code=500, detail="Failed to create project")

@app.get(f"{settings.API_V1_STR}/projects", response_model=List[ProjectResponse])
async def get_projects(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Get projects for current user"""
    query = db.query(Project).filter(Project.client_id == current_user["id"])
    
    if status:
        query = query.filter(Project.status == status)
    
    projects = query.offset(skip).limit(limit).all()
    return [ProjectResponse.from_orm(project) for project in projects]

@app.get(f"{settings.API_V1_STR}/projects/{{project_id}}", response_model=ProjectDetailResponse)
async def get_project(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Get project details with bids"""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.client_id == current_user["id"]
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Get bids for this project
    bids = db.query(Bid).filter(Bid.project_id == project_id).all()
    
    return ProjectDetailResponse(
        **project.__dict__,
        bids=[BidResponse.from_orm(bid) for bid in bids]
    )

# Bidding Endpoints

@app.post(f"{settings.API_V1_STR}/bids", response_model=BidResponse)
async def submit_bid(
    bid: BidCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Submit a bid for a project"""
    try:
        # Verify project exists and is open
        project = db.query(Project).filter(Project.id == bid.project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if project.status != "open":
            raise HTTPException(status_code=400, detail="Project is not accepting bids")
        
        # Create bid record
        db_bid = Bid(
            project_id=bid.project_id,
            provider_id=current_user["id"],
            price=bid.price,
            delivery_time=bid.delivery_time,
            quality_score=bid.quality_score,
            proposal_text=bid.proposal_text
        )
        
        db.add(db_bid)
        db.commit()
        db.refresh(db_bid)
        
        # Generate AI predictions in background
        background_tasks.add_task(generate_bid_predictions, db_bid.id)
        
        logger.info(f"Bid {db_bid.id} submitted for project {bid.project_id}")
        return BidResponse.from_orm(db_bid)
        
    except Exception as e:
        logger.error(f"Error submitting bid: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit bid")

@app.get(f"{settings.API_V1_STR}/bids/optimize")
async def get_bid_optimization(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Get AI-powered bid optimization recommendations"""
    try:
        # Get project details
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Get existing bids for competition analysis
        existing_bids = db.query(Bid).filter(Bid.project_id == project_id).all()
        
        # Get provider data
        provider = db.query(Provider).filter(Provider.id == current_user["id"]).first()
        if not provider:
            raise HTTPException(status_code=404, detail="Provider profile not found")
        
        # Generate optimization recommendations
        recommendations = await generate_optimization_recommendations(
            project, existing_bids, provider, db
        )
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating bid optimization: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate recommendations")

# Model Management Endpoints

@app.post(f"{settings.API_V1_STR}/models/train")
async def train_models(
    background_tasks: BackgroundTasks,
    force_retrain: bool = False,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Trigger model training"""
    if not current_user.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Check if models need retraining
    last_training = db.query(PredictionModel).filter(
        PredictionModel.is_active == True
    ).first()
    
    if last_training and not force_retrain:
        time_since_training = datetime.utcnow() - last_training.created_at
        if time_since_training.hours < settings.MODEL_TRAINING_INTERVAL:
            raise HTTPException(
                status_code=400, 
                detail=f"Models were trained {time_since_training.hours} hours ago"
            )
    
    # Start training in background
    background_tasks.add_task(train_models_background, db)
    
    return {"message": "Model training started", "status": "in_progress"}

@app.get(f"{settings.API_V1_STR}/models/performance")
async def get_model_performance(
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Get model performance metrics"""
    models = db.query(PredictionModel).order_by(PredictionModel.created_at.desc()).limit(10).all()
    
    performance_data = []
    for model in models:
        performance_data.append({
            "id": model.id,
            "name": model.name,
            "version": model.version,
            "accuracy": model.accuracy_score,
            "is_active": model.is_active,
            "created_at": model.created_at
        })
    
    return {
        "models": performance_data,
        "active_model": model_manager.active_model,
        "model_manager_performance": model_manager.model_performances
    }

@app.get(f"{settings.API_V1_STR}/analytics/dashboard")
async def get_dashboard_analytics(
    days: int = 30,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Get dashboard analytics data"""
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get user's projects and bids
        if current_user.get("role") == "client":
            projects = db.query(Project).filter(
                Project.client_id == current_user["id"],
                Project.created_at >= start_date
            ).all()
            
            total_projects = len(projects)
            
            # Get bids for user's projects
            project_ids = [p.id for p in projects]
            bids = db.query(Bid).filter(Bid.project_id.in_(project_ids)).all() if project_ids else []
            
            # Calculate metrics
            avg_bids_per_project = len(bids) / total_projects if total_projects > 0 else 0
            completed_projects = len([p for p in projects if p.status == "closed"])
            
            analytics = {
                "user_role": "client",
                "total_projects": total_projects,
                "completed_projects": completed_projects,
                "total_bids_received": len(bids),
                "avg_bids_per_project": round(avg_bids_per_project, 2),
                "project_categories": {},
                "budget_distribution": {}
            }
            
            # Category breakdown
            for project in projects:
                category = project.category
                analytics["project_categories"][category] = analytics["project_categories"].get(category, 0) + 1
            
            # Budget distribution
            for project in projects:
                budget_range = "< $50k" if project.budget_max < 50000 else \
                              "$50k - $100k" if project.budget_max < 100000 else \
                              "$100k - $500k" if project.budget_max < 500000 else "> $500k"
                analytics["budget_distribution"][budget_range] = analytics["budget_distribution"].get(budget_range, 0) + 1
        
        else:  # Provider
            bids = db.query(Bid).filter(
                Bid.provider_id == current_user["id"],
                Bid.submitted_at >= start_date
            ).all()
            
            total_bids = len(bids)
            won_bids = len([b for b in bids if b.is_winner])
            win_rate = (won_bids / total_bids * 100) if total_bids > 0 else 0
            
            # Calculate average metrics
            avg_bid_price = np.mean([b.price for b in bids]) if bids else 0
            avg_delivery_time = np.mean([b.delivery_time for b in bids]) if bids else 0
            
            analytics = {
                "user_role": "provider",
                "total_bids": total_bids,
                "won_bids": won_bids,
                "win_rate": round(win_rate, 2),
                "avg_bid_price": round(avg_bid_price, 2),
                "avg_delivery_time": round(avg_delivery_time, 1),
                "category_performance": {},
                "monthly_trends": {}
            }
            
            # Category performance
            for bid in bids:
                # Would need to join with Project to get category
                # Simplified for now
                category = "General"
                if category not in analytics["category_performance"]:
                    analytics["category_performance"][category] = {"total": 0, "won": 0}
                analytics["category_performance"][category]["total"] += 1
                if bid.is_winner:
                    analytics["category_performance"][category]["won"] += 1
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error generating dashboard analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate analytics")

@app.get(f"{settings.API_V1_STR}/analytics/market-intelligence")
async def get_market_intelligence(
    category: Optional[str] = None,
    complexity_min: Optional[int] = None,
    complexity_max: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Get market intelligence and competitive analysis"""
    try:
        # Build query
        query = db.query(Bid).join(Project)
        
        if category:
            query = query.filter(Project.category == category)
        
        if complexity_min:
            query = query.filter(Project.complexity >= complexity_min)
        
        if complexity_max:
            query = query.filter(Project.complexity <= complexity_max)
        
        # Get recent bids (last 90 days)
        ninety_days_ago = datetime.utcnow() - timedelta(days=90)
        recent_bids = query.filter(Bid.submitted_at >= ninety_days_ago).all()
        
        if not recent_bids:
            return {"message": "No recent data available for the specified criteria"}
        
        # Calculate market statistics
        prices = [bid.price for bid in recent_bids]
        delivery_times = [bid.delivery_time for bid in recent_bids]
        quality_scores = [bid.quality_score for bid in recent_bids]
        
        # Winning vs losing bid analysis
        winning_prices = [bid.price for bid in recent_bids if bid.is_winner]
        losing_prices = [bid.price for bid in recent_bids if not bid.is_winner]
        
        market_intelligence = {
            "sample_size": len(recent_bids),
            "time_period": "Last 90 days",
            "price_statistics": {
                "mean": round(np.mean(prices), 2),
                "median": round(np.median(prices), 2),
                "std": round(np.std(prices), 2),
                "min": round(np.min(prices), 2),
                "max": round(np.max(prices), 2),
                "percentiles": {
                    "25th": round(np.percentile(prices, 25), 2),
                    "75th": round(np.percentile(prices, 75), 2),
                    "90th": round(np.percentile(prices, 90), 2)
                }
            },
            "delivery_statistics": {
                "mean_days": round(np.mean(delivery_times), 1),
                "median_days": round(np.median(delivery_times), 1),
                "fastest": int(np.min(delivery_times)),
                "slowest": int(np.max(delivery_times))
            },
            "quality_statistics": {
                "mean_score": round(np.mean(quality_scores), 2),
                "median_score": round(np.median(quality_scores), 2)
            },
            "competitive_insights": {
                "avg_winning_price": round(np.mean(winning_prices), 2) if winning_prices else 0,
                "avg_losing_price": round(np.mean(losing_prices), 2) if losing_prices else 0,
                "winning_price_advantage": round(
                    (np.mean(losing_prices) - np.mean(winning_prices)) / np.mean(losing_prices) * 100, 2
                ) if winning_prices and losing_prices else 0
            }
        }
        
        return market_intelligence
        
    except Exception as e:
        logger.error(f"Error generating market intelligence: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate market intelligence")

# Background Tasks

async def generate_bid_predictions(bid_id: str):
    """Generate AI predictions for a bid"""
    try:
        db = next(get_db())
        
        # Get bid and related data
        bid = db.query(Bid).filter(Bid.id == bid_id).first()
        if not bid:
            return
        
        project = db.query(Project).filter(Project.id == bid.project_id).first()
        provider = db.query(Provider).filter(Provider.id == bid.provider_id).first()
        
        # Get competing bids
        competing_bids = db.query(Bid).filter(
            Bid.project_id == bid.project_id,
            Bid.id != bid_id
        ).all()
        
        # Generate predictions if we have an active model
        if model_manager.active_model:
            # Create feature vector
            historical_data = pd.read_sql(
                "SELECT * FROM bids WHERE submitted_at < %s",
                db.bind,
                params=[bid.submitted_at]
            )
            
            feature_vector, _ = feature_engineer.create_feature_vector(
                bid_data=bid.__dict__,
                project_data=project.__dict__,
                provider_data=provider.__dict__,
                competition_data=[b.__dict__ for b in competing_bids],
                historical_data=historical_data
            )
            
            # Get prediction
            win_probability = model_manager.predict_proba(feature_vector.reshape(1, -1))[0][1]
            
            # Update bid with prediction
            bid.predicted_win_probability = float(win_probability)
            bid.ai_confidence_score = 0.8  # Placeholder confidence
            
            db.commit()
            
            logger.info(f"Generated prediction for bid {bid_id}: {win_probability:.3f}")
        
    except Exception as e:
        logger.error(f"Error generating predictions for bid {bid_id}: {e}")
    finally:
        db.close()

async def generate_optimization_recommendations(
    project: Project, 
    existing_bids: List[Bid], 
    provider: Provider, 
    db: Session
) -> Dict[str, Any]:
    """Generate AI-powered bid optimization recommendations"""
    try:
        # Get historical data for feature engineering
        historical_data = pd.read_sql(
            "SELECT * FROM bids b JOIN projects p ON b.project_id = p.id WHERE p.category = %s",
            db.bind,
            params=[project.category]
        )
        
        if len(historical_data) < 10:
            # Not enough data for reliable predictions
            return {
                "status": "insufficient_data",
                "message": "Not enough historical data for reliable recommendations",
                "basic_recommendations": {
                    "suggested_price_range": {
                        "min": project.budget_min * 0.8 if project.budget_min else project.budget_max * 0.6,
                        "max": project.budget_max * 0.9 if project.budget_max else 100000
                    },
                    "suggested_delivery": "30-45 days",
                    "quality_focus": "Emphasize past experience and quality metrics"
                }
            }
        
        # Analyze competition
        competition_analysis = {
            "num_competitors": len(existing_bids),
            "price_range": {
                "min": min([b.price for b in existing_bids]) if existing_bids else 0,
                "max": max([b.price for b in existing_bids]) if existing_bids else 0,
                "avg": np.mean([b.price for b in existing_bids]) if existing_bids else 0
            },
            "delivery_range": {
                "fastest": min([b.delivery_time for b in existing_bids]) if existing_bids else 0,
                "slowest": max([b.delivery_time for b in existing_bids]) if existing_bids else 0,
                "avg": np.mean([b.delivery_time for b in existing_bids]) if existing_bids else 0
            }
        }
        
        # Generate price optimization scenarios
        price_scenarios = []
        base_price = project.budget_max * 0.7 if project.budget_max else 50000
        
        for multiplier in [0.8, 0.9, 1.0, 1.1, 1.2]:
            scenario_price = base_price * multiplier
            
            # Create feature vector for this scenario
            scenario_bid = {
                'price': scenario_price,
                'delivery_time': 30,  # Default
                'quality_score': provider.quality_score or 7
            }
            
            try:
                feature_vector, _ = feature_engineer.create_feature_vector(
                    bid_data=scenario_bid,
                    project_data=project.__dict__,
                    provider_data=provider.__dict__,
                    competition_data=[b.__dict__ for b in existing_bids],
                    historical_data=historical_data
                )
                
                # Get win probability if model is available
                if model_manager.active_model:
                    win_prob = model_manager.predict_proba(feature_vector.reshape(1, -1))[0][1]
                else:
                    # Simple heuristic if no model
                    avg_competitor_price = competition_analysis["price_range"]["avg"]
                    if avg_competitor_price > 0:
                        win_prob = max(0.1, min(0.9, 1 - (scenario_price - avg_competitor_price) / avg_competitor_price))
                    else:
                        win_prob = 0.5
                
                # Calculate expected value
                profit_margin = 0.3  # Assume 30% profit margin
                expected_profit = scenario_price * profit_margin * win_prob
                
                price_scenarios.append({
                    "price": round(scenario_price, 2),
                    "price_multiplier": multiplier,
                    "win_probability": round(win_prob, 3),
                    "expected_profit": round(expected_profit, 2),
                    "competitiveness": "High" if win_prob > 0.7 else "Medium" if win_prob > 0.4 else "Low"
                })
                
            except Exception as e:
                logger.warning(f"Error in price scenario calculation: {e}")
                continue
        
        # Find optimal price scenario
        optimal_scenario = max(price_scenarios, key=lambda x: x["expected_profit"]) if price_scenarios else None
        
        # Generate delivery time recommendations
        delivery_recommendations = {
            "conservative": 45,  # Safe delivery time
            "competitive": int(competition_analysis["delivery_range"]["avg"]) - 5 if competition_analysis["delivery_range"]["avg"] > 0 else 30,
            "aggressive": max(15, int(competition_analysis["delivery_range"]["fastest"]) - 3) if competition_analysis["delivery_range"]["fastest"] > 0 else 20
        }
        
        # Generate final recommendations
        recommendations = {
            "status": "success",
            "competition_analysis": competition_analysis,
            "optimal_strategy": {
                "recommended_price": optimal_scenario["price"] if optimal_scenario else base_price,
                "win_probability": optimal_scenario["win_probability"] if optimal_scenario else 0.5,
                "expected_profit": optimal_scenario["expected_profit"] if optimal_scenario else 0,
                "reasoning": "Based on competitive analysis and AI predictions"
            },
            "price_scenarios": price_scenarios,
            "delivery_recommendations": delivery_recommendations,
            "strategic_insights": {
                "market_position": "competitive" if len(existing_bids) > 3 else "favorable",
                "key_differentiators": [
                    "Competitive pricing",
                    "Quality track record", 
                    "Reliable delivery"
                ],
                "risk_factors": [
                    "High competition" if len(existing_bids) > 5 else "Moderate competition",
                    "Price pressure" if competition_analysis["price_range"]["avg"] < project.budget_max * 0.8 else "Normal pricing"
                ]
            }
        }
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating optimization recommendations: {e}")
        raise

async def train_models_background(db: Session):
    """Background task to train ML models"""
    try:
        logger.info("Starting background model training...")
        
        # Get training data
        training_data = pd.read_sql(
            """
            SELECT b.*, p.category, p.complexity, p.budget_min, p.budget_max,
                   pr.reputation_score, pr.historical_success_rate, pr.quality_score
            FROM bids b 
            JOIN projects p ON b.project_id = p.id 
            JOIN providers pr ON b.provider_id = pr.id
            WHERE b.submitted_at >= NOW() - INTERVAL '1 year'
            """,
            db.bind
        )
        
        if len(training_data) < settings.MIN_TRAINING_DATA_SIZE:
            logger.warning(f"Insufficient training data: {len(training_data)} samples")
            return
        
        # Initialize training pipeline
        training_pipeline = ModelTrainingPipeline(feature_engineer, model_manager)
        
        # Run training
        results = training_pipeline.run_training_pipeline(training_data)
        
        # Save models
        model_manager.save_models("trained_models")
        
        # Update database with training results
        for model_name, metrics in results.items():
            db_model = PredictionModel(
                name=model_name,
                version=f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                model_type="supervised" if "ensemble" in model_name else "deep_learning",
                accuracy_score=metrics.get("accuracy", 0),
                is_active=(model_name == model_manager.active_model),
                hyperparameters={"training_samples": len(training_data)},
                training_data_size=len(training_data)
            )
            
            db.add(db_model)
        
        db.commit()
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in background model training: {e}")
        db.rollback()
    finally:
        db.close()

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.utcnow().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.utcnow().isoformat()}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)