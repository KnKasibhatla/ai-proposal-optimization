# schemas.py
from pydantic import BaseModel, EmailStr, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

# Enums
class ProjectStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    AWARDED = "awarded"

class UserRole(str, Enum):
    CLIENT = "client"
    PROVIDER = "provider"
    ADMIN = "admin"

class ProjectCategory(str, Enum):
    SOFTWARE_DEVELOPMENT = "software_development"
    DESIGN = "design"
    MARKETING = "marketing"
    CONSULTING = "consulting"
    MANUFACTURING = "manufacturing"
    LOGISTICS = "logistics"
    OTHER = "other"

# Base Schemas
class BaseSchema(BaseModel):
    class Config:
        orm_mode = True
        validate_assignment = True

# User Schemas
class UserBase(BaseSchema):
    email: EmailStr
    name: str
    role: UserRole

class UserCreate(UserBase):
    password: str
    company_name: Optional[str] = None
    industry: Optional[str] = None

class UserResponse(UserBase):
    id: str
    created_at: datetime
    is_active: bool = True

class UserLogin(BaseSchema):
    email: EmailStr
    password: str

class Token(BaseSchema):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse

# Client Schemas
class ClientBase(BaseSchema):
    name: str
    email: EmailStr
    industry: Optional[str] = None
    company_size: Optional[str] = None

class ClientCreate(ClientBase):
    pass

class ClientResponse(ClientBase):
    id: str
    created_at: datetime

# Provider Schemas
class ProviderBase(BaseSchema):
    name: str
    email: EmailStr
    industry_expertise: Optional[List[str]] = []
    reputation_score: Optional[float] = 0.0
    historical_success_rate: Optional[float] = 0.0
    average_delivery_time: Optional[int] = 30
    quality_score: Optional[float] = 5.0

class ProviderCreate(ProviderBase):
    pass

class ProviderResponse(ProviderBase):
    id: str
    created_at: datetime

class ProviderUpdate(BaseSchema):
    name: Optional[str] = None
    industry_expertise: Optional[List[str]] = None
    quality_score: Optional[float] = None

# Project Schemas
class ProjectBase(BaseSchema):
    title: str
    description: Optional[str] = None
    category: ProjectCategory
    complexity: int
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    deadline: datetime

    @validator('complexity')
    def validate_complexity(cls, v):
        if not 1 <= v <= 10:
            raise ValueError('Complexity must be between 1 and 10')
        return v

    @validator('budget_max')
    def validate_budget(cls, v, values):
        if v is not None and 'budget_min' in values and values['budget_min'] is not None:
            if v <= values['budget_min']:
                raise ValueError('Budget max must be greater than budget min')
        return v

class ProjectCreate(ProjectBase):
    pass

class ProjectUpdate(BaseSchema):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[ProjectStatus] = None
    deadline: Optional[datetime] = None

class ProjectResponse(ProjectBase):
    id: str
    client_id: str
    status: ProjectStatus
    created_at: datetime

# Bid Schemas
class BidBase(BaseSchema):
    project_id: str
    price: float
    delivery_time: int
    quality_score: int
    proposal_text: Optional[str] = None

    @validator('price')
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return v

    @validator('delivery_time')
    def validate_delivery_time(cls, v):
        if v <= 0:
            raise ValueError('Delivery time must be positive')
        return v

    @validator('quality_score')
    def validate_quality_score(cls, v):
        if not 1 <= v <= 10:
            raise ValueError('Quality score must be between 1 and 10')
        return v

class BidCreate(BidBase):
    pass

class BidUpdate(BaseSchema):
    price: Optional[float] = None
    delivery_time: Optional[int] = None
    quality_score: Optional[int] = None
    proposal_text: Optional[str] = None

class BidResponse(BidBase):
    id: str
    provider_id: str
    is_winner: bool = False
    ai_confidence_score: Optional[float] = None
    predicted_win_probability: Optional[float] = None
    submitted_at: datetime

# Project with Bids
class ProjectDetailResponse(ProjectResponse):
    bids: List[BidResponse] = []

# Optimization Schemas
class OptimizationRequest(BaseSchema):
    project_id: str
    max_price: Optional[float] = None
    min_delivery_time: Optional[int] = None
    quality_preference: Optional[float] = 1.0

class PriceScenario(BaseSchema):
    price: float
    price_multiplier: float
    win_probability: float
    expected_profit: float
    competitiveness: str

class OptimizationResponse(BaseSchema):
    status: str
    competition_analysis: Dict[str, Any]
    optimal_strategy: Dict[str, Any]
    price_scenarios: List[PriceScenario]
    delivery_recommendations: Dict[str, int]
    strategic_insights: Dict[str, Any]

# Analytics Schemas
class DashboardAnalytics(BaseSchema):
    user_role: str
    total_projects: Optional[int] = None
    total_bids: Optional[int] = None
    win_rate: Optional[float] = None
    avg_bid_price: Optional[float] = None
    project_categories: Optional[Dict[str, int]] = {}
    budget_distribution: Optional[Dict[str, int]] = {}
    monthly_trends: Optional[Dict[str, Any]] = {}

class MarketIntelligence(BaseSchema):
    sample_size: int
    time_period: str
    price_statistics: Dict[str, Any]
    delivery_statistics: Dict[str, Any]
    quality_statistics: Dict[str, Any]
    competitive_insights: Dict[str, Any]

# Model Management Schemas
class ModelPerformance(BaseSchema):
    id: str
    name: str
    version: str
    accuracy: Optional[float] = None
    is_active: bool
    created_at: datetime

class TrainingRequest(BaseSchema):
    force_retrain: bool = False
    model_types: Optional[List[str]] = ["ensemble", "deep_nn"]

class TrainingResponse(BaseSchema):
    message: str
    status: str
    estimated_duration: Optional[str] = None

# Feature Importance Schema
class FeatureImportanceResponse(BaseSchema):
    feature_name: str
    importance_score: float
    feature_type: str

# Error Schemas
class ErrorResponse(BaseSchema):
    detail: str
    timestamp: datetime
    error_code: Optional[str] = None

# Pagination Schemas
class PaginationParams(BaseSchema):
    skip: int = 0
    limit: int = 100

    @validator('limit')
    def validate_limit(cls, v):
        if v > 1000:
            raise ValueError('Limit cannot exceed 1000')
        return v

class PaginatedResponse(BaseSchema):
    items: List[Any]
    total: int
    skip: int
    limit: int
    has_more: bool

# Auction Schemas
class AuctionSession(BaseSchema):
    project_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    num_participants: int = 0
    winning_bid_id: Optional[str] = None
    ai_predictions: Optional[Dict[str, Any]] = {}
    market_conditions: Optional[Dict[str, Any]] = {}

class AuctionResult(BaseSchema):
    project_id: str
    winner_id: Optional[str] = None
    winning_price: Optional[float] = None
    num_bidders: int
    competition_level: str
    market_efficiency: float

# Real-time Updates Schema
class BidUpdate(BaseSchema):
    bid_id: str
    project_id: str
    event_type: str  # "new_bid", "bid_update", "auction_end"
    data: Dict[str, Any]
    timestamp: datetime

# auth.py
import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Optional
import secrets
from models import Provider, Client
from config import settings

class AuthManager:
    """Authentication and authorization manager"""
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.ALGORITHM
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash password"""
        return self.pwd_context.hash(password)
    
    def create_access_token(self, data: Dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def get_current_user(self, token: str) -> Dict:
        """Get current user from token"""
        try:
            payload = self.verify_token(token)
            user_id: str = payload.get("sub")
            user_role: str = payload.get("role")
            
            if user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials"
                )
            
            # Return user info
            return {
                "id": user_id,
                "role": user_role,
                "email": payload.get("email"),
                "name": payload.get("name"),
                "is_admin": payload.get("is_admin", False)
            }
            
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
    
    def authenticate_user(self, db: Session, email: str, password: str) -> Optional[Dict]:
        """Authenticate user with email and password"""
        # Try to find user in providers table
        provider = db.query(Provider).filter(Provider.email == email).first()
        if provider and hasattr(provider, 'password_hash'):
            if self.verify_password(password, provider.password_hash):
                return {
                    "id": provider.id,
                    "email": provider.email,
                    "name": provider.name,
                    "role": "provider"
                }
        
        # Try to find user in clients table
        client = db.query(Client).filter(Client.email == email).first()
        if client and hasattr(client, 'password_hash'):
            if self.verify_password(password, client.password_hash):
                return {
                    "id": client.id,
                    "email": client.email,
                    "name": client.name,
                    "role": "client"
                }
        
        return None
    
    def create_user(self, db: Session, user_data: Dict) -> Dict:
        """Create a new user account"""
        try:
            # Hash password
            hashed_password = self.get_password_hash(user_data["password"])
            
            if user_data["role"] == "provider":
                # Create provider
                db_user = Provider(
                    name=user_data["name"],
                    email=user_data["email"],
                    password_hash=hashed_password,
                    industry_expertise=user_data.get("industry_expertise", []),
                    reputation_score=0.0,
                    historical_success_rate=0.0,
                    quality_score=5.0
                )
            else:  # client
                # Create client
                db_user = Client(
                    name=user_data["name"],
                    email=user_data["email"],
                    password_hash=hashed_password,
                    industry=user_data.get("industry"),
                    company_size=user_data.get("company_size")
                )
            
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
            
            return {
                "id": db_user.id,
                "email": db_user.email,
                "name": db_user.name,
                "role": user_data["role"]
            }
            
        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to create user: {str(e)}"
            )
    
    def generate_api_key(self) -> str:
        """Generate API key for programmatic access"""
        return secrets.token_urlsafe(32)
    
    def validate_api_key(self, api_key: str, db: Session) -> Optional[Dict]:
        """Validate API key and return user info"""
        # This would check against an API keys table
        # For now, return None (not implemented)
        return None

# middleware.py
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import time
import logging
from typing import Dict, Optional
import redis
import json

logger = logging.getLogger(__name__)

class RateLimitMiddleware:
    """Rate limiting middleware"""
    
    def __init__(self, redis_client, requests_per_minute: int = 60):
        self.redis_client = redis_client
        self.requests_per_minute = requests_per_minute
    
    async def __call__(self, request: Request, call_next):
        """Rate limiting logic"""
        client_ip = request.client.host
        key = f"rate_limit:{client_ip}"
        
        try:
            # Get current request count
            current_requests = self.redis_client.get(key)
            
            if current_requests is None:
                # First request from this IP
                self.redis_client.setex(key, 60, 1)  # Expire in 60 seconds
            else:
                current_requests = int(current_requests)
                if current_requests >= self.requests_per_minute:
                    raise HTTPException(
                        status_code=429,
                        detail="Rate limit exceeded. Please try again later."
                    )
                
                # Increment request count
                self.redis_client.incr(key)
            
            response = await call_next(request)
            return response
            
        except redis.RedisError:
            # If Redis is down, allow the request
            logger.warning("Redis unavailable for rate limiting")
            response = await call_next(request)
            return response

class LoggingMiddleware:
    """Request/response logging middleware"""
    
    async def __call__(self, request: Request, call_next):
        """Log requests and responses"""
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")
        
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
        
        return response

# websocket.py
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List
import json
import logging

logger = logging.getLogger(__name__)

class ConnectionManager:
    """WebSocket connection manager for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.user_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str, room: str = "general"):
        """Accept WebSocket connection"""
        await websocket.accept()
        
        if room not in self.active_connections:
            self.active_connections[room] = []
        
        self.active_connections[room].append(websocket)
        self.user_connections[user_id] = websocket
        
        logger.info(f"User {user_id} connected to room {room}")
    
    def disconnect(self, websocket: WebSocket, user_id: str, room: str = "general"):
        """Remove WebSocket connection"""
        if room in self.active_connections:
            self.active_connections[room].remove(websocket)
        
        if user_id in self.user_connections:
            del self.user_connections[user_id]
        
        logger.info(f"User {user_id} disconnected from room {room}")
    
    async def send_personal_message(self, message: Dict, user_id: str):
        """Send message to specific user"""
        if user_id in self.user_connections:
            websocket = self.user_connections[user_id]
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to user {user_id}: {e}")
    
    async def broadcast_to_room(self, message: Dict, room: str = "general"):
        """Broadcast message to all connections in a room"""
        if room in self.active_connections:
            disconnected = []
            
            for connection in self.active_connections[room]:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error broadcasting to room {room}: {e}")
                    disconnected.append(connection)
            
            # Remove disconnected connections
            for connection in disconnected:
                self.active_connections[room].remove(connection)
    
    async def notify_new_bid(self, bid_data: Dict, project_id: str):
        """Notify about new bid submission"""
        message = {
            "type": "new_bid",
            "project_id": project_id,
            "data": bid_data,
            "timestamp": bid_data.get("submitted_at")
        }
        
        await self.broadcast_to_room(message, f"project_{project_id}")
    
    async def notify_auction_end(self, project_id: str, winner_data: Dict):
        """Notify about auction end"""
        message = {
            "type": "auction_end",
            "project_id": project_id,
            "winner": winner_data,
            "timestamp": winner_data.get("timestamp")
        }
        
        await self.broadcast_to_room(message, f"project_{project_id}")

# websocket_routes.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from websocket import ConnectionManager
from auth import AuthManager

router = APIRouter()
manager = ConnectionManager()
auth_manager = AuthManager()

@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, token: str = None):
    """WebSocket endpoint for real-time communication"""
    try:
        # Validate user authentication
        if token:
            user_info = auth_manager.get_current_user(token)
            if user_info["id"] != user_id:
                await websocket.close(code=1008)  # Policy violation
                return
        
        await manager.connect(websocket, user_id)
        
        # Send welcome message
        await manager.send_personal_message({
            "type": "connection_established",
            "message": "Connected to real-time updates",
            "user_id": user_id
        }, user_id)
        
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Handle different message types
            if message_data.get("type") == "join_room":
                room = message_data.get("room", "general")
                await manager.connect(websocket, user_id, room)
            
            elif message_data.get("type") == "ping":
                await manager.send_personal_message({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                }, user_id)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        logger.info(f"WebSocket disconnected for user {user_id}")
    
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        await websocket.close(code=1011)  # Internal error

# Additional auth routes for the main app
auth_router = APIRouter(prefix="/auth", tags=["Authentication"])

@auth_router.post("/register", response_model=Token)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    auth_manager = AuthManager()
    
    # Check if user already exists
    existing_provider = db.query(Provider).filter(Provider.email == user_data.email).first()
    existing_client = db.query(Client).filter(Client.email == user_data.email).first()
    
    if existing_provider or existing_client:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    # Create user
    user = auth_manager.create_user(db, user_data.dict())
    
    # Create access token
    access_token_expires = timedelta(minutes=auth_manager.access_token_expire_minutes)
    access_token = auth_manager.create_access_token(
        data={
            "sub": user["id"],
            "email": user["email"],
            "role": user["role"],
            "name": user["name"]
        },
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=auth_manager.access_token_expire_minutes * 60,
        user=UserResponse(**user, created_at=datetime.utcnow(), is_active=True)
    )

@auth_router.post("/login", response_model=Token)
async def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
    """Authenticate user and return token"""
    auth_manager = AuthManager()
    
    # Authenticate user
    user = auth_manager.authenticate_user(db, user_credentials.email, user_credentials.password)
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=auth_manager.access_token_expire_minutes)
    access_token = auth_manager.create_access_token(
        data={
            "sub": user["id"],
            "email": user["email"],
            "role": user["role"],
            "name": user["name"]
        },
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=auth_manager.access_token_expire_minutes * 60,
        user=UserResponse(**user, created_at=datetime.utcnow(), is_active=True)
    )

@auth_router.post("/refresh")
async def refresh_token(current_user: Dict = Depends(get_current_user)):
    """Refresh access token"""
    auth_manager = AuthManager()
    
    # Create new access token
    access_token_expires = timedelta(minutes=auth_manager.access_token_expire_minutes)
    access_token = auth_manager.create_access_token(
        data={
            "sub": current_user["id"],
            "email": current_user["email"],
            "role": current_user["role"],
            "name": current_user["name"]
        },
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=auth_manager.access_token_expire_minutes * 60,
        user=UserResponse(**current_user, created_at=datetime.utcnow(), is_active=True)
    )

@auth_router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: Dict = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(**current_user, created_at=datetime.utcnow(), is_active=True)

@auth_router.post("/logout")
async def logout(current_user: Dict = Depends(get_current_user)):
    """Logout user (client-side token removal)"""
    return {"message": "Successfully logged out"}

# Import these into main.py:
# app.include_router(auth_router, prefix=settings.API_V1_STR)
# app.include_router(router, prefix="/ws")