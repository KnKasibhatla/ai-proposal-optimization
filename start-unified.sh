#!/bin/bash

# AI Proposal Optimization Platform - Unified Startup Script
# This script handles both FastAPI and Flask backends with proper error handling

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}==================================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}==================================================${NC}"
}

# Function to cleanup background processes on exit
cleanup() {
    print_status "Shutting down services..."
    
    # Kill background processes
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
        print_status "Frontend stopped"
    fi
    
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
        print_status "Backend stopped"
    fi
    
    if [ ! -z "$FASTAPI_PID" ]; then
        kill $FASTAPI_PID 2>/dev/null || true
        print_status "FastAPI stopped"
    fi
    
    print_success "All services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

print_header "AI Proposal Optimization Platform - Unified Startup"
echo ""

# Check prerequisites
print_status "Checking prerequisites..."

# Check Node.js
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js 16.0 or higher."
    echo "   Visit: https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 16 ]; then
    print_error "Node.js version 16.0 or higher is required. Current version: $(node -v)"
    echo "   Please upgrade Node.js from https://nodejs.org/"
    exit 1
fi
print_success "Node.js version: $(node -v)"

# Check npm
if ! command -v npm &> /dev/null; then
    print_error "npm is not installed. Please install npm."
    exit 1
fi
print_success "npm version: $(npm -v)"

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi
print_success "Python version: $(python3 --version)"

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    print_error "package.json not found. Please run this script from the project root directory."
    exit 1
fi

echo ""

# Setup Python Environment
print_header "Setting up Python Environment"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
print_status "Installing Python dependencies..."

# Install basic requirements first
if [ -f "requirements_api.txt" ]; then
    print_status "Installing API requirements..."
    pip install -r requirements_api.txt
fi

# Install backend requirements
if [ -f "src/backend/requirements.txt" ]; then
    print_status "Installing backend requirements..."
    pip install -r src/backend/requirements.txt
fi

# Install enhanced requirements if available
if [ -f "src/backend/requirements_enhanced.txt" ]; then
    print_status "Installing enhanced requirements..."
    pip install -r src/backend/requirements_enhanced.txt
fi

# Handle typing-extensions conflict
print_status "Resolving dependency conflicts..."
pip install --upgrade typing-extensions==4.5.0 2>/dev/null || true

print_success "Python dependencies installed"

# Create necessary directories
mkdir -p data/uploads data/models logs src/backend/data/uploads src/backend/data/models src/backend/logs

echo ""

# Setup Frontend
print_header "Setting up Frontend"

# Install Node.js dependencies if needed
if [ ! -d "node_modules" ]; then
    print_status "Installing Node.js dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        print_error "Failed to install Node.js dependencies"
        exit 1
    fi
    print_success "Node.js dependencies installed"
else
    print_status "Node.js dependencies already installed"
fi

echo ""

# Start Services
print_header "Starting Services"

# Choose backend type based on available files
BACKEND_TYPE="flask"
if [ -f "api_main.py" ] && [ -f "requirements_api.txt" ]; then
    print_status "FastAPI backend detected (api_main.py)"
    BACKEND_TYPE="fastapi"
else
    print_status "Flask backend detected (src/backend/app.py)"
    BACKEND_TYPE="flask"
fi

# Start Backend
if [ "$BACKEND_TYPE" = "fastapi" ]; then
    print_status "Starting FastAPI backend on port 8000..."
    python -m uvicorn api_main:app --host 0.0.0.0 --port 8000 --reload &
    FASTAPI_PID=$!
    print_success "FastAPI backend started (PID: $FASTAPI_PID)"
    
    # Also start Flask backend for compatibility
    print_status "Starting Flask backend on port 5000..."
    cd src/backend
    export FLASK_APP=app.py
    export PYTHONPATH=$PWD
    python app.py &
    BACKEND_PID=$!
    cd ../..
    print_success "Flask backend started (PID: $BACKEND_PID)"
    
else
    print_status "Starting Flask backend on port 5000..."
    cd src/backend
    export FLASK_APP=app.py
    export PYTHONPATH=$PWD
    python app.py &
    BACKEND_PID=$!
    cd ../..
    print_success "Flask backend started (PID: $BACKEND_PID)"
    
    # Start enhanced API if available
    if [ -f "src/backend/enhanced_api.py" ]; then
        print_status "Starting Enhanced API on port 8000..."
        cd src/backend
        python enhanced_api.py &
        FASTAPI_PID=$!
        cd ../..
        print_success "Enhanced API started (PID: $FASTAPI_PID)"
    fi
fi

# Start Frontend
print_status "Starting React frontend on port 3000..."
NODE_OPTIONS="--max-old-space-size=8192" npm start &
FRONTEND_PID=$!
print_success "Frontend started (PID: $FRONTEND_PID)"

# Wait a moment for services to start
sleep 5

echo ""

# Display final status
print_header "Platform Startup Complete"
echo ""
print_success "🎉 AI Proposal Optimization Platform is now running!"
echo ""
echo -e "${CYAN}📱 Frontend (React):${NC}     http://localhost:3000"
if [ "$BACKEND_TYPE" = "fastapi" ]; then
    echo -e "${CYAN}🚀 FastAPI Backend:${NC}     http://localhost:8000"
    echo -e "${CYAN}🔧 Flask Backend:${NC}       http://localhost:5000"
else
    echo -e "${CYAN}🔧 Flask Backend:${NC}       http://localhost:5000"
    if [ ! -z "$FASTAPI_PID" ]; then
        echo -e "${CYAN}🧠 Enhanced API:${NC}     http://localhost:8000"
    fi
fi
echo ""
echo -e "${YELLOW}📊 Available Features:${NC}"
echo "   • Data Upload & Validation"
echo "   • AI-Powered Bidding Engine"
echo "   • Data Analysis Dashboard"
echo "   • Model Validation & Optimization"
echo "   • Competitive Analysis"
echo ""
if [ "$BACKEND_TYPE" = "fastapi" ]; then
    echo -e "${YELLOW}🔧 API Endpoints:${NC}"
    echo "   • FastAPI: http://localhost:8000/docs"
    echo "   • Flask: http://localhost:5000/api/"
else
    echo -e "${YELLOW}🔧 API Endpoints:${NC}"
    echo "   • Flask: http://localhost:5000/api/"
    if [ ! -z "$FASTAPI_PID" ]; then
        echo "   • Enhanced API: http://localhost:8000/api/v2/"
    fi
fi
echo ""
echo -e "${YELLOW}📁 File Upload:${NC}        http://localhost:5000/upload"
echo -e "${YELLOW}🎯 Smart Predict:${NC}       http://localhost:5000/predict"
echo ""
echo -e "${GREEN}✅ All services are running successfully!${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Keep the script running and monitor processes
while true; do
    # Check if processes are still running
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        print_error "Frontend process has stopped"
        cleanup
    fi
    
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        print_error "Backend process has stopped"
        cleanup
    fi
    
    if [ ! -z "$FASTAPI_PID" ] && ! kill -0 $FASTAPI_PID 2>/dev/null; then
        print_error "FastAPI/Enhanced API process has stopped"
        cleanup
    fi
    
    sleep 10
done 