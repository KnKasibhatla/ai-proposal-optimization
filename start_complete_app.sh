#!/bin/bash

# AI Bidding Prediction Platform - Complete App Startup Script
# This script starts both the backend (Flask) and frontend (HTML) applications

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

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to wait for a service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=15
    local attempt=1
    
    print_status "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            print_success "$service_name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_warning "$service_name may not be fully ready yet, but continuing..."
    return 0
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
    
    print_success "All services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

print_header "AI Bidding Prediction Platform - Complete App Startup"
echo ""

# Check prerequisites
print_status "Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi
print_success "Python version: $(python3 --version)"

# Check if we're in the right directory
if [ ! -f "backend/app.py" ]; then
    print_error "backend/app.py not found. Please run this script from the project root directory."
    exit 1
fi

if [ ! -f "frontend/public/advanced-app.html" ]; then
    print_error "frontend/public/advanced-app.html not found. Please run this script from the project root directory."
    exit 1
fi

# Check if ports are available
print_status "Checking port availability..."

BACKEND_PORT=5000
FRONTEND_PORT=8080

if check_port $BACKEND_PORT; then
    print_warning "Port $BACKEND_PORT is already in use. Attempting to kill existing processes..."
    pkill -f "python.*app.py" 2>/dev/null || true
    sleep 2
fi

if check_port $FRONTEND_PORT; then
    print_warning "Port $FRONTEND_PORT is already in use. Attempting to kill existing processes..."
    pkill -f "python.*http.server.*$FRONTEND_PORT" 2>/dev/null || true
    sleep 2
fi

echo ""

# Start Backend Service
print_header "Starting Backend Service"

# Navigate to backend directory
cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade requirements
print_status "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    # Install basic requirements
    pip install flask flask-cors pandas numpy scikit-learn matplotlib plotly
fi

# Create necessary directories
mkdir -p data/uploads data/models logs

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development
export PYTHONPATH=$PWD

# Start the backend (Flask app)
print_status "Starting Flask backend server on port $BACKEND_PORT..."
python3 app.py &
BACKEND_PID=$!
print_success "Backend started (PID: $BACKEND_PID)"

# Wait a moment for backend to start
sleep 3

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    print_error "Backend failed to start"
    exit 1
fi

# Wait for backend to be ready
wait_for_service "http://localhost:$BACKEND_PORT" "Backend"

# Return to project root
cd ..

echo ""

# Start Frontend Service
print_header "Starting Frontend Service"

# Navigate to frontend directory
cd frontend/public

# Start the frontend using Python's built-in HTTP server
print_status "Starting HTML frontend server on port $FRONTEND_PORT..."
python3 -m http.server $FRONTEND_PORT &
FRONTEND_PID=$!
print_success "Frontend started (PID: $FRONTEND_PID)"

# Wait for frontend to be ready
sleep 2
wait_for_service "http://localhost:$FRONTEND_PORT" "Frontend"

# Return to project root
cd ../..

echo ""

# Display final status
print_header "Platform Startup Complete"
echo ""
print_success "🎉 AI Bidding Prediction Platform is now running!"
echo ""
echo -e "${CYAN}🔧 Backend (Flask API):${NC}    http://localhost:$BACKEND_PORT"
echo -e "${CYAN}📱 Frontend (HTML):${NC}        http://localhost:$FRONTEND_PORT"
echo ""
echo -e "${YELLOW}📊 Available Applications:${NC}"
echo -e "   • Main App:              http://localhost:$FRONTEND_PORT/advanced-app.html"
echo -e "   • Simple Test:           http://localhost:$FRONTEND_PORT/simple-test.html"
echo -e "   • Status Check:          http://localhost:$FRONTEND_PORT/status.html"
echo ""
echo -e "${YELLOW}🔧 API Endpoints:${NC}"
echo -e "   • Health Check:          http://localhost:$BACKEND_PORT/"
echo -e "   • Upload Data:           http://localhost:$BACKEND_PORT/upload"
echo -e "   • Make Prediction:       http://localhost:$BACKEND_PORT/api/predict"
echo -e "   • Smart Pricing:         http://localhost:$BACKEND_PORT/api/predict-pricing-fixed"
echo -e "   • Model Performance:     http://localhost:$BACKEND_PORT/api/model/performance"
echo ""
echo -e "${YELLOW}📊 Key Features:${NC}"
echo "   ✅ Data Upload & Validation"
echo "   ✅ AI-Powered Bidding Predictions"
echo "   ✅ Smart Pricing Recommendations"
echo "   ✅ Model Performance Analysis"
echo "   ✅ Competitive Analysis"
echo "   ✅ Historical Validation"
echo "   ✅ Dynamic Training Analysis"
echo ""
echo -e "${GREEN}✅ All services are running successfully!${NC}"
echo ""
echo -e "${YELLOW}🚀 Quick Start:${NC}"
echo "   1. Open: http://localhost:$FRONTEND_PORT/advanced-app.html"
echo "   2. Upload your historical bidding data"
echo "   3. Use the Smart Pricing tab to get AI recommendations"
echo "   4. Explore model validation and competitive analysis"
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
    
    sleep 10
done
