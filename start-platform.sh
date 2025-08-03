#!/bin/bash

# AI Proposal Optimization Platform - Full Stack Startup Script
# This script starts both the frontend (React) and backend (Flask) applications

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
    local max_attempts=30
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
    
    print_error "$service_name failed to start within $((max_attempts * 2)) seconds"
    return 1
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
    
    if [ ! -z "$ENHANCED_API_PID" ]; then
        kill $ENHANCED_API_PID 2>/dev/null || true
        print_status "Enhanced API stopped"
    fi
    
    print_success "All services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

print_header "AI Proposal Optimization Platform - Full Stack Startup"
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
if [ ! -f "package.json" ] || [ ! -f "src/backend/app.py" ]; then
    print_error "Please run this script from the project root directory."
    exit 1
fi

# Check if ports are available
print_status "Checking port availability..."

FRONTEND_PORT=3000
BACKEND_PORT=5000
ENHANCED_API_PORT=8000

if check_port $FRONTEND_PORT; then
    print_warning "Port $FRONTEND_PORT is already in use. Frontend may not start properly."
fi

if check_port $BACKEND_PORT; then
    print_warning "Port $BACKEND_PORT is already in use. Backend may not start properly."
fi

if check_port $ENHANCED_API_PORT; then
    print_warning "Port $ENHANCED_API_PORT is already in use. Enhanced API may not start properly."
fi

echo ""

# Start Backend Services
print_header "Starting Backend Services"

# Navigate to backend directory
cd src/backend

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
if [ -f "requirements_enhanced.txt" ]; then
    pip install -r requirements_enhanced.txt
else
    pip install -r requirements.txt
fi

# Create necessary directories
mkdir -p data/uploads data/models logs

# Set environment variables
export FLASK_APP=app.py
export PYTHONPATH=$PWD

# Start the main backend (Flask app)
print_status "Starting main backend server on port $BACKEND_PORT..."
python app.py &
BACKEND_PID=$!
print_success "Main backend started (PID: $BACKEND_PID)"

# Start the enhanced API
print_status "Starting enhanced API server on port $ENHANCED_API_PORT..."
python enhanced_api.py &
ENHANCED_API_PID=$!
print_success "Enhanced API started (PID: $ENHANCED_API_PID)"

# Wait a moment for backend services to start
sleep 3

# Check if backend services are running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    print_error "Main backend failed to start"
    exit 1
fi

if ! kill -0 $ENHANCED_API_PID 2>/dev/null; then
    print_error "Enhanced API failed to start"
    exit 1
fi

# Wait for backend services to be ready
if wait_for_service "http://localhost:$BACKEND_PORT" "Main Backend"; then
    print_success "Main backend is ready at http://localhost:$BACKEND_PORT"
else
    print_warning "Main backend may not be fully ready yet"
fi

if wait_for_service "http://localhost:$ENHANCED_API_PORT/api/v2/health" "Enhanced API"; then
    print_success "Enhanced API is ready at http://localhost:$ENHANCED_API_PORT"
else
    print_warning "Enhanced API may not be fully ready yet"
fi

# Return to project root
cd ../..

echo ""

# Start Frontend
print_header "Starting Frontend Application"

# Check if node_modules exists
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

# Start the frontend
print_status "Starting React frontend on port $FRONTEND_PORT..."
npm start &
FRONTEND_PID=$!
print_success "Frontend started (PID: $FRONTEND_PID)"

# Wait for frontend to be ready
sleep 5

if wait_for_service "http://localhost:$FRONTEND_PORT" "Frontend"; then
    print_success "Frontend is ready at http://localhost:$FRONTEND_PORT"
else
    print_warning "Frontend may not be fully ready yet"
fi

echo ""

# Display final status
print_header "Platform Startup Complete"
echo ""
print_success "🎉 AI Proposal Optimization Platform is now running!"
echo ""
echo -e "${CYAN}📱 Frontend (React):${NC}     http://localhost:$FRONTEND_PORT"
echo -e "${CYAN}🔧 Main Backend:${NC}         http://localhost:$BACKEND_PORT"
echo -e "${CYAN}🧠 Enhanced API:${NC}         http://localhost:$ENHANCED_API_PORT"
echo ""
echo -e "${YELLOW}📊 Available Features:${NC}"
echo "   • Data Upload & Validation"
echo "   • AI-Powered Bidding Engine"
echo "   • Data Analysis Dashboard"
echo "   • Model Validation & Optimization"
echo "   • Competitive Analysis"
echo ""
echo -e "${YELLOW}🔧 API Endpoints:${NC}"
echo "   • Main API: http://localhost:$BACKEND_PORT/api/"
echo "   • Enhanced API: http://localhost:$ENHANCED_API_PORT/api/v2/"
echo ""
echo -e "${YELLOW}📁 File Upload:${NC}        http://localhost:$BACKEND_PORT/upload"
echo -e "${YELLOW}🎯 Smart Predict:${NC}       http://localhost:$BACKEND_PORT/predict"
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
        print_error "Main backend process has stopped"
        cleanup
    fi
    
    if ! kill -0 $ENHANCED_API_PID 2>/dev/null; then
        print_error "Enhanced API process has stopped"
        cleanup
    fi
    
    sleep 10
done 