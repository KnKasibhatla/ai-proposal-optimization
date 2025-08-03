#!/bin/bash

# Complete AI Proposal Optimization Platform Startup Script
# This script starts all services and ensures everything works together

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Function to cleanup background processes on exit
cleanup() {
    print_status "Shutting down services..."
    
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
        print_status "Frontend stopped"
    fi
    
    if [ ! -z "$FLASK_PID" ]; then
        kill $FLASK_PID 2>/dev/null || true
        print_status "Flask backend stopped"
    fi
    
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null || true
        print_status "API backend stopped"
    fi
    
    print_success "All services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

print_status "Starting Complete AI Proposal Optimization Platform"

# Check prerequisites
print_status "Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi
print_success "Python version: $(python3 --version)"

# Setup Python Environment
print_status "Setting up Python Environment"

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

# Install basic requirements
pip install flask flask-cors pandas numpy werkzeug

print_success "Python dependencies installed"

# Create necessary directories
mkdir -p data/uploads data/models logs

# Kill any existing processes
print_status "Cleaning up existing processes..."
pkill -f "python.*app.py" 2>/dev/null || true
pkill -f "python.*simple_api.py" 2>/dev/null || true
pkill -f "python.*http.server" 2>/dev/null || true
sleep 2

# Start Flask Backend (Port 5000)
print_status "Starting Flask backend on port 5000..."
cd src/backend
export FLASK_APP=app.py
export PYTHONPATH=$PWD
python app.py &
FLASK_PID=$!
cd ../..
print_success "Flask backend started (PID: $FLASK_PID)"

# Start Simple API Backend (Port 8000)
print_status "Starting Simple API backend on port 8000..."
python simple_api.py &
API_PID=$!
print_success "Simple API backend started (PID: $API_PID)"

# Wait for backend services to start
print_status "Waiting for backend services to start..."
sleep 5

# Start HTML Frontend (Port 3000)
print_status "Starting HTML frontend on port 3000..."
python3 -m http.server 3000 &
FRONTEND_PID=$!
print_success "HTML frontend started (PID: $FRONTEND_PID)"

# Wait a moment for services to start
sleep 3

# Test services
print_status "Testing services..."

# Test Flask backend
if curl -s http://localhost:5000 > /dev/null; then
    print_success "Flask backend (port 5000) is responding"
else
    print_warning "Flask backend (port 5000) may not be responding"
fi

# Test Simple API backend
if curl -s http://localhost:8000 > /dev/null; then
    print_success "Simple API backend (port 8000) is responding"
else
    print_warning "Simple API backend (port 8000) may not be responding"
fi

# Test HTML frontend
if curl -s http://localhost:3000/simple-frontend.html > /dev/null; then
    print_success "HTML frontend (port 3000) is responding"
else
    print_warning "HTML frontend (port 3000) may not be responding"
fi

# Display final status
print_success "🎉 Complete AI Proposal Optimization Platform is now running!"
echo ""
echo -e "${BLUE}📱 HTML Frontend:${NC}        http://localhost:3000/simple-frontend.html"
echo -e "${BLUE}🔧 Flask Backend:${NC}        http://localhost:5000"
echo -e "${BLUE}🚀 Simple API:${NC}           http://localhost:8000"
echo ""
echo -e "${YELLOW}📋 Available Features:${NC}"
echo "   • Data Upload & Validation"
echo "   • Data Analysis Dashboard"
echo "   • AI-Powered Bidding Prediction"
echo "   • Model Validation"
echo "   • Competitive Analysis"
echo "   • Project Management"
echo "   • Customer Management"
echo ""
echo -e "${YELLOW}🔧 API Endpoints:${NC}"
echo "   • Health Check: http://localhost:8000/api/v2/health"
echo "   • File Upload: http://localhost:8000/api/v2/upload/"
echo "   • Prediction: http://localhost:8000/api/v2/prediction/single"
echo "   • Projects: http://localhost:8000/api/v2/projects"
echo "   • Customers: http://localhost:8000/api/v2/customers"
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
    
    if ! kill -0 $FLASK_PID 2>/dev/null; then
        print_error "Flask backend process has stopped"
        cleanup
    fi
    
    if ! kill -0 $API_PID 2>/dev/null; then
        print_error "API backend process has stopped"
        cleanup
    fi
    
    sleep 10
done 