#!/bin/bash
cd "$(dirname "$0")"

# Kill any existing processes on port 5001
lsof -ti:5001 | xargs kill -9 2>/dev/null || true

# Start the server
cd backend
export PORT=5001
python3 app.py &
SERVER_PID=$!

# Wait for server to start
echo "⏳ Starting server..."
sleep 3

# Upload test data if it exists
if [ -f "../test_data.csv" ]; then
    echo "📤 Uploading test data..."
    curl -X POST -F "file=@../test_data.csv" http://localhost:5001/api/upload > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ Test data uploaded successfully"
    else
        echo "⚠️  Failed to upload test data"
    fi
fi

echo "🌐 Server running at: http://localhost:5001"
echo "🎯 Advanced App: http://localhost:5001/advanced-app.html"
echo "⏹️  Press Ctrl+C to stop"

# Wait for the server process
wait $SERVER_PID