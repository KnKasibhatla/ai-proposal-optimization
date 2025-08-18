#!/bin/bash
echo "🔍 Checking AI Proposal Optimization Platform Status..."
echo ""

# Check if server is running
if curl -s http://localhost:5001/ > /dev/null 2>&1; then
    echo "✅ Server is running at http://localhost:5001"
    
    # Check if frontend is accessible
    if curl -s http://localhost:5001/advanced-app.html | grep -q "Advanced AI Proposal"; then
        echo "✅ Frontend is accessible at http://localhost:5001/advanced-app.html"
    else
        echo "❌ Frontend not accessible"
    fi
    
    # Check if data is uploaded
    if curl -s http://localhost:5001/api/debug/data-status | grep -q '"data_available":true'; then
        echo "✅ Test data is loaded"
    else
        echo "⚠️  No data loaded - upload data via the frontend"
    fi
    
    echo ""
    echo "🎯 Ready to use! Open: http://localhost:5001"
    
else
    echo "❌ Server is not running"
    echo "💡 Run: ./start_server.sh"
fi