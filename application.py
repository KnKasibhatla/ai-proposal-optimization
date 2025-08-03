#!/usr/bin/env python3
"""
AWS Elastic Beanstalk entry point for the AI Bid Optimization Platform
"""

import os
import sys

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Import the Flask application
from backend.app import app as application

# Production configuration
application.config['ENV'] = 'production'
application.config['DEBUG'] = False
application.config['TESTING'] = False

# Ensure upload directories exist
upload_dir = os.path.join(os.path.dirname(__file__), 'backend', 'data', 'uploads')
os.makedirs(upload_dir, exist_ok=True)

# Add a route to serve the main application
@application.route('/')
def index():
    """Serve the main application page"""
    try:
        with open(os.path.join(os.path.dirname(__file__), 'frontend', 'public', 'advanced-app.html'), 'r') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return """
        <h1>🚀 AI Bid Optimization Platform</h1>
        <p>Welcome to your AI-powered bidding platform!</p>
        <p><a href="/frontend/public/advanced-app.html">Access Main Application</a></p>
        <p><a href="/frontend/public/simple-upload.html">Simple Upload Interface</a></p>
        """

if __name__ == "__main__":
    # This is used when running locally
    # Elastic Beanstalk will use the 'application' variable
    port = int(os.environ.get('PORT', 5000))
    application.run(debug=False, host='0.0.0.0', port=port)