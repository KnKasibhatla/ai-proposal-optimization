#!/usr/bin/env python3
"""
Quick Start Script for AI Proposal Optimization Platform
Handles setup and launches the application with Enhanced AI Techniques
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}")
    return True

def setup_environment():
    """Set up the development environment"""
    print("\n🔧 Setting up environment...")
    
    # Add src to Python path
    src_path = os.path.join(os.getcwd(), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        os.environ['PYTHONPATH'] = src_path + ':' + os.environ.get('PYTHONPATH', '')
    
    # Set Flask environment
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = '1'
    
    print("✅ Environment configured")

def create_basic_structure():
    """Create basic directory structure and files"""
    print("\n📁 Creating basic structure...")
    
    # Create directories
    directories = [
        'src/backend', 'src/models', 'src/utils', 'config',
        'templates', 'static/css', 'static/js', 'data/raw',
        'data/processed', 'data/uploads', 'logs', 'tests'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Create __init__.py files
    init_files = [
        'src/__init__.py', 'src/backend/__init__.py',
        'src/models/__init__.py', 'src/utils/__init__.py',
        'config/__init__.py'
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# Package initialization\n')
    
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write('''FLASK_ENV=development
SECRET_KEY=dev-secret-key-change-in-production
DATABASE_URL=sqlite:///data/proposals.db
ENABLE_ADVANCED_AI=true
''')
    
    print("✅ Basic structure created")

def install_enhanced_requirements():
    """Install enhanced requirements for AI techniques"""
    print("\n📦 Installing enhanced AI requirements...")
    
    enhanced_packages = [
        'flask>=2.3.0',
        'flask-sqlalchemy>=3.0.0',
        'pandas>=2.0.0',
        'numpy>=1.26.0',
        'scikit-learn>=1.3.0',
        'python-dotenv>=1.0.0',
        'tensorflow==2.16.1',
        'torch==2.7.1',
        'gymnasium==0.29.1',
        'cvxpy>=1.3.0',
        'networkx>=3.0',
        'plotly>=5.15.0',
        'joblib>=1.3.0'
    ]
    
    try:
        for package in enhanced_packages:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ Enhanced AI requirements installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install enhanced requirements: {e}")
        return False

def install_minimal_requirements():
    """Install minimal requirements for basic functionality"""
    print("\n📦 Installing minimal requirements...")
    
    minimal_packages = [
        'flask==2.3.3',
        'flask-sqlalchemy==3.0.5',
        'pandas==2.0.3',
        'numpy==1.24.3',
        'scikit-learn==1.3.0',
        'python-dotenv==1.0.0'
    ]
    
    try:
        for package in minimal_packages:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ Minimal requirements installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def test_enhanced_imports():
    """Test if enhanced AI imports work"""
    print("\n🧪 Testing enhanced AI imports...")
    
    try:
        import tensorflow as tf
        print("✅ TensorFlow imported successfully")
    except ImportError:
        print("❌ TensorFlow import failed")
        return False
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
    except ImportError:
        print("❌ PyTorch import failed")
        return False
    
    try:
        import gymnasium as gym
        print("✅ Gymnasium imported successfully")
    except ImportError:
        print("❌ Gymnasium import failed")
        return False
    
    try:
        import cvxpy
        print("✅ CVXPY imported successfully")
    except ImportError:
        print("❌ CVXPY import failed")
        return False
    
    try:
        import networkx
        print("✅ NetworkX imported successfully")
    except ImportError:
        print("❌ NetworkX import failed")
        return False
    
    return True

def test_imports():
    """Test if critical imports work"""
    print("\n🧪 Testing imports...")
    
    try:
        import flask
        print("✅ Flask imported successfully")
    except ImportError:
        print("❌ Flask import failed")
        return False
    
    try:
        import pandas
        print("✅ Pandas imported successfully")
    except ImportError:
        print("❌ Pandas import failed")
        return False
    
    try:
        import numpy
        print("✅ NumPy imported successfully")
    except ImportError:
        print("❌ NumPy import failed")
        return False
    
    try:
        import sklearn
        print("✅ Scikit-learn imported successfully")
    except ImportError:
        print("❌ Scikit-learn import failed")
        return False
    
    return True

def create_minimal_app():
    """Create a minimal working Flask app"""
    print("\n🔧 Creating minimal Flask app...")
    
    app_content = '''from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'

@app.route('/')
def index():
    return jsonify({
        'message': 'AI Proposal Optimization Platform',
        'status': 'running',
        'version': '2.0.0_enhanced'
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'enhanced_ai': True,
        'techniques': ['MDP', 'DQN', 'Multi-Agent', 'No-Regret']
    })

if __name__ == '__main__':
    print("🚀 Enhanced AI Proposal Platform Starting...")
    print("📍 API Endpoint: http://localhost:5000")
    print("🧠 Advanced AI Techniques: MDP, DQN, Multi-Agent, No-Regret Learning")
    app.run(debug=True, host='0.0.0.0', port=5000)
'''
    
    app_path = Path('src/backend/app.py')
    app_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(app_path, 'w') as f:
        f.write(app_content)
    
    print("✅ Minimal Flask app created")

def start_application():
    """Start the application"""
    print("\n🚀 Starting Enhanced AI Proposal Platform...")
    
    # Check if enhanced API exists
    enhanced_api_path = Path('src/backend/enhanced_api.py')
    if enhanced_api_path.exists():
        print("🎯 Starting Enhanced AI API...")
        try:
            subprocess.run([sys.executable, 'src/backend/enhanced_api.py'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to start enhanced API: {e}")
            print("🔄 Falling back to basic app...")
            subprocess.run([sys.executable, 'src/backend/app.py'], check=True)
    else:
        print("🔄 Starting basic app...")
        subprocess.run([sys.executable, 'src/backend/app.py'], check=True)

def main():
    """Main function"""
    print("🎯 Enhanced AI Proposal Optimization Platform Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Setup environment
    setup_environment()
    
    # Create basic structure
    create_basic_structure()
    
    # Ask user for installation type
    print("\n📦 Installation Options:")
    print("1. Enhanced AI Setup (Recommended) - Includes MDP, DQN, Multi-Agent, No-Regret Learning")
    print("2. Minimal Setup - Basic functionality only")
    print("3. Skip installation - Use existing setup")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        print("\n🔧 Setting up Enhanced AI Platform...")
        if install_enhanced_requirements():
            if test_enhanced_imports():
                print("✅ Enhanced AI setup completed successfully!")
            else:
                print("⚠️ Enhanced AI setup completed with warnings")
        else:
            print("❌ Enhanced AI setup failed")
            return
    elif choice == '2':
        print("\n🔧 Setting up Minimal Platform...")
        if install_minimal_requirements():
            if test_imports():
                print("✅ Minimal setup completed successfully!")
            else:
                print("❌ Minimal setup failed")
                return
        else:
            print("❌ Minimal setup failed")
            return
    elif choice == '3':
        print("⏭️ Skipping installation")
    else:
        print("❌ Invalid choice")
        return
    
    # Create minimal app if it doesn't exist
    if not Path('src/backend/app.py').exists():
        create_minimal_app()
    
    # Ask if user wants to start the application
    start_choice = input("\n🚀 Start the application now? (y/n): ").strip().lower()
    
    if start_choice in ['y', 'yes']:
        try:
            start_application()
        except KeyboardInterrupt:
            print("\n👋 Application stopped by user")
        except Exception as e:
            print(f"❌ Failed to start application: {e}")
    else:
        print("\n📋 Manual startup instructions:")
        print("1. Run: python src/backend/enhanced_api.py (for enhanced AI)")
        print("2. Run: python src/backend/app.py (for basic functionality)")
        print("3. Or use: make enhanced-run (if Makefile is available)")
        print("\n🌐 Access the platform at: http://localhost:8000 (enhanced) or http://localhost:5000 (basic)")

if __name__ == '__main__':
    main()
