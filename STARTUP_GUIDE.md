# AI Proposal Optimization Platform - Startup Guide

## Quick Start (Automatic)
Run the automated startup script:
```bash
./start-app-simple.sh
```

This will open two new terminal windows - one for the backend and one for the frontend.

## Manual Startup

### Option 1: Start Backend First
1. Open a terminal and navigate to the backend directory:
   ```bash
   cd /Users/nkasibhatla/WorkingPredictor/backend
   python3 app.py
   ```

2. Open another terminal and navigate to the frontend directory:
   ```bash
   cd /Users/nkasibhatla/WorkingPredictor/frontend
   npm start
   ```

### Option 2: Use the Original Script
```bash
./start-frontend-fixed.sh
```

## Access the Application

- **Frontend (React UI)**: http://localhost:3000
- **Backend (Flask API)**: http://localhost:5000

## Features Available

### Dashboard
- View system statistics
- Check API connection status
- Quick access to main features

### Data Upload
- Upload CSV or Excel files with proposal data
- View data summary after upload
- Required columns: proposal_id, client_id, bid_amount, win_loss, industry

### Proposal Analyzer
- Analyze new proposals for win probability
- Get AI-powered recommendations
- View risk scores and optimization suggestions

## Troubleshooting

### Frontend Issues
- If you see "default React app", make sure both backend and frontend are running
- If compilation fails, try: `rm -rf node_modules package-lock.json && npm install --legacy-peer-deps`

### Backend Issues
- Make sure Python 3 is installed: `python3 --version`
- Install required packages: `pip3 install flask pandas numpy`

### Port Conflicts
- If ports 3000 or 5000 are in use, kill existing processes:
  ```bash
  lsof -ti:3000 | xargs kill -9
  lsof -ti:5000 | xargs kill -9
  ```

## Data Format

Your CSV file should include these columns:
- `proposal_id`: Unique identifier
- `client_id`: Client identifier  
- `bid_amount`: Proposed amount
- `win_loss`: "win" or "loss"
- `industry`: Industry category (optional)

## Support

If you encounter issues:
1. Check that both services are running
2. Verify the API connection in the frontend dashboard
3. Check browser console for errors
4. Ensure data is uploaded before analyzing proposals
