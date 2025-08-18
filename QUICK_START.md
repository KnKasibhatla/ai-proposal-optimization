# 🚀 AI Proposal Optimization Platform - Quick Start

## URLs
- **Main Application**: http://localhost:5001
- **Advanced App**: http://localhost:5001/advanced-app.html

## Quick Commands

### Start the Server
```bash
./start_server.sh
```

### Check Status
```bash
./check_status.sh
```

### Stop the Server
```bash
pkill -f "python.*app.py"
```

## Features Status
- ✅ **Smart Pricing** - Generates varied prices based on project type
- ✅ **Model Validation** - Shows real metrics (90% accuracy, detailed precision/recall/F1)
- ✅ **Competitive Intelligence** - Dynamic competitive analysis
- ✅ **Data Upload** - Handles CSV files with proper column mapping
- ✅ **Frontend Interface** - Full web interface with all features

## Test Data
- Sample data automatically uploaded: `test_data.csv` (2,361 records)
- Upload your own data via the web interface

## Fixed Issues
1. ✅ Model Validation 0% precision/recall/F1 → Now shows accurate metrics
2. ✅ Smart Pricing static values → Now varies by project type  
3. ✅ Competitive Intelligence static → Now shows dynamic values
4. ✅ Frontend accessibility → Now served correctly on port 5001
5. ✅ Data column mapping → Handles both 'price' and 'bid_amount' columns

## Port Notes
- Uses port 5001 (port 5000 conflicts with macOS AirPlay)
- Server automatically kills existing processes on startup