# 🚀 Free Web Hosting Deployment Guide

Deploy your AI Proposal Optimization Platform to the web for **FREE** using these services.

## 🌐 Deployment Options

### Option 1: Render.com + Netlify (Recommended)

**Backend on Render.com (Free):**
1. Go to [render.com](https://render.com) and sign up
2. Click "New +" → "Web Service"
3. Connect your GitHub: `https://github.com/KnKasibhatla/ai-proposal-optimization`
4. Settings:
   - **Name**: `ai-proposal-backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python backend/app.py`
   - **Plan**: Free
5. Click "Create Web Service"

**Frontend on Netlify (Free):**
1. Go to [netlify.com](https://netlify.com) and sign up
2. Click "Add new site" → "Import an existing project"
3. Connect your GitHub: `https://github.com/KnKasibhatla/ai-proposal-optimization`
4. Settings:
   - **Base directory**: Leave empty
   - **Build command**: Leave empty
   - **Publish directory**: `frontend/public`
5. Click "Deploy site"

### Option 2: Railway.app (Alternative)

**Full-Stack on Railway:**
1. Go to [railway.app](https://railway.app) and sign up
2. Click "New Project" → "Deploy from GitHub repo"
3. Select: `https://github.com/KnKasibhatla/ai-proposal-optimization`
4. Railway will auto-detect and deploy both frontend and backend

### Option 3: GitHub Pages + Render

**Frontend on GitHub Pages:**
1. Go to your repository: `https://github.com/KnKasibhatla/ai-proposal-optimization`
2. Click "Settings" → "Pages"
3. Source: "Deploy from a branch"
4. Branch: `main`
5. Folder: `/ (root)`
6. Click "Save"

**Backend on Render:** (Same as Option 1)

## 🔧 After Deployment

### Update Frontend API URL
Once your backend is deployed on Render, update the frontend:

1. Note your Render backend URL (e.g., `https://ai-proposal-backend.onrender.com`)
2. The frontend will automatically detect and use this URL in production

### Test Your Deployment
1. Visit your deployed frontend URL
2. Upload sample data using the "Upload Data" tab
3. Test predictions using the "Smart Pricing" tab
4. Verify all features work correctly

## 📋 Deployment URLs

After deployment, you'll have:
- **Frontend**: `https://your-site-name.netlify.app`
- **Backend API**: `https://ai-proposal-backend.onrender.com`

## 🆓 Free Tier Limits

**Render.com:**
- 750 hours/month (always free)
- Sleeps after 15 minutes of inactivity
- Wakes up on first request (may take 30 seconds)

**Netlify:**
- 100GB bandwidth/month
- 300 build minutes/month
- Custom domains supported

**GitHub Pages:**
- 1GB storage
- 100GB bandwidth/month
- Public repositories only

## 🚀 Quick Deploy Commands

```bash
# Commit any changes
git add .
git commit -m "Prepare for deployment"
git push

# Your code is now ready for deployment!
```

## 🔒 Security Notes

- Environment variables are automatically handled
- No API keys exposed in frontend code
- HTTPS enabled by default on all platforms

## 📞 Support

If you encounter issues:
1. Check the deployment logs in your hosting platform
2. Ensure all files are committed to GitHub
3. Verify the `requirements.txt` includes all dependencies