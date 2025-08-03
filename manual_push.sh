#!/bin/bash

echo "🚀 Manual GitHub Push Script"
echo "============================="
echo ""

# Pre-configured repository details
repo_url="https://github.com/KnKasibhatla/ai-proposal-optimization.git"

echo "📝 Repository: $repo_url"
echo ""

echo "🔧 Your current token may need updated permissions."
echo ""
echo "📋 To push manually:"
echo ""
echo "1️⃣  Create a new GitHub Personal Access Token:"
echo "   - Go to: https://github.com/settings/tokens"
echo "   - Click 'Generate new token (classic)'"
echo "   - Select scopes: ✅ repo (full control)"
echo "   - Copy the new token"
echo ""
echo "2️⃣  Push using Git credentials:"
echo "   Username: KnKasibhatla"
echo "   Password: [paste your new token]"
echo ""
echo "3️⃣  Run this command:"
echo "   git push -u origin main"
echo ""
echo "🔄 Or try this one-liner (replace YOUR_TOKEN):"
echo "   git push https://KnKasibhatla:YOUR_TOKEN@github.com/KnKasibhatla/ai-proposal-optimization.git main"
echo ""

# Try simple push that will prompt for credentials
echo "🎯 Attempting push (will prompt for username/password):"
echo ""
git push -u origin main

echo ""
echo "💡 If prompted:"
echo "   Username: KnKasibhatla"
echo "   Password: [your GitHub token]"