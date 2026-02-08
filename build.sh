#!/usr/bin/env bash
# Build script for Render deployment
# 1. Install Node dependencies and build React frontend
# 2. Install Python dependencies
# 3. Train the NLP model (if not already trained)

set -e

echo "=== Installing Node.js dependencies ==="
npm install --legacy-peer-deps

echo "=== Building React frontend ==="
npm run build

echo "=== Installing Python dependencies ==="
pip install -r backend/requirements.txt

echo "=== Training NLP model ==="
cd backend
python -m chatbot.train
cd ..

echo "=== Build complete ==="
