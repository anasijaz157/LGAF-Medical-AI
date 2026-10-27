#!/bin/bash
set -e

echo "====================================================================="
echo "          LGAF Medical AI - Local Deployment Launcher (Linux/Mac)    "
echo "====================================================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.10+."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source .venv/bin/activate

# Install requirements
echo "[INFO] Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Run Streamlit
echo "====================================================================="
echo "Starting LGAF Medical AI Suite on http://localhost:8501"
echo "====================================================================="
streamlit run app.py
