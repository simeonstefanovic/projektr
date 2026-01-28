#!/bin/bash

echo "============================================"
echo "   MATAN Analysis Tool - Setup"
echo "============================================"
echo

cd "$(dirname "$0")"

echo "[1/3] Creating virtual environment..."
python3 -m venv .venv

echo "[2/3] Installing dependencies..."
.venv/bin/pip install -r requirements.txt -q

echo "[3/3] Setup complete!"
echo
echo "============================================"
echo "Now you can run the analysis with: ./run.sh"
echo "============================================"
