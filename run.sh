#!/bin/bash

echo "============================================"
echo "   MATAN Analysis Tool"
echo "   Analiza uspjesnosti MA1 i MA2"
echo "============================================"
echo

cd "$(dirname "$0")"

if [ ! -f ".venv/bin/python" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo "Please run: ./setup.sh"
    exit 1
fi

echo "Starting analysis..."
echo

.venv/bin/python main.py

echo
echo "============================================"
echo "Analysis complete! Check the output folder."
echo "============================================"
