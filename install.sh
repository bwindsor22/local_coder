#!/usr/bin/env bash
set -e

echo "Creating virtual environment..."
python3 -m venv local_env
source local_env/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install \
  requests \
  rich \
  pydantic \
  tiktoken \
  watchdog

echo "Done."

echo ""
echo "To use:"
echo "source local_env/bin/activate"
echo "python local_coder.py"
