#!/usr/bin/env bash
set -e

# ──────────────────────────────────────────────────────────────────────────────
# local-coder installer
#
# Prerequisites (must be done before running this script):
#
#   1. Install Ollama:  https://ollama.com/download
#      macOS:   brew install ollama   (then 'brew services start ollama')
#      Linux:   curl -fsSL https://ollama.com/install.sh | sh
#
#   2. Pull the required models:
#      ollama pull qwen3-coder:30b   (~18 GB, primary coding model)
#      ollama pull llava-phi3:latest (~3 GB, vision model for screenshots)
#
#   3. Make sure the Ollama server is running before launching local-coder:
#      ollama serve       (runs on http://localhost:11434 by default)
#      — or start it as a system service (see Ollama docs).
#
#   Hardware notes:
#     qwen3-coder:30b Q4_K_M requires ~20 GB RAM (runs on Apple Silicon M-series
#     or a GPU with 24 GB VRAM).  A smaller model can be configured by editing
#     the MODEL = "..." line at the top of local_coder.py.
# ──────────────────────────────────────────────────────────────────────────────

echo "Creating virtual environment..."
python3 -m venv local_env
source local_env/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing Python dependencies..."
pip install \
  requests \
  rich \
  pydantic \
  tiktoken \
  watchdog

echo ""
echo "Installation complete."
echo ""
echo "────────────────────────────────────────────────"
echo "Before starting local-coder, ensure Ollama is running:"
echo ""
echo "  ollama serve"
echo ""
echo "  # (first time only — pull the models):"
echo "  ollama pull qwen3-coder:30b"
echo "  ollama pull llava-phi3:latest"
echo ""
echo "Then run:"
echo "  source local_env/bin/activate"
echo "  python local_coder.py --project /path/to/project"
echo ""
echo "Or run a single task non-interactively:"
echo "  python local_coder.py --project /path/to/project --task 'Add a dark mode toggle'"
echo "────────────────────────────────────────────────"
