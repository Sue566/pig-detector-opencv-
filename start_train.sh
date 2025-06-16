#!/bin/bash
# Simple setup and training script for macOS
set -e
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/train.py --config config.yaml
