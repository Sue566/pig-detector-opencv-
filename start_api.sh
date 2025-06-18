#!/bin/bash
set -e
uvicorn scripts.api:app --host 0.0.0.0 --port 8093
