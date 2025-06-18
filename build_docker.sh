#!/bin/bash
# Build Docker image for pig detector
set -e
IMAGE_NAME=pig-detector

docker build -t $IMAGE_NAME .
echo "Docker image '$IMAGE_NAME' built successfully."
