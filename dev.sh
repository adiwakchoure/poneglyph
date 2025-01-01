#!/bin/bash

# Start Redis and MinIO containers
docker-compose up -d redis minio

# Wait for services to be healthy
echo "Waiting for Redis and MinIO to be ready..."
sleep 5

# Start API in development mode
python3 dev.py
