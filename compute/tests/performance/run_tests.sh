#!/bin/bash

# Set environment variables for testing
export GOMAXPROCS=8  # Adjust based on your CPU cores
export GOGC=100      # Default GC setting

echo "Running HNSW performance tests..."
go test -v -timeout 30m

echo -e "\nRunning with different GOGC settings..."
for gc in 50 200 400; do
    echo -e "\nTesting with GOGC=$gc"
    GOGC=$gc go test -v -run TestHNSWPerformance -timeout 30m
done

echo -e "\nRunning accuracy tests..."
go test -v -run TestHNSWAccuracy
