"""
FastAPI application for vector search with optimized HNSW implementation.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from typing import List, Dict, Optional
import logging
import os
import threading

# Import HNSW implementation
import sys
import os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.hnsw.core import HNSW

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vector Search API")

# Environment variables
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
MINIO_HOST = os.getenv("MINIO_HOST", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

# Global state
indices: Dict[int, HNSW] = {}
index_lock = threading.Lock()

class VectorRequest(BaseModel):
    id: int
    vector: List[float]
    dimension: int

class SearchRequest(BaseModel):
    vector: List[float]
    dimension: int
    k: int = Field(default=10, gt=0)

class SearchResult(BaseModel):
    id: int
    distance: float

def get_or_create_index(dimension: int) -> HNSW:
    """Get existing index or create new one with optimized settings."""
    with index_lock:
        if dimension not in indices:
            # Calculate optimal parameters based on dimension
            M = min(64, max(16, int(dimension ** 0.5)))  # Scale M with dimension
            ef_construction = max(100, min(400, dimension * 8))  # Scale ef with dimension
            
            indices[dimension] = HNSW(
                dimension=dimension,
                max_elements=1000000,  # Default to 1M vectors
                M=M,
                ef_construction=ef_construction,
                ef=40,  # Default ef for search
                redis_host=REDIS_HOST,
                redis_port=REDIS_PORT,
                minio_endpoint=MINIO_HOST,
                minio_access_key=MINIO_ACCESS_KEY,
                minio_secret_key=MINIO_SECRET_KEY,
                bucket_prefix="vectors",  # Will create bucket like vectors-512
                n_jobs=-1,  # Use all CPU cores
                batch_size=1000  # Batch size for insertions
            )
            logger.info(f"Created new index for {dimension}d vectors with M={M}, ef_construction={ef_construction}")
        return indices[dimension]

@app.post("/vectors/")
async def add_vector(request: VectorRequest):
    """Add vector to the index."""
    try:
        vector = np.array(request.vector, dtype=np.float32)
        index = get_or_create_index(request.dimension)
        index.insert(request.id, vector)
        return {"status": "success", "message": f"Vector {request.id} added successfully"}
    except Exception as e:
        logger.error(f"Error adding vector: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search/")
async def search_vectors(request: SearchRequest):
    """Search for similar vectors."""
    try:
        vector = np.array(request.vector, dtype=np.float32)
        index = get_or_create_index(request.dimension)
        
        results = index.search(vector, k=request.k)
        
        return [
            SearchResult(id=id, distance=float(dist))
            for dist, id in results
        ]
    except Exception as e:
        logger.error(f"Error searching vectors: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/clear")
async def clear_all_indices():
    """Clear all indices."""
    try:
        with index_lock:
            indices.clear()
        return {"status": "success", "message": "All indices cleared"}
    except Exception as e:
        logger.error(f"Error clearing indices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear/{dimension}")
async def clear_index(dimension: int):
    """Clear specific dimension index."""
    try:
        with index_lock:
            if dimension in indices:
                del indices[dimension]
        return {"status": "success", "message": f"Index for dimension {dimension} cleared"}
    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
