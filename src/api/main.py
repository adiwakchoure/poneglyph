"""
FastAPI service for vector search operations.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import redis
from minio import Minio
from minio.error import S3Error
import pickle
import os
from io import BytesIO
from src.hnsw.core import HNSW

app = FastAPI(title="Vector Search Service")

# Initialize Redis connection
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0
)

# Initialize MinIO client
minio_client = Minio(
    os.getenv("MINIO_HOST", "localhost:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    secure=False
)

# Initialize HNSW index
VECTOR_DIM = int(os.getenv("VECTOR_DIM", 128))
index = HNSW(dim=VECTOR_DIM)

BUCKET_NAME = "vectors"

# Ensure bucket exists
try:
    if not minio_client.bucket_exists(BUCKET_NAME):
        minio_client.make_bucket(BUCKET_NAME)
except S3Error as e:
    print(f"Error creating bucket: {e}")

class Vector(BaseModel):
    id: int
    vector: List[float]

class SearchQuery(BaseModel):
    vector: List[float]
    k: int = 10
    ef: int = 50

class SearchResult(BaseModel):
    id: int
    distance: float

@app.post("/vectors/", status_code=201)
async def add_vector(vector_data: Vector):
    """Add a vector to the index and store in MinIO."""
    try:
        # Convert vector to numpy array
        vector = np.array(vector_data.vector, dtype=np.float32)
        
        # Add to HNSW index
        index.add(vector_data.id, vector)
        
        # Store vector in MinIO
        vector_bytes = pickle.dumps(vector)
        vector_data_stream = BytesIO(vector_bytes)
        
        minio_client.put_object(
            BUCKET_NAME,
            f"vector_{vector_data.id}",
            vector_data_stream,
            length=len(vector_bytes)
        )
        
        # Store metadata in Redis
        redis_client.hset(
            f"vector:{vector_data.id}",
            mapping={
                "dim": str(VECTOR_DIM),
                "created_at": str(np.datetime64('now'))
            }
        )
        
        return {"message": "Vector added successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/", response_model=List[SearchResult])
async def search_vectors(query: SearchQuery):
    """Search for similar vectors."""
    try:
        query_vector = np.array(query.vector, dtype=np.float32)
        results = index.search(query_vector, k=query.k, ef=query.ef)
        
        return [
            SearchResult(id=idx, distance=float(dist))
            for dist, idx in results
        ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vectors/{vector_id}")
async def get_vector(vector_id: int):
    """Retrieve a vector by ID."""
    try:
        # Get vector from MinIO
        response = minio_client.get_object(
            BUCKET_NAME,
            f"vector_{vector_id}"
        )
        vector_bytes = response.read()
        vector = pickle.loads(vector_bytes)
        
        # Get metadata from Redis
        metadata = redis_client.hgetall(f"vector:{vector_id}")
        
        return {
            "id": vector_id,
            "vector": vector.tolist(),
            "metadata": {k.decode(): v.decode() for k, v in metadata.items()}
        }
    
    except minio.error.NoSuchKey:
        raise HTTPException(status_code=404, detail="Vector not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
