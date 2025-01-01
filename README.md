# Poneglyph: Distributed Vector Search System

Poneglyph is a high-performance distributed vector search system built on HNSW (Hierarchical Navigable Small World) graphs. It provides fast and accurate approximate nearest neighbor search for high-dimensional vectors, with support for distributed storage and parallel processing.

## System Architecture

### Core Components

1. **HNSW Implementation (`src/hnsw/core.py`)**
   - Custom HNSW graph implementation optimized for distributed systems
   - Multi-level graph structure for logarithmic search complexity
   - Optimized distance calculations using NumPy
   - Thread-safe operations with proper locking mechanisms
   - Custom serialization for efficient vector storage

2. **FastAPI Service (`src/api/main.py`)**
   - RESTful API for vector operations
   - Endpoints for vector insertion and search
   - Health checks and index management
   - Asynchronous request handling
   - Proper error handling and logging

3. **Storage Layer**
   - **MinIO**: Object storage for vector data
     - Efficient binary storage of vectors
     - Scalable and distributed
     - S3-compatible API
   - **Redis**: Metadata and graph structure
     - Fast access to graph connections
     - In-memory performance
     - Atomic operations for thread safety

### Key Features

1. **Optimized Vector Operations**
   - Efficient vector insertion with O(log N) complexity
   - Fast approximate nearest neighbor search
   - Batch processing capabilities
   - Custom serialization for NumPy arrays

2. **Distributed Architecture**
   - Horizontally scalable storage with MinIO
   - Distributed metadata with Redis
   - Connection pooling for better resource utilization
   - Thread-safe operations

3. **Performance Optimizations**
   - Dynamic parameter tuning based on dimensionality
   - Efficient memory management
   - Optimized distance calculations
   - Custom serialization protocols

## Technical Details

### HNSW Implementation

```python
class HNSW:
    def __init__(self, dimension, max_elements, M=16, ef_construction=200, ef=10):
        # M: Maximum number of connections per element per layer
        # ef_construction: Size of dynamic candidate list during construction
        # ef: Size of dynamic candidate list during search
```

1. **Graph Construction**
   - Multi-layer graph structure
   - Layer count: log(N) * ml where ml = 1/ln(M)
   - Each element connects to M nearest neighbors
   - Bottom layer has 2*M connections for better recall

2. **Search Algorithm**
   - Greedy search with backtracking
   - Layer-wise traversal from top to bottom
   - Dynamic candidate list size (ef parameter)
   - Distance calculations optimized with NumPy

3. **Vector Storage**
   - Custom VectorData class with efficient serialization
   - MinIO for binary vector storage
   - Redis for graph structure and metadata
   - Connection pooling for better performance

### API Endpoints

1. **Vector Operations**
   ```
   POST /vectors/
   {
     "id": int,
     "vector": List[float],
     "dimension": int
   }
   
   POST /search/
   {
     "vector": List[float],
     "dimension": int,
     "k": int
   }
   ```

2. **Index Management**
   ```
   POST /clear
   POST /clear/{dimension}
   GET /health
   ```

## Deployment

### Docker Compose Setup

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      REDIS_HOST: redis
      REDIS_PORT: 6379
      MINIO_HOST: minio:9000
      
  redis:
    image: redis:alpine
    
  minio:
    image: minio/minio
    ports:
      - "9000:9000"
      - "9001:9001"
```

### Environment Variables

- `REDIS_HOST`: Redis server hostname
- `REDIS_PORT`: Redis server port
- `MINIO_HOST`: MinIO server endpoint
- `MINIO_ACCESS_KEY`: MinIO access key
- `MINIO_SECRET_KEY`: MinIO secret key
- `VECTOR_DIM`: Default vector dimension

## Performance Characteristics

1. **Time Complexity**
   - Insert: O(log N)
   - Search: O(log N)
   - Space: O(N * M * log N)

2. **Memory Usage**
   - Vector storage: O(N * dimension)
   - Graph structure: O(N * M * log N)
   - Connection pools: O(max_connections)

3. **Scalability**
   - Horizontal scaling with MinIO
   - Redis cluster support
   - Connection pooling
   - Thread-safe operations

## Future Improvements

1. **Optimization Opportunities**
   - Implement vector quantization
   - Add support for multiple indices
   - Implement batch search operations
   - Add support for vector deletion

2. **Feature Additions**
   - Real-time index updates
   - Vector metadata support
   - Custom distance metrics
   - Index persistence and recovery

## References

1. Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.
2. Bernhardsson, E. (2018). Annoy: Approximate Nearest Neighbors in C++/Python.
3. Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs.
