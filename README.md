# Poneglyph: High-Performance Vector Search Database

Poneglyph is a high-performance vector search database built with Go, implementing the HNSW (Hierarchical Navigable Small World) algorithm for efficient similarity search. It provides a distributed architecture with separate storage and compute layers, making it highly scalable and maintainable.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│  RESTful API    │────▶│   Go Compute     │────▶│  MinIO Storage │
│    Interface    │     │     Service      │     │                │
└─────────────────┘     └──────────────────┘     └────────────────┘
         │                       │                        │
         │                       ▼                        │
         │              ┌──────────────────┐             │
         └──────────▶  │  Redis Cache/    │   ◀─────────┘
                       │  Index Storage    │
                       └──────────────────┘
```

### Core Components

1. **HNSW Index (Compute Layer)**
   - Implements Hierarchical Navigable Small World graph
   - Multi-level graph structure for logarithmic search complexity
   - Thread-safe operations with fine-grained locking
   - Optimized distance calculations
   - Configurable parameters for performance tuning

2. **Storage Layer**
   - Abstract interface for storage backends
   - MinIO implementation for vector persistence
   - Efficient binary serialization
   - Support for any S3-compatible storage
   - Batch operations for better performance

3. **Cache Layer**
   - Redis-based caching
   - Read-through caching strategy
   - Configurable expiration
   - Batch operations support
   - Thread-safe operations

4. **API Layer**
   - RESTful endpoints
   - High-performance Fiber framework
   - Comprehensive error handling
   - Request validation
   - CORS support

## Technical Details

### HNSW Algorithm Implementation

The HNSW algorithm creates a multi-layer graph structure where:
- Each layer is a navigable small world graph
- Higher layers are sparser, providing "highways" for quick navigation
- Lower layers are denser, providing precise local search
- Search complexity: O(log N) for insertions and queries

Key optimizations:
1. Priority queue-based neighbor selection
2. Fine-grained locking for concurrent operations
3. Efficient distance calculations
4. Batch processing support
5. Configurable graph parameters

### Storage Optimization

1. **Binary Serialization**
   - Compact vector representation
   - Efficient serialization/deserialization
   - Minimized storage overhead

2. **Caching Strategy**
   - Read-through caching
   - Batch operations
   - Automatic cache invalidation
   - Memory-efficient storage

3. **Concurrent Operations**
   - Thread-safe implementations
   - Connection pooling
   - Batch processing
   - Error handling and recovery

## API Endpoints

### Vector Operations

```http
# Insert a vector
POST /api/v1/vectors
{
    "id": 1,
    "vector": [0.1, 0.2, 0.3, ...]
}

# Batch insert vectors
POST /api/v1/vectors/batch
{
    "vectors": {
        "1": [0.1, 0.2, ...],
        "2": [0.3, 0.4, ...],
        ...
    }
}

# Search similar vectors
POST /api/v1/search
{
    "vector": [0.1, 0.2, 0.3, ...],
    "k": 10
}

# Delete a vector
DELETE /api/v1/vectors/:id

# Rebuild index
POST /api/v1/index/rebuild
```

## Setup and Configuration

1. **Prerequisites**
   - Docker and Docker Compose
   - Go 1.21+ (for development)

2. **Environment Variables**
   ```env
   # MinIO Configuration
   MINIO_ENDPOINT=minio:9000
   MINIO_ACCESS_KEY=minioadmin
   MINIO_SECRET_KEY=minioadmin
   MINIO_BUCKET=vectors
   
   # Redis Configuration
   REDIS_ADDR=redis:6379
   REDIS_PASSWORD=
   REDIS_DB=0
   
   # Vector Service Configuration
   VECTOR_DIMENSION=128
   SERVER_ADDR=:3000
   ```

3. **Running with Docker Compose**
   ```bash
   docker-compose up --build
   ```

## Development

1. Initialize Go module:
   ```bash
   cd compute
   go mod init github.com/yourusername/poneglyph/compute
   go mod tidy
   ```

2. Run tests:
   ```bash
   go test ./...
   ```

3. Build locally:
   ```bash
   go build -o server ./src/main.go
   ```

## Performance Considerations

1. **HNSW Parameters**
   - `maxLevel`: Maximum number of layers (default: 16)
   - `efConstruction`: Size of dynamic candidate list (default: 100)
   - `M`: Maximum number of connections per node (default: 16)

2. **Caching Strategy**
   - Configurable cache expiration
   - Memory-efficient serialization
   - Batch operations for better throughput

3. **Storage Optimization**
   - Binary serialization for vectors
   - Connection pooling
   - Batch operations

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License
