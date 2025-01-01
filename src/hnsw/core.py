"""
Optimized HNSW implementation with proper beam search and neighbor selection.
"""
import numpy as np
from typing import List, Dict, Set, Optional, Tuple, Any, DefaultDict
import heapq
import logging
import threading
import queue
import time
from dataclasses import dataclass, field
import multiprocessing as mp
from redis import Redis, ConnectionPool
from redis.connection import ConnectionError
from minio import Minio
import json
import io
import pickle
from collections import defaultdict
import math
import random
from functools import lru_cache
import urllib3
import copy

logger = logging.getLogger(__name__)

@dataclass
class VectorData:
    """Vector data container with efficient serialization."""
    id: int
    vector: np.ndarray
    level: int = field(default=0)
    neighbors: Dict[int, List[int]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize neighbors dictionary if not provided."""
        if not self.neighbors:
            self.neighbors = {}
    
    def __getstate__(self):
        """Custom serialization to handle numpy array."""
        state = self.__dict__.copy()
        if self.vector is not None:
            state['vector'] = self.vector.tobytes()
        return state
    
    def __setstate__(self, state):
        """Custom deserialization to handle numpy array."""
        vector_bytes = state.pop('vector', None)
        self.__dict__.update(state)
        if vector_bytes is not None:
            self.vector = np.frombuffer(vector_bytes, dtype=np.float32)
        else:
            self.vector = None

class DistanceCache:
    """Cache for distance calculations."""
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache = {}
        
    def get(self, key: Tuple[int, int]) -> Optional[float]:
        """Get cached distance."""
        return self.cache.get(key)
        
    def put(self, key: Tuple[int, int], value: float):
        """Put distance in cache."""
        if len(self.cache) >= self.max_size:
            # Remove random item
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
        
    def clear(self):
        """Clear the cache."""
        self.cache.clear()

class LRUCache:
    def __init__(self, maxsize=100):
        self.maxsize = maxsize
        self.cache = {}
        self.order = []

    def get(self, key):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        else:
            return None

    def set(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.maxsize:
            lru_key = self.order.pop(0)
            del self.cache[lru_key]
        self.cache[key] = value
        self.order.append(key)

class ConnectionManager:
    """Manages connection pools for Redis and MinIO."""
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        minio_endpoint: str = "localhost:9000",
        minio_access_key: str = "minioadmin",
        minio_secret_key: str = "minioadmin"
    ):
        """Initialize connection pools."""
        try:
            # Initialize Redis pool
            self.redis_pool = ConnectionPool(
                host=redis_host,
                port=redis_port,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            
            # Test Redis connection
            redis_client = Redis(connection_pool=self.redis_pool)
            redis_client.ping()
            logger.info(f"Successfully connected to Redis at {redis_host}:{redis_port}")
            
            # Initialize MinIO client
            self.minio_client = Minio(
                minio_endpoint,
                access_key=minio_access_key,
                secret_key=minio_secret_key,
                secure=False,
                http_client=urllib3.PoolManager(
                    timeout=urllib3.Timeout(connect=5, read=10),
                    retries=urllib3.Retry(
                        total=3,
                        backoff_factor=0.2
                    )
                )
            )
            
            # Test MinIO connection
            self.minio_client.list_buckets()
            logger.info(f"Successfully connected to MinIO at {minio_endpoint}")
            
        except ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize connections: {str(e)}")
            raise
    
    def get_redis(self) -> Redis:
        """Get Redis client from pool."""
        return Redis(connection_pool=self.redis_pool)
    
    def get_minio(self) -> Minio:
        """Get MinIO client."""
        return self.minio_client
    
    def release_redis(self, client: Redis):
        """Release Redis client back to pool."""
        try:
            client.close()
        except Exception as e:
            logger.error(f"Error releasing Redis client: {str(e)}")
    
    def release_minio(self, client: Minio):
        """Release MinIO client."""
        # MinIO client doesn't need explicit release
        pass

class HNSW:
    """Optimized HNSW implementation with proper beam search."""
    def __init__(
        self,
        dimension: int,
        max_elements: int,
        M: int = 16,
        ef_construction: int = 200,
        ef: int = 50,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        minio_endpoint: str = "localhost:9000",
        minio_access_key: str = "minioadmin",
        minio_secret_key: str = "minioadmin",
        bucket_name: str = None
    ):
        """Initialize HNSW index."""
        self._validate_params(dimension, max_elements, M, ef_construction, ef)
        
        self.dimension = dimension
        self.max_elements = max_elements
        self.M = M
        self.ef_construction = ef_construction
        self.ef = ef
        self.M_scale = 1.0 / math.log(M)  # Precompute for level generation
        
        # Initialize connection manager
        self.conn_manager = ConnectionManager(
            redis_host, redis_port,
            minio_endpoint, minio_access_key, minio_secret_key
        )
        
        # Set bucket name based on dimension if not provided
        self.bucket_name = bucket_name or f"vectors-{dimension}"
        
        # Initialize MinIO bucket
        self._init_bucket()
        
        # Initialize caches
        self.vector_cache = {}
        self.cache_size = 1000  # Adjust based on memory constraints
        self.distance_cache = DistanceCache(max_size=10000)
        
        # Initialize batching
        self._vector_batch = {}
        self._batch_size = 100
        self._neighbor_updates = {}
        self.mlock = threading.Lock()
        self.rlock = threading.Lock()

    def _flush_vector_batch(self):
        """Flush batched vectors to storage."""
        if not self._vector_batch:
            return
            
        try:
            # Batch MinIO operations
            with self.mlock:
                minio_client = self.conn_manager.get_minio()
                for vid, vector_data in self._vector_batch.items():
                    vector_bytes = pickle.dumps(vector_data)
                    vector_stream = io.BytesIO(vector_bytes)
                    minio_client.put_object(
                        self.bucket_name,
                        f"vector_{vid}",
                        vector_stream,
                        len(vector_bytes)
                    )

            # Batch Redis operations
            with self.rlock:
                redis_client = self.conn_manager.get_redis()
                pipe = redis_client.pipeline()
                for vid in self._vector_batch:
                    pipe.sadd("hnsw:vectors", str(vid))
                pipe.execute()

            self._vector_batch.clear()

        except Exception as e:
            logger.error(f"Failed to flush vector batch: {e}")
            raise

    def _store_vector(self, vector_data: VectorData) -> None:
        """Store vector with proper error handling."""
        try:
            # Ensure bucket exists
            self._init_bucket()
            
            # Store in Redis
            redis_client = self.conn_manager.get_redis()
            try:
                # Prepare vector data for storage
                vector_copy = copy.deepcopy(vector_data)
                vector_bytes = vector_copy.vector.tobytes()
                vector_copy.vector = None  # Don't store vector in Redis
                
                # Store metadata in Redis
                redis_key = f"hnsw:vector:{vector_data.id}"
                try:
                    redis_data = pickle.dumps(vector_copy, protocol=pickle.HIGHEST_PROTOCOL)
                    redis_client.set(redis_key, redis_data)
                except Exception as e:
                    logger.error(f"Failed to serialize vector {vector_data.id} metadata: {str(e)}")
                    raise
                
                # Store vector in MinIO
                minio_client = self.conn_manager.get_minio()
                try:
                    object_name = f"vector_{vector_data.id}"
                    try:
                        minio_client.put_object(
                            self.bucket_name,
                            object_name,
                            io.BytesIO(vector_bytes),
                            len(vector_bytes)
                        )
                    except Exception as e:
                        logger.error(f"Failed to store vector {vector_data.id} in MinIO: {str(e)}")
                        # Try to clean up Redis key
                        redis_client.delete(redis_key)
                        raise
                    
                    # Update cache
                    if len(self.vector_cache) >= self.cache_size:
                        remove_key = next(iter(self.vector_cache))
                        del self.vector_cache[remove_key]
                    self.vector_cache[vector_data.id] = vector_data
                    
                finally:
                    self.conn_manager.release_minio(minio_client)
                    
            finally:
                self.conn_manager.release_redis(redis_client)
                
        except Exception as e:
            logger.error(f"Failed to store vector {vector_data.id}: {str(e)}")
            raise

    def _update_neighbors(self, vector_id: int, neighbors: Dict[int, List[int]]):
        """Batch neighbor updates."""
        self._neighbor_updates[vector_id] = neighbors
        if len(self._neighbor_updates) >= self._batch_size:
            self._flush_neighbor_updates()

    def _flush_neighbor_updates(self):
        """Flush batched neighbor updates."""
        if not self._neighbor_updates:
            return

        try:
            with self.rlock:
                redis_client = self.conn_manager.get_redis()
                pipe = redis_client.pipeline()
                
                for vid, neighbors in self._neighbor_updates.items():
                    key = f"vector:{vid}:neighbors"
                    pipe.delete(key)
                    if neighbors:
                        pipe.hmset(key, neighbors)
                
                pipe.execute()
            
            self._neighbor_updates.clear()

        except Exception as e:
            logger.error(f"Failed to flush neighbor updates: {e}")
            raise

    def _distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine distance between two vectors."""
        # Both vectors should already be normalized
        dot_product = np.dot(v1, v2)
        # Clamp to [-1, 1] to handle numerical errors
        dot_product = max(-1.0, min(1.0, dot_product))
        # Convert similarity [-1, 1] to distance [0, 2]
        return 1.0 - dot_product

    def _get_neighbors(self, vector_id: int, level: int) -> List[int]:
        """Get neighbors of a vector at a specific level."""
        vector = self._get_vector(vector_id)
        if vector is None or level not in vector.neighbors:
            return []
        return vector.neighbors[level]

    def _random_level(self) -> int:
        """Generate random level using exponential distribution."""
        return int(-math.log(random.random()) * self.M_scale)

    def _init_bucket(self):
        """Initialize MinIO bucket."""
        try:
            minio_client = self.conn_manager.get_minio()
            try:
                # Create bucket if it doesn't exist
                if not minio_client.bucket_exists(self.bucket_name):
                    minio_client.make_bucket(self.bucket_name)
                    logger.info(f"Created MinIO bucket: {self.bucket_name}")
                else:
                    logger.info(f"Using existing MinIO bucket: {self.bucket_name}")
            finally:
                self.conn_manager.release_minio(minio_client)
        except Exception as e:
            logger.error(f"Failed to initialize MinIO bucket: {str(e)}")
            raise

    def _compute_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute L2 distance with caching."""
        return float(np.linalg.norm(v1 - v2))
    
    def _get_distance(self, v1_id: int, v2_id: int, v1_data: Optional[VectorData] = None,
                     v2_data: Optional[VectorData] = None) -> float:
        """Get distance from cache or compute it."""
        # Check cache first
        cache_key = tuple(sorted([v1_id, v2_id]))
        cached_dist = self.distance_cache.get(cache_key)
        if cached_dist is not None:
            return cached_dist
        
        # Get vector data if not provided
        if v1_data is None:
            v1_data = self._get_vector(v1_id)
        if v2_data is None:
            v2_data = self._get_vector(v2_id)
        
        if v1_data is None or v2_data is None:
            return float('inf')
        
        # Compute and cache distance
        distance = float(np.linalg.norm(v1_data.vector - v2_data.vector))
        self.distance_cache.put(cache_key, distance)
        return distance
    
    def _select_neighbors(self, candidates: List[Tuple[float, int]], M: int, level: int, query_vector: np.ndarray = None) -> List[int]:
        """Select best neighbors using distance-based heuristic."""
        if not candidates:
            return []
        
        # Sort candidates by distance
        candidates.sort()
        
        # For upper levels, use simple distance-based selection
        if level > 0:
            return [c[1] for c in candidates[:M]]
        
        # For base layer, use diversity-aware selection
        selected = []
        remaining = candidates.copy()
        
        # Always select closest point
        if remaining:
            selected.append(remaining.pop(0)[1])
        
        # Select remaining points using distance and diversity
        while len(selected) < M and remaining:
            best_idx = -1
            best_score = float('inf')
            
            for i, (dist, idx) in enumerate(remaining):
                # Calculate diversity score
                max_sim = 0.0
                valid_selection = True
                
                for sel_id in selected:
                    sel_vector = self._get_vector(sel_id)
                    curr_vector = self._get_vector(idx)
                    
                    if sel_vector is None or curr_vector is None:
                        valid_selection = False
                        break
                    
                    sim = 1.0 / (1.0 + self._distance(curr_vector.vector, sel_vector.vector))
                    max_sim = max(max_sim, sim)
                
                if not valid_selection:
                    continue
                
                # Combine distance and diversity
                score = dist * (1.0 + max_sim)
                
                if score < best_score:
                    best_score = score
                    best_idx = i
            
            if best_idx != -1:
                selected.append(remaining.pop(best_idx)[1])
            else:
                break
        
        return selected

    def _search_layer(self, query_vector: np.ndarray, ep: int, ef: int, level: int) -> List[Tuple[float, int]]:
        """Search for nearest neighbors in a single layer."""
        visited = set([ep])
        candidates = []  # min-heap for candidates
        results = []     # max-heap for results
        
        # Initialize with entry point
        if ep is not None:
            ep_vector = self._get_vector(ep)
            if ep_vector is not None:
                dist = self._distance(query_vector, ep_vector.vector)
                heapq.heappush(candidates, (dist, ep))
                heapq.heappush(results, (-dist, ep))
        
        # Search while we have candidates and can improve results
        while candidates:
            current_dist, current_id = heapq.heappop(candidates)
            
            # Stop if we can't improve results
            if len(results) >= ef and -results[0][0] < current_dist:
                break
            
            # Get neighbors of current point
            neighbors = self._get_neighbors(current_id, level)
            
            # Process each neighbor
            for neighbor_id in neighbors:
                if neighbor_id in visited:
                    continue
                    
                visited.add(neighbor_id)
                
                # Get neighbor vector
                neighbor_vector = self._get_vector(neighbor_id)
                if neighbor_vector is None:
                    continue
                
                # Calculate distance
                dist = self._distance(query_vector, neighbor_vector.vector)
                
                # Update candidates and results
                if len(results) < ef or dist < -results[0][0]:
                    heapq.heappush(candidates, (dist, neighbor_id))
                    heapq.heappush(results, (-dist, neighbor_id))
                    
                    # Maintain ef size
                    if len(results) > ef:
                        heapq.heappop(results)
        
        # Convert results to ascending distance order
        return sorted([(dist, idx) for dist, idx in [(-d, i) for d, i in results]])

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def _get_vector(self, vector_id: int) -> Optional[VectorData]:
        """Get vector data with proper error handling."""
        try:
            # Check cache first
            if vector_id in self.vector_cache:
                return self.vector_cache[vector_id]
            
            # Get from Redis
            redis_client = self.conn_manager.get_redis()
            try:
                vector_data_bytes = redis_client.get(f"hnsw:vector:{vector_id}")
                if vector_data_bytes is None:
                    logger.error(f"Vector {vector_id} not found in Redis")
                    return None
                
                # Load metadata
                try:
                    vector_data = pickle.loads(vector_data_bytes)
                except Exception as e:
                    logger.error(f"Failed to deserialize vector {vector_id} metadata: {str(e)}")
                    return None
                
                # Get vector from MinIO
                minio_client = self.conn_manager.get_minio()
                try:
                    try:
                        response = minio_client.get_object(
                            self.bucket_name,
                            f"vector_{vector_id}"
                        )
                        vector_bytes = response.read()
                        vector_data.vector = np.frombuffer(vector_bytes, dtype=np.float32)
                    except Exception as e:
                        logger.error(f"Failed to get vector {vector_id} from MinIO: {str(e)}")
                        return None
                    
                    # Update cache
                    if len(self.vector_cache) >= self.cache_size:
                        remove_key = next(iter(self.vector_cache))
                        del self.vector_cache[remove_key]
                    self.vector_cache[vector_id] = vector_data
                    
                    return vector_data
                    
                finally:
                    self.conn_manager.release_minio(minio_client)
                    
            finally:
                self.conn_manager.release_redis(redis_client)
                
        except Exception as e:
            logger.error(f"Error getting vector {vector_id}: {str(e)}")
            return None

    def _cleanup_storage(self):
        """Clean up all storage."""
        try:
            # Clean Redis
            redis_client = self.conn_manager.get_redis()
            try:
                keys = redis_client.keys("hnsw:*")
                if keys:
                    redis_client.delete(*keys)
            finally:
                self.conn_manager.release_redis(redis_client)
            
            # Clean MinIO
            minio_client = self.conn_manager.get_minio()
            try:
                if minio_client.bucket_exists(self.bucket_name):
                    objects = minio_client.list_objects(self.bucket_name)
                    for obj in objects:
                        minio_client.remove_object(self.bucket_name, obj.object_name)
            finally:
                self.conn_manager.release_minio(minio_client)
                
            # Clear caches
            self.vector_cache.clear()
            self.distance_cache.cache.clear()
            self._vector_batch.clear()
            self._neighbor_updates.clear()
            
        except Exception as e:
            logger.error(f"Failed to clean storage: {str(e)}")

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[float, int]]:
        """Search for k nearest neighbors of query vector."""
        query_vector = self._normalize_vector(query_vector)
        
        # Get entry point
        entry_point = self._get_entry_point()
        if entry_point is None:
            return []
            
        # Initialize variables
        ep = entry_point
        ep_dist = float('inf')
        if ep is not None:
            ep_vector = self._get_vector(ep)
            if ep_vector is not None:
                ep_dist = self._distance(query_vector, ep_vector.vector)
        
        # Search through layers
        for level in range(self.max_level, -1, -1):
            # Search current layer
            candidates = self._search_layer(
                query_vector,
                ep,
                max(self.ef, k) if level == 0 else 1,
                level
            )
            
            # Update entry point for next layer
            if candidates:
                ep = candidates[0][1]
                ep_dist = candidates[0][0]
        
        # Get final results from bottom layer
        results = self._search_layer(query_vector, ep, max(self.ef, k), 0)
        
        # Return top k results
        return results[:k]

    def _update_connections(self, vector_id: int, neighbor_ids: List[int], level: int) -> None:
        """Update connections between vectors at a given level."""
        for neighbor_id in neighbor_ids:
            neighbor = self._get_vector(neighbor_id)
            if neighbor:
                if level not in neighbor.neighbors:
                    neighbor.neighbors[level] = []
                if vector_id not in neighbor.neighbors[level]:
                    neighbor.neighbors[level].append(vector_id)
                    self._store_vector(neighbor)

    def _get_entry_point(self) -> Optional[int]:
        """Get the entry point for the index."""
        try:
            redis_client = self.conn_manager.get_redis()
            try:
                ep = redis_client.get("hnsw:entry_point")
                if ep is not None:
                    return int(ep)
                return None
            finally:
                self.conn_manager.release_redis(redis_client)
        except Exception as e:
            logger.error(f"Failed to get entry point: {str(e)}")
            return None

    def _set_entry_point(self, vector_id: int):
        """Set the entry point for the index."""
        try:
            redis_client = self.conn_manager.get_redis()
            try:
                redis_client.set("hnsw:entry_point", str(vector_id))
            finally:
                self.conn_manager.release_redis(redis_client)
        except Exception as e:
            logger.error(f"Failed to set entry point: {str(e)}")
            raise

    def _get_max_level(self) -> int:
        """Get the maximum level in the index."""
        try:
            redis_client = self.conn_manager.get_redis()
            try:
                max_level = redis_client.get("hnsw:max_level")
                if max_level is not None:
                    return int(max_level)
                return -1
            finally:
                self.conn_manager.release_redis(redis_client)
        except Exception as e:
            logger.error(f"Failed to get max level: {str(e)}")
            return -1

    def _set_max_level(self, level: int):
        """Set the maximum level in the index."""
        try:
            redis_client = self.conn_manager.get_redis()
            try:
                redis_client.set("hnsw:max_level", str(level))
            finally:
                self.conn_manager.release_redis(redis_client)
        except Exception as e:
            logger.error(f"Failed to set max level: {str(e)}")
            raise

    def insert(self, vector_id: int, vector: np.ndarray) -> None:
        """Insert a vector into the index."""
        if not isinstance(vector, np.ndarray):
            raise ValueError("Vector must be a numpy array")
        
        if vector.shape != (self.dimension,):
            raise ValueError(f"Vector dimension mismatch. Expected {self.dimension}, got {vector.shape}")
        
        try:
            # Normalize vector
            vector = self._normalize_vector(vector)
            
            # Create new vector data
            new_vector = VectorData(
                id=vector_id,
                vector=vector,
                level=self._random_level()
            )
            
            # Get current max level and entry point
            max_level = self._get_max_level()
            ep = self._get_entry_point()
            
            # Special case for first vector
            if ep is None:
                self._store_vector(new_vector)
                self._set_entry_point(vector_id)
                self._set_max_level(new_vector.level)
                return
            
            # Get entry point vector
            ep_vector = self._get_vector(ep)
            if ep_vector is None:
                # If entry point is invalid, make this the new entry point
                self._store_vector(new_vector)
                self._set_entry_point(vector_id)
                self._set_max_level(new_vector.level)
                return
            
            # Search for insert
            curr_ep = ep
            curr_dist = float('inf')
            
            # Find entry point for each level
            for level in range(max_level, new_vector.level, -1):
                candidates = self._search_layer(vector, curr_ep, 1, level)
                if candidates:
                    curr_ep, curr_dist = candidates[0]
            
            # For each level the vector has
            for level in range(min(new_vector.level, max_level), -1, -1):
                # Find nearest neighbors
                candidates = self._search_layer(vector, curr_ep, self.ef_construction, level)
                if not candidates:
                    continue
                
                # Select neighbors
                curr_neighbors = self._select_neighbors(candidates, self.M, level, vector)
                if not curr_neighbors:
                    continue
                
                # Update connections
                new_vector.neighbors[level] = curr_neighbors
                
                # Update reverse connections
                for neighbor_id in curr_neighbors:
                    neighbor = self._get_vector(neighbor_id)
                    if neighbor is None:
                        continue
                    
                    # Get all potential neighbors
                    potential_neighbors = []
                    for n in neighbor.neighbors.get(level, []):
                        n_vector = self._get_vector(n)
                        if n_vector is not None:
                            dist = self._distance(neighbor.vector, n_vector.vector)
                            potential_neighbors.append((dist, n))
                    
                    # Add the new vector
                    dist = self._distance(neighbor.vector, new_vector.vector)
                    potential_neighbors.append((dist, new_vector.id))
                    
                    # Select best neighbors
                    neighbor.neighbors[level] = self._select_neighbors(
                        potential_neighbors,
                        self.M,
                        level
                    )
                    self._store_vector(neighbor)
            
            # Store the new vector
            self._store_vector(new_vector)
            
            # Update entry point and max level if needed
            if new_vector.level > max_level:
                self._set_entry_point(vector_id)
                self._set_max_level(new_vector.level)
            
        except Exception as e:
            logger.error(f"Failed to insert vector {vector_id}: {str(e)}")
            raise

    def _validate_params(self, dimension: int, max_elements: int, M: int,
                        ef_construction: int, ef: int):
        """Validate initialization parameters."""
        if dimension <= 0:
            raise ValueError("Dimension must be positive")
        if max_elements <= 0:
            raise ValueError("Max elements must be positive")
        if M <= 0:
            raise ValueError("M must be positive")
        if ef_construction <= 0:
            raise ValueError("ef_construction must be positive")
        if ef <= 0:
            raise ValueError("ef must be positive")
    
    def _validate_vector(self, vector: np.ndarray):
        """Validate vector format and dimension."""
        if not isinstance(vector, np.ndarray):
            raise ValueError("Vector must be a numpy array")
        if vector.shape != (self.dimension,):
            raise ValueError(f"Vector dimension must be {self.dimension}")
