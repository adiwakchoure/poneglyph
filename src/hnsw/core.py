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
from redis.connection import ConnectionPool
from redis import Redis
from minio import Minio
import json
import io
import pickle
from collections import defaultdict
import math
import random
from functools import lru_cache

logger = logging.getLogger(__name__)

@dataclass
class VectorData:
    """Vector data container with efficient serialization."""
    id: int
    vector: np.ndarray
    neighbors: DefaultDict[int, List[int]] = field(default_factory=lambda: defaultdict(list))
    level: int = field(default=-1)
    
    def __getstate__(self):
        """Custom serialization to handle numpy array and defaultdict."""
        state = self.__dict__.copy()
        state['vector'] = self.vector.tobytes()
        state['neighbors'] = dict(self.neighbors)
        return state
    
    def __setstate__(self, state):
        """Custom deserialization to restore numpy array and defaultdict."""
        vector_bytes = state.pop('vector')
        neighbors_dict = state.pop('neighbors')
        self.__dict__.update(state)
        self.vector = np.frombuffer(vector_bytes, dtype=np.float32)
        self.neighbors = defaultdict(list, neighbors_dict)

class DistanceCache:
    """Cache for distance computations."""
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size
        self._lock = threading.Lock()
    
    def get(self, v1_id: int, v2_id: int) -> Optional[float]:
        """Get cached distance."""
        key = tuple(sorted([v1_id, v2_id]))
        with self._lock:
            return self.cache.get(key)
    
    def set(self, v1_id: int, v2_id: int, distance: float):
        """Set distance in cache."""
        key = tuple(sorted([v1_id, v2_id]))
        with self._lock:
            if len(self.cache) >= self.max_size:
                # Remove random entry if full
                del self.cache[random.choice(list(self.cache.keys()))]
            self.cache[key] = distance

class ConnectionManager:
    """Manages connection pools for Redis and MinIO."""
    def __init__(
        self,
        redis_host: str,
        redis_port: int,
        minio_endpoint: str,
        minio_access_key: str,
        minio_secret_key: str,
        max_connections: int = 50
    ):
        # Redis connection pool
        self.redis_pool = ConnectionPool(
            host=redis_host,
            port=redis_port,
            max_connections=max_connections,
            decode_responses=True
        )
        
        # MinIO connection parameters
        self.minio_params = {
            "endpoint": minio_endpoint,
            "access_key": minio_access_key,
            "secret_key": minio_secret_key,
            "secure": False
        }
        
        # Connection caches
        self._redis_cache = queue.Queue(maxsize=max_connections)
        self._minio_cache = queue.Queue(maxsize=max_connections)
        
        # Initialize connection caches
        for _ in range(max_connections // 2):
            self._redis_cache.put(Redis(connection_pool=self.redis_pool))
            self._minio_cache.put(Minio(**self.minio_params))
    
    def get_redis(self) -> Redis:
        """Get a Redis connection from cache."""
        try:
            return self._redis_cache.get_nowait()
        except queue.Empty:
            return Redis(connection_pool=self.redis_pool)
    
    def get_minio(self) -> Minio:
        """Get a MinIO client from cache."""
        try:
            return self._minio_cache.get_nowait()
        except queue.Empty:
            return Minio(**self.minio_params)
    
    def release_redis(self, conn: Redis):
        """Return Redis connection to cache."""
        try:
            self._redis_cache.put_nowait(conn)
        except queue.Full:
            conn.close()
    
    def release_minio(self, client: Minio):
        """Return MinIO client to cache."""
        try:
            self._minio_cache.put_nowait(client)
        except queue.Full:
            pass

class HNSW:
    """Optimized HNSW implementation with proper beam search."""
    def __init__(
        self,
        dimension: int,
        max_elements: int,
        M: int = 16,
        ef_construction: int = 200,
        ef: int = 40,
        ml: Optional[float] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        minio_endpoint: str = "localhost:9000",
        minio_access_key: str = "minioadmin",
        minio_secret_key: str = "minioadmin",
        bucket_prefix: str = "vectors",
        n_jobs: int = -1,
        batch_size: int = 1000
    ):
        self._validate_params(dimension, max_elements, M, ef_construction, ef)
        
        self.dimension = dimension
        self.max_elements = max_elements
        self.M = M
        self.ef_construction = ef_construction
        self.ef = ef
        self.ml = ml if ml is not None else 1/np.log2(M)
        
        # Create DNS-compliant bucket name
        self.bucket_name = f"{bucket_prefix}-{dimension}"
        
        # Initialize managers
        self.conn_manager = ConnectionManager(
            redis_host, redis_port,
            minio_endpoint, minio_access_key, minio_secret_key
        )
        
        # Initialize caches
        self.distance_cache = DistanceCache()
        
        # Initialize storage
        self._init_storage()
        
        # Initialize vector counter
        redis_client = self.conn_manager.get_redis()
        try:
            if not redis_client.exists("hnsw:vector_count"):
                redis_client.set("hnsw:vector_count", 0)
        finally:
            self.conn_manager.release_redis(redis_client)
    
    def _init_storage(self):
        """Initialize storage with proper error handling."""
        try:
            # Initialize MinIO bucket
            minio_client = self.conn_manager.get_minio()
            try:
                if not minio_client.bucket_exists(self.bucket_name):
                    minio_client.make_bucket(self.bucket_name)
            finally:
                self.conn_manager.release_minio(minio_client)
            
            # Initialize Redis structures
            redis_client = self.conn_manager.get_redis()
            try:
                redis_client.set("hnsw:dimension", self.dimension)
                redis_client.set("hnsw:max_elements", self.max_elements)
                if not redis_client.exists("hnsw:entry_point"):
                    redis_client.set("hnsw:entry_point", -1)
            finally:
                self.conn_manager.release_redis(redis_client)
        except Exception as e:
            logger.error(f"Error initializing storage: {e}")
            raise
    
    @lru_cache(maxsize=1024)
    def _compute_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute L2 distance with caching."""
        return float(np.linalg.norm(v1 - v2))
    
    def _get_distance(self, v1_id: int, v2_id: int, v1_data: Optional[VectorData] = None,
                     v2_data: Optional[VectorData] = None) -> float:
        """Get distance from cache or compute it."""
        # Check cache first
        cache_key = tuple(sorted([v1_id, v2_id]))
        cached_dist = self.distance_cache.get(v1_id, v2_id)
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
        self.distance_cache.set(v1_id, v2_id, distance)
        return distance
    
    def _select_neighbors(
        self,
        query_data: VectorData,
        candidates: List[VectorData],
        M: int,
        level: int,
        ef: int,
        visited: Set[int]
    ) -> List[int]:
        """Select nearest neighbors using beam search with improved exploration."""
        if not candidates:
            return []
        
        # Use level-specific visited set
        level_visited = {query_data.id}  # Only track visits for this level
        
        # Initialize candidate set W and result set
        W = []  # Working set of candidates (min-heap)
        result = []  # Result set (sorted by distance)
        
        # Initialize with candidates
        for c in candidates:
            if c.id not in level_visited:
                level_visited.add(c.id)
                dist = self._get_distance(query_data.id, c.id, query_data, c)
                if dist != float('inf'):
                    heapq.heappush(W, (dist, c.id))
                    result.append((dist, c.id))
        
        # Sort initial results
        result.sort()
        if len(result) > ef:
            result = result[:ef]
        
        # Continue search while W not empty
        while W:
            # Get closest unexpanded element
            current_dist, current_id = heapq.heappop(W)
            
            # Stop if we can't improve results
            if result and current_dist > result[-1][0] * 1.1:  # Allow 10% slack for better recall
                break
                
            # Get neighbors of current point
            current_vector = self._get_vector(current_id)
            if not current_vector:
                continue
                
            # Consider neighbors from current and lower levels
            neighbors_to_explore = set()
            for l in range(level, -1, -1):  # Consider lower levels too
                if l in current_vector.neighbors:
                    neighbors_to_explore.update(current_vector.neighbors[l])
            
            # Explore neighbors
            for neighbor_id in neighbors_to_explore:
                if neighbor_id not in level_visited:
                    level_visited.add(neighbor_id)
                    neighbor = self._get_vector(neighbor_id)
                    if neighbor:
                        dist = self._get_distance(query_data.id, neighbor_id, query_data, neighbor)
                        if dist != float('inf'):
                            # Add to W if better than worst in W
                            if len(W) < ef or dist < (-W[0][0]):
                                heapq.heappush(W, (dist, neighbor_id))
                            
                            # Add to result set
                            result.append((dist, neighbor_id))
                            result.sort()
                            if len(result) > ef:
                                result.pop()
        
        # Return M closest neighbors
        return [id for _, id in result[:M]] if result else []
    
    def search(self, query: np.ndarray, k: int = 1) -> List[Tuple[float, int]]:
        """Search with improved beam search and better exploration."""
        self._validate_vector(query)
        
        # Get entry point
        redis_client = self.conn_manager.get_redis()
        try:
            entry_point = int(redis_client.get("hnsw:entry_point") or -1)
            if entry_point == -1:
                logger.warning("No entry point found. Graph might be empty.")
                return []
        finally:
            self.conn_manager.release_redis(redis_client)
        
        # Create temporary query vector
        query_data = VectorData(-1, query.copy())
        
        # Get entry point vector
        ep_vector = self._get_vector(entry_point)
        if not ep_vector:
            logger.error(f"Failed to retrieve entry point vector {entry_point}")
            return []
        
        # Initialize visited set
        visited = {-1}
        
        # Start with entry point
        ep_dist = self._get_distance(-1, entry_point, query_data, ep_vector)
        current_neighbors = [(ep_dist, entry_point)]
        
        # Track best candidates across all levels
        best_candidates = []
        
        # Search through levels
        for level in range(ep_vector.level, -1, -1):
            logger.debug(f"Searching level {level}")
            
            # Get candidates at current level
            candidates = []
            for _, vid in current_neighbors:
                v = self._get_vector(vid)
                if v:
                    # Consider neighbors from current and upper levels
                    for l in range(level, v.level + 1):
                        if l in v.neighbors:
                            for nid in v.neighbors[l]:
                                if nid not in visited:
                                    neighbor = self._get_vector(nid)
                                    if neighbor:
                                        candidates.append(neighbor)
                                        visited.add(nid)
            
            if not candidates and level > 0:
                logger.debug(f"No candidates at level {level}, continuing with current neighbors")
                continue
                
            # Select neighbors using beam search
            neighbor_ids = self._select_neighbors(
                query_data,
                candidates,
                self.ef if level == 0 else self.M,
                level,
                max(self.ef, k),  # Use larger ef for better recall
                visited
            )
            
            # Update current neighbors and track best candidates
            current_neighbors = []
            for vid in neighbor_ids:
                dist = self._get_distance(-1, vid, query_data)
                current_neighbors.append((dist, vid))
                best_candidates.append((dist, vid))
            
            current_neighbors.sort()  # Keep sorted for efficiency
        
        # Return k nearest neighbors from all discovered candidates
        return heapq.nsmallest(k, best_candidates)
    
    def _store_vector(self, vector_data: VectorData):
        """Store vector with proper error handling."""
        try:
            # Store in MinIO
            minio_client = self.conn_manager.get_minio()
            try:
                data = pickle.dumps(vector_data)
                data_stream = io.BytesIO(data)
                minio_client.put_object(
                    self.bucket_name,
                    f"vector_{vector_data.id}",
                    data_stream,
                    len(data)
                )
            finally:
                self.conn_manager.release_minio(minio_client)
            
            # Store metadata in Redis
            redis_client = self.conn_manager.get_redis()
            try:
                key = f"vector:{vector_data.id}"
                redis_client.hset(key, mapping={
                    "dimension": self.dimension,
                    "level": vector_data.level,
                    "neighbors": json.dumps(dict(vector_data.neighbors))
                })
            finally:
                self.conn_manager.release_redis(redis_client)
        except Exception as e:
            logger.error(f"Error storing vector {vector_data.id}: {e}")
            raise
    
    def _get_vector(self, vector_id: int) -> Optional[VectorData]:
        """Get vector with proper error handling."""
        try:
            minio_client = self.conn_manager.get_minio()
            try:
                data = minio_client.get_object(
                    self.bucket_name,
                    f"vector_{vector_id}"
                ).read()
                return pickle.loads(data)
            finally:
                self.conn_manager.release_minio(minio_client)
        except Exception as e:
            logger.error(f"Error getting vector {vector_id}: {e}")
            return None
    
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

    def insert(self, vector_id: int, vector: np.ndarray) -> bool:
        """Insert a vector with improved level generation and connection management."""
        try:
            self._validate_vector(vector)
            
            # Check if vector already exists
            redis_client = self.conn_manager.get_redis()
            try:
                if redis_client.sismember("hnsw:vectors", vector_id):
                    logger.warning(f"Vector {vector_id} already exists")
                    return False
            finally:
                self.conn_manager.release_redis(redis_client)
            
            # Generate random level
            level = self._generate_random_level()
            logger.debug(f"Generated level {level} for vector {vector_id}")
            
            # Create vector data
            vector_data = VectorData(vector_id, vector.copy(), level=level)
            
            # Get entry point
            redis_client = self.conn_manager.get_redis()
            try:
                entry_point = int(redis_client.get("hnsw:entry_point") or -1)
                
                # If this is the first vector, make it the entry point
                if entry_point == -1:
                    self._store_vector(vector_data)
                    redis_client.set("hnsw:entry_point", vector_id)
                    redis_client.sadd("hnsw:vectors", vector_id)
                    redis_client.incr("hnsw:vector_count")
                    logger.info(f"Inserted first vector {vector_id} as entry point")
                    return True
                    
                # Get entry point vector
                ep_vector = self._get_vector(entry_point)
                if not ep_vector:
                    logger.error(f"Failed to retrieve entry point vector {entry_point}")
                    return False
                    
                # Update entry point if necessary
                if level > ep_vector.level:
                    redis_client.set("hnsw:entry_point", vector_id)
                    logger.debug(f"Updated entry point to {vector_id}")
                    
                # Insert vector into graph
                self._insert_vector_to_graph(vector_data, entry_point)
                
                # Store vector and update metadata
                self._store_vector(vector_data)
                redis_client.sadd("hnsw:vectors", vector_id)
                redis_client.incr("hnsw:vector_count")
                
                logger.info(f"Successfully inserted vector {vector_id} at level {level}")
                return True
                
            finally:
                self.conn_manager.release_redis(redis_client)
                
        except Exception as e:
            logger.error(f"Failed to insert vector {vector_id}: {str(e)}")
            return False

    def _insert_vector_to_graph(self, vector_data: VectorData, entry_point: int):
        """Insert vector into graph with improved neighbor selection."""
        current_neighbors = []
        
        # Get entry point vector
        ep_vector = self._get_vector(entry_point)
        if not ep_vector:
            return
            
        # Initialize with entry point
        ep_dist = self._get_distance(vector_data.id, entry_point, vector_data, ep_vector)
        current_neighbors = [(ep_dist, entry_point)]
        
        # Insert from top to bottom
        for level in range(min(vector_data.level, ep_vector.level), -1, -1):
            # Get candidates at current level
            candidates = []
            visited = {vector_data.id}
            
            # Get candidates from current neighbors
            for _, vid in current_neighbors:
                v = self._get_vector(vid)
                if v and level in v.neighbors:
                    for nid in v.neighbors[level]:
                        if nid not in visited:
                            neighbor = self._get_vector(nid)
                            if neighbor:
                                candidates.append(neighbor)
                                visited.add(nid)
            
            # Select and connect to neighbors
            neighbor_ids = self._select_neighbors(
                vector_data,
                candidates,
                self.M if level > 0 else self.M * 2,  # Use more connections at bottom level
                level,
                self.ef_construction,
                visited
            )
            
            # Create bidirectional connections
            vector_data.neighbors[level] = []  # Clear existing neighbors at this level
            for neighbor_id in neighbor_ids:
                neighbor = self._get_vector(neighbor_id)
                if neighbor:
                    # Add forward connection
                    vector_data.neighbors[level].append(neighbor_id)
                    
                    # Add backward connection with pruning
                    if level not in neighbor.neighbors:
                        neighbor.neighbors[level] = []
                
                    # Add bidirectional connection if not already present
                    if vector_data.id not in neighbor.neighbors[level]:
                        neighbor.neighbors[level].append(vector_data.id)
                        
                        # Prune neighbor's connections if needed
                        if len(neighbor.neighbors[level]) > self.M * 2:
                            # Get all neighbors including the new one
                            all_neighbors = []
                            for nid in neighbor.neighbors[level]:
                                n = self._get_vector(nid)
                                if n:
                                    dist = self._get_distance(neighbor.id, nid, neighbor, n)
                                    all_neighbors.append((dist, nid))
                            
                            # Keep only the closest M*2 neighbors
                            all_neighbors.sort()
                            neighbor.neighbors[level] = [nid for _, nid in all_neighbors[:self.M * 2]]
                        
                        # Store updated neighbor
                        self._store_vector(neighbor)
            
            # Update current neighbors for next level
            current_neighbors = [(self._get_distance(vector_data.id, vid), vid) for vid in neighbor_ids]
            current_neighbors.sort()  # Keep sorted for efficiency

    def _generate_random_level(self) -> int:
        """Generate random level using exponential distribution."""
        return int(-math.log(random.random()) * self.ml)
