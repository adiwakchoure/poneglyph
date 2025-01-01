"""
Performance testing framework for HNSW vector search with accuracy metrics.
"""
import sys
import os
import numpy as np
import time
import json
import logging
import psutil
import threading
from typing import Dict, List, Tuple, Any, Set
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sklearn.neighbors import NearestNeighbors
import redis
from minio import Minio

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.hnsw.core import HNSW

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BruteForceKNN:
    """Ground truth nearest neighbor search using scikit-learn."""
    
    def __init__(self, vectors: List[np.ndarray]):
        """Initialize brute force KNN."""
        self.vectors = np.array(vectors)
        self.knn = NearestNeighbors(
            n_neighbors=max(1, min(100, len(vectors))),
            algorithm='brute',
            metric='euclidean'
        )
        self.knn.fit(self.vectors)
    
    def search(self, query: np.ndarray, k: int) -> List[Tuple[float, int]]:
        """Find k nearest neighbors."""
        distances, indices = self.knn.kneighbors(
            query.reshape(1, -1),
            n_neighbors=min(k, len(self.vectors))
        )
        return list(zip(distances[0], indices[0]))

class AccuracyMetrics:
    """Compute accuracy metrics for approximate nearest neighbor search."""
    
    @staticmethod
    def compute_recall(
        ground_truth: List[Set[int]],
        approximate: List[Set[int]]
    ) -> float:
        """Compute recall@k."""
        if not ground_truth or not approximate:
            return 0.0
        
        recalls = []
        for gt, ap in zip(ground_truth, approximate):
            if not gt:
                continue
            intersection = len(gt.intersection(ap))
            recalls.append(intersection / len(gt))
        
        return float(np.mean(recalls)) if recalls else 0.0
    
    @staticmethod
    def compute_precision(
        ground_truth: List[Set[int]],
        approximate: List[Set[int]]
    ) -> float:
        """Compute precision@k."""
        if not ground_truth or not approximate:
            return 0.0
        
        precisions = []
        for gt, ap in zip(ground_truth, approximate):
            if not ap:
                continue
            intersection = len(gt.intersection(ap))
            precisions.append(intersection / len(ap))
        
        return float(np.mean(precisions)) if precisions else 0.0

class PerformanceTest:
    """Performance testing framework for HNSW."""
    
    def __init__(
        self,
        dimension: int = 128,
        max_elements: int = 100000,
        M: int = 16,
        ef_construction: int = 200,
        ef: int = 40,
        n_test_queries: int = 100,
        k_values: List[int] = None,
        test_sizes: List[int] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        minio_endpoint: str = "localhost:9000",
        minio_access_key: str = "minioadmin",
        minio_secret_key: str = "minioadmin"
    ):
        """Initialize test parameters."""
        self.dimension = dimension
        self.max_elements = max_elements
        self.M = M
        self.ef_construction = ef_construction
        self.ef = ef
        self.n_test_queries = n_test_queries
        self.k_values = k_values or [1, 5, 10, 20, 50]
        self.test_sizes = test_sizes or [1000]
        
        # Storage configuration
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.minio_endpoint = minio_endpoint
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        
        # Initialize Redis client
        self.redis = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            decode_responses=True
        )
        
        # Initialize MinIO client
        self.minio = Minio(
            self.minio_endpoint,
            access_key=self.minio_access_key,
            secret_key=self.minio_secret_key,
            secure=False
        )
        
        # Ensure MinIO bucket exists
        bucket_name = f"vectors-{dimension}"
        try:
            if not self.minio.bucket_exists(bucket_name):
                self.minio.make_bucket(bucket_name)
        except Exception as e:
            logger.error(f"Failed to initialize MinIO bucket: {str(e)}")
            raise
    
    def _cleanup_storage(self):
        """Clean up Redis and MinIO storage."""
        try:
            # Clean Redis
            self.redis.flushdb()
            
            # Clean MinIO
            bucket_name = f"vectors-{self.dimension}"
            try:
                objects = self.minio.list_objects(bucket_name)
                for obj in objects:
                    self.minio.remove_object(bucket_name, obj.object_name)
            except Exception as e:
                logger.error(f"Failed to clean MinIO bucket: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to cleanup storage: {str(e)}")
    
    def _compute_stats(self, values: List[float]) -> Dict[str, float]:
        """Compute statistics."""
        if not values:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0
            }
        
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99))
        }
    
    def run_tests(self) -> Dict[str, Any]:
        """Run performance tests with proper cleanup."""
        try:
            # Clean up before tests
            self._cleanup_storage()
            
            results = {}
            for size in self.test_sizes:
                logger.info(f"Running test with {size} vectors")
                
                # Generate test data
                vectors = np.random.rand(size, self.dimension).astype(np.float32)
                queries = np.random.rand(self.n_test_queries, self.dimension).astype(np.float32)
                
                # Initialize HNSW
                index = HNSW(
                    dimension=self.dimension,
                    max_elements=self.max_elements,
                    M=self.M,
                    ef_construction=self.ef_construction,
                    ef=self.ef,
                    redis_host=self.redis_host,
                    redis_port=self.redis_port,
                    minio_endpoint=self.minio_endpoint,
                    minio_access_key=self.minio_access_key,
                    minio_secret_key=self.minio_secret_key
                )
                
                # Measure insertion time
                insertion_times = []
                for i in tqdm(range(size), desc="Inserting vectors"):
                    start_time = time.time()
                    success = index.insert(i, vectors[i])
                    if not success:
                        logger.error(f"Failed to insert vector {i}")
                    insertion_times.append(time.time() - start_time)
                
                # Initialize ground truth
                ground_truth = BruteForceKNN(vectors)
                
                # Test results for this size
                size_results = {
                    "insertion": self._compute_stats(insertion_times),
                    "search": {},
                    "accuracy": {},
                    "memory": {"mean": 0, "max": 0},
                    "cpu": {"mean": 0, "max": 0},
                    "errors": []
                }
                
                # Test each k value
                for k in self.k_values:
                    logger.info(f"Testing k={k}")
                    
                    # Measure search time and accuracy
                    search_times = []
                    gt_neighbors = []
                    ap_neighbors = []
                    
                    for query in tqdm(queries, desc=f"Searching with k={k}"):
                        # Get ground truth
                        gt_results = ground_truth.search(query, k)
                        gt_neighbors.append(set(idx for _, idx in gt_results))
                        
                        # Get approximate results
                        start_time = time.time()
                        ap_results = index.search(query, k)
                        search_times.append(time.time() - start_time)
                        ap_neighbors.append(set(idx for _, idx in ap_results))
                    
                    # Compute metrics
                    size_results["search"][k] = self._compute_stats(search_times)
                    size_results["accuracy"][k] = {
                        "recall": AccuracyMetrics.compute_recall(gt_neighbors, ap_neighbors),
                        "precision": AccuracyMetrics.compute_precision(gt_neighbors, ap_neighbors)
                    }
                
                # Monitor resource usage
                process = psutil.Process()
                size_results["memory"]["mean"] = process.memory_info().rss / 1024 / 1024  # MB
                size_results["memory"]["max"] = size_results["memory"]["mean"]
                size_results["cpu"]["mean"] = process.cpu_percent()
                size_results["cpu"]["max"] = size_results["cpu"]["mean"]
                
                results[size] = size_results
            
            return results
            
        finally:
            # Clean up after tests
            self._cleanup_storage()

def main():
    """Run performance tests."""
    # Test configuration
    config = {
        "dimension": 128,
        "max_elements": 100000,
        "M": 16,
        "ef_construction": 200,
        "ef": 40,
        "n_test_queries": 100,
        "k_values": [1, 5, 10, 20, 50],
        "test_sizes": [1000],
        "redis_host": "localhost",
        "redis_port": 6379,
        "minio_endpoint": "localhost:9000",
        "minio_access_key": "minioadmin",
        "minio_secret_key": "minioadmin"
    }
    
    # Create and run tests
    test = PerformanceTest(**config)
    results = test.run_tests()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = Path(f"hnsw_performance_{timestamp}.json")
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {result_file}")

if __name__ == "__main__":
    main()
