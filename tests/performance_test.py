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
import urllib3

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

class HNSWPerformanceTest:
    """Test HNSW index performance."""
    
    def __init__(self):
        """Initialize test parameters."""
        # Index parameters
        self.dimension = 512
        self.max_elements = 10000
        self.M = 16
        self.ef_construction = 200
        self.ef = 50
        
        # Test parameters
        self.test_sizes = [100]  # Start with smaller sizes
        self.k_values = [1, 10, 50]
        self.n_test_queries = 10
        
        # Storage parameters
        self.redis_host = "localhost"
        self.redis_port = 6379
        self.minio_endpoint = "localhost:9000"
        
    def _cosine_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine distance between vectors."""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return float('inf')
        
        similarity = dot_product / (norm_v1 * norm_v2)
        distance = 1 - similarity
        
        return float(distance)
    
    def _get_true_neighbors(self, query: np.ndarray, vectors: List[np.ndarray], k: int) -> Set[int]:
        """Get true neighbors for a query."""
        distances = [(self._cosine_distance(query, vector), i) for i, vector in enumerate(vectors)]
        distances.sort()
        return set(i for _, i in distances[:k])

    def run_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        results = {}
        
        # Run tests for each size
        for size in self.test_sizes:
            logger.info(f"Running test with {size} vectors")
            
            # Initialize HNSW index
            index = HNSW(
                dimension=self.dimension,
                max_elements=self.max_elements,
                M=self.M,
                ef_construction=self.ef_construction,
                ef=self.ef,
                redis_host=self.redis_host,
                redis_port=self.redis_port,
                minio_endpoint=self.minio_endpoint,
                minio_access_key="minioadmin",
                minio_secret_key="minioadmin"
            )
            
            # Clean up storage first
            index._cleanup_storage()
            
            try:
                # Generate random vectors
                vectors = []
                for i in tqdm(range(size), desc="Inserting vectors"):
                    vector = np.random.randn(self.dimension).astype(np.float32)
                    vector = vector / np.linalg.norm(vector)  # Normalize
                    vectors.append(vector)
                    index.insert(i, vector)
                
                # Ensure all vectors are stored
                index._flush_vector_batch()
                
                # Test each k value
                for k in self.k_values:
                    logger.info(f"Testing k={k}")
                    
                    # Run search queries
                    search_times = []
                    recall_rates = []
                    
                    for _ in tqdm(range(self.n_test_queries), desc=f"Searching with k={k}"):
                        # Generate random query
                        query = np.random.randn(self.dimension).astype(np.float32)
                        query = query / np.linalg.norm(query)
                        
                        # Measure search time
                        start_time = time.time()
                        results_k = index.search(query, k)
                        search_time = time.time() - start_time
                        search_times.append(search_time)
                        
                        # Calculate recall
                        true_neighbors = self._get_true_neighbors(query, vectors, k)
                        found_neighbors = {r[1] for r in results_k}
                        recall = len(found_neighbors.intersection(true_neighbors)) / k
                        recall_rates.append(recall)
                    
                    # Store results
                    results[f"size={size},k={k}"] = {
                        "avg_search_time": np.mean(search_times),
                        "avg_recall": np.mean(recall_rates)
                    }
                    
                    # Log results
                    logger.info(f"Results for size={size}, k={k}:")
                    logger.info(f"Average search time: {results[f'size={size},k={k}']['avg_search_time']:.6f} seconds")
                    logger.info(f"Average recall: {results[f'size={size},k={k}']['avg_recall']:.6f}")
            
            finally:
                # Clean up after test
                index._cleanup_storage()
        
        return results

def main():
    """Run performance tests."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    test = HNSWPerformanceTest()
    results = test.run_tests()
    
    # Print final results
    print("\nFinal Results:")
    for test_key, test_results in results.items():
        print(f"\n{test_key}:")
        print(f"Average search time: {test_results['avg_search_time']:.6f} seconds")
        print(f"Average recall: {test_results['avg_recall']:.6f}")

if __name__ == "__main__":
    main()
