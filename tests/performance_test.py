"""
Performance testing script for the vector search service.
"""
import numpy as np
import requests
import time
from typing import List, Tuple
import concurrent.futures
import matplotlib.pyplot as plt

def generate_random_vectors(n: int, dim: int) -> List[np.ndarray]:
    """Generate n random vectors of dimension dim."""
    return [np.random.randn(dim).astype(np.float32) for _ in range(n)]

def test_insertion(vectors: List[np.ndarray]) -> List[Tuple[int, float]]:
    """Test vector insertion performance."""
    results = []
    url = "http://localhost:8000/vectors/"
    
    for i, vector in enumerate(vectors):
        start_time = time.time()
        response = requests.post(url, json={
            "id": i,
            "vector": vector.tolist()
        })
        end_time = time.time()
        
        if response.status_code == 201:
            results.append((i, end_time - start_time))
        else:
            print(f"Error inserting vector {i}: {response.text}")
    
    return results

def test_search(query_vectors: List[np.ndarray], k: int = 10) -> List[float]:
    """Test search performance."""
    results = []
    url = "http://localhost:8000/search/"
    
    for vector in query_vectors:
        start_time = time.time()
        response = requests.post(url, json={
            "vector": vector.tolist(),
            "k": k
        })
        end_time = time.time()
        
        if response.status_code == 200:
            results.append(end_time - start_time)
        else:
            print(f"Error in search: {response.text}")
    
    return results

def plot_results(insert_times: List[Tuple[int, float]], search_times: List[float]):
    """Plot performance results."""
    # Insertion time plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot([x[0] for x in insert_times], [x[1] for x in insert_times])
    plt.title("Insertion Time vs Vector Count")
    plt.xlabel("Vector Count")
    plt.ylabel("Time (seconds)")
    
    # Search time plot
    plt.subplot(1, 2, 2)
    plt.boxplot(search_times)
    plt.title("Search Time Distribution")
    plt.ylabel("Time (seconds)")
    
    plt.tight_layout()
    plt.savefig("performance_results.png")
    plt.close()

def main():
    # Test parameters
    n_vectors = 1000
    vector_dim = 128
    n_queries = 100
    
    print(f"Generating {n_vectors} random vectors...")
    vectors = generate_random_vectors(n_vectors, vector_dim)
    
    print("Testing insertion performance...")
    insert_results = test_insertion(vectors)
    
    print("Testing search performance...")
    query_vectors = generate_random_vectors(n_queries, vector_dim)
    search_results = test_search(query_vectors)
    
    print("\nResults Summary:")
    print(f"Average insertion time: {np.mean([x[1] for x in insert_results]):.4f} seconds")
    print(f"Average search time: {np.mean(search_results):.4f} seconds")
    
    print("\nGenerating performance plots...")
    plot_results(insert_results, search_results)
    print("Results saved to performance_results.png")

if __name__ == "__main__":
    main()
