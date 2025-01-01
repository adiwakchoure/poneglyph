"""
Core implementation of HNSW (Hierarchical Navigable Small World) algorithm.
"""
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import heapq
import threading

@dataclass
class Node:
    id: int
    vector: np.ndarray
    neighbors: Dict[int, Set[int]]  # layer -> set of neighbor ids

class HNSW:
    def __init__(
        self,
        dim: int,
        M: int = 16,  # Max number of connections per layer
        ef_construction: int = 200,  # Size of dynamic candidate list
        num_layers: int = 4,
        distance_func: str = "l2"
    ):
        self.dim = dim
        self.M = M
        self.M0 = 2 * M  # More connections for ground layer
        self.ef_construction = ef_construction
        self.num_layers = num_layers
        self.nodes: Dict[int, Node] = {}
        self.entry_point = None
        self.lock = threading.Lock()
        
        if distance_func == "l2":
            self.distance = self._l2_distance
        elif distance_func == "cosine":
            self.distance = self._cosine_distance
        else:
            raise ValueError("Unsupported distance function")

    def _l2_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.linalg.norm(a - b)

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _get_random_level(self) -> int:
        level = 0
        while np.random.random() < 0.5 and level < self.num_layers - 1:
            level += 1
        return level

    def _search_layer(
        self,
        query: np.ndarray,
        entry: int,
        ef: int,
        layer: int
    ) -> List[Tuple[float, int]]:
        visited = {entry}
        candidates = [(self.distance(query, self.nodes[entry].vector), entry)]
        heapq.heapify(candidates)
        results = candidates.copy()

        while candidates:
            dist, current = heapq.heappop(candidates)
            furthest_dist = results[0][0]  # Results are in reverse order
            
            if dist > furthest_dist:
                break
                
            for neighbor in self.nodes[current].neighbors.get(layer, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist = self.distance(query, self.nodes[neighbor].vector)
                    
                    if len(results) < ef or dist < -results[0][0]:
                        heapq.heappush(candidates, (dist, neighbor))
                        heapq.heappush(results, (-dist, neighbor))
                        
                        if len(results) > ef:
                            heapq.heappop(results)

        return [(-dist, idx) for dist, idx in sorted(results)]

    def add(self, vector_id: int, vector: np.ndarray) -> None:
        """Add a new vector to the index."""
        with self.lock:
            vector = np.array(vector, dtype=np.float32)
            if len(vector.shape) != 1 or vector.shape[0] != self.dim:
                raise ValueError(f"Vector must have shape ({self.dim},)")

            level = self._get_random_level()
            node = Node(id=vector_id, vector=vector, neighbors={})

            if not self.entry_point:
                self.entry_point = vector_id
                self.nodes[vector_id] = node
                return

            # Find entry points for all layers
            curr_node = self.entry_point
            curr_dist = self.distance(vector, self.nodes[curr_node].vector)
            
            for layer in range(self.num_layers - 1, -1, -1):
                changed = True
                while changed:
                    changed = False
                    neighbors = self.nodes[curr_node].neighbors.get(layer, set())
                    
                    for neighbor in neighbors:
                        dist = self.distance(vector, self.nodes[neighbor].vector)
                        if dist < curr_dist:
                            curr_node = neighbor
                            curr_dist = dist
                            changed = True
                
                if layer <= level:
                    # Add connections for this layer
                    candidates = self._search_layer(
                        vector, curr_node, self.ef_construction, layer
                    )
                    
                    # Select M nearest neighbors
                    M = self.M0 if layer == 0 else self.M
                    selected = candidates[:M]
                    
                    # Add bidirectional connections
                    node.neighbors[layer] = set(idx for _, idx in selected)
                    for _, idx in selected:
                        if layer not in self.nodes[idx].neighbors:
                            self.nodes[idx].neighbors[layer] = set()
                        self.nodes[idx].neighbors[layer].add(vector_id)

            self.nodes[vector_id] = node
            
            # Update entry point if necessary
            if level > len(self.nodes[self.entry_point].neighbors):
                self.entry_point = vector_id

    def search(self, query: np.ndarray, k: int = 1, ef: int = 50) -> List[Tuple[float, int]]:
        """Search for k nearest neighbors of the query vector."""
        query = np.array(query, dtype=np.float32)
        if len(query.shape) != 1 or query.shape[0] != self.dim:
            raise ValueError(f"Query must have shape ({self.dim},)")

        if not self.entry_point:
            return []

        curr_node = self.entry_point
        
        # Search through layers
        for layer in range(len(self.nodes[self.entry_point].neighbors), -1, -1):
            candidates = self._search_layer(query, curr_node, 1, layer)
            curr_node = candidates[0][1]

        # Search bottom layer more thoroughly
        return self._search_layer(query, curr_node, ef, 0)[:k]
