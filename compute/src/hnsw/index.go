package hnsw

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
)

const (
	defaultMaxLevel    = 16
	defaultLevelMult   = 1.0 / math.Ln2 // Using Ln2 instead of Log(2)
	defaultEfConstruction = 100
	defaultM            = 10  // Maximum number of connections per layer
	defaultEfSearch     = 50  // Size of dynamic candidate list for search
)

// minInt returns the minimum of two integers
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Node represents a single point in the HNSW graph
type Node struct {
	ID        uint64
	Vector    []float32
	Neighbors map[int][]uint64 // level -> neighbors
	mutex     sync.RWMutex
}

// HNSWIndex represents the main index structure
type HNSWIndex struct {
	maxLevel        int
	levelMult       float64
	efConstruction  int
	m              int     // Maximum number of connections per layer
	efSearch       int     // Size of dynamic candidate list for search
	entryPoint     *Node
	nodes          map[uint64]*Node
	mutex          sync.RWMutex
	dimension      int
}

// NewHNSWIndex creates a new HNSW index
func NewHNSWIndex(dimension int) *HNSWIndex {
	return &HNSWIndex{
		maxLevel:       defaultMaxLevel,
		levelMult:      defaultLevelMult,
		efConstruction: defaultEfConstruction,
		m:             defaultM,
		efSearch:      defaultEfSearch,
		nodes:         make(map[uint64]*Node),
		dimension:     dimension,
	}
}

// generateLevel generates a random level for a new node
func (h *HNSWIndex) generateLevel() int {
	level := int(-math.Log(rand.Float64()) * h.levelMult)
	if level > h.maxLevel {
		level = h.maxLevel
	}
	return level
}

// distance calculates Euclidean distance between two vectors
func distance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return float32(math.Sqrt(float64(sum)))
}

// searchLayer performs a greedy search within a single layer
func (h *HNSWIndex) searchLayer(query []float32, entryPoint *Node, ef int, level int) []*item {
	visited := make(map[uint64]bool)
	candidates := &candidateQueue{}
	heap.Init(candidates)

	// Initialize with entry point
	entryDist := distance(query, entryPoint.Vector)
	heap.Push(candidates, &item{
		node:     entryPoint,
		distance: entryDist,
	})

	results := &resultQueue{}
	heap.Init(results)
	heap.Push(results, &item{
		node:     entryPoint,
		distance: entryDist,
	})

	visited[entryPoint.ID] = true

	for candidates.Len() > 0 {
		current := heap.Pop(candidates).(*item)
		furthestDist := float32(0)
		if results.Len() > 0 {
			furthestDist = (*results)[0].distance // Get max distance from result queue
		}

		if current.distance > furthestDist && results.Len() >= ef {
			break
		}

		// Check neighbors at current level
		current.node.mutex.RLock()
		neighbors := current.node.Neighbors[level]
		current.node.mutex.RUnlock()

		for _, neighborID := range neighbors {
			if visited[neighborID] {
				continue
			}
			visited[neighborID] = true

			neighbor := h.nodes[neighborID]
			if neighbor == nil {
				continue
			}

			dist := distance(query, neighbor.Vector)
			neighborItem := &item{node: neighbor, distance: dist}
			
			// Add to results if it's closer than the furthest result
			if results.Len() < ef {
				heap.Push(results, neighborItem)
				heap.Push(candidates, neighborItem)
			} else if dist < furthestDist {
				heap.Pop(results) // Remove furthest result
				heap.Push(results, neighborItem)
				heap.Push(candidates, neighborItem)
			}
		}
	}

	// Convert results to slice
	resultItems := make([]*item, results.Len())
	for i := len(resultItems) - 1; i >= 0; i-- {
		resultItems[i] = heap.Pop(results).(*item)
	}

	return resultItems
}

// Search finds k nearest neighbors for the query vector
func (h *HNSWIndex) Search(query []float32, k int) ([]uint64, []float32, error) {
	if len(query) != h.dimension {
		return nil, nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", h.dimension, len(query))
	}

	h.mutex.RLock()
	entryPoint := h.entryPoint
	if entryPoint == nil {
		h.mutex.RUnlock()
		return nil, nil, fmt.Errorf("empty index")
	}
	maxLevel := h.getMaxLevel()
	h.mutex.RUnlock()

	// Find entry point for search
	currObj := entryPoint
	currDist := distance(query, currObj.Vector)

	// Search from top to bottom to find good entry point
	for level := maxLevel; level > 0; level-- {
		changed := true
		for changed {
			changed = false

			currObj.mutex.RLock()
			neighbors := currObj.Neighbors[level]
			currObj.mutex.RUnlock()

			for _, neighborID := range neighbors {
				neighbor := h.nodes[neighborID]
				if neighbor == nil {
					continue
				}

				dist := distance(query, neighbor.Vector)
				if dist < currDist {
					currDist = dist
					currObj = neighbor
					changed = true
				}
			}
		}
	}

	// Get ef nearest elements from the lowest layer
	ef := k * 2 // Use a larger ef value to explore more candidates
	candidates := h.searchLayer(query, currObj, ef, 0)
	if len(candidates) == 0 {
		return nil, nil, fmt.Errorf("no results found")
	}

	// Sort candidates by distance
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].distance < candidates[j].distance
	})

	// Take k closest neighbors
	numResults := minInt(k, len(candidates))
	ids := make([]uint64, numResults)
	distances := make([]float32, numResults)
	for i := 0; i < numResults; i++ {
		ids[i] = candidates[i].node.ID
		distances[i] = candidates[i].distance
	}

	return ids, distances, nil
}

// Insert adds a new vector to the index
func (h *HNSWIndex) Insert(id uint64, vector []float32) error {
	if len(vector) != h.dimension {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", h.dimension, len(vector))
	}

	node := &Node{
		ID:        id,
		Vector:    vector,
		Neighbors: make(map[int][]uint64),
	}

	h.mutex.Lock()
	if h.entryPoint == nil {
		h.entryPoint = node
		h.nodes[id] = node
		h.mutex.Unlock()
		return nil
	}
	h.mutex.Unlock()

	// Generate random level
	level := h.generateLevel()

	h.mutex.RLock()
	maxLevel := h.getMaxLevel()
	currObj := h.entryPoint
	h.mutex.RUnlock()

	// Find entry point
	currDist := distance(vector, currObj.Vector)

	// Search from top to bottom
	for lc := maxLevel; lc > level; lc-- {
		changed := true
		for changed {
			changed = false

			currObj.mutex.RLock()
			neighbors := currObj.Neighbors[lc]
			currObj.mutex.RUnlock()

			for _, neighborID := range neighbors {
				neighbor := h.nodes[neighborID]
				if neighbor == nil {
					continue
				}

				dist := distance(vector, neighbor.Vector)
				if dist < currDist {
					currDist = dist
					currObj = neighbor
					changed = true
				}
			}
		}
	}

	// Insert connections for all levels from 0 to level
	for lc := minInt(level, maxLevel); lc >= 0; lc-- {
		// Find neighbors at current level
		candidates := h.searchLayer(vector, currObj, h.efConstruction, lc)

		// Select M closest neighbors
		maxNeighbors := h.m
		if lc == 0 {
			maxNeighbors = h.m * 2 // Use more connections at level 0
		}

		selectedNeighbors := make([]uint64, 0, maxNeighbors)
		for i := 0; i < len(candidates) && i < maxNeighbors; i++ {
			selectedNeighbors = append(selectedNeighbors, candidates[i].node.ID)
		}

		// Add bidirectional connections
		node.mutex.Lock()
		node.Neighbors[lc] = selectedNeighbors
		node.mutex.Unlock()

		for _, neighborID := range selectedNeighbors {
			neighbor := h.nodes[neighborID]
			if neighbor == nil {
				continue
			}

			neighbor.mutex.Lock()
			if neighbor.Neighbors[lc] == nil {
				neighbor.Neighbors[lc] = make([]uint64, 0, maxNeighbors)
			}

			// Add connection to new node
			neighbor.Neighbors[lc] = append(neighbor.Neighbors[lc], id)

			// Ensure neighbor doesn't have too many connections
			if len(neighbor.Neighbors[lc]) > maxNeighbors {
				// Keep only the closest neighbors
				candidates := make([]*item, 0, len(neighbor.Neighbors[lc]))
				for _, nID := range neighbor.Neighbors[lc] {
					n := h.nodes[nID]
					if n != nil {
						candidates = append(candidates, &item{
							node:     n,
							distance: distance(neighbor.Vector, n.Vector),
						})
					}
				}

				// Sort by distance
				sort.Slice(candidates, func(i, j int) bool {
					return candidates[i].distance < candidates[j].distance
				})

				// Keep only the closest neighbors
				newNeighbors := make([]uint64, 0, maxNeighbors)
				for i := 0; i < maxNeighbors && i < len(candidates); i++ {
					newNeighbors = append(newNeighbors, candidates[i].node.ID)
				}
				neighbor.Neighbors[lc] = newNeighbors
			}
			neighbor.mutex.Unlock()
		}
	}

	h.mutex.Lock()
	h.nodes[id] = node
	if level > maxLevel {
		h.entryPoint = node
	}
	h.mutex.Unlock()

	return nil
}

// Delete removes a vector from the index
func (h *HNSWIndex) Delete(id uint64, vector []float32) error {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	node, exists := h.nodes[id]
	if !exists {
		return fmt.Errorf("node with ID %d not found", id)
	}

	// Remove all connections to this node
	for level := range node.Neighbors {
		// Remove connections from neighbors to this node
		for _, neighborID := range node.Neighbors[level] {
			if neighbor, ok := h.nodes[neighborID]; ok {
				neighbor.mutex.Lock()
				newNeighbors := make([]uint64, 0)
				for _, nID := range neighbor.Neighbors[level] {
					if nID != id {
						newNeighbors = append(newNeighbors, nID)
					}
				}
				neighbor.Neighbors[level] = newNeighbors
				neighbor.mutex.Unlock()
			}
		}
	}

	// If this node is the entry point, find a new entry point
	if h.entryPoint == node {
		if len(h.nodes) > 1 {
			// Find another node with the highest level
			maxLevel := -1
			var newEntryPoint *Node
			for _, n := range h.nodes {
				if n.ID != id && len(n.Neighbors) > maxLevel {
					maxLevel = len(n.Neighbors)
					newEntryPoint = n
				}
			}
			h.entryPoint = newEntryPoint
		} else {
			h.entryPoint = nil
		}
	}

	// Remove the node from the index
	delete(h.nodes, id)

	return nil
}

func (h *HNSWIndex) getMaxLevel() int {
	h.mutex.RLock()
	defer h.mutex.RUnlock()

	if h.entryPoint == nil {
		return 0
	}
	return len(h.entryPoint.Neighbors) - 1
}

// Helper types and functions for priority queues
type item struct {
	node     *Node
	distance float32
}

// candidateQueue is a min-heap (closest neighbors first)
type candidateQueue []*item

func (pq candidateQueue) Len() int           { return len(pq) }
func (pq candidateQueue) Less(i, j int) bool { return pq[i].distance < pq[j].distance }
func (pq candidateQueue) Swap(i, j int)      { pq[i], pq[j] = pq[j], pq[i] }

func (pq *candidateQueue) Push(x interface{}) {
	*pq = append(*pq, x.(*item))
}

func (pq *candidateQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	*pq = old[0 : n-1]
	return item
}

// resultQueue is a max-heap (furthest neighbors first)
type resultQueue []*item

func (pq resultQueue) Len() int           { return len(pq) }
func (pq resultQueue) Less(i, j int) bool { return pq[i].distance > pq[j].distance }
func (pq resultQueue) Swap(i, j int)      { pq[i], pq[j] = pq[j], pq[i] }

func (pq *resultQueue) Push(x interface{}) {
	*pq = append(*pq, x.(*item))
}

func (pq *resultQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	*pq = old[0 : n-1]
	return item
}

// Dimension returns the dimension of vectors in the index
func (h *HNSWIndex) Dimension() int {
	return h.dimension
}
