package performance

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"testing"
	"time"

	"github.com/adi/poneglyph/compute/src/hnsw"
)

const (
	dimension     = 128
	numVectors    = 100000
	searchK       = 10
	numSearches   = 1000
	reportMemory  = true
)

func generateRandomVector(dim int) []float32 {
	vector := make([]float32, dim)
	for i := range vector {
		vector[i] = rand.Float32()
	}
	return vector
}

func generateTestData(numVectors, dim int) ([][]float32, []uint64) {
	vectors := make([][]float32, numVectors)
	ids := make([]uint64, numVectors)
	
	for i := 0; i < numVectors; i++ {
		vectors[i] = generateRandomVector(dim)
		ids[i] = uint64(i)
	}
	
	return vectors, ids
}

func printMemStats() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("Alloc = %v MiB", bToMb(m.Alloc))
	fmt.Printf("\tTotalAlloc = %v MiB", bToMb(m.TotalAlloc))
	fmt.Printf("\tSys = %v MiB", bToMb(m.Sys))
	fmt.Printf("\tNumGC = %v\n", m.NumGC)
}

func bToMb(b uint64) uint64 {
	return b / 1024 / 1024
}

func TestHNSWPerformance(t *testing.T) {
	// Generate test data
	vectors, ids := generateTestData(numVectors, dimension)
	
	// Create index
	index := hnsw.NewHNSWIndex(dimension)
	
	// Test insertion performance
	start := time.Now()
	for i := 0; i < numVectors; i++ {
		if err := index.Insert(ids[i], vectors[i]); err != nil {
			t.Fatalf("Failed to insert vector %d: %v", i, err)
		}
		
		if (i+1)%10000 == 0 {
			elapsed := time.Since(start)
			rate := float64(i+1) / elapsed.Seconds()
			fmt.Printf("Inserted %d vectors in %v (%.2f vectors/sec)\n", i+1, elapsed, rate)
			if reportMemory {
				printMemStats()
			}
		}
	}
	
	totalInsertTime := time.Since(start)
	fmt.Printf("\nTotal insertion time: %v\n", totalInsertTime)
	fmt.Printf("Average insertion time: %v/vector\n", totalInsertTime/time.Duration(numVectors))
	
	if reportMemory {
		fmt.Println("\nMemory usage after insertion:")
		printMemStats()
	}
	
	// Test search performance
	fmt.Println("\nTesting search performance...")
	searchTimes := make([]time.Duration, numSearches)
	
	for i := 0; i < numSearches; i++ {
		query := generateRandomVector(dimension)
		
		start := time.Now()
		ids, distances, err := index.Search(query, searchK)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		searchTimes[i] = time.Since(start)
		
		if len(ids) != searchK || len(distances) != searchK {
			t.Errorf("Expected %d results, got %d ids and %d distances", searchK, len(ids), len(distances))
		}
		
		if (i+1)%100 == 0 {
			fmt.Printf("Completed %d searches\n", i+1)
		}
	}
	
	// Calculate search statistics
	var totalSearchTime time.Duration
	for _, t := range searchTimes {
		totalSearchTime += t
	}
	
	avgSearchTime := totalSearchTime / time.Duration(numSearches)
	fmt.Printf("\nAverage search time: %v\n", avgSearchTime)
	
	// Test deletion performance
	fmt.Println("\nTesting deletion performance...")
	start = time.Now()
	for i := 0; i < numVectors/10; i++ { // Delete 10% of vectors
		if err := index.Delete(ids[i], vectors[i]); err != nil {
			t.Fatalf("Failed to delete vector %d: %v", i, err)
		}
		
		if (i+1)%1000 == 0 {
			elapsed := time.Since(start)
			rate := float64(i+1) / elapsed.Seconds()
			fmt.Printf("Deleted %d vectors in %v (%.2f vectors/sec)\n", i+1, elapsed, rate)
		}
	}
	
	totalDeleteTime := time.Since(start)
	fmt.Printf("\nTotal deletion time: %v\n", totalDeleteTime)
	fmt.Printf("Average deletion time: %v/vector\n", totalDeleteTime/time.Duration(numVectors/10))
	
	if reportMemory {
		fmt.Println("\nFinal memory usage:")
		printMemStats()
	}
}

func TestHNSWAccuracy(t *testing.T) {
	// Generate a small dataset for accuracy testing
	const (
		testDim = 8
		testNum = 1000
		k = 5
	)
	
	vectors, ids := generateTestData(testNum, testDim)
	
	// Create index
	index := hnsw.NewHNSWIndex(testDim)
	
	// Insert vectors
	for i := 0; i < testNum; i++ {
		if err := index.Insert(ids[i], vectors[i]); err != nil {
			t.Fatalf("Failed to insert vector %d: %v", i, err)
		}
	}
	
	// Test accuracy by comparing with brute force search
	fmt.Println("\nTesting search accuracy...")
	
	numQueries := 100
	var totalRecall float64
	
	for i := 0; i < numQueries; i++ {
		query := generateRandomVector(testDim)
		
		// Get HNSW results
		hnswIDs, _, err := index.Search(query, k)
		if err != nil {
			t.Fatalf("HNSW search failed: %v", err)
		}
		
		// Get ground truth (brute force)
		groundTruth := bruteForceSearch(query, vectors, ids, k)
		
		// Calculate recall
		recall := calculateRecall(hnswIDs, groundTruth)
		totalRecall += recall
		
		if (i+1)%10 == 0 {
			fmt.Printf("Completed %d accuracy tests (current recall: %.2f)\n", i+1, recall)
		}
	}
	
	avgRecall := totalRecall / float64(numQueries)
	fmt.Printf("\nAverage recall@%d: %.4f\n", k, avgRecall)
	
	if avgRecall < 0.8 {
		t.Errorf("Low recall: %.4f (expected > 0.8)", avgRecall)
	}
}

func bruteForceSearch(query []float32, vectors [][]float32, ids []uint64, k int) []uint64 {
	type distanceItem struct {
		id       uint64
		distance float32
	}
	
	// Calculate distances to all vectors
	distances := make([]distanceItem, len(vectors))
	for i := range vectors {
		distances[i] = distanceItem{
			id:       ids[i],
			distance: calculateDistance(query, vectors[i]),
		}
	}
	
	// Sort by distance
	sort.Slice(distances, func(i, j int) bool {
		return distances[i].distance < distances[j].distance
	})
	
	// Get top k
	result := make([]uint64, k)
	for i := 0; i < k; i++ {
		result[i] = distances[i].id
	}
	
	return result
}

func calculateDistance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return float32(math.Sqrt(float64(sum)))
}

func calculateRecall(result, groundTruth []uint64) float64 {
	// Convert ground truth to map for O(1) lookup
	truthMap := make(map[uint64]bool)
	for _, id := range groundTruth {
		truthMap[id] = true
	}
	
	// Count matches
	var matches int
	for _, id := range result {
		if truthMap[id] {
			matches++
		}
	}
	
	return float64(matches) / float64(len(groundTruth))
}
