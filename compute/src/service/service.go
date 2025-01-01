package service

import (
	"context"
	"fmt"
	"sync"

	"github.com/adi/poneglyph/compute/src/hnsw"
	"github.com/adi/poneglyph/compute/src/storage"
)

type VectorService struct {
	index   *hnsw.HNSWIndex
	storage storage.VectorStorage
	mu      sync.RWMutex
}

func NewVectorService(dimension int, storage storage.VectorStorage) *VectorService {
	return &VectorService{
		index:   hnsw.NewHNSWIndex(dimension),
		storage: storage,
	}
}

func (s *VectorService) Insert(ctx context.Context, id uint64, vector []float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Store vector in storage
	if err := s.storage.StoreVector(ctx, id, vector); err != nil {
		return fmt.Errorf("failed to store vector: %w", err)
	}

	// Add to index
	if err := s.index.Insert(id, vector); err != nil {
		// Try to rollback storage
		if delErr := s.storage.DeleteVector(ctx, id); delErr != nil {
			return fmt.Errorf("failed to insert vector and rollback failed: %v, rollback error: %v", err, delErr)
		}
		return fmt.Errorf("failed to insert vector: %w", err)
	}

	return nil
}

func (s *VectorService) Search(ctx context.Context, query []float32, k int) ([]uint64, []float32, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Search in index
	ids, distances, err := s.index.Search(query, k)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to search in index: %w", err)
	}

	return ids, distances, nil
}

func (s *VectorService) BatchInsert(ctx context.Context, vectors map[uint64][]float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Store vectors in storage
	if err := s.storage.BatchStore(ctx, vectors); err != nil {
		return fmt.Errorf("failed to store vectors: %w", err)
	}

	// Add to index
	for id, vector := range vectors {
		if err := s.index.Insert(id, vector); err != nil {
			// Try to rollback storage
			for rollbackID := range vectors {
				if delErr := s.storage.DeleteVector(ctx, rollbackID); delErr != nil {
					return fmt.Errorf("failed to insert vectors and rollback failed: %v, rollback error: %v", err, delErr)
				}
			}
			return fmt.Errorf("failed to insert vector: %w", err)
		}
	}

	return nil
}

func (s *VectorService) Delete(ctx context.Context, id uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Get vector from storage
	vector, err := s.storage.GetVector(ctx, id)
	if err != nil {
		return fmt.Errorf("failed to get vector: %w", err)
	}

	// Remove from index
	if err := s.index.Delete(id, vector); err != nil {
		return fmt.Errorf("failed to delete vector from index: %w", err)
	}

	// Remove from storage
	if err := s.storage.DeleteVector(ctx, id); err != nil {
		return fmt.Errorf("failed to delete vector from storage: %w", err)
	}

	return nil
}

// RebuildIndex rebuilds the HNSW index from storage
// This is useful after restarts or when vectors are deleted
func (s *VectorService) RebuildIndex(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Get all vectors from storage
	vectors, err := s.storage.BatchGet(ctx, []uint64{})
	if err != nil {
		return fmt.Errorf("failed to get vectors from storage: %w", err)
	}

	// Create new index
	dimension := len(vectors[0])
	newIndex := hnsw.NewHNSWIndex(dimension)

	// Insert vectors into new index
	for id, vector := range vectors {
		if err := newIndex.Insert(id, vector); err != nil {
			return fmt.Errorf("failed to insert vector into new index: %w", err)
		}
	}

	// Replace old index with new one
	s.index = newIndex

	return nil
}
