package storage

import "context"

// VectorStorage defines the interface for vector storage operations
type VectorStorage interface {
	// StoreVector stores a vector with the given ID
	StoreVector(ctx context.Context, id uint64, vector []float32) error
	
	// GetVector retrieves a vector by its ID
	GetVector(ctx context.Context, id uint64) ([]float32, error)
	
	// DeleteVector removes a vector by its ID
	DeleteVector(ctx context.Context, id uint64) error
	
	// BatchStore stores multiple vectors in a single operation
	BatchStore(ctx context.Context, vectors map[uint64][]float32) error
	
	// BatchGet retrieves multiple vectors in a single operation
	BatchGet(ctx context.Context, ids []uint64) (map[uint64][]float32, error)
}

// StorageConfig holds configuration for storage implementations
type StorageConfig struct {
	Endpoint   string
	AccessKey  string
	SecretKey  string
	BucketName string
	Region     string
	Secure     bool
}
