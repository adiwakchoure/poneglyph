package storage

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"sync"

	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

type MinioStorage struct {
	client     *minio.Client
	bucketName string
}

func NewMinioStorage(cfg *StorageConfig) (VectorStorage, error) {
	client, err := minio.New(cfg.Endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(cfg.AccessKey, cfg.SecretKey, ""),
		Secure: cfg.Secure,
		Region: cfg.Region,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create minio client: %w", err)
	}

	exists, err := client.BucketExists(context.Background(), cfg.BucketName)
	if err != nil {
		return nil, fmt.Errorf("failed to check bucket existence: %w", err)
	}

	if !exists {
		err = client.MakeBucket(context.Background(), cfg.BucketName, minio.MakeBucketOptions{Region: cfg.Region})
		if err != nil {
			return nil, fmt.Errorf("failed to create bucket: %w", err)
		}
	}

	return &MinioStorage{
		client:     client,
		bucketName: cfg.BucketName,
	}, nil
}

func (s *MinioStorage) StoreVector(ctx context.Context, id uint64, vector []float32) error {
	buf := new(bytes.Buffer)
	
	if err := binary.Write(buf, binary.LittleEndian, uint32(len(vector))); err != nil {
		return fmt.Errorf("failed to write vector dimension: %w", err)
	}
	
	if err := binary.Write(buf, binary.LittleEndian, vector); err != nil {
		return fmt.Errorf("failed to write vector data: %w", err)
	}

	objectName := fmt.Sprintf("vectors/%d", id)
	_, err := s.client.PutObject(
		ctx,
		s.bucketName,
		objectName,
		buf,
		int64(buf.Len()),
		minio.PutObjectOptions{ContentType: "application/octet-stream"},
	)
	
	if err != nil {
		return fmt.Errorf("failed to store vector: %w", err)
	}

	return nil
}

func (s *MinioStorage) GetVector(ctx context.Context, id uint64) ([]float32, error) {
	objectName := fmt.Sprintf("vectors/%d", id)
	
	obj, err := s.client.GetObject(
		ctx,
		s.bucketName,
		objectName,
		minio.GetObjectOptions{},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to get vector: %w", err)
	}
	defer obj.Close()

	var dimension uint32
	if err := binary.Read(obj, binary.LittleEndian, &dimension); err != nil {
		return nil, fmt.Errorf("failed to read vector dimension: %w", err)
	}

	vector := make([]float32, dimension)
	if err := binary.Read(obj, binary.LittleEndian, vector); err != nil {
		return nil, fmt.Errorf("failed to read vector data: %w", err)
	}

	return vector, nil
}

func (s *MinioStorage) DeleteVector(ctx context.Context, id uint64) error {
	objectName := fmt.Sprintf("vectors/%d", id)
	err := s.client.RemoveObject(
		ctx,
		s.bucketName,
		objectName,
		minio.RemoveObjectOptions{},
	)
	if err != nil {
		return fmt.Errorf("failed to delete vector: %w", err)
	}
	return nil
}

func (s *MinioStorage) BatchStore(ctx context.Context, vectors map[uint64][]float32) error {
	var wg sync.WaitGroup
	errChan := make(chan error, len(vectors))

	for id, vector := range vectors {
		wg.Add(1)
		go func(id uint64, vector []float32) {
			defer wg.Done()
			if err := s.StoreVector(ctx, id, vector); err != nil {
				errChan <- fmt.Errorf("failed to store vector %d: %w", id, err)
			}
		}(id, vector)
	}

	wg.Wait()
	close(errChan)

	// Collect errors
	var errors []error
	for err := range errChan {
		errors = append(errors, err)
	}

	if len(errors) > 0 {
		return fmt.Errorf("batch store failed with %d errors: %v", len(errors), errors)
	}

	return nil
}

func (s *MinioStorage) BatchGet(ctx context.Context, ids []uint64) (map[uint64][]float32, error) {
	results := make(map[uint64][]float32)
	var mu sync.Mutex
	var wg sync.WaitGroup
	errChan := make(chan error, len(ids))

	for _, id := range ids {
		wg.Add(1)
		go func(id uint64) {
			defer wg.Done()
			vector, err := s.GetVector(ctx, id)
			if err != nil {
				errChan <- fmt.Errorf("failed to get vector %d: %w", id, err)
				return
			}

			mu.Lock()
			results[id] = vector
			mu.Unlock()
		}(id)
	}

	wg.Wait()
	close(errChan)

	// Collect errors
	var errors []error
	for err := range errChan {
		errors = append(errors, err)
	}

	if len(errors) > 0 {
		return nil, fmt.Errorf("batch get failed with %d errors: %v", len(errors), errors)
	}

	return results, nil
}
