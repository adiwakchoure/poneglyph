package storage

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"time"

	"github.com/redis/go-redis/v9"
)

type RedisCache struct {
	client        *redis.Client
	storage       VectorStorage
	expiration    time.Duration
	keyPrefix     string
}

type RedisCacheConfig struct {
	Addr       string
	Password   string
	DB         int
	Expiration time.Duration
	KeyPrefix  string
}

func NewRedisCache(cfg *RedisCacheConfig, storage VectorStorage) (*RedisCache, error) {
	client := redis.NewClient(&redis.Options{
		Addr:     cfg.Addr,
		Password: cfg.Password,
		DB:       cfg.DB,
	})

	// Test connection
	if err := client.Ping(context.Background()).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	return &RedisCache{
		client:        client,
		storage:       storage,
		expiration:    cfg.Expiration,
		keyPrefix:     cfg.KeyPrefix,
	}, nil
}

func (c *RedisCache) vectorKey(id uint64) string {
	return fmt.Sprintf("%s:vector:%d", c.keyPrefix, id)
}

func (c *RedisCache) serializeVector(vector []float32) []byte {
	buf := make([]byte, 4+len(vector)*4)
	binary.LittleEndian.PutUint32(buf, uint32(len(vector)))
	for i, v := range vector {
		binary.LittleEndian.PutUint32(buf[4+i*4:], math.Float32bits(v))
	}
	return buf
}

func (c *RedisCache) deserializeVector(data []byte) ([]float32, error) {
	if len(data) < 4 {
		return nil, fmt.Errorf("invalid vector data")
	}

	dimension := binary.LittleEndian.Uint32(data)
	if len(data) != int(4+dimension*4) {
		return nil, fmt.Errorf("invalid vector data length")
	}

	vector := make([]float32, dimension)
	for i := range vector {
		bits := binary.LittleEndian.Uint32(data[4+i*4:])
		vector[i] = math.Float32frombits(bits)
	}
	return vector, nil
}

func (c *RedisCache) StoreVector(ctx context.Context, id uint64, vector []float32) error {
	// Store in underlying storage first
	if err := c.storage.StoreVector(ctx, id, vector); err != nil {
		return err
	}

	// Cache the vector
	data := c.serializeVector(vector)
	if err := c.client.Set(ctx, c.vectorKey(id), data, c.expiration).Err(); err != nil {
		return fmt.Errorf("failed to cache vector: %w", err)
	}

	return nil
}

func (c *RedisCache) GetVector(ctx context.Context, id uint64) ([]float32, error) {
	// Try cache first
	data, err := c.client.Get(ctx, c.vectorKey(id)).Bytes()
	if err == nil {
		return c.deserializeVector(data)
	}
	if err != redis.Nil {
		return nil, fmt.Errorf("failed to get vector from cache: %w", err)
	}

	// Cache miss, get from storage
	vector, err := c.storage.GetVector(ctx, id)
	if err != nil {
		return nil, err
	}

	// Cache the vector
	data = c.serializeVector(vector)
	if err := c.client.Set(ctx, c.vectorKey(id), data, c.expiration).Err(); err != nil {
		// Log error but don't fail the request
		fmt.Printf("failed to cache vector %d: %v\n", id, err)
	}

	return vector, nil
}

func (c *RedisCache) DeleteVector(ctx context.Context, id uint64) error {
	// Delete from storage first
	if err := c.storage.DeleteVector(ctx, id); err != nil {
		return err
	}

	// Delete from cache
	if err := c.client.Del(ctx, c.vectorKey(id)).Err(); err != nil {
		return fmt.Errorf("failed to delete vector from cache: %w", err)
	}

	return nil
}

func (c *RedisCache) BatchStore(ctx context.Context, vectors map[uint64][]float32) error {
	// Store in underlying storage first
	if err := c.storage.BatchStore(ctx, vectors); err != nil {
		return err
	}

	// Cache all vectors
	pipe := c.client.Pipeline()
	for id, vector := range vectors {
		data := c.serializeVector(vector)
		pipe.Set(ctx, c.vectorKey(id), data, c.expiration)
	}

	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("failed to cache vectors: %w", err)
	}

	return nil
}

func (c *RedisCache) BatchGet(ctx context.Context, ids []uint64) (map[uint64][]float32, error) {
	results := make(map[uint64][]float32)
	missingIDs := make([]uint64, 0)

	// Try cache first
	pipe := c.client.Pipeline()
	for _, id := range ids {
		pipe.Get(ctx, c.vectorKey(id))
	}
	cmds, err := pipe.Exec(ctx)
	if err != nil && err != redis.Nil {
		return nil, fmt.Errorf("failed to get vectors from cache: %w", err)
	}

	// Process cache hits and collect misses
	for i, cmd := range cmds {
		id := ids[i]
		if cmd.Err() == redis.Nil {
			missingIDs = append(missingIDs, id)
			continue
		}
		if cmd.Err() != nil {
			return nil, fmt.Errorf("failed to get vector %d from cache: %w", id, cmd.Err())
		}

		data, err := cmd.(*redis.StringCmd).Bytes()
		if err != nil {
			return nil, fmt.Errorf("failed to get vector %d data: %w", id, err)
		}

		vector, err := c.deserializeVector(data)
		if err != nil {
			return nil, fmt.Errorf("failed to deserialize vector %d: %w", id, err)
		}

		results[id] = vector
	}

	// Get missing vectors from storage
	if len(missingIDs) > 0 {
		missing, err := c.storage.BatchGet(ctx, missingIDs)
		if err != nil {
			return nil, err
		}

		// Cache missing vectors
		pipe := c.client.Pipeline()
		for id, vector := range missing {
			data := c.serializeVector(vector)
			pipe.Set(ctx, c.vectorKey(id), data, c.expiration)
			results[id] = vector
		}

		if _, err := pipe.Exec(ctx); err != nil {
			// Log error but don't fail the request
			fmt.Printf("failed to cache missing vectors: %v\n", err)
		}
	}

	return results, nil
}
