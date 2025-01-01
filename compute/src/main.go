package main

import (
	"log"
	"os"
	"strconv"
	"time"

	"github.com/adi/poneglyph/compute/src/api"
	"github.com/adi/poneglyph/compute/src/service"
	"github.com/adi/poneglyph/compute/src/storage"
)

func main() {
	// Initialize storage
	storageConfig := &storage.StorageConfig{
		Endpoint:   getEnv("MINIO_ENDPOINT", "localhost:9000"),
		AccessKey:  getEnv("MINIO_ACCESS_KEY", "minioadmin"),
		SecretKey:  getEnv("MINIO_SECRET_KEY", "minioadmin"),
		BucketName: getEnv("MINIO_BUCKET", "vectors"),
		Region:     getEnv("MINIO_REGION", "us-east-1"),
		Secure:     getEnvBool("MINIO_SECURE", false),
	}

	baseStorage, err := storage.NewMinioStorage(storageConfig)
	if err != nil {
		log.Fatalf("Failed to initialize storage: %v", err)
	}

	// Initialize Redis cache
	cacheConfig := &storage.RedisCacheConfig{
		Addr:       getEnv("REDIS_ADDR", "localhost:6379"),
		Password:   getEnv("REDIS_PASSWORD", ""),
		DB:         getEnvInt("REDIS_DB", 0),
		Expiration: time.Duration(getEnvInt("REDIS_EXPIRATION", 3600)) * time.Second,
		KeyPrefix:  getEnv("REDIS_PREFIX", "vectors"),
	}

	cachedStorage, err := storage.NewRedisCache(cacheConfig, baseStorage)
	if err != nil {
		log.Fatalf("Failed to initialize cache: %v", err)
	}

	// Initialize vector service
	vectorDimension := getEnvInt("VECTOR_DIMENSION", 128)
	vectorService := service.NewVectorService(vectorDimension, cachedStorage)

	// Initialize HTTP server
	server := api.NewServer(vectorService)

	// Start server
	addr := getEnv("SERVER_ADDR", ":3000")
	log.Printf("Starting server on %s", addr)
	if err := server.Start(addr); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

func getEnv(key, defaultValue string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	intValue, err := strconv.Atoi(value)
	if err != nil {
		return defaultValue
	}
	return intValue
}

func getEnvBool(key string, defaultValue bool) bool {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	boolValue, err := strconv.ParseBool(value)
	if err != nil {
		return defaultValue
	}
	return boolValue
}
