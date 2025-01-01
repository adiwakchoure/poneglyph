package hnsw

import "errors"

var (
	ErrInvalidDimension = errors.New("invalid vector dimension")
	ErrEmptyIndex      = errors.New("index is empty")
)
