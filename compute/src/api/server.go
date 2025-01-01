package api

import (
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"
	"github.com/adi/poneglyph/compute/src/service"
)

type Server struct {
	app     *fiber.App
	service *service.VectorService
}

type VectorRequest struct {
	ID     uint64    `json:"id"`
	Vector []float32 `json:"vector"`
}

type SearchRequest struct {
	Vector []float32 `json:"vector"`
	K      int       `json:"k"`
}

type SearchResponse struct {
	IDs       []uint64  `json:"ids"`
	Distances []float32 `json:"distances"`
}

type BatchInsertRequest struct {
	Vectors map[uint64][]float32 `json:"vectors"`
}

func NewServer(service *service.VectorService) *Server {
	app := fiber.New(fiber.Config{
		ErrorHandler: func(c *fiber.Ctx, err error) error {
			code := fiber.StatusInternalServerError
			if e, ok := err.(*fiber.Error); ok {
				code = e.Code
			}
			return c.Status(code).JSON(fiber.Map{
				"error": err.Error(),
			})
		},
	})

	// Middleware
	app.Use(recover.New())
	app.Use(logger.New())
	app.Use(cors.New())

	server := &Server{
		app:     app,
		service: service,
	}

	// Routes
	app.Post("/vectors", server.insertVector)
	app.Post("/vectors/batch", server.batchInsert)
	app.Post("/vectors/search", server.search)
	app.Delete("/vectors/:id", server.deleteVector)
	app.Post("/rebuild", server.rebuildIndex)

	return server
}

func (s *Server) insertVector(c *fiber.Ctx) error {
	var req VectorRequest
	if err := c.BodyParser(&req); err != nil {
		return fiber.NewError(fiber.StatusBadRequest, "Invalid request body")
	}

	if err := s.service.Insert(c.Context(), req.ID, req.Vector); err != nil {
		return err
	}

	return c.SendStatus(fiber.StatusCreated)
}

func (s *Server) batchInsert(c *fiber.Ctx) error {
	var req BatchInsertRequest
	if err := c.BodyParser(&req); err != nil {
		return fiber.NewError(fiber.StatusBadRequest, "Invalid request body")
	}

	if err := s.service.BatchInsert(c.Context(), req.Vectors); err != nil {
		return err
	}

	return c.SendStatus(fiber.StatusCreated)
}

func (s *Server) search(c *fiber.Ctx) error {
	var req SearchRequest
	if err := c.BodyParser(&req); err != nil {
		return fiber.NewError(fiber.StatusBadRequest, "Invalid request body")
	}

	if req.K <= 0 {
		return fiber.NewError(fiber.StatusBadRequest, "K must be greater than 0")
	}

	ids, distances, err := s.service.Search(c.Context(), req.Vector, req.K)
	if err != nil {
		return err
	}

	return c.JSON(SearchResponse{
		IDs:       ids,
		Distances: distances,
	})
}

func (s *Server) deleteVector(c *fiber.Ctx) error {
	id, err := c.ParamsInt("id")
	if err != nil {
		return fiber.NewError(fiber.StatusBadRequest, "Invalid ID")
	}

	if err := s.service.Delete(c.Context(), uint64(id)); err != nil {
		return err
	}

	return c.SendStatus(fiber.StatusNoContent)
}

func (s *Server) rebuildIndex(c *fiber.Ctx) error {
	if err := s.service.RebuildIndex(c.Context()); err != nil {
		return err
	}
	return c.SendStatus(fiber.StatusOK)
}

func (s *Server) Start(addr string) error {
	return s.app.Listen(addr)
}

func (s *Server) Shutdown() error {
	return s.app.Shutdown()
}
