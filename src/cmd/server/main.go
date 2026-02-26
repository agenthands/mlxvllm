package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"github.com/agenthands/mlxvllm/internal/api"
	"github.com/agenthands/mlxvllm/internal/config"
	"github.com/agenthands/mlxvllm/internal/model"
)

var (
	configPath = flag.String("config", "./models/config.yaml", "Path to configuration file")
)

func main() {
	flag.Parse()

	// Load configuration
	cfg, err := config.LoadConfig(*configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Initialize model registry
	registry := model.NewRegistry(cfg)

	// Preload configured models
	for name, mcfg := range cfg.Models {
		if mcfg.Preload {
			log.Printf("Preloading model: %s", name)
			if err := registry.LoadModel(name); err != nil {
				log.Printf("Warning: failed to preload %s: %v", name, err)
			}
		}
	}

	// Create API handler
	handler := api.NewHandler(registry)

	// Start server
	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	server := api.NewServer(addr, handler)

	// Handle shutdown gracefully
	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
		<-sigChan

		log.Println("Shutting down...")
		server.Shutdown(context.Background())
	}()

	if err := server.Start(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Server error: %v", err)
	}
}
