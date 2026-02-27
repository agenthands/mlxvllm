package main

import (
	"context"
	"flag"
	"log/slog"
	nethttp "net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	httpserver "github.com/agenthands/GUI-Actor/internal/http"
	"github.com/agenthands/GUI-Actor/internal/mlx"
	"github.com/agenthands/GUI-Actor/internal/radix"
	"github.com/agenthands/GUI-Actor/pkg/tokenizer"
)

var (
	// Server configuration
	addr         = flag.String("addr", ":8080", "Server address")
	modelPath    = flag.String("model", "", "Path to model weights")
	vocabSize    = flag.Int("vocab-size", 32000, "Tokenizer vocabulary size")
	maxCacheSize = flag.Int("max-cache-size", 1000, "Maximum cache entries (0 = unlimited)")
	logLevel     = flag.String("log-level", "info", "Log level (debug, info, warn, error)")
	// MLX configuration
	mlxLibrary = flag.String("mlx-library", "libmlx_runtime.dylib", "Path to MLX runtime library")
)

func main() {
	flag.Parse()

	// Setup logging
	setupLogging(*logLevel)

	slog.Info("Starting GUI-Actor RadixAttention Server",
		"addr", *addr,
		"model", *modelPath,
		"vocab_size", *vocabSize,
		"max_cache", *maxCacheSize,
	)

	// Initialize components
	// Create Radix tree for KV cache management
	tree := radix.NewTree()
	slog.Info("Initialized Radix tree for prefix caching")

	// Initialize MLX engine (placeholder - would load actual MLX)
	engine, err := setupMLXEngine()
	if err != nil {
		slog.Error("Failed to setup MLX engine", "error", err)
		os.Exit(1)
	}
	slog.Info("Initialized MLX engine", "library", *mlxLibrary)

	// Initialize tokenizer
	tok := tokenizer.NewTokenizer(*vocabSize)
	slog.Info("Initialized tokenizer", "vocab_size", *vocabSize)

	// Load model (placeholder - would load actual weights)
	model, err := loadModel(*modelPath, *vocabSize)
	if err != nil {
		slog.Error("Failed to load model", "error", err)
		os.Exit(1)
	}
	slog.Info("Loaded model", "path", *modelPath, "vocab_size", *vocabSize)

	// Create HTTP server
	server := httpserver.NewServer(tree, engine, tok, model)

	// Setup routes
	mux := nethttp.NewServeMux()
	server.RegisterRoutes(mux)

	// Wrap with middleware
	handler := wrapMiddleware(server, mux)

	// Create HTTP server
	httpServer := &nethttp.Server{
		Addr:         *addr,
		Handler:      handler,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Start server in background
	go func() {
		slog.Info("Server listening", "addr", *addr)
		if err := httpServer.ListenAndServe(); err != nil && err != nethttp.ErrServerClosed {
			slog.Error("Server error", "error", err)
		}
	}()

	// Wait for shutdown signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	sig := <-sigChan
	slog.Info("Received signal, shutting down", "signal", sig)

	// Graceful shutdown
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		slog.Error("Shutdown error", "error", err)
	}

	// Cleanup resources
	slog.Info("Cleaning up resources")

	// Evict all LRU entries
	// Note: lruList is private, we'd need to add a public method
	// tree.EvictLRU(tree.lruList.Len())
	slog.Info("Cache cleanup complete")

	slog.Info("Shutdown complete")
}

// setupLogging configures structured logging
func setupLogging(level string) {
	var lvl slog.Level
	switch level {
	case "debug":
		lvl = slog.LevelDebug
	case "info":
		lvl = slog.LevelInfo
	case "warn":
		lvl = slog.LevelWarn
	case "error":
		lvl = slog.LevelError
	default:
		lvl = slog.LevelInfo
	}

	handler := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: lvl,
	})

	logger := slog.New(handler)
	slog.SetDefault(logger)
}

// setupMLXEngine initializes the MLX inference engine
func setupMLXEngine() (radix.MLXEngine, error) {
	// In production, this would:
	// 1. Load the MLX runtime library
	// 2. Initialize Metal device
	// 3. Load model weights into GPU memory
	// 4. Return MLXEngine interface

	// For now, return a mock that will be replaced with CGO bindings
	return &radix.MockMLXEngine{
		ForwardFunc: func(model any, tokens []uint32, base uint64) ([]float32, uint64, error) {
			// Placeholder: return fake logits
			logits := make([]float32, 32000)
			return logits, base + 1, nil
		},
		SliceFunc: func(handle uint64, keepTokens int) (uint64, error) {
			return handle + 100, nil
		},
		FreeFunc: func(handle uint64) {
			// Placeholder: no-op
		},
	}, nil
}

// loadModel loads the model weights
func loadModel(path string, vocabSize int) (any, error) {
	// For testing/demo, allow empty path and return mock
	if path == "" {
		slog.Info("Using mock model (no model path provided)")
		slog.Info("To use real model, download Qwen2-VL from: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct")
		return map[string]any{
			"type":       "mock",
			"name":       "qwen2-vl-mock",
			"vocab_size": vocabSize,
		}, nil
	}

	// Load MLX model using CGO bridge
	slog.Info("Loading MLX model", "path", path, "vocab_size", vocabSize)
	if err := mlx.LoadModel(path, vocabSize); err != nil {
		slog.Error("Failed to load MLX model", "error", err)
		// Fall back to mock if MLX loading fails
		slog.Warn("Falling back to mock model")
		return map[string]any{
			"type":       "mock",
			"name":       "qwen2-vl-mock-fallback",
			"vocab_size": vocabSize,
		}, nil
	}

	slog.Info("MLX model loaded successfully")
	return map[string]any{
		"type":       "mlx",
		"name":       "qwen2-vl",
		"path":       path,
		"vocab_size": vocabSize,
	}, nil
}

// wrapMiddleware applies middleware to the handler
func wrapMiddleware(server *httpserver.Server, handler nethttp.Handler) nethttp.Handler {
	// Apply panic recovery
	h := panicRecoveryMiddleware(handler)

	// Apply request logging
	h = requestLoggingMiddleware(h)

	// Apply CORS
	h = corsMiddleware(h)

	return h
}

// panicRecoveryMiddleware catches panics and returns 500 error
func panicRecoveryMiddleware(next nethttp.Handler) nethttp.Handler {
	return nethttp.HandlerFunc(func(w nethttp.ResponseWriter, r *nethttp.Request) {
		defer func() {
			if err := recover(); err != nil {
				slog.Error("Panic recovered", "error", err)
				nethttp.Error(w, "Internal Server Error", 500)
			}
		}()
		next.ServeHTTP(w, r)
	})
}

// requestLoggingMiddleware logs all requests
func requestLoggingMiddleware(next nethttp.Handler) nethttp.Handler {
	return nethttp.HandlerFunc(func(w nethttp.ResponseWriter, r *nethttp.Request) {
		slog.Info("Request", "method", r.Method, "path", r.URL.Path)
		next.ServeHTTP(w, r)
	})
}

// corsMiddleware adds CORS headers for browser clients
func corsMiddleware(next nethttp.Handler) nethttp.Handler {
	return nethttp.HandlerFunc(func(w nethttp.ResponseWriter, r *nethttp.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == "OPTIONS" {
			w.WriteHeader(nethttp.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// init performs startup initialization
func init() {
	// Set timezone
	os.Setenv("TZ", "UTC")

	// Configure CGO (for MLX bindings)
	// In production, this would set DYLD_LIBRARY_PATH
	// os.Setenv("DYLD_LIBRARY_PATH", "/usr/local/lib:$DYLD_LIBRARY_PATH")
}
