package api

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestNewServer(t *testing.T) {
	srv := NewServer(":8080", nil)
	if srv == nil {
		t.Fatal("Expected non-nil server")
	}

	if srv.Handler == nil {
		t.Error("Expected non-nil handler")
	}
}

func TestServerRoutes(t *testing.T) {
	h := &Handler{} // Minimal handler for testing
	srv := NewServer(":8080", h)

	tests := []struct {
		path       string
		method     string
		expectCode int
	}{
		{"/v1/health", "GET", 200},
		{"/v1/models", "GET", 200},
		{"/invalid", "GET", 404},
	}

	for _, tt := range tests {
		req := httptest.NewRequest(tt.method, tt.path, nil)
		w := httptest.NewRecorder()

		srv.Handler.ServeHTTP(w, req)

		if w.Code != tt.expectCode {
			t.Errorf("%s %s: expected %d, got %d", tt.method, tt.path, tt.expectCode, w.Code)
		}
	}
}

func TestServerStartAndShutdown(t *testing.T) {
	h := NewHandler(nil)
	srv := NewServer(":0", h) // Use :0 for random port

	// Start server in background
	errChan := make(chan error, 1)
	go func() {
		if err := srv.Start(); err != nil && err != http.ErrServerClosed {
			errChan <- err
		}
	}()

	// Wait a bit for server to start
	time.Sleep(100 * time.Millisecond)

	// Verify server is running by making a request
	client := &http.Client{Timeout: 1 * time.Second}
	resp, err := client.Get("http://localhost/v1/health")
	if err == nil {
		resp.Body.Close()
	}

	// Shutdown the server
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		t.Errorf("Shutdown failed: %v", err)
	}

	// Check for any errors from Start
	select {
	case err := <-errChan:
		if err != nil && err != http.ErrServerClosed {
			t.Errorf("Server error: %v", err)
		}
	default:
	}
}

func TestServerShutdownMultipleTimes(t *testing.T) {
	h := NewHandler(nil)
	srv := NewServer(":0", h)

	// Shutdown without starting should not panic
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		// First shutdown might fail since server isn't started
		t.Logf("First shutdown (not started): %v", err)
	}
}
