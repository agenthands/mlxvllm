package api

import (
	"net/http/httptest"
	"testing"
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
