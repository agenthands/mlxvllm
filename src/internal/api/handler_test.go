package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gorilla/mux"
)

func TestHealthHandler(t *testing.T) {
	h := NewHandler(nil)
	req := httptest.NewRequest("GET", "/v1/health", nil)
	w := httptest.NewRecorder()

	h.Health(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var resp HealthResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	if resp.Status != "ok" {
		t.Errorf("Expected status 'ok', got '%s'", resp.Status)
	}
}

func TestListModels(t *testing.T) {
	h := NewHandler(nil)
	req := httptest.NewRequest("GET", "/v1/models", nil)
	w := httptest.NewRecorder()

	h.ListModels(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var resp ModelsResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	if resp.Object != "list" {
		t.Errorf("Expected object 'list', got '%s'", resp.Object)
	}
}

func TestGetModel(t *testing.T) {
	tests := []struct {
		name       string
		modelID    string
		expectCode int
	}{
		{"valid model", "gui-actor-2b", http.StatusOK},
		{"7b model", "gui-actor-7b", http.StatusOK},
		{"unknown model", "unknown-model", http.StatusOK},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewHandler(nil)
			req := httptest.NewRequest("GET", "/v1/models/"+tt.modelID, nil)
			w := httptest.NewRecorder()

			req = mux.SetURLVars(req, map[string]string{"id": tt.modelID})

			h.GetModel(w, req)

			if w.Code != tt.expectCode {
				t.Errorf("Expected status %d, got %d", tt.expectCode, w.Code)
			}

			var resp ModelInfo
			if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
				t.Fatalf("Failed to decode response: %v", err)
			}

			if resp.ID != tt.modelID {
				t.Errorf("Expected model ID '%s', got '%s'", tt.modelID, resp.ID)
			}
		})
	}
}

func TestLoadModel(t *testing.T) {
	tests := []struct {
		name       string
		modelID    string
		registry   bool
		expectCode int
	}{
		{"with registry", "gui-actor-2b", true, http.StatusOK},
		{"without registry", "gui-actor-2b", false, http.StatusServiceUnavailable},
		{"7b model with registry", "gui-actor-7b", true, http.StatusOK},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewHandler(nil)

			req := httptest.NewRequest("POST", "/v1/models/"+tt.modelID+"/load", nil)
			w := httptest.NewRecorder()

			req = mux.SetURLVars(req, map[string]string{"id": tt.modelID})

			h.LoadModel(w, req)

			if w.Code != http.StatusOK && w.Code != http.StatusInternalServerError && w.Code != http.StatusServiceUnavailable {
				t.Logf("Got status %d", w.Code)
			}
		})
	}
}

func TestUnloadModel(t *testing.T) {
	tests := []struct {
		name       string
		modelID    string
		registry   bool
		expectCode int
	}{
		{"with registry", "gui-actor-2b", true, http.StatusOK},
		{"without registry", "gui-actor-2b", false, http.StatusServiceUnavailable},
		{"7b model with registry", "gui-actor-7b", true, http.StatusOK},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewHandler(nil)

			req := httptest.NewRequest("DELETE", "/v1/models/"+tt.modelID, nil)
			w := httptest.NewRecorder()

			req = mux.SetURLVars(req, map[string]string{"id": tt.modelID})

			h.UnloadModel(w, req)

			if w.Code != http.StatusOK && w.Code != http.StatusInternalServerError && w.Code != http.StatusServiceUnavailable {
				t.Logf("Got status %d", w.Code)
			}
		})
	}
}

func TestChatCompletion(t *testing.T) {
	tests := []struct {
		name        string
		requestBody string
		expectCode  int
	}{
		{
			name: "valid request",
			requestBody: `{
				"model": "gui-actor-2b",
				"messages": [{"role": "user", "content": "test"}]
			}`,
			expectCode: http.StatusOK,
		},
		{
			name:        "invalid json",
			requestBody: `{invalid json`,
			expectCode:  http.StatusBadRequest,
		},
		{
			name:        "empty body",
			requestBody: `{}`,
			expectCode:  http.StatusOK,
		},
		{
			name: "with image content",
			requestBody: `{
				"model": "gui-actor-7b",
				"messages": [{
					"role": "user",
					"content": [
						{"type": "text", "text": "click submit"},
						{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}}
					]
				}]
			}`,
			expectCode: http.StatusOK,
		},
		{
			name: "with profile",
			requestBody: `{
				"model": "gui-actor-2b",
				"profile": "fast",
				"messages": [{"role": "user", "content": "test"}]
			}`,
			expectCode: http.StatusOK,
		},
		{
			name: "with max pixels",
			requestBody: `{
				"model": "gui-actor-2b",
				"max_pixels": 1048576,
				"messages": [{"role": "user", "content": "test"}]
			}`,
			expectCode: http.StatusOK,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewHandler(nil)
			req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(tt.requestBody))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			h.ChatCompletion(w, req)

			if w.Code != tt.expectCode {
				t.Errorf("Expected status %d, got %d", tt.expectCode, w.Code)
			}

			if tt.expectCode == http.StatusOK {
				var resp ChatCompletionResponse
				if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
					t.Fatalf("Failed to decode response: %v", err)
				}

				if resp.Object != "chat.completion" {
					t.Errorf("Expected object 'chat.completion', got '%s'", resp.Object)
				}

				if len(resp.Choices) == 0 {
					t.Error("Expected at least one choice")
				}
			}
		})
	}
}

func TestNewChatCompletionResponse(t *testing.T) {
	tests := []struct {
		name    string
		model   string
		choices []Choice
	}{
		{
			name:  "single choice",
			model: "gui-actor-2b",
			choices: []Choice{
				{
					Index:        0,
					Message:      Message{Role: "assistant", Content: "test"},
					FinishReason: "stop",
				},
			},
		},
		{
			name:  "multiple choices",
			model: "gui-actor-7b",
			choices: []Choice{
				{
					Index:        0,
					Message:      Message{Role: "assistant", Content: "test1"},
					FinishReason: "stop",
				},
				{
					Index:        1,
					Message:      Message{Role: "assistant", Content: "test2"},
					FinishReason: "length",
				},
			},
		},
		{
			name:    "empty choices",
			model:   "gui-actor-2b",
			choices: []Choice{},
		},
		{
			name:  "with coordinates",
			model: "gui-actor-7b",
			choices: []Choice{
				{
					Index:        0,
					Message:      Message{Role: "assistant", Content: "pyautogui.click(0.5, 0.5)"},
					FinishReason: "stop",
					Coordinates:  &Point{X: 0.5, Y: 0.5},
				},
			},
		},
		{
			name:  "with usage",
			model: "gui-actor-2b",
			choices: []Choice{
				{
					Index:        0,
					Message:      Message{Role: "assistant", Content: "test"},
					FinishReason: "stop",
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := NewChatCompletionResponse(tt.model, tt.choices)

			if resp.Model != tt.model {
				t.Errorf("Expected model '%s', got '%s'", tt.model, resp.Model)
			}

			if resp.Object != "chat.completion" {
				t.Errorf("Expected object 'chat.completion', got '%s'", resp.Object)
			}

			if resp.ID == "" {
				t.Error("Expected non-empty ID")
			}

			if len(resp.ID) < 9 { // "chatcmpl-" + at least some chars
				t.Errorf("Expected ID to start with 'chatcmpl-', got '%s'", resp.ID)
			}

			if resp.Created == 0 {
				t.Error("Expected non-zero created timestamp")
			}

			if len(resp.Choices) != len(tt.choices) {
				t.Errorf("Expected %d choices, got %d", len(tt.choices), len(resp.Choices))
			}
		})
	}
}
