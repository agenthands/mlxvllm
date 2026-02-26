package api

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"github.com/agenthands/gui-actor/internal/model"
)

type Handler struct {
	registry *model.Registry
	startTime time.Time
}

func NewHandler(registry *model.Registry) *Handler {
	return &Handler{
		registry: registry,
		startTime: time.Now(),
	}
}

// Health returns the server health status
func (h *Handler) Health(w http.ResponseWriter, r *http.Request) {
	resp := HealthResponse{
		Status:   "ok",
		Uptime:   int64(time.Since(h.startTime).Seconds()),
		MemoryGB: 0, // TODO: get actual memory usage
		Models:   0, // TODO: get loaded model count
	}

	writeJSON(w, http.StatusOK, resp)
}

// ListModels returns available models
func (h *Handler) ListModels(w http.ResponseWriter, r *http.Request) {
	resp := ModelsResponse{
		Object: "list",
		Data: []ModelInfo{
			{
				ID:     "gui-actor-2b",
				Object: "model",
				Loaded: false, // TODO: check registry
			},
			{
				ID:     "gui-actor-7b",
				Object: "model",
				Loaded: false,
			},
		},
	}

	writeJSON(w, http.StatusOK, resp)
}

// GetModel returns model status
func (h *Handler) GetModel(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	modelID := vars["id"]

	resp := ModelInfo{
		ID:     modelID,
		Object: "model",
		Loaded: false, // TODO: check registry
	}

	writeJSON(w, http.StatusOK, resp)
}

// LoadModel loads a model into memory
func (h *Handler) LoadModel(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	modelID := vars["id"]

	if h.registry == nil {
		writeError(w, http.StatusServiceUnavailable, "registry not available")
		return
	}

	if err := h.registry.LoadModel(modelID); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	writeJSON(w, http.StatusOK, map[string]string{"status": "loaded"})
}

// UnloadModel unloads a model from memory
func (h *Handler) UnloadModel(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	modelID := vars["id"]

	if h.registry == nil {
		writeError(w, http.StatusServiceUnavailable, "registry not available")
		return
	}

	if err := h.registry.UnloadModel(modelID); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	writeJSON(w, http.StatusOK, map[string]string{"status": "unloaded"})
}

// ChatCompletion handles inference requests
func (h *Handler) ChatCompletion(w http.ResponseWriter, r *http.Request) {
	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	// TODO: implement actual inference
	resp := NewChatCompletionResponse(req.Model, []Choice{
		{
			Index:        0,
			Message:      Message{Role: "assistant", Content: "pyautogui.click(0.5, 0.5)"},
			FinishReason: "stop",
			Coordinates:  &Point{X: 0.5, Y: 0.5},
		},
	})

	writeJSON(w, http.StatusOK, resp)
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, ErrorResponse{
		Error: ErrorDetail{
			Message: message,
			Type:    "invalid_request_error",
		},
	})
}
