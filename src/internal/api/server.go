package api

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/gorilla/mux"
)

type Server struct {
	httpServer *http.Server
	Handler    http.Handler
}

func NewServer(addr string, handler *Handler) *Server {
	r := mux.NewRouter()
	api := r.PathPrefix("/v1").Subrouter()

	// Register routes
	api.HandleFunc("/health", handler.Health).Methods("GET")
	api.HandleFunc("/models", handler.ListModels).Methods("GET")
	api.HandleFunc("/models/{id}", handler.GetModel).Methods("GET")
	api.HandleFunc("/models/{id}/load", handler.LoadModel).Methods("POST")
	api.HandleFunc("/models/{id}", handler.UnloadModel).Methods("DELETE")
	api.HandleFunc("/chat/completions", handler.ChatCompletion).Methods("POST")

	httpSrv := &http.Server{
		Addr:         addr,
		Handler:      r,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	return &Server{
		httpServer: httpSrv,
		Handler:    r,
	}
}

func (s *Server) Start() error {
	fmt.Printf("Server listening on %s\n", s.httpServer.Addr)
	return s.httpServer.ListenAndServe()
}

func (s *Server) Shutdown(ctx context.Context) error {
	return s.httpServer.Shutdown(ctx)
}
