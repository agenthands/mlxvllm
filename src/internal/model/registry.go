package model

import (
	"fmt"
	"sync"

	"github.com/agenthands/gui-actor/internal/config"
)

type ModelStatus struct {
	Name     string
	Loaded   bool
	Path     string
	MemoryGB float64
	LastUsed int64 // Unix timestamp
}

type Model interface {
	ID() string
	IsLoaded() bool
	Unload() error
}

type Registry struct {
	mu     sync.RWMutex
	cfg    *config.Config
	models map[string]*ModelStatus
	loaded map[string]Model
	totalGB float64
}

func NewRegistry(cfg *config.Config) *Registry {
	reg := &Registry{
		cfg:    cfg,
		models: make(map[string]*ModelStatus),
		loaded: make(map[string]Model),
	}

	// Register all enabled models
	for name, mcfg := range cfg.Models {
		if mcfg.Enabled {
			reg.models[name] = &ModelStatus{
				Name:     name,
				Loaded:   false,
				Path:     mcfg.Path,
				MemoryGB: estimateMemoryGB(name),
			}
		}
	}

	return reg
}

func (r *Registry) HasModel(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.models[name]
	return ok
}

func (r *Registry) LoadModel(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	status, ok := r.models[name]
	if !ok {
		return fmt.Errorf("model %s not found", name)
	}

	if status.Loaded {
		return nil // Already loaded
	}

	// Check memory constraints
	if r.totalGB+status.MemoryGB > 32 { // TODO: parse cfg.Memory.MaxTotalGB
		r.makeRoom(status.MemoryGB)
	}

	// Load model (placeholder)
	model := &GUIActorModel{
		name:   name,
		path:   status.Path,
		loaded: true,
	}

	r.loaded[name] = model
	status.Loaded = true
	r.totalGB += status.MemoryGB

	return nil
}

func (r *Registry) UnloadModel(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	model, ok := r.loaded[name]
	if !ok {
		return fmt.Errorf("model %s not loaded", name)
	}

	if err := model.Unload(); err != nil {
		return err
	}

	status := r.models[name]
	status.Loaded = false
	r.totalGB -= status.MemoryGB
	delete(r.loaded, name)

	return nil
}

func (r *Registry) GetModel(name string) (Model, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	model, ok := r.loaded[name]
	if !ok {
		return nil, fmt.Errorf("model %s not loaded", name)
	}

	return model, nil
}

func (r *Registry) makeRoom(requiredGB float64) {
	// LRU eviction logic
	// TODO: implement actual LRU
}

func estimateMemoryGB(name string) float64 {
	switch name {
	case "gui-actor-2b":
		return 4.0
	case "gui-actor-7b":
		return 14.0
	default:
		return 8.0
	}
}

// GUIActorModel is a placeholder model implementation
type GUIActorModel struct {
	name   string
	path   string
	loaded bool
}

func (m *GUIActorModel) ID() string {
	return m.name
}

func (m *GUIActorModel) IsLoaded() bool {
	return m.loaded
}

func (m *GUIActorModel) Unload() error {
	m.loaded = false
	return nil
}
