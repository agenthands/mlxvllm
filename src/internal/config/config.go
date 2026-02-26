package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

type ServerConfig struct {
	Host         string `yaml:"host"`
	Port         int    `yaml:"port"`
	DefaultModel string `yaml:"default_model"`
}

type ModelConfig struct {
	Path             string `yaml:"path"`
	Enabled          bool   `yaml:"enabled"`
	Preload          bool   `yaml:"preload"`
	MinPixels        int    `yaml:"min_pixels"`
	MaxPixels        int    `yaml:"max_pixels"`
	MaxContextLength int    `yaml:"max_context_length"`
	MemoryLimitGB    int    `yaml:"memory_limit_gb"`
}

type ProfileConfig struct {
	MaxPixels        int `yaml:"max_pixels"`
	MaxContextLength int `yaml:"max_context_length"`
}

type MemoryConfig struct {
	MaxTotalGB     string   `yaml:"max_total_gb"`
	UnloadStrategy string   `yaml:"unload_strategy"`
	KeepModels     []string `yaml:"keep_models"`
}

type LoggingConfig struct {
	Level  string `yaml:"level"`
	Format string `yaml:"format"`
}

type Config struct {
	Server   ServerConfig              `yaml:"server"`
	Models   map[string]ModelConfig    `yaml:"models"`
	Profiles map[string]ProfileConfig  `yaml:"profiles"`
	Memory   MemoryConfig              `yaml:"memory"`
	Logging  LoggingConfig             `yaml:"logging"`
}

func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	return &cfg, nil
}
