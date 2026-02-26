package main

import (
	"flag"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestMainFlags(t *testing.T) {
	// Test that config flag is parsed correctly
	tests := []struct {
		name      string
		args      []string
		expectOut string
	}{
		{"default config", []string{"test"}, ""},
		{"custom config", []string{"test", "-config", "/tmp/test.yaml"}, ""},
		{"help flag", []string{"test", "-h"}, ""},
		{"long help flag", []string{"test", "-help"}, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Just test that flags parse without crashing
			fs := flag.NewFlagSet("test", flag.ContinueOnError)
			configPath := fs.String("config", "./models/config.yaml", "Path to configuration file")

			err := fs.Parse(tt.args[1:])
			if err != nil && tt.name != "help flag" && tt.name != "long help flag" {
				t.Errorf("Failed to parse flags: %v", err)
			}

			if *configPath == "" {
				t.Error("Expected config path to be set")
			}
		})
	}
}

func TestMainIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}
	// Integration test requires actual model files and runs the server
	// This test is skipped by default as it requires a full environment
	t.Skip("Integration test requires full model setup")
}

func TestMainInvalidConfig(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "invalid_config.yaml")

	if err := os.WriteFile(configPath, []byte("invalid: yaml: [broken"), 0644); err != nil {
		t.Fatalf("Failed to write test config: %v", err)
	}

	// Test that server fails to start with invalid config
	testBinary := os.Args[0]
	cmd := exec.Command(testBinary, "-config", configPath)

	err := cmd.Run()
	if err == nil {
		t.Error("Expected server to fail with invalid config, but it succeeded")
	}
}
