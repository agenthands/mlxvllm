# Makefile for MLX RadixAttention Server
.PHONY: all test clean build

CGO_ENABLED=1
GO_TAGS=mlx
BUILD_DIR := ./bin
SERVER := $(BUILD_DIR)/server

all: build

build:
	mkdir -p $(BUILD_DIR)
	cd src && go build -tags=$(GO_TAGS) -o ../$(SERVER) ./cmd/server

test-short:
	cd src && go test ./... -short

test-integration:
	cd src && go test ./internal/mlx/... -run Integration

test-coverage:
	cd src && go test ./... -coverprofile=../coverage.out -covermode=atomic
	go tool cover -html=../coverage.out -o ../coverage.html

clean:
	rm -rf $(BUILD_DIR) coverage.out coverage.html
	go clean -cache
