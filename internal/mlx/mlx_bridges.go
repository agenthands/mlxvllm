//go:build !mlx_mock

package mlx

/*
#cgo CFLAGS: -I.
#cgo LDFLAGS: -lmlx_runtime

#include "mlx_api.h"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"unsafe"
)

// Constants from C API
const (
	RootCacheHandle = uint64(C.MLX_ROOT_CACHE_HANDLE)
	Success         = int(C.MLX_SUCCESS)
	ErrorInvalidHandle = int(C.MLX_ERROR_INVALID_HANDLE)
	ErrorOutOfMemory   = int(C.MLX_ERROR_OUT_OF_MEMORY)
	ErrorInvalidTokens  = int(C.MLX_ERROR_INVALID_TOKENS)
	ErrorComputationFailed = int(C.MLX_ERROR_COMPUTATION_FAILED)
	ErrorModelNotLoaded = int(C.MLX_ERROR_MODEL_NOT_LOADED)
)

// ForwardWithCache executes MLX inference with KV cache
// This is a Go wrapper around the C API
func ForwardWithCache(
	modelHandle uintptr,
	tokens []uint32,
	baseCacheHandle uint64,
) ([]float32, uint64, error) {
	if len(tokens) == 0 {
		return nil, 0, nil
	}

	// Allocate output buffers
	outLogits := make([]float32, 32000) // TODO: get vocab size from model
	var outCacheHandle C.uint64_t
	var outErrorMsg *C.char

	// Call C function
	ret := C.MLXForwardWithCache(
		C.uintptr_t(modelHandle),
		(*C.uint32_t)(unsafe.Pointer(&tokens[0])),
		C.int(len(tokens)),
		C.uint64_t(baseCacheHandle),
		(*C.float)(unsafe.Pointer(&outLogits[0])),
		C.int(len(outLogits)),
		&outCacheHandle,
		&outErrorMsg,
	)

	// Check for errors
	if ret != C.MLX_SUCCESS {
		if outErrorMsg != nil {
			errMsg := C.GoString(outErrorMsg)
			C.MLXFreeError(outErrorMsg)
			return nil, 0, errors.New(errMsg)
		}
		return nil, 0, errors.New("MLX error: unknown failure")
	}

	return outLogits, uint64(outCacheHandle), nil
}

// SliceCache creates a zero-copy view of an existing cache
func SliceCache(cacheHandle uint64, keepTokens int) (uint64, error) {
	var outSlicedHandle C.uint64_t
	var outErrorMsg *C.char

	ret := C.MLXSliceCache(
		C.uint64_t(cacheHandle),
		C.int(keepTokens),
		&outSlicedHandle,
		&outErrorMsg,
	)

	if ret != C.MLX_SUCCESS {
		if outErrorMsg != nil {
			errMsg := C.GoString(outErrorMsg)
			C.MLXFreeError(outErrorMsg)
			return 0, errors.New(errMsg)
		}
		return 0, errors.New("MLX error: unknown failure")
	}

	return uint64(outSlicedHandle), nil
}

// FreeCache releases a cache handle
func FreeCache(cacheHandle uint64) {
	C.MLXFreeCache(C.uint64_t(cacheHandle))
}

// FreeError frees an error message
func FreeError(errMsg *C.char) {
	C.MLXFreeError(errMsg)
}
