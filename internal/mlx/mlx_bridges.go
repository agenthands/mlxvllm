//go:build !mlx_mock

package mlx

/*
#cgo CFLAGS: -I.
#cgo darwin LDFLAGS: -L. -lmlx_runtime -framework Metal -framework Foundation -framework CoreGraphics
#cgo darwin LDFLAGS: -lstdc++ -Wl,-rpath,/Users/Janis_Vizulis/go/src/github.com/agenthands/mlxvllm/internal/mlx

#include <stdlib.h>
#include "mlx_api.h"
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
// logits must be pre-allocated by caller with size vocab_size
// This enables zero-copy: C++ writes directly to Go's memory
func ForwardWithCache(
	modelHandle uintptr,
	tokens []uint32,
	baseCacheHandle uint64,
	logits []float32,
) (uint64, error) {
	if len(tokens) == 0 {
		return 0, nil
	}

	if len(logits) == 0 {
		return 0, errors.New("logits buffer must be pre-allocated")
	}

	var outCacheHandle C.uint64_t
	var outErrorMsg *C.char

	// Call C function
	ret := C.MLXForwardWithCache(
		C.uintptr_t(modelHandle),
		(*C.uint32_t)(unsafe.Pointer(&tokens[0])),
		C.int(len(tokens)),
		C.uint64_t(baseCacheHandle),
		(*C.float)(unsafe.Pointer(&logits[0])),
		C.int(len(logits)),
		&outCacheHandle,
		&outErrorMsg,
	)

	// Check for errors
	if ret != C.MLX_SUCCESS {
		if outErrorMsg != nil {
			errMsg := C.GoString(outErrorMsg)
			C.MLXFreeError(outErrorMsg)
			return 0, errors.New(errMsg)
		}
		return 0, errors.New("MLX error: unknown failure")
	}

	return uint64(outCacheHandle), nil
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

// LoadModel loads an MLX model for inference
func LoadModel(modelPath string, vocabSize int) error {
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	ret := C.MLXLoadModel(cPath, C.int(vocabSize))

	if ret != C.MLX_SUCCESS {
		return errors.New("MLX error: failed to load model")
	}

	return nil
}
