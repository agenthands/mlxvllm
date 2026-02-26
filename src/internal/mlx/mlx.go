package mlx

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -lm
#include "mlx.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

type Device string

const (
	DeviceCPU Device = "cpu"
	DeviceGPU Device = "metal"
)

func Init() error {
	ret := C.mlx_init()
	if ret == 0 {
		return fmt.Errorf("failed to initialize MLX")
	}
	return nil
}

func Shutdown() {
	C.mlx_shutdown()
}

func IsInitialized() bool {
	return C.mlx_is_initialized() == 1
}

func GetDefaultDevice() string {
	buf := make([]byte, 32)
	C.mlx_get_default_device((*C.char)(unsafe.Pointer(&buf[0])), 32)
	return string(buf)
}
