#ifndef MLX_H
#define MLX_H

#include <stdint.h>

// Device types
#define MLX_DEVICE_CPU 0
#define MLX_DEVICE_GPU 1

// Context management
int mlx_init(void);
void mlx_shutdown(void);
int mlx_is_initialized(void);
int mlx_get_default_device(char* device, int max_len);

// Model loading
typedef void* mlx_model_t;
mlx_model_t mlx_load_model(const char* path, const char* device);
void mlx_unload_model(mlx_model_t model);

// Inference
typedef struct {
    float* data;
    int64_t* shape;
    int ndim;
    int dtype;
} mlx_array_t;

mlx_array_t* mlx_forward(mlx_model_t model, mlx_array_t** inputs, int num_inputs);
void mlx_free_array(mlx_array_t* arr);

#endif // MLX_H
