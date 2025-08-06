#ifndef GPU_MEM_FUSE_H
#define GPU_MEM_FUSE_H

#include <fuse3/fuse.h>
#include <cuda.h>
#include <glib.h>
#include <pthread.h>
#include <stdbool.h>
#include <time.h>

// Configuration constants
#define MAX_PATH_LEN 512

#define UNUSED(x) (void)(x)

// Simple file entry - tracks files and their GPU allocations
typedef struct {
    char path[MAX_PATH_LEN];
    CUmemGenericAllocationHandle gpu_handle;  // 0 means no GPU memory allocated
    CUmemFabricHandle fabric_handle;          // 0 means no fabric handle allocated
    size_t size;                              // 0 means no GPU memory allocated
    time_t created_time;
    time_t access_time;
    time_t modify_time;
    pthread_mutex_t mutex;
} gpu_file_t;

// Main FUSE context
typedef struct {
    char *mount_point;
    GHashTable *files;            // path -> gpu_file_t*
    pthread_mutex_t global_mutex;
    CUdevice cuda_device;
} gpu_fuse_context_t;

// Function declarations
int gpu_fuse_init_cuda(gpu_fuse_context_t *ctx);
//gpu_file_t *gpu_fuse_get_file(gpu_fuse_context_t *ctx, const char *path);
int gpu_fuse_cleanup_gpu_memory(gpu_file_t *file);

#endif // GPU_MEM_FUSE_H