#ifndef GPU_MEM_FUSE_H
#define GPU_MEM_FUSE_H

#define FUSE_USE_VERSION 31

#include <fuse3/fuse.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <assert.h>
#include <unistd.h>
#include <sys/xattr.h>
#include <pthread.h>
#include <glib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <time.h>

// Configuration constants
#define MAX_PATH_LEN 512
#define MAX_ALLOCATIONS 1024
#define XATTR_GPU_SIZE "user.gpu.size"
#define XATTR_GPU_DURABLE "user.gpu.durable"
#define METADATA_DIR ".metadata"

// Allocation states
typedef enum {
    GPU_ALLOC_PENDING,    // Size specified but not allocated yet
    GPU_ALLOC_ACTIVE,     // Allocated and ready to use
    GPU_ALLOC_DURABLE,    // Marked as durable (survives crashes)
    GPU_ALLOC_TRANSIENT   // Will be cleaned up on process exit
} gpu_alloc_state_t;

// GPU allocation structure
typedef struct {
    char path[MAX_PATH_LEN];
    CUmemGenericAllocationHandle handle;
    CUdeviceptr device_ptr;
    size_t size;
    int refcount;
    gpu_alloc_state_t state;
    int export_fd;              // POSIX FD for durability
    time_t created_time;
    time_t last_access;
    pthread_mutex_t mutex;
} gpu_allocation_t;

// Pending allocation (size specified via xattr)
typedef struct {
    char path[MAX_PATH_LEN];
    size_t size;
    bool is_durable;
    time_t created;
} pending_allocation_t;

// Main FUSE context
typedef struct {
    char *mount_point;
    GHashTable *allocations;      // path -> gpu_allocation_t*
    GHashTable *pending_allocs;   // path -> pending_allocation_t*
    GHashTable *handle_map;       // handle -> gpu_allocation_t*
    pthread_mutex_t global_mutex;
    CUdevice cuda_device;
    bool recovery_mode;
    char *persistence_file;
} gpu_fuse_context_t;

// Function declarations
int gpu_fuse_init_cuda(gpu_fuse_context_t *ctx);
int gpu_fuse_create_allocation(gpu_fuse_context_t *ctx, const char *path, 
                              size_t size, bool is_durable);
gpu_allocation_t *gpu_fuse_get_allocation(gpu_fuse_context_t *ctx, const char *path);
int gpu_fuse_make_durable(gpu_fuse_context_t *ctx, gpu_allocation_t *alloc);
int gpu_fuse_cleanup_allocation(gpu_fuse_context_t *ctx, gpu_allocation_t *alloc);
int gpu_fuse_save_state(gpu_fuse_context_t *ctx);
int gpu_fuse_restore_state(gpu_fuse_context_t *ctx);

// FUSE operation declarations
static int gpu_fuse_getattr(const char *path, struct stat *stbuf, struct fuse_file_info *fi);
static int gpu_fuse_readdir(const char *path, void *buf, fuse_fill_dir_t filler,
                           off_t offset, struct fuse_file_info *fi, enum fuse_readdir_flags flags);
static int gpu_fuse_open(const char *path, struct fuse_file_info *fi);
static int gpu_fuse_create(const char *path, mode_t mode, struct fuse_file_info *fi);
static int gpu_fuse_read(const char *path, char *buf, size_t size, off_t offset,
                        struct fuse_file_info *fi);
static int gpu_fuse_write(const char *path, const char *buf, size_t size,
                         off_t offset, struct fuse_file_info *fi);
static int gpu_fuse_release(const char *path, struct fuse_file_info *fi);
static int gpu_fuse_setxattr(const char *path, const char *name, const char *value,
                            size_t size, int flags);
static int gpu_fuse_getxattr(const char *path, const char *name, char *value, size_t size);
static int gpu_fuse_listxattr(const char *path, char *list, size_t size);
static int gpu_fuse_unlink(const char *path);
static int gpu_fuse_mkdir(const char *path, mode_t mode);
static void *gpu_fuse_init(struct fuse_conn_info *conn, struct fuse_config *cfg);
static void gpu_fuse_destroy(void *private_data);

// Global context
extern gpu_fuse_context_t *g_gpu_ctx;

#endif // GPU_MEM_FUSE_H