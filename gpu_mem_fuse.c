#include "gpu_mem_fuse.h"

// Global context
gpu_fuse_context_t *g_gpu_ctx = NULL;

// Helper function to check if path is metadata
static bool is_metadata_path(const char *path) {
    return strstr(path, METADATA_DIR) != NULL;
}

// Initialize CUDA context
int gpu_fuse_init_cuda(gpu_fuse_context_t *ctx) {
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to initialize CUDA: %d\n", result);
        return -1;
    }

    printf("CUDA initialized successfully\n");
    return 0;
}

// Create a new GPU allocation
int gpu_fuse_create_allocation(gpu_fuse_context_t *ctx, const char *path, 
                              size_t size, bool is_durable) {
    if (!ctx || !path || size == 0) {
        return -EINVAL;
    }
    
    pthread_mutex_lock(&ctx->global_mutex);
    
    // Check if allocation already exists
    if (g_hash_table_lookup(ctx->allocations, path)) {
        pthread_mutex_unlock(&ctx->global_mutex);
        return -EEXIST;
    }
    
    // Create new allocation
    gpu_allocation_t *alloc = calloc(1, sizeof(gpu_allocation_t));
    if (!alloc) {
        pthread_mutex_unlock(&ctx->global_mutex);
        return -ENOMEM;
    }
    
    strncpy(alloc->path, path, MAX_PATH_LEN - 1);
    alloc->size = size;
    alloc->refcount = 1;
    alloc->state = is_durable ? GPU_ALLOC_DURABLE : GPU_ALLOC_TRANSIENT;
    alloc->created_time = time(NULL);
    alloc->last_access = alloc->created_time;
    alloc->export_fd = -1;
    pthread_mutex_init(&alloc->mutex, NULL);

    // Create CUDA memory allocation
    CUmemAllocationProp props = {};
    props.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    props.location.id = ctx->cuda_device;
    
    CUresult result = cuMemCreate(&alloc->handle, size, &props, 0);
    if (result != CUDA_SUCCESS) {
        printf("Failed to create CUDA memory: %d\n", result);
        free(alloc);
        pthread_mutex_unlock(&ctx->global_mutex);
        return -ENOMEM;
    }
    
    // Export handle for durability if needed
    if (is_durable) {
        result = cuMemExportToShareableHandle(&alloc->export_fd, alloc->handle, 
                                            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
        if (result != CUDA_SUCCESS) {
            printf("Failed to export handle: %d\n", result);
            // Continue without durability
            alloc->state = GPU_ALLOC_TRANSIENT;
        }
    }
    
    // Add to hash tables
    char *path_key = strdup(path);
    g_hash_table_insert(ctx->allocations, path_key, alloc);
    g_hash_table_insert(ctx->handle_map, GINT_TO_POINTER((gintptr)alloc->handle), alloc);
    
    printf("Created GPU allocation: %s, size: %zu, ptr: %p, durable: %d\n", 
           path, size, (void*)alloc->device_ptr, is_durable);
    
    pthread_mutex_unlock(&ctx->global_mutex);
    return 0;
}

// Get allocation by path
gpu_allocation_t *gpu_fuse_get_allocation(gpu_fuse_context_t *ctx, const char *path) {
    if (!ctx || !path) return NULL;
    
    pthread_mutex_lock(&ctx->global_mutex);
    gpu_allocation_t *alloc = g_hash_table_lookup(ctx->allocations, path);
    if (alloc) {
        alloc->last_access = time(NULL);
    }
    pthread_mutex_unlock(&ctx->global_mutex);
    return alloc;
}

// Make allocation durable
int gpu_fuse_make_durable(gpu_fuse_context_t *ctx, gpu_allocation_t *alloc) {
    if (!ctx || !alloc) return -EINVAL;
    
    pthread_mutex_lock(&alloc->mutex);
    
    if (alloc->state == GPU_ALLOC_DURABLE) {
        pthread_mutex_unlock(&alloc->mutex);
        return 0; // Already durable
    }
    
    if (alloc->export_fd == -1) {
        CUresult result = cuMemExportToShareableHandle(&alloc->export_fd, alloc->handle,
                                                     CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
        if (result != CUDA_SUCCESS) {
            pthread_mutex_unlock(&alloc->mutex);
            return -EIO;
        }
    }
    
    alloc->state = GPU_ALLOC_DURABLE;
    alloc->refcount++;
    
    pthread_mutex_unlock(&alloc->mutex);
    printf("Made allocation durable: %s\n", alloc->path);
    return 0;
}

// Cleanup allocation
int gpu_fuse_cleanup_allocation(gpu_fuse_context_t *ctx, gpu_allocation_t *alloc) {
    if (!ctx || !alloc) return -EINVAL;
    
    pthread_mutex_lock(&ctx->global_mutex);
    pthread_mutex_lock(&alloc->mutex);
    
    alloc->refcount--;
    
    if (alloc->refcount > 0 && alloc->state == GPU_ALLOC_DURABLE) {
        pthread_mutex_unlock(&alloc->mutex);
        pthread_mutex_unlock(&ctx->global_mutex);
        return 0; // Still has references and is durable
    }
    
    printf("Cleaning up allocation: %s\n", alloc->path);
    
    // Cleanup CUDA resources
    if (alloc->handle) {
        cuMemRelease(alloc->handle);
    }
    
    if (alloc->export_fd != -1) {
        close(alloc->export_fd);
    }
    
    // Remove from hash tables
    g_hash_table_remove(ctx->allocations, alloc->path);
    g_hash_table_remove(ctx->handle_map, GINT_TO_POINTER((gintptr)alloc->handle));
    
    pthread_mutex_unlock(&alloc->mutex);
    pthread_mutex_destroy(&alloc->mutex);
    free(alloc);
    
    pthread_mutex_unlock(&ctx->global_mutex);
    return 0;
}

// FUSE operation implementations
static int gpu_fuse_getattr(const char *path, struct stat *stbuf, struct fuse_file_info *fi) {
    (void) fi;
    memset(stbuf, 0, sizeof(struct stat));
    
    if (strcmp(path, "/") == 0) {
        stbuf->st_mode = S_IFDIR | 0755;
        stbuf->st_nlink = 2;
        return 0;
    }
    
    if (is_metadata_path(path)) {
        stbuf->st_mode = S_IFDIR | 0755;
        stbuf->st_nlink = 2;
        return 0;
    }
    
    gpu_allocation_t *alloc = gpu_fuse_get_allocation(g_gpu_ctx, path);
    if (alloc) {
        stbuf->st_mode = S_IFREG | 0644;
        stbuf->st_nlink = 1;
        stbuf->st_size = alloc->size;
        stbuf->st_atime = alloc->last_access;
        stbuf->st_mtime = alloc->created_time;
        stbuf->st_ctime = alloc->created_time;
        return 0;
    }
    
    // Check if it's a pending allocation
    pthread_mutex_lock(&g_gpu_ctx->global_mutex);
    pending_allocation_t *pending = g_hash_table_lookup(g_gpu_ctx->pending_allocs, path);
    pthread_mutex_unlock(&g_gpu_ctx->global_mutex);
    
    if (pending) {
        stbuf->st_mode = S_IFREG | 0644;
        stbuf->st_nlink = 1;
        stbuf->st_size = 0; // Pending allocation has no content yet
        stbuf->st_atime = pending->created;
        stbuf->st_mtime = pending->created;
        stbuf->st_ctime = pending->created;
        return 0;
    }
    
    return -ENOENT;
}

static int gpu_fuse_readdir(const char *path, void *buf, fuse_fill_dir_t filler,
                           off_t offset, struct fuse_file_info *fi, enum fuse_readdir_flags flags) {
    (void) offset;
    (void) fi;
    (void) flags;
    
    if (strcmp(path, "/") != 0) {
        return -ENOENT;
    }
    
    filler(buf, ".", NULL, 0, 0);
    filler(buf, "..", NULL, 0, 0);
    filler(buf, METADATA_DIR, NULL, 0, 0);
    
    // List all allocations
    pthread_mutex_lock(&g_gpu_ctx->global_mutex);
    
    GHashTableIter iter;
    gpointer key, value;
    
    g_hash_table_iter_init(&iter, g_gpu_ctx->allocations);
    while (g_hash_table_iter_next(&iter, &key, &value)) {
        char *alloc_path = (char*)key;
        if (alloc_path[0] == '/') alloc_path++; // Skip leading slash
        filler(buf, alloc_path, NULL, 0, 0);
    }
    
    g_hash_table_iter_init(&iter, g_gpu_ctx->pending_allocs);
    while (g_hash_table_iter_next(&iter, &key, &value)) {
        char *alloc_path = (char*)key;
        if (alloc_path[0] == '/') alloc_path++; // Skip leading slash
        filler(buf, alloc_path, NULL, 0, 0);
    }
    
    pthread_mutex_unlock(&g_gpu_ctx->global_mutex);
    
    return 0;
}

static int gpu_fuse_open(const char *path, struct fuse_file_info *fi) {
    gpu_allocation_t *alloc = gpu_fuse_get_allocation(g_gpu_ctx, path);
    if (!alloc) {
        return -ENOENT;
    }
    
    pthread_mutex_lock(&alloc->mutex);
    alloc->refcount++;
    pthread_mutex_unlock(&alloc->mutex);
    
    return 0;
}

static int gpu_fuse_create(const char *path, mode_t mode, struct fuse_file_info *fi) {
    (void) mode;
    (void) fi;
    
    // Check if we have a pending allocation with size specified
    pthread_mutex_lock(&g_gpu_ctx->global_mutex);
    pending_allocation_t *pending = g_hash_table_lookup(g_gpu_ctx->pending_allocs, path);
    
    if (pending && pending->size > 0) {
        // Create the actual allocation
        size_t size = pending->size;
        bool is_durable = pending->is_durable;
        
        g_hash_table_remove(g_gpu_ctx->pending_allocs, path);
        pthread_mutex_unlock(&g_gpu_ctx->global_mutex);
        
        return gpu_fuse_create_allocation(g_gpu_ctx, path, size, is_durable);
    }
    
    pthread_mutex_unlock(&g_gpu_ctx->global_mutex);
    
    // Create pending allocation (size will be specified via xattr)
    pending_allocation_t *new_pending = malloc(sizeof(pending_allocation_t));
    if (!new_pending) {
        return -ENOMEM;
    }
    
    strncpy(new_pending->path, path, MAX_PATH_LEN - 1);
    new_pending->size = 0;
    new_pending->is_durable = false;
    new_pending->created = time(NULL);
    
    pthread_mutex_lock(&g_gpu_ctx->global_mutex);
    char *path_key = strdup(path);
    g_hash_table_insert(g_gpu_ctx->pending_allocs, path_key, new_pending);
    pthread_mutex_unlock(&g_gpu_ctx->global_mutex);
    
    return 0;
}

static int gpu_fuse_read(const char *path, char *buf, size_t size, off_t offset,
                        struct fuse_file_info *fi) {
    (void) fi;
    
    gpu_allocation_t *alloc = gpu_fuse_get_allocation(g_gpu_ctx, path);
    if (!alloc) {
        return -ENOENT;
    }
    
    if (offset >= (off_t)alloc->size) {
        return 0;
    }
    
    if (offset + size > alloc->size) {
        size = alloc->size - offset;
    }
    
    // For now, just return allocation info as text
    char info[256];
    snprintf(info, sizeof(info), 
             "GPU Allocation Info:\n"
             "Path: %s\n"
             "Size: %zu bytes\n"
             "Device Pointer: %p\n"
             "State: %s\n"
             "Refcount: %d\n",
             alloc->path, alloc->size, (void*)alloc->device_ptr,
             (alloc->state == GPU_ALLOC_DURABLE) ? "durable" : "transient",
             alloc->refcount);
    
    size_t info_len = strlen(info);
    if (offset >= (off_t)info_len) {
        return 0;
    }
    
    if (offset + size > info_len) {
        size = info_len - offset;
    }
    
    memcpy(buf, info + offset, size);
    return size;
}

static int gpu_fuse_write(const char *path, const char *buf, size_t size,
                         off_t offset, struct fuse_file_info *fi) {
    (void) buf;
    (void) offset;
    (void) fi;
    
    gpu_allocation_t *alloc = gpu_fuse_get_allocation(g_gpu_ctx, path);
    if (!alloc) {
        return -ENOENT;
    }
    
    // For now, just acknowledge the write
    return size;
}

static int gpu_fuse_release(const char *path, struct fuse_file_info *fi) {
    (void) fi;
    
    gpu_allocation_t *alloc = gpu_fuse_get_allocation(g_gpu_ctx, path);
    if (!alloc) {
        return -ENOENT;
    }
    
    pthread_mutex_lock(&alloc->mutex);
    alloc->refcount--;
    
    if (alloc->refcount <= 0 && alloc->state == GPU_ALLOC_TRANSIENT) {
        pthread_mutex_unlock(&alloc->mutex);
        gpu_fuse_cleanup_allocation(g_gpu_ctx, alloc);
    } else {
        pthread_mutex_unlock(&alloc->mutex);
    }
    
    return 0;
}

static int gpu_fuse_setxattr(const char *path, const char *name, const char *value,
                            size_t size, int flags) {
    (void) flags;
    
    if (strcmp(name, XATTR_GPU_SIZE) == 0) {
        size_t alloc_size = strtoull(value, NULL, 10);
        if (alloc_size == 0) {
            return -EINVAL;
        }
        
        pthread_mutex_lock(&g_gpu_ctx->global_mutex);
        pending_allocation_t *pending = g_hash_table_lookup(g_gpu_ctx->pending_allocs, path);
        
        if (!pending) {
            // Create new pending allocation
            pending = malloc(sizeof(pending_allocation_t));
            if (!pending) {
                pthread_mutex_unlock(&g_gpu_ctx->global_mutex);
                return -ENOMEM;
            }
            
            strncpy(pending->path, path, MAX_PATH_LEN - 1);
            pending->size = alloc_size;
            pending->is_durable = false;
            pending->created = time(NULL);
            
            char *path_key = strdup(path);
            g_hash_table_insert(g_gpu_ctx->pending_allocs, path_key, pending);
        } else {
            pending->size = alloc_size;
        }
        
        pthread_mutex_unlock(&g_gpu_ctx->global_mutex);
        printf("Set allocation size for %s: %zu bytes\n", path, alloc_size);
        return 0;
    }
    
    if (strcmp(name, XATTR_GPU_DURABLE) == 0) {
        bool is_durable = (strcmp(value, "1") == 0 || strcmp(value, "true") == 0);
        
        // Check if it's an existing allocation
        gpu_allocation_t *alloc = gpu_fuse_get_allocation(g_gpu_ctx, path);
        if (alloc) {
            if (is_durable) {
                return gpu_fuse_make_durable(g_gpu_ctx, alloc);
            } else {
                pthread_mutex_lock(&alloc->mutex);
                alloc->state = GPU_ALLOC_TRANSIENT;
                pthread_mutex_unlock(&alloc->mutex);
                return 0;
            }
        }
        
        // Set for pending allocation
        pthread_mutex_lock(&g_gpu_ctx->global_mutex);
        pending_allocation_t *pending = g_hash_table_lookup(g_gpu_ctx->pending_allocs, path);
        if (pending) {
            pending->is_durable = is_durable;
        }
        pthread_mutex_unlock(&g_gpu_ctx->global_mutex);
        
        printf("Set durability for %s: %s\n", path, is_durable ? "true" : "false");
        return 0;
    }
    
    return -ENODATA;
}

static int gpu_fuse_getxattr(const char *path, const char *name, char *value, size_t size) {
    if (strcmp(name, XATTR_GPU_SIZE) == 0) {
        gpu_allocation_t *alloc = gpu_fuse_get_allocation(g_gpu_ctx, path);
        if (alloc) {
            char size_str[32];
            snprintf(size_str, sizeof(size_str), "%zu", alloc->size);
            size_t len = strlen(size_str);
            
            if (size == 0) return len;
            if (size < len) return -ERANGE;
            
            strcpy(value, size_str);
            return len;
        }
        
        pthread_mutex_lock(&g_gpu_ctx->global_mutex);
        pending_allocation_t *pending = g_hash_table_lookup(g_gpu_ctx->pending_allocs, path);
        if (pending && pending->size > 0) {
            char size_str[32];
            snprintf(size_str, sizeof(size_str), "%zu", pending->size);
            size_t len = strlen(size_str);
            
            pthread_mutex_unlock(&g_gpu_ctx->global_mutex);
            
            if (size == 0) return len;
            if (size < len) return -ERANGE;
            
            strcpy(value, size_str);
            return len;
        }
        pthread_mutex_unlock(&g_gpu_ctx->global_mutex);
    }
    
    if (strcmp(name, XATTR_GPU_DURABLE) == 0) {
        gpu_allocation_t *alloc = gpu_fuse_get_allocation(g_gpu_ctx, path);
        if (alloc) {
            const char *durable_str = (alloc->state == GPU_ALLOC_DURABLE) ? "true" : "false";
            size_t len = strlen(durable_str);
            
            if (size == 0) return len;
            if (size < len) return -ERANGE;
            
            strcpy(value, durable_str);
            return len;
        }
    }
    
    return -ENODATA;
}

static int gpu_fuse_listxattr(const char *path, char *list, size_t size) {
    gpu_allocation_t *alloc = gpu_fuse_get_allocation(g_gpu_ctx, path);
    if (!alloc) {
        pthread_mutex_lock(&g_gpu_ctx->global_mutex);
        pending_allocation_t *pending = g_hash_table_lookup(g_gpu_ctx->pending_allocs, path);
        pthread_mutex_unlock(&g_gpu_ctx->global_mutex);
        
        if (!pending) {
            return -ENOENT;
        }
    }
    
    const char *attrs = XATTR_GPU_SIZE "\0" XATTR_GPU_DURABLE "\0";
    size_t attrs_len = strlen(XATTR_GPU_SIZE) + 1 + strlen(XATTR_GPU_DURABLE) + 1;
    
    if (size == 0) return attrs_len;
    if (size < attrs_len) return -ERANGE;
    
    memcpy(list, attrs, attrs_len);
    return attrs_len;
}

static int gpu_fuse_unlink(const char *path) {
    gpu_allocation_t *alloc = gpu_fuse_get_allocation(g_gpu_ctx, path);
    if (alloc) {
        return gpu_fuse_cleanup_allocation(g_gpu_ctx, alloc);
    }
    
    // Remove pending allocation
    pthread_mutex_lock(&g_gpu_ctx->global_mutex);
    if (g_hash_table_remove(g_gpu_ctx->pending_allocs, path)) {
        pthread_mutex_unlock(&g_gpu_ctx->global_mutex);
        return 0;
    }
    pthread_mutex_unlock(&g_gpu_ctx->global_mutex);
    
    return -ENOENT;
}

static int gpu_fuse_mkdir(const char *path, mode_t mode) {
    (void) mode;
    
    if (strstr(path, METADATA_DIR)) {
        return 0; // Allow metadata directory creation
    }
    
    return -EACCES; // Don't allow other directories
}

static void *gpu_fuse_init(struct fuse_conn_info *conn, struct fuse_config *cfg) {
    (void) conn;
    (void) cfg;
    printf("GPU Memory FUSE filesystem initialized\n");
    return g_gpu_ctx;
}

static void gpu_fuse_destroy(void *private_data) {
    (void) private_data;
    
    if (g_gpu_ctx) {
        printf("Destroying GPU Memory FUSE filesystem\n");
        
        // Cleanup all allocations
        pthread_mutex_lock(&g_gpu_ctx->global_mutex);
        
        GHashTableIter iter;
        gpointer key, value;
        
        g_hash_table_iter_init(&iter, g_gpu_ctx->allocations);
        while (g_hash_table_iter_next(&iter, &key, &value)) {
            gpu_allocation_t *alloc = (gpu_allocation_t*)value;
            if (alloc->state == GPU_ALLOC_TRANSIENT) {
                gpu_fuse_cleanup_allocation(g_gpu_ctx, alloc);
            }
        }
        
        pthread_mutex_unlock(&g_gpu_ctx->global_mutex);
        
        // Cleanup hash tables
        g_hash_table_destroy(g_gpu_ctx->allocations);
        g_hash_table_destroy(g_gpu_ctx->pending_allocs);
        g_hash_table_destroy(g_gpu_ctx->handle_map);
        
        pthread_mutex_destroy(&g_gpu_ctx->global_mutex);
        
        free(g_gpu_ctx->mount_point);
        free(g_gpu_ctx->persistence_file);
        free(g_gpu_ctx);
        g_gpu_ctx = NULL;
    }
}

// FUSE operations structure
static struct fuse_operations gpu_fuse_ops = {
    .getattr    = gpu_fuse_getattr,
    .readdir    = gpu_fuse_readdir,
    .open       = gpu_fuse_open,
    .create     = gpu_fuse_create,
    .read       = gpu_fuse_read,
    .write      = gpu_fuse_write,
    .release    = gpu_fuse_release,
    .setxattr   = gpu_fuse_setxattr,
    .getxattr   = gpu_fuse_getxattr,
    .listxattr  = gpu_fuse_listxattr,
    .unlink     = gpu_fuse_unlink,
    .mkdir      = gpu_fuse_mkdir,
    .init       = gpu_fuse_init,
    .destroy    = gpu_fuse_destroy,
};

// Main function
int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <mountpoint> [FUSE options]\n", argv[0]);
        return 1;
    }
    
    // Initialize global context
    g_gpu_ctx = calloc(1, sizeof(gpu_fuse_context_t));
    if (!g_gpu_ctx) {
        fprintf(stderr, "Failed to allocate context\n");
        return 1;
    }
    
    g_gpu_ctx->mount_point = strdup(argv[1]);
    g_gpu_ctx->allocations = g_hash_table_new_full(g_str_hash, g_str_equal, free, NULL);
    g_gpu_ctx->pending_allocs = g_hash_table_new_full(g_str_hash, g_str_equal, free, free);
    g_gpu_ctx->handle_map = g_hash_table_new(g_direct_hash, g_direct_equal);
    pthread_mutex_init(&g_gpu_ctx->global_mutex, NULL);
    
    // Initialize CUDA
    if (gpu_fuse_init_cuda(g_gpu_ctx) != 0) {
        fprintf(stderr, "Failed to initialize CUDA\n");
        return 1;
    }
    
    printf("Starting GPU Memory FUSE filesystem on %s\n", argv[1]);
    
    // Start FUSE
    return fuse_main(argc, argv, &gpu_fuse_ops, NULL);
}