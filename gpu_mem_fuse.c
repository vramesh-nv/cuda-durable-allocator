#define FUSE_USE_VERSION 31

#include "gpu_mem_fuse.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fuse3/fuse.h>
#include <glib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <fcntl.h>

// Global context
static gpu_fuse_context_t *g_gpu_ctx = NULL;

// CUDA initialization
int gpu_fuse_init_cuda(gpu_fuse_context_t *ctx)
{
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        printf("Failed to initialize CUDA: %d\n", result);
        return -1;
    }
    
    result = cuDeviceGet(&ctx->cuda_device, 0);
    if (result != CUDA_SUCCESS) {
        printf("Failed to get CUDA device: %d\n", result);
        return -1;
    }
    
    printf("CUDA initialized successfully\n");
    return 0;
}

// Helper function to get file by path
gpu_file_t *gpu_fuse_get_file_from_path(gpu_fuse_context_t *ctx, const char *path)
{
    pthread_mutex_lock(&ctx->global_mutex);
    gpu_file_t *file = g_hash_table_lookup(ctx->files, path);
    pthread_mutex_unlock(&ctx->global_mutex);
    return file;
}

// Cleanup GPU memory for a file
int gpu_fuse_cleanup_gpu_memory(gpu_file_t *file)
{
    if (file->gpu_handle != 0) {
        CUresult result = cuMemRelease(file->gpu_handle);
        if (result != CUDA_SUCCESS) {
            printf("Failed to release GPU memory: %d\n", result);
            return -1;
        }
        file->gpu_handle = 0;
        file->size = 0;
        printf("Released GPU memory for %s\n", file->path);
    }
    return 0;
}

// FUSE getattr - check file attributes
static int gpu_fuse_getattr(const char *path, struct stat *stbuf, struct fuse_file_info *fi)
{
    UNUSED(fi);
    
    memset(stbuf, 0, sizeof(struct stat));
    
    if (strcmp(path, "/") == 0) {
        stbuf->st_mode = S_IFDIR | 0755;
        stbuf->st_nlink = 2;
        return 0;
    }
    
    gpu_file_t *file = gpu_fuse_get_file_from_path(g_gpu_ctx, path);
    if (file) {
        pthread_mutex_lock(&file->mutex);
        stbuf->st_mode = S_IFREG | 0644;
        stbuf->st_nlink = 1;
        stbuf->st_size = file->size;
        stbuf->st_atime = file->access_time;
        stbuf->st_mtime = file->modify_time;
        stbuf->st_ctime = file->created_time;
        pthread_mutex_unlock(&file->mutex);
        return 0;
    }
    
    return -ENOENT;
}

// FUSE readdir - list directory contents
static int gpu_fuse_readdir(const char *path, void *buf, fuse_fill_dir_t filler,
                           off_t offset, struct fuse_file_info *fi, enum fuse_readdir_flags flags)
{
    UNUSED(offset);
    UNUSED(fi);
    UNUSED(flags);
    
    if (strcmp(path, "/") != 0) {
        return -ENOENT;
    }
    
    filler(buf, ".", NULL, 0, 0);
    filler(buf, "..", NULL, 0, 0);
    
    // List all files
    pthread_mutex_lock(&g_gpu_ctx->global_mutex);
    
    GHashTableIter iter;
    gpointer key, value;
    
    g_hash_table_iter_init(&iter, g_gpu_ctx->files);
    while (g_hash_table_iter_next(&iter, &key, &value)) {
        char *file_path = (char*)key;
        if (file_path[0] == '/') file_path++; // Skip leading slash
        filler(buf, file_path, NULL, 0, 0);
    }
    
    pthread_mutex_unlock(&g_gpu_ctx->global_mutex);
    
    return 0;
}

// FUSE create - create a new file path (no GPU memory allocated yet)
static int gpu_fuse_create(const char *path, mode_t mode, struct fuse_file_info *fi)
{
    UNUSED(mode);
    //UNUSED(fi);
    
    // Check if file already exists
    gpu_file_t *existing = gpu_fuse_get_file_from_path(g_gpu_ctx, path);
    if (existing) {
        printf("File %s already exists\n", path);
        return 0;  // File already exists, return success
    }
    
    // Create a new file entry (no GPU memory allocated yet)
    gpu_file_t *new_file = malloc(sizeof(gpu_file_t));
    if (!new_file) {
        return -ENOMEM;
    }
    
    strncpy(new_file->path, path, MAX_PATH_LEN - 1);
    new_file->path[MAX_PATH_LEN - 1] = '\0';
    new_file->gpu_handle = 0;          // No GPU memory allocated yet
    new_file->size = 0;                // No size yet
    time_t current_time = time(NULL);
    new_file->created_time = current_time;
    new_file->access_time = current_time;
    new_file->modify_time = current_time;
    pthread_mutex_init(&new_file->mutex, NULL);
    
    pthread_mutex_lock(&g_gpu_ctx->global_mutex);
    char *path_key = strdup(path);
    g_hash_table_insert(g_gpu_ctx->files, path_key, new_file);
    pthread_mutex_unlock(&g_gpu_ctx->global_mutex);

    printf("Created file entry %s (no GPU memory allocated yet)\n", path);
    return 0;
}

// FUSE truncate - allocate/deallocate GPU memory based on size
static int gpu_fuse_truncate(const char *path, off_t size, struct fuse_file_info *fi)
{
    UNUSED(fi);
    
    printf("gpu_fuse_truncate called: path=%s, size=%ld\n", path, size);
    
    if (size < 0) {
        return -EINVAL;
    }
    
    // Get the file
    gpu_file_t *file = gpu_fuse_get_file_from_path(g_gpu_ctx, path);
    if (!file) {
        return -ENOENT;  // File doesn't exist
    }
    
    pthread_mutex_lock(&file->mutex);
    
    if (size == 0) {
        // Truncate to 0 - deallocate GPU memory if allocated
        if (file->gpu_handle != 0) {
            printf("Deallocating GPU memory for %s\n", path);
            CUresult result = cuMemRelease(file->gpu_handle);
            if (result != CUDA_SUCCESS) {
                printf("cuMemRelease failed: %d\n", result);
                pthread_mutex_unlock(&file->mutex);
                return -EIO;
            }
            file->gpu_handle = 0;
        }
        file->size = 0;
        file->modify_time = time(NULL);  // Update modification time
        pthread_mutex_unlock(&file->mutex);
        printf("File %s truncated to 0 (GPU memory deallocated)\n", path);
        return 0;
    }
    
    if (file->size == 0 && file->gpu_handle == 0) {
        // This is a new allocation - create GPU memory
        printf("Allocating GPU memory for %s with size %ld bytes\n", path, size);
        
        // Setup allocation properties
        CUmemAllocationProp props = {};
        props.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        props.location.id = g_gpu_ctx->cuda_device;
        props.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        
        CUmemGenericAllocationHandle gpu_handle;
        CUresult result = cuMemCreate(&gpu_handle, size, &props, 0);
        if (result != CUDA_SUCCESS) {
            printf("cuMemCreate failed: %d\n", result);
            pthread_mutex_unlock(&file->mutex);
            return -ENOMEM;
        }

        CUmemFabricHandle fabricHandle;
        result = cuMemExportToShareableHandle((void *)&fabricHandle, gpu_handle, CU_MEM_HANDLE_TYPE_FABRIC, 0);
        if (result != CUDA_SUCCESS) {
            printf("cuMemExportToShareableHandle failed: %d\n", result);
            pthread_mutex_unlock(&file->mutex);
            return -ENOMEM;
        }

        memcpy(&file->fabric_handle, &fabricHandle, sizeof(CUmemFabricHandle));
        file->gpu_handle = gpu_handle;
        file->size = size;
        file->modify_time = time(NULL);  // Update modification time
        
        printf("GPU memory allocated for %s: size=%zu, handle=%llu\n", 
               path, file->size, (unsigned long long)file->gpu_handle);
    } else if (file->size != (size_t)size) {
        // Resize not supported
        printf("Resize not supported for %s (current: %zu, requested: %ld)\n", 
               path, file->size, size);
        pthread_mutex_unlock(&file->mutex);
        return -ENOTSUP;
    } else {
        printf("File %s already has size %ld\n", path, size);
    }
    
    pthread_mutex_unlock(&file->mutex);
    return 0;
}

// FUSE init - initialize filesystem
static void *gpu_fuse_init(struct fuse_conn_info *conn, struct fuse_config *cfg)
{
    UNUSED(conn);
    UNUSED(cfg);
    
    printf("GPU Memory FUSE filesystem initialized\n");
    return NULL;
}

// FUSE utimens - set file timestamps
static int gpu_fuse_utimens(const char *path, const struct timespec ts[2], struct fuse_file_info *fi) {
    UNUSED(fi);
    
    gpu_file_t *file = gpu_fuse_get_file_from_path(g_gpu_ctx, path);
    if (!file) {
        return -ENOENT;
    }
    
    pthread_mutex_lock(&file->mutex);
    
    // ts[0] is access time, ts[1] is modification time
    if (ts) {
        if (ts[0].tv_nsec != UTIME_OMIT) {
            file->access_time = ts[0].tv_sec;
        }
        if (ts[1].tv_nsec != UTIME_OMIT) {
            file->modify_time = ts[1].tv_sec;
        }
    } else {
        // If ts is NULL, set both times to current time
        time_t current_time = time(NULL);
        file->access_time = current_time;
        file->modify_time = current_time;
    }
    
    pthread_mutex_unlock(&file->mutex);
    
    printf("Updated timestamps for %s\n", path);
    return 0;
}

// FUSE open - open file for reading/writing
static int gpu_fuse_open(const char *path, struct fuse_file_info *fi)
{
    printf("gpu_fuse_open called: path=%s, flags=%d\n", path, fi->flags);
    
    gpu_file_t *file = gpu_fuse_get_file_from_path(g_gpu_ctx, path);
    if (!file) {
        return -ENOENT;
    }
    
    // File exists, allow opening
    return 0;
}

// FUSE getxattr - get extended attributes
static int gpu_fuse_getxattr(const char *path, const char *name, char *value, size_t size)
{
    printf("gpu_fuse_getxattr called: path=%s, name=%s, size=%zu\n", path, name, size);
    
    gpu_file_t *file = gpu_fuse_get_file_from_path(g_gpu_ctx, path);
    if (!file) {
        return -ENOENT;
    }
    
    pthread_mutex_lock(&file->mutex);
    
    if (strcmp(name, "user.fabric_handle") == 0) {
        // Return the fabric handle
        if (file->gpu_handle == 0) {
            pthread_mutex_unlock(&file->mutex);
            return -ENODATA;  // No GPU allocation
        }
        
        if (size == 0) {
            // Caller is asking for the size of the attribute
            pthread_mutex_unlock(&file->mutex);
            return sizeof(CUmemFabricHandle);
        }
        
        if (size < sizeof(CUmemFabricHandle)) {
            pthread_mutex_unlock(&file->mutex);
            return -ERANGE;  // Buffer too small
        }
        
        memcpy(value, &file->fabric_handle, sizeof(CUmemFabricHandle));
        pthread_mutex_unlock(&file->mutex);
        printf("Returned fabric handle via getxattr: %zu bytes\n", sizeof(CUmemFabricHandle));
        return sizeof(CUmemFabricHandle);
        
    } else if (strcmp(name, "user.allocation_size") == 0) {
        // Return the allocation size as a string
        if (file->gpu_handle == 0) {
            pthread_mutex_unlock(&file->mutex);
            return -ENODATA;  // No GPU allocation
        }
        
        char size_str[32];
        int len = snprintf(size_str, sizeof(size_str), "%zu", file->size);
        
        if (size == 0) {
            // Caller is asking for the size of the attribute
            pthread_mutex_unlock(&file->mutex);
            return len;
        }
        
        if (size < (size_t)len + 1) {
            pthread_mutex_unlock(&file->mutex);
            return -ERANGE;  // Buffer too small
        }
        
        strcpy(value, size_str);
        pthread_mutex_unlock(&file->mutex);
        printf("Returned allocation size via getxattr: %s bytes\n", size_str);
        return len;  
    }
    
    pthread_mutex_unlock(&file->mutex);
    return -ENODATA;  // Attribute not found
}

// FUSE listxattr - list extended attributes
static int gpu_fuse_listxattr(const char *path, char *list, size_t size)
{
    printf("gpu_fuse_listxattr called: path=%s, size=%zu\n", path, size);
    
    gpu_file_t *file = gpu_fuse_get_file_from_path(g_gpu_ctx, path);
    if (!file) {
        return -ENOENT;
    }
    
    const char *attrs = "user.fabric_handle\0user.allocation_size\0";
    size_t attrs_len = strlen("user.fabric_handle") + 1 + 
                       strlen("user.allocation_size") + 1;
    
    if (size == 0) {
        // Caller is asking for the size needed
        return attrs_len;
    }
    
    if (size < attrs_len) {
        return -ERANGE;  // Buffer too small
    }
    
    memcpy(list, attrs, attrs_len);
    printf("Listed extended attributes: fabric_handle, allocation_size\n");
    return attrs_len;
}

// FUSE destroy - cleanup filesystem
static void gpu_fuse_destroy(void *private_data)
{
    UNUSED(private_data);
    
    if (g_gpu_ctx) {
        printf("Destroying GPU Memory FUSE filesystem\n");
        
        // Cleanup all files and their GPU memory
        pthread_mutex_lock(&g_gpu_ctx->global_mutex);
        
        GHashTableIter iter;
        gpointer key, value;
        
        g_hash_table_iter_init(&iter, g_gpu_ctx->files);
        while (g_hash_table_iter_next(&iter, &key, &value)) {
            gpu_file_t *file = (gpu_file_t*)value;
            gpu_fuse_cleanup_gpu_memory(file);
            pthread_mutex_destroy(&file->mutex);
        }
        
        pthread_mutex_unlock(&g_gpu_ctx->global_mutex);
        
        // Cleanup hash table
        g_hash_table_destroy(g_gpu_ctx->files);
        
        pthread_mutex_destroy(&g_gpu_ctx->global_mutex);
        
        free(g_gpu_ctx->mount_point);
        free(g_gpu_ctx);
        g_gpu_ctx = NULL;
    }
}

// FUSE read - read from file
// Probably not needed since we can use getxattr to get the fabric handle. This is just for testing.
static int gpu_fuse_read(const char *path, char *buf, size_t size, off_t offset, struct fuse_file_info *fi) {
    UNUSED(fi);
    
    printf("gpu_fuse_read called: path=%s, size=%zu, offset=%ld\n", path, size, offset);
    gpu_file_t *file = gpu_fuse_get_file_from_path(g_gpu_ctx, path);
    if (!file) {
        return -ENOENT;
    }

    // Check if GPU memory is allocated
    if (file->gpu_handle == 0) {
        printf("No GPU memory allocated for %s\n", path);
        return -ENODATA;
    }

    // Only support reading the fabric handle at offset 0
    if (offset != 0) {
        return 0;  // EOF for any offset > 0
    }

    // Read the fabric handle
    if (size >= sizeof(CUmemFabricHandle)) {
        memcpy(buf, &file->fabric_handle, sizeof(CUmemFabricHandle));
        printf("Read fabric handle for %s: %zu bytes\n", path, sizeof(CUmemFabricHandle));
        return sizeof(CUmemFabricHandle);  // Return actual bytes read
    } else {
        // Partial read not supported for fabric handle
        return -EINVAL;
    }
}



// FUSE operations structure - minimal set needed for create + truncate workflow
static struct fuse_operations gpu_fuse_ops = {
    .getattr    = gpu_fuse_getattr,  // Required to check if file exists
    .readdir    = gpu_fuse_readdir,  // Required for ls to work
    .create     = gpu_fuse_create,   // Required to create files
    .open       = gpu_fuse_open,     // Required to open files for reading/writing
    .truncate   = gpu_fuse_truncate, // Required for truncate -s SIZE
    .utimens    = gpu_fuse_utimens,  // Required to avoid touch warnings
    .getxattr   = gpu_fuse_getxattr, // Get extended attributes (fabric handle, size)
    .listxattr  = gpu_fuse_listxattr,// List available extended attributes
    .init       = gpu_fuse_init,     // Required for filesystem initialization
    .destroy    = gpu_fuse_destroy,  // Required for cleanup
    .read       = gpu_fuse_read,     // Required for read
};

// Main function
int main(int argc, char *argv[])
{
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
    g_gpu_ctx->files = g_hash_table_new_full(g_str_hash, g_str_equal, free, free);
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