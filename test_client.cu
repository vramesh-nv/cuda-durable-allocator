#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <cuda.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/xattr.h>
#include <getopt.h>
#include <assert.h>

#define CUDA_CHECK(err) do { \
    cudaError_t _err = (err); \
    if (_err != cudaSuccess) { \
        printf("CUDA error %d: %s\n", _err, cudaGetErrorString(_err)); \
        return -1; \
    } \
} while (0)

#define CUDA_CHECK_DRV(err) do { \
    CUresult _err = (err); \
    if (_err != CUDA_SUCCESS) { \
        const char* error_str; \
        cuGetErrorString(_err, &error_str); \
        printf("CUDA error %d: %s\n", _err, error_str); \
        return -1; \
    } \
} while (0)

// Test client for the GPU Memory FUSE filesystem
// Tests the simplified create + truncate workflow

#define TEST_MOUNT_PATH "./test_mount"

void print_test_header(const char *test_name) {
    printf("\n=== %s ===\n", test_name);
}

void print_error(const char *operation) {
    printf("ERROR in %s: %s\n", operation, strerror(errno));
}

void print_usage(const char *program_name) {
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("Options:\n");
    printf("  --parent    Run as parent process (creates allocation and waits for child)\n");
    printf("  --child     Run as child process (accesses existing allocation)\n");
    printf("  --help      Show this help message\n");
    printf("\nExample:\n");
    printf("  # Terminal 1 (parent):\n");
    printf("  %s --parent\n", program_name);
    printf("  \n");
    printf("  # Terminal 2 (child):\n");
    printf("  %s --child\n", program_name);
}

__global__ void kernel_write(void *ptr, size_t size) {
    for (size_t i = 0; i < size; i++) {
        ((char *)ptr)[i] = (unsigned char)i;
    }
}

__global__ void kernel_read(void *ptr, size_t size) {
    for (size_t i = 0; i < size; i++) {
        unsigned char *ptr_char = (unsigned char *)ptr;
        //printf("%d, %zd\n", (int)ptr_char[i], i);
        assert(ptr_char[i] == (unsigned char)i);
    }
}

static CUdeviceptr
get_va_from_fabric_handle(CUmemFabricHandle fabric_handle, size_t allocation_size, size_t granularity) {
    CUmemGenericAllocationHandle gpu_handle;
    CUDA_CHECK_DRV(cuMemImportFromShareableHandle(&gpu_handle, (void *)&fabric_handle, CU_MEM_HANDLE_TYPE_FABRIC));
    
    CUdeviceptr va = 0U;
    CUDA_CHECK_DRV(cuMemAddressReserve(&va, allocation_size, granularity, 0U, 0));

    CUDA_CHECK_DRV(cuMemMap(va, allocation_size, 0, gpu_handle, 0));
    CUDA_CHECK_DRV(cuMemRelease(gpu_handle));

    CUmemAccessDesc accessDesc;
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = 0;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUDA_CHECK_DRV(cuMemSetAccess(va, allocation_size, &accessDesc, 1));

    return va;
}

int test_parent_process() {
    print_test_header("PARENT PROCESS - Creating GPU Allocation");
    
    char path[256];
    snprintf(path, sizeof(path), "%s/shared_gpu_buffer", TEST_MOUNT_PATH);
    
    // 1. Create the file (no GPU memory allocated yet)
    printf("1. Creating file (no GPU memory yet)...\n");
    mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH; 
    int fd = creat(path, mode);
    if (fd < 0) {
        print_error("open/create");
        return -1;
    }
    close(fd);  // Close the creation file descriptor

    printf("2. Truncating to 8MB to allocate GPU memory...\n");
    if (truncate(path, 8 * 1024 * 1024) != 0) {
        print_error("truncate");
        return -1;
    }

    // 3. Get allocation size using getxattr
    printf("3. Getting allocation size using getxattr...\n");
    char size_str[64];
    ssize_t size_len = getxattr(path, "user.allocation_size", size_str, sizeof(size_str) - 1);
    if (size_len < 0) {
        print_error("getxattr allocation_size");
        return -1;
    }
    size_str[size_len] = '\0';
    size_t allocation_size = atol(size_str);
    printf("   Retrieved allocation size: %s bytes (%.2f MB)\n", 
           size_str, allocation_size / (1024.0 * 1024.0));

    // 4. List all available extended attributes
    printf("4. Listing available extended attributes...\n");
    char attr_list[1024];
    ssize_t list_size = listxattr(path, attr_list, sizeof(attr_list));
    if (list_size < 0) {
        print_error("listxattr");
    } else {
        printf("   Available attributes (%zd bytes):\n", list_size);
        char *attr = attr_list;
        while (attr < attr_list + list_size) {
            printf("   - %s\n", attr);
            attr += strlen(attr) + 1;
        }
    }

    CUmemFabricHandle fabric_handle;
    ssize_t bytes_read = getxattr(path, "user.fabric_handle", &fabric_handle, sizeof(CUmemFabricHandle));
    if (bytes_read != sizeof(CUmemFabricHandle)) {
        printf("getxattr failed: expected %zu bytes, got %zd bytes\n", sizeof(CUmemFabricHandle), bytes_read);
        print_error("getxattr");
        return -1;
    }
    
    printf("4. Successfully read fabric handle (%zu bytes)\n", sizeof(CUmemFabricHandle));
    
    // 5. Initialize CUDA
    CUDA_CHECK_DRV(cuInit(0));
    
    CUdeviceptr va = get_va_from_fabric_handle(fabric_handle, allocation_size, allocation_size);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    kernel_write<<<1, 1, 0, stream>>>((void *)va, allocation_size);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    // Wait for user input (simulating child process completion)
    //getchar();
    
    printf("5. Parent process completed.\n");
    printf("✅ PARENT PROCESS completed successfully!\n");
    return 0;
}

int test_child_process() {
    print_test_header("CHILD PROCESS - Accessing Shared GPU Allocation");
    
    char path[256];
    snprintf(path, sizeof(path), "%s/shared_gpu_buffer", TEST_MOUNT_PATH);
    
    // 1. Check if the allocation exists
    printf("1. Checking if shared allocation exists...\n");
    struct stat st;
    if (stat(path, &st) != 0) {
        print_error("stat - allocation not found");
        printf("   Make sure to run the parent process first!\n");
        return -1;
    }
    printf("   Found allocation: %ld bytes (%.2f MB)\n", 
           st.st_size, st.st_size / (1024.0 * 1024.0));

    // 2. Get allocation size using getxattr
    printf("2. Getting allocation size using getxattr...\n");
    char size_str[64];
    ssize_t size_len = getxattr(path, "user.allocation_size", size_str, sizeof(size_str) - 1);
    if (size_len < 0) {
        print_error("getxattr allocation_size");
        return -1;
    }
    size_str[size_len] = '\0';
    size_t allocation_size = atol(size_str);
    printf("   Child sees allocation size: %s bytes (%.2f MB)\n", 
           size_str, allocation_size / (1024.0 * 1024.0));

    CUmemFabricHandle fabric_handle;
    ssize_t bytes_read = getxattr(path, "user.fabric_handle", &fabric_handle, sizeof(CUmemFabricHandle));
    if (bytes_read != sizeof(CUmemFabricHandle)) {
        printf("getxattr failed: expected %zu bytes, got %zd bytes\n", sizeof(CUmemFabricHandle), bytes_read);
        print_error("getxattr");
        return -1;
    }
    
    printf("4. Successfully read fabric handle (%zu bytes)\n", sizeof(CUmemFabricHandle));
    
    // 5. Initialize CUDA
    CUDA_CHECK_DRV(cuInit(0));
    
    CUdeviceptr va = get_va_from_fabric_handle(fabric_handle, allocation_size, allocation_size);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 7. Write to the shared memory from child
    kernel_read<<<1, 1, 0, stream>>>((void *)va, allocation_size);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    printf("5. Successfully wrote to shared GPU memory from child!\n");
    printf("✅ CHILD PROCESS completed successfully!\n");
    return 0;
}

int main(int argc, char *argv[]) {
    printf("GPU Memory FUSE Filesystem Test Client\n");
    printf("======================================\n");
    
    // Define long options
    static struct option long_options[] = {
        {"parent", no_argument, 0, 'p'},
        {"child",  no_argument, 0, 'c'},
        {"help",   no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    int option_index = 0;
    int opt;
    enum { MODE_NONE, MODE_PARENT, MODE_CHILD } mode = MODE_NONE;
    
    // Parse command line options
    while ((opt = getopt_long(argc, argv, "pch", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'p':
                mode = MODE_PARENT;
                break;
            case 'c':
                mode = MODE_CHILD;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    // Check if mode was specified
    if (mode == MODE_NONE) {
        printf("Error: You must specify either --parent or --child\n\n");
        print_usage(argv[0]);
        return 1;
    }
    
    // Check if mount point exists
    struct stat st;
    if (stat(TEST_MOUNT_PATH, &st) != 0) {
        printf("Error: Mount point %s does not exist.\n", TEST_MOUNT_PATH);
        printf("Please make sure the FUSE filesystem is running.\n");
        return 1;
    }

    // Run the appropriate test based on mode
    int result = 0;
    switch (mode) {
        case MODE_PARENT:
            result = test_parent_process();
            break;
        case MODE_CHILD:
            result = test_child_process();
            break;
        default:
            // Should never reach here
            result = 1;
            break;
    }

    return result;
}