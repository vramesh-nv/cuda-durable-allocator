# GPU Durable Memory Allocator with FUSE

A FUSE-based filesystem that provides durable GPU memory allocations using CUDA Virtual Memory Management (VMM) APIs. This allocator allows GPU memory to survive process crashes when marked as durable, enabling robust GPU applications.

## Features

- **Durable GPU Memory**: Allocations marked as durable survive process crashes
- **POSIX Interface**: Standard filesystem operations for GPU memory management
- **Extended Attributes**: Use xattr to specify allocation size and durability
- **Reference Counting**: Automatic cleanup of non-durable allocations
- **CUDA VMM Integration**: Uses `cuMemCreate`, `cuMemExportToShareableHandle`, and `cuMemImportFromShareableHandle`
- **Cross-Process Sharing**: Share GPU memory allocations via filesystem paths

## Architecture

The allocator maps GPU memory allocations to filesystem paths:

```
/gpu_allocator/
├── allocation1          # GPU memory allocation (file)
├── allocation2          # Another allocation
├── temp_buffer         # Transient allocation
└── .metadata/          # Internal metadata
```

### Allocation Lifecycle

1. **Size Specification**: Set allocation size via extended attribute
2. **Creation**: Create file to trigger GPU memory allocation  
3. **Durability Control**: Mark as durable/transient via extended attribute
4. **Usage**: Access allocation info via file operations
5. **Cleanup**: Remove file or let refcount reach zero

### Durability Model

- **Durable Allocations**: High refcount, survive crashes, recoverable by path
- **Transient Allocations**: Normal refcount, cleaned up on process exit

## Dependencies

- **libfuse3-dev**: FUSE filesystem development library
- **libglib2.0-dev**: GLib hash tables and utilities
- **CUDA Toolkit**: CUDA development headers and libraries
- **build-essential**: GCC and build tools

### Installation on Ubuntu/Debian

```bash
sudo apt update
sudo apt install libfuse3-dev libglib2.0-dev build-essential
# Install CUDA Toolkit from NVIDIA
```

## Building

```bash
# Check dependencies
make check-deps

# Build the filesystem
make

# Build with debug symbols
make debug
```

## Usage

### Starting the Filesystem

```bash
# Create mount point
mkdir gpu_allocator_mount

# Start filesystem (foreground with debug)
./build/gpu_mem_fuse gpu_allocator_mount -f -d

# Or run in background
./build/gpu_mem_fuse gpu_allocator_mount
```

### Basic Operations

```bash
# 1. Specify allocation size (1MB)
setfattr -n user.gpu.size -v "1048576" gpu_allocator_mount/my_buffer

# 2. Create the allocation
touch gpu_allocator_mount/my_buffer

# 3. Check allocation info
cat gpu_allocator_mount/my_buffer

# 4. Make allocation durable (survives crashes)
setfattr -n user.gpu.durable -v "true" gpu_allocator_mount/my_buffer

# 5. List all allocations
ls -la gpu_allocator_mount/

# 6. Check extended attributes
getfattr -d gpu_allocator_mount/my_buffer

# 7. Remove allocation
rm gpu_allocator_mount/my_buffer
```

### Extended Attributes

- **user.gpu.size**: Allocation size in bytes (set before creation)
- **user.gpu.durable**: "true"/"false" for durability control

### Example Workflow

```bash
# Create a 64MB durable allocation for neural network weights
setfattr -n user.gpu.size -v "67108864" gpu_allocator_mount/nn_weights
setfattr -n user.gpu.durable -v "true" gpu_allocator_mount/nn_weights
touch gpu_allocator_mount/nn_weights

# Create a 4MB transient buffer for temporary data
setfattr -n user.gpu.size -v "4194304" gpu_allocator_mount/temp_buffer
touch gpu_allocator_mount/temp_buffer

# List allocations
ls -la gpu_allocator_mount/
cat gpu_allocator_mount/nn_weights
```

## API Integration

### From C/C++ Applications

```c
#include <sys/xattr.h>
#include <fcntl.h>

// Create a durable 1MB GPU allocation
const char *path = "/mnt/gpu_allocator/my_allocation";

// Set size
setxattr(path, "user.gpu.size", "1048576", 7, 0);

// Mark as durable
setxattr(path, "user.gpu.durable", "true", 4, 0);

// Create allocation
int fd = open(path, O_CREAT | O_RDWR, 0644);

// Read allocation info
char info[512];
read(fd, info, sizeof(info));
printf("GPU Allocation: %s\n", info);

close(fd);
```

### From Python

```python
import os
import xattr

# Set allocation size (1MB)
xattr.setxattr('gpu_allocator_mount/python_buffer', 
               'user.gpu.size', b'1048576')

# Create allocation
with open('gpu_allocator_mount/python_buffer', 'w') as f:
    pass

# Make durable
xattr.setxattr('gpu_allocator_mount/python_buffer',
               'user.gpu.durable', b'true')

# Read allocation info
with open('gpu_allocator_mount/python_buffer', 'r') as f:
    print(f.read())
```

## Testing

```bash
# Terminal 1: Start filesystem
make test

# Terminal 2: Run test commands
make test-usage

# Cleanup
make test-clean
```

## Implementation Details

### CUDA Integration

The allocator uses CUDA Virtual Memory Management APIs:

- **cuMemCreate()**: Create physical memory allocation
- **cuMemAddressReserve()**: Reserve virtual address space
- **cuMemMap()**: Map physical to virtual memory
- **cuMemExportToShareableHandle()**: Export for durability
- **cuMemImportFromShareableHandle()**: Import after crash

### Memory Layout

```c
typedef struct {
    char path[MAX_PATH_LEN];
    CUmemGenericAllocationHandle handle;    // CUDA memory handle
    CUdeviceptr device_ptr;                 // GPU virtual address
    size_t size;                            // Allocation size
    int refcount;                           // Reference count
    gpu_alloc_state_t state;               // DURABLE/TRANSIENT
    int export_fd;                         // POSIX FD for durability
    time_t created_time;
    time_t last_access;
    pthread_mutex_t mutex;
} gpu_allocation_t;
```

### Crash Recovery

1. **Export Phase**: Durable allocations export POSIX file descriptors
2. **Crash**: Process dies, transient allocations are cleaned up
3. **Recovery**: New process imports handles from exported file descriptors
4. **Restoration**: Memory mappings are restored to previous state

## Limitations

- Currently supports single GPU device
- Requires CUDA-compatible GPU with VMM support
- File system must be unmounted cleanly for proper cleanup
- Memory mappings are not directly accessible via mmap() yet

## Future Enhancements

- [ ] Direct mmap() support for GPU memory
- [ ] Multi-GPU support
- [ ] Persistent allocation registry
- [ ] Memory usage statistics
- [ ] Compression for large allocations
- [ ] Integration with existing CUDA memory pools

## Troubleshooting

### Common Issues

1. **Permission Denied**: Ensure user has access to CUDA devices
2. **CUDA Not Found**: Install CUDA Toolkit and set paths
3. **FUSE Errors**: Check if FUSE module is loaded (`modprobe fuse`)
4. **Mount Busy**: Unmount with `fusermount3 -u <mountpoint>`

### Debug Mode

```bash
# Run with verbose debugging
./build/gpu_mem_fuse mountpoint -f -d -o debug
```

### Log Output

The filesystem prints detailed information about:
- CUDA initialization
- Allocation creation/destruction
- Durability state changes
- Reference count updates

## License

This implementation is provided as a prototype for educational and research purposes.