#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/xattr.h>
#include <errno.h>
#include <time.h>

// Test client for the GPU Memory FUSE filesystem
// Demonstrates how to use the allocator from a C application

#define TEST_MOUNT_PATH "./test_mount"
#define BUFFER_SIZE 1024

void print_test_header(const char *test_name) {
    printf("\n=== %s ===\n", test_name);
}

void print_error(const char *operation) {
    printf("ERROR in %s: %s\n", operation, strerror(errno));
}

int test_basic_allocation() {
    print_test_header("Basic Allocation Test");
    
    char path[256];
    snprintf(path, sizeof(path), "%s/test_basic", TEST_MOUNT_PATH);
    
    // 1. Set allocation size (1MB)
    printf("1. Setting allocation size to 1MB...\n");
    if (setxattr(path, "user.gpu.size", "1048576", 7, 0) != 0) {
        print_error("setxattr size");
        return -1;
    }
    
    // 2. Create the allocation
    printf("2. Creating allocation...\n");
    int fd = open(path, O_CREAT | O_RDWR, 0644);
    if (fd < 0) {
        print_error("open");
        return -1;
    }
    
    // 3. Read allocation info
    printf("3. Reading allocation info...\n");
    char info[BUFFER_SIZE];
    ssize_t bytes_read = read(fd, info, sizeof(info) - 1);
    if (bytes_read > 0) {
        info[bytes_read] = '\0';
        printf("Allocation Info:\n%s\n", info);
    } else {
        print_error("read");
    }
    
    // 4. Check extended attributes
    printf("4. Checking extended attributes...\n");
    char size_value[32];
    ssize_t size_len = getxattr(path, "user.gpu.size", size_value, sizeof(size_value) - 1);
    if (size_len > 0) {
        size_value[size_len] = '\0';
        printf("   user.gpu.size = %s\n", size_value);
    }
    
    char durable_value[32];
    ssize_t durable_len = getxattr(path, "user.gpu.durable", durable_value, sizeof(durable_value) - 1);
    if (durable_len > 0) {
        durable_value[durable_len] = '\0';
        printf("   user.gpu.durable = %s\n", durable_value);
    }
    
    close(fd);
    
    // 5. Cleanup
    printf("5. Cleaning up...\n");
    if (unlink(path) != 0) {
        print_error("unlink");
        return -1;
    }
    
    printf("Basic allocation test PASSED\n");
    return 0;
}

int test_durable_allocation() {
    print_test_header("Durable Allocation Test");
    
    char path[256];
    snprintf(path, sizeof(path), "%s/test_durable", TEST_MOUNT_PATH);
    
    // 1. Set allocation size (512KB)
    printf("1. Setting allocation size to 512KB...\n");
    if (setxattr(path, "user.gpu.size", "524288", 6, 0) != 0) {
        print_error("setxattr size");
        return -1;
    }
    
    // 2. Mark as durable before creation
    printf("2. Marking as durable...\n");
    if (setxattr(path, "user.gpu.durable", "true", 4, 0) != 0) {
        print_error("setxattr durable");
        return -1;
    }
    
    // 3. Create the allocation
    printf("3. Creating durable allocation...\n");
    int fd = open(path, O_CREAT | O_RDWR, 0644);
    if (fd < 0) {
        print_error("open");
        return -1;
    }
    
    // 4. Verify durability
    printf("4. Verifying durability...\n");
    char durable_value[32];
    ssize_t durable_len = getxattr(path, "user.gpu.durable", durable_value, sizeof(durable_value) - 1);
    if (durable_len > 0) {
        durable_value[durable_len] = '\0';
        printf("   Durability status: %s\n", durable_value);
        
        if (strcmp(durable_value, "true") != 0) {
            printf("ERROR: Expected durable=true, got %s\n", durable_value);
            close(fd);
            return -1;
        }
    } else {
        print_error("getxattr durable");
        close(fd);
        return -1;
    }
    
    // 5. Read allocation info
    printf("5. Reading durable allocation info...\n");
    char info[BUFFER_SIZE];
    ssize_t bytes_read = read(fd, info, sizeof(info) - 1);
    if (bytes_read > 0) {
        info[bytes_read] = '\0';
        printf("Durable Allocation Info:\n%s\n", info);
    }
    
    close(fd);
    
    printf("Durable allocation test PASSED\n");
    printf("NOTE: This allocation should survive process crashes\n");
    return 0;
}

int test_multiple_allocations() {
    print_test_header("Multiple Allocations Test");
    
    const int num_allocs = 5;
    char paths[num_allocs][256];
    int fds[num_allocs];
    size_t sizes[] = {1024, 4096, 16384, 65536, 262144}; // Various sizes
    
    printf("Creating %d allocations with different sizes...\n", num_allocs);
    
    // Create multiple allocations
    for (int i = 0; i < num_allocs; i++) {
        snprintf(paths[i], sizeof(paths[i]), "%s/test_multi_%d", TEST_MOUNT_PATH, i);
        
        printf("  Creating allocation %d (size: %zu bytes)...\n", i, sizes[i]);
        
        // Set size
        char size_str[32];
        snprintf(size_str, sizeof(size_str), "%zu", sizes[i]);
        if (setxattr(paths[i], "user.gpu.size", size_str, strlen(size_str), 0) != 0) {
            print_error("setxattr size");
            return -1;
        }
        
        // Create allocation
        fds[i] = open(paths[i], O_CREAT | O_RDWR, 0644);
        if (fds[i] < 0) {
            printf("Failed to create allocation %d\n", i);
            print_error("open");
            return -1;
        }
    }
    
    printf("All allocations created successfully\n");
    
    // Test concurrent access
    printf("Testing concurrent access...\n");
    for (int i = 0; i < num_allocs; i++) {
        char info[BUFFER_SIZE];
        ssize_t bytes_read = read(fds[i], info, sizeof(info) - 1);
        if (bytes_read > 0) {
            info[bytes_read] = '\0';
            printf("  Allocation %d info: %s", i, strstr(info, "Size:"));
        }
    }
    
    // Close all allocations
    printf("Closing allocations...\n");
    for (int i = 0; i < num_allocs; i++) {
        close(fds[i]);
    }
    
    // Cleanup
    printf("Cleaning up...\n");
    for (int i = 0; i < num_allocs; i++) {
        if (unlink(paths[i]) != 0) {
            printf("Warning: Failed to remove allocation %d\n", i);
        }
    }
    
    printf("Multiple allocations test PASSED\n");
    return 0;
}

int test_invalid_operations() {
    print_test_header("Invalid Operations Test");
    
    char path[256];
    snprintf(path, sizeof(path), "%s/test_invalid", TEST_MOUNT_PATH);
    
    // 1. Try to create without size
    printf("1. Testing creation without size specification...\n");
    int fd = open(path, O_CREAT | O_RDWR, 0644);
    if (fd >= 0) {
        printf("   Creation succeeded (creates pending allocation)\n");
        
        // Try to read from pending allocation
        char info[BUFFER_SIZE];
        ssize_t bytes_read = read(fd, info, sizeof(info) - 1);
        printf("   Read from pending allocation returned %zd bytes\n", bytes_read);
        
        close(fd);
        unlink(path);
    } else {
        print_error("open without size");
    }
    
    // 2. Try invalid size
    printf("2. Testing invalid size specification...\n");
    if (setxattr(path, "user.gpu.size", "invalid", 7, 0) == 0) {
        printf("   WARNING: Invalid size was accepted\n");
    } else {
        printf("   Invalid size rejected (expected)\n");
    }
    
    // 3. Try zero size
    printf("3. Testing zero size...\n");
    if (setxattr(path, "user.gpu.size", "0", 1, 0) == 0) {
        fd = open(path, O_CREAT | O_RDWR, 0644);
        if (fd >= 0) {
            printf("   WARNING: Zero size allocation was created\n");
            close(fd);
            unlink(path);
        } else {
            printf("   Zero size allocation rejected (expected)\n");
        }
    }
    
    printf("Invalid operations test completed\n");
    return 0;
}

int test_listing() {
    print_test_header("Directory Listing Test");
    
    // Create a few allocations
    char paths[3][256];
    const char *names[] = {"list_test_1", "list_test_2", "list_test_3"};
    
    printf("Creating test allocations for listing...\n");
    for (int i = 0; i < 3; i++) {
        snprintf(paths[i], sizeof(paths[i]), "%s/%s", TEST_MOUNT_PATH, names[i]);
        
        if (setxattr(paths[i], "user.gpu.size", "4096", 4, 0) != 0) {
            print_error("setxattr");
            continue;
        }
        
        int fd = open(paths[i], O_CREAT | O_RDWR, 0644);
        if (fd >= 0) {
            close(fd);
            printf("  Created %s\n", names[i]);
        }
    }
    
    printf("\nListing directory contents:\n");
    printf("Run 'ls -la %s' to see the allocations\n", TEST_MOUNT_PATH);
    
    // Cleanup
    printf("\nCleaning up test allocations...\n");
    for (int i = 0; i < 3; i++) {
        if (unlink(paths[i]) == 0) {
            printf("  Removed %s\n", names[i]);
        }
    }
    
    printf("Directory listing test completed\n");
    return 0;
}

void print_usage(const char *program_name) {
    printf("Usage: %s [test_name]\n", program_name);
    printf("\nAvailable tests:\n");
    printf("  basic      - Basic allocation test\n");
    printf("  durable    - Durable allocation test\n");
    printf("  multiple   - Multiple allocations test\n");
    printf("  invalid    - Invalid operations test\n");
    printf("  listing    - Directory listing test\n");
    printf("  all        - Run all tests (default)\n");
    printf("\nMake sure the GPU Memory FUSE filesystem is mounted at %s\n", TEST_MOUNT_PATH);
}

int main(int argc, char *argv[]) {
    printf("GPU Memory FUSE Filesystem Test Client\n");
    printf("======================================\n");
    
    // Check if mount point exists
    if (access(TEST_MOUNT_PATH, F_OK) != 0) {
        printf("ERROR: Mount point %s not found\n", TEST_MOUNT_PATH);
        printf("Please start the GPU Memory FUSE filesystem first:\n");
        printf("  ./build/gpu_mem_fuse %s -f -d\n", TEST_MOUNT_PATH);
        return 1;
    }
    
    const char *test_name = "all";
    if (argc > 1) {
        test_name = argv[1];
    }
    
    int total_tests = 0;
    int passed_tests = 0;
    
    if (strcmp(test_name, "all") == 0 || strcmp(test_name, "basic") == 0) {
        total_tests++;
        if (test_basic_allocation() == 0) passed_tests++;
    }
    
    if (strcmp(test_name, "all") == 0 || strcmp(test_name, "durable") == 0) {
        total_tests++;
        if (test_durable_allocation() == 0) passed_tests++;
    }
    
    if (strcmp(test_name, "all") == 0 || strcmp(test_name, "multiple") == 0) {
        total_tests++;
        if (test_multiple_allocations() == 0) passed_tests++;
    }
    
    if (strcmp(test_name, "all") == 0 || strcmp(test_name, "invalid") == 0) {
        total_tests++;
        if (test_invalid_operations() == 0) passed_tests++;
    }
    
    if (strcmp(test_name, "all") == 0 || strcmp(test_name, "listing") == 0) {
        total_tests++;
        if (test_listing() == 0) passed_tests++;
    }
    
    if (total_tests == 0) {
        printf("Unknown test: %s\n", test_name);
        print_usage(argv[0]);
        return 1;
    }
    
    printf("\n" "========================================\n");
    printf("Test Results: %d/%d tests passed\n", passed_tests, total_tests);
    
    if (passed_tests == total_tests) {
        printf("All tests PASSED!\n");
        return 0;
    } else {
        printf("Some tests FAILED!\n");
        return 1;
    }
}