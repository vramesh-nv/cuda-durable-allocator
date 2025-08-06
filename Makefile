CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -std=c11 -D_GNU_SOURCE -DFUSE_USE_VERSION=31
NVCCFLAGS = -std=c++11 -Xcompiler -fPIC
LDFLAGS = -lfuse3 -lglib-2.0 -lcuda -lcudart -lpthread -L/usr/local/cuda/lib64

# Use pkg-config for proper dependency management
INCLUDES = $(shell pkg-config --cflags fuse3 glib-2.0)
CUDA_INCLUDES = -I/usr/local/cuda/include

SRCDIR = .
BUILDDIR = build
SOURCES = gpu_mem_fuse.c
OBJECTS = $(SOURCES:%.c=$(BUILDDIR)/%.o)
TARGET = $(BUILDDIR)/gpu_mem_fuse

# Test client (CUDA)
TEST_CLIENT_SRC = test_client.cu
TEST_CLIENT_OBJ = $(BUILDDIR)/test_client.o
TEST_CLIENT_TARGET = $(BUILDDIR)/test_client

.PHONY: all clean install uninstall test

all: $(TARGET) $(TEST_CLIENT_TARGET)

$(TARGET): $(OBJECTS) | $(BUILDDIR)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

$(TEST_CLIENT_TARGET): $(TEST_CLIENT_OBJ) | $(BUILDDIR)
	$(NVCC) $(TEST_CLIENT_OBJ) -o $@ $(LDFLAGS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(CUDA_INCLUDES) -c $< -o $@

$(BUILDDIR)/test_client.o: $(SRCDIR)/test_client.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $(CUDA_INCLUDES) -c $< -o $@

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)

install: $(TARGET)
	@echo "Installing GPU Memory FUSE to /usr/local/bin/"
	sudo cp $(TARGET) /usr/local/bin/
	sudo chmod +x /usr/local/bin/gpu_mem_fuse

uninstall:
	sudo rm -f /usr/local/bin/gpu_mem_fuse

# Test targets
test: $(TARGET)
	@echo "Creating test mount point..."
	mkdir -p ./test_mount
	@echo "Starting GPU Memory FUSE filesystem..."
	@echo "Run 'make test-usage' in another terminal to test the filesystem"
	./$(TARGET) ./test_mount -f -d

test-usage:
	@echo "Testing GPU Memory FUSE filesystem..."
	@echo ""
	@echo "1. Setting allocation size via extended attribute:"
	setfattr -n user.gpu.size -v "1048576" ./test_mount/my_buffer
	@echo ""
	@echo "2. Creating the allocation:"
	touch ./test_mount/my_buffer
	@echo ""
	@echo "3. Checking allocation info:"
	cat ./test_mount/my_buffer
	@echo ""
	@echo "4. Listing extended attributes:"
	getfattr -d ./test_mount/my_buffer
	@echo ""
	@echo "5. Making allocation durable:"
	setfattr -n user.gpu.durable -v "true" ./test_mount/my_buffer
	@echo ""
	@echo "6. Listing all allocations:"
	ls -la ./test_mount/
	@echo ""
	@echo "7. Cleaning up:"
	rm ./test_mount/my_buffer

test-client: $(TEST_CLIENT_TARGET)
	@echo "Running test client..."
	./$(TEST_CLIENT_TARGET)

test-clean:
	@echo "Cleaning up test environment..."
	fusermount3 -u ./test_mount 2>/dev/null || true
	rmdir ./test_mount 2>/dev/null || true

# Development helpers
debug: CFLAGS += -g -DDEBUG
debug: $(TARGET)

format:
	clang-format -i *.c *.h

check-deps:
	@echo "Checking dependencies..."
	@which gcc > /dev/null || (echo "gcc not found" && exit 1)
	@pkg-config --exists fuse3 || (echo "libfuse3-dev not installed" && exit 1)
	@pkg-config --exists glib-2.0 || (echo "libglib2.0-dev not installed" && exit 1)
	@test -f /usr/local/cuda/include/cuda.h || (echo "CUDA headers not found" && exit 1)
	@echo "All dependencies found!"

help:
	@echo "GPU Memory FUSE Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build the GPU Memory FUSE filesystem and test client"
	@echo "  clean       - Remove build files"
	@echo "  install     - Install to /usr/local/bin (requires sudo)"
	@echo "  uninstall   - Remove from /usr/local/bin (requires sudo)"
	@echo "  test        - Start filesystem in foreground for testing"
	@echo "  test-usage  - Run test commands (run in separate terminal)"
	@echo "  test-client - Run automated test client"
	@echo "  test-clean  - Cleanup test environment"
	@echo "  debug       - Build with debug symbols"
	@echo "  format      - Format code with clang-format"
	@echo "  check-deps  - Check if all dependencies are installed"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Dependencies:"
	@echo "  - libfuse3-dev"
	@echo "  - libglib2.0-dev"
	@echo "  - CUDA development toolkit"
	@echo "  - build-essential"