# GPU Object Detection - Makefile
# Target: NVIDIA A100 (sm_80)
# Author: Group U

# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80 -Xcompiler -Wall
CXX = g++
CXX_FLAGS = -O3 -Wall

# Directories
SRC_DIR = .
BUILD_DIR = ../build
BIN_DIR = ../bin

# Target executable
TARGET = $(BIN_DIR)/detector

# Source files
CUDA_SOURCES = main.cu support.cu kernel.cu
HEADERS = support.h

# Object files
OBJECTS = $(BUILD_DIR)/main.o $(BUILD_DIR)/support.o

# Default target
all: directories $(TARGET)

# Create necessary directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

# Link executable
$(TARGET): $(OBJECTS)
	@echo "Linking $@..."
	$(NVCC) $(NVCC_FLAGS) $^ -o $@
	@echo "Build complete: $@"

# Compile main.cu (includes kernel.cu)
$(BUILD_DIR)/main.o: main.cu kernel.cu $(HEADERS)
	@echo "Compiling main.cu..."
	$(NVCC) $(NVCC_FLAGS) -c main.cu -o $@

# Compile support.cu
$(BUILD_DIR)/support.o: support.cu $(HEADERS)
	@echo "Compiling support.cu..."
	$(NVCC) $(NVCC_FLAGS) -c support.cu -o $@

# Run the detector
run: $(TARGET)
	@echo "Running CUDA kernel tests..."
	cd .. && $(TARGET)

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@rm -rf $(BIN_DIR)
	@echo "Clean complete"

# Clean and rebuild
rebuild: clean all

# Display device information
deviceQuery:
	@echo "Querying CUDA devices..."
	@nvidia-smi
	@echo ""
	@nvcc --version

# Help target
help:
	@echo "GPU Object Detection - Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build the detector (default)"
	@echo "  run          - Build and run CUDA kernel tests"
	@echo "  clean        - Remove build artifacts"
	@echo "  rebuild      - Clean and rebuild"
	@echo "  deviceQuery  - Display CUDA device information"
	@echo "  help         - Display this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make         - Build project"
	@echo "  make run     - Build and run"
	@echo "  make clean   - Clean build files"

.PHONY: all directories run clean rebuild deviceQuery help
