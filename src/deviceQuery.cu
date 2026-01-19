#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    printf("Detected %d CUDA-capable device(s)\n\n", deviceCount);
    
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        
        printf("Device %d: \"%s\"\n", dev, deviceProp.name);
        printf("  Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total Global Memory: %.2f GB\n", deviceProp.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
        printf("  Multiprocessors: %d\n", deviceProp.multiProcessorCount);
        printf("  CUDA Cores: %d\n", deviceProp.multiProcessorCount * 64); // A100: 64 cores/SM
        printf("  Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max Block Dimensions: [%d, %d, %d]\n", 
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Max Grid Dimensions: [%d, %d, %d]\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("  Shared Memory per Block: %zu KB\n", deviceProp.sharedMemPerBlock / 1024);
        printf("  Registers per Block: %d K\n", deviceProp.regsPerBlock / 1024);
        printf("  Warp Size: %d\n", deviceProp.warpSize);
        printf("  Memory Clock Rate: %.2f GHz\n", deviceProp.memoryClockRate / 1e6);
        printf("  Memory Bus Width: %d-bit\n", deviceProp.memoryBusWidth);
        printf("  Peak Memory Bandwidth: %.2f GB/s\n\n",
               2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) / 1.0e6);
    }
    
    printf("TEST PASSED\n");
    return 0;
}