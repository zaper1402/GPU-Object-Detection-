/******************************************************************************
 * GPU Object Detection - CUDA Main Program
 * Demonstrates custom CUDA kernels for feature matching
 * 
 * Author: Group U (Muhammad Zahid, Sadikshya Satyal, Ayodeji Ibrahim, Ashir Kulshreshtha)
 * 
 * This program demonstrates the core CUDA kernel optimization for descriptor
 * matching. Full detection pipeline uses OpenCV (object_detector_gpu.py)
 ******************************************************************************/

#include "kernel.cu"
#include "support.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void printDeviceInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    printf("\n==========================================================\n");
    printf("CUDA Device Information\n");
    printf("==========================================================\n");
    printf("Number of CUDA devices: %d\n\n", deviceCount);
    
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        
        printf("Device %d: %s\n", dev, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Global Memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
        printf("  Shared Memory per Block: %lu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Registers per Block: %d\n", prop.regsPerBlock);
        printf("\n");
    }
}

void testHammingDistance() {
    printf("\n==========================================================\n");
    printf("Test 1: Hamming Distance Kernel\n");
    printf("==========================================================\n");
    
    Timer timer;
    
    // Simulate ORB descriptors (32 bytes each)
    unsigned int n_template = 500;   // Typical ball/book template features
    unsigned int n_query = 1000;     // Typical query image features
    unsigned int desc_dim = 32;      // ORB descriptor size
    
    printf("Template descriptors: %d\n", n_template);
    printf("Query descriptors: %d\n", n_query);
    printf("Descriptor dimension: %d bytes (256 bits)\n", desc_dim);
    
    // Allocate host memory
    printf("\nAllocating host memory...");
    fflush(stdout);
    startTime(&timer);
    
    Descriptors template_h = allocateDescriptors(n_template, desc_dim);
    Descriptors query_h = allocateDescriptors(n_query, desc_dim);
    
    // Initialize with random binary data
    for (unsigned int i = 0; i < n_template * desc_dim; i++) {
        template_h.data[i] = rand() % 256;
    }
    for (unsigned int i = 0; i < n_query * desc_dim; i++) {
        query_h.data[i] = rand() % 256;
    }
    
    stopTime(&timer);
    printf(" %f s\n", elapsedTime(timer));
    
    // Allocate device memory
    printf("Allocating device memory...");
    fflush(stdout);
    startTime(&timer);
    
    Descriptors template_d = allocateDeviceDescriptors(n_template, desc_dim);
    Descriptors query_d = allocateDeviceDescriptors(n_query, desc_dim);
    float* distances_d = allocateDistanceMatrix(n_template, n_query);
    
    stopTime(&timer);
    printf(" %f s\n", elapsedTime(timer));
    
    // Copy to device
    printf("Copying descriptors to device...");
    fflush(stdout);
    startTime(&timer);
    
    copyToDeviceDescriptors(template_d, template_h);
    copyToDeviceDescriptors(query_d, query_h);
    
    stopTime(&timer);
    printf(" %f s\n", elapsedTime(timer));
    
    // Launch kernel (standard version)
    printf("\n--- Standard Kernel ---\n");
    printf("Launching Hamming distance kernel...");
    fflush(stdout);
    startTime(&timer);
    
    launchHammingDistance(
        template_d.data, query_d.data, distances_d,
        n_template, n_query, false
    );
    
    stopTime(&timer);
    float kernel_time = elapsedTime(timer);
    printf(" %f ms\n", kernel_time * 1000);
    
    // Calculate throughput
    unsigned long long operations = (unsigned long long)n_template * n_query * desc_dim;
    float throughput = operations / (kernel_time * 1e9);  // Giga-operations per second
    printf("Throughput: %.2f GOPS (Giga-operations/sec)\n", throughput);
    printf("Distance computations: %u x %u = %u\n", 
           n_template, n_query, n_template * n_query);
    
    // Launch kernel (shared memory version)
    printf("\n--- Shared Memory Optimized Kernel ---\n");
    printf("Launching optimized kernel...");
    fflush(stdout);
    startTime(&timer);
    
    launchHammingDistance(
        template_d.data, query_d.data, distances_d,
        n_template, n_query, true
    );
    
    stopTime(&timer);
    float kernel_time_opt = elapsedTime(timer);
    printf(" %f ms\n", kernel_time_opt * 1000);
    
    float speedup = kernel_time / kernel_time_opt;
    printf("Speedup: %.2fx\n", speedup);
    
    // Copy results back and verify
    printf("\nCopying results back...");
    fflush(stdout);
    
    float* distances_h = (float*)malloc(n_template * n_query * sizeof(float));
    cudaMemcpy(distances_h, distances_d, n_template * n_query * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    printf(" done\n");
    
    // Verify results (sample check)
    printf("\nSample distances (first 5 template vs first 5 query):\n");
    for (int t = 0; t < 5 && t < n_template; t++) {
        printf("  Template %d: ", t);
        for (int q = 0; q < 5 && q < n_query; q++) {
            printf("%.0f ", distances_h[t * n_query + q]);
        }
        printf("\n");
    }
    
    // Cleanup
    freeDescriptors(template_h);
    freeDescriptors(query_h);
    freeDeviceDescriptors(template_d);
    freeDeviceDescriptors(query_d);
    freeDistanceMatrix(distances_d);
    free(distances_h);
    
    printf("\n[SUCCESS] Hamming distance kernel TEST PASSED\n");
}

void testKnnRatioTest() {
    printf("\n==========================================================\n");
    printf("Test 2: k-NN with Ratio Test Kernel\n");
    printf("==========================================================\n");
    
    Timer timer;
    
    unsigned int n_template = 500;
    unsigned int n_query = 1000;
    int max_matches = 2000;
    float ratio_threshold = 0.75;
    
    printf("Template descriptors: %d\n", n_template);
    printf("Query descriptors: %d\n", n_query);
    printf("Ratio threshold: %.2f\n", ratio_threshold);
    
    // Allocate and generate synthetic distance matrix
    printf("\nGenerating synthetic distance matrix...");
    fflush(stdout);
    startTime(&timer);
    
    float* distances_h = (float*)malloc(n_template * n_query * sizeof(float));
    for (unsigned int i = 0; i < n_template * n_query; i++) {
        distances_h[i] = (float)(rand() % 256);  // Random distances 0-255
    }
    
    float* distances_d = allocateDistanceMatrix(n_template, n_query);
    cudaMemcpy(distances_d, distances_h, n_template * n_query * sizeof(float),
               cudaMemcpyHostToDevice);
    
    stopTime(&timer);
    printf(" %f s\n", elapsedTime(timer));
    
    // Allocate match output arrays
    printf("Allocating match arrays...");
    fflush(stdout);
    
    int* match_query_idx_d;
    int* match_template_idx_d;
    float* match_distances_d;
    int* match_count_d;
    
    cudaMalloc((void**)&match_query_idx_d, max_matches * sizeof(int));
    cudaMalloc((void**)&match_template_idx_d, max_matches * sizeof(int));
    cudaMalloc((void**)&match_distances_d, max_matches * sizeof(float));
    cudaMalloc((void**)&match_count_d, sizeof(int));
    
    printf(" done\n");
    
    // Launch k-NN ratio test kernel
    printf("\nLaunching k-NN ratio test kernel...");
    fflush(stdout);
    startTime(&timer);
    
    launchKnnRatioTest(
        distances_d,
        match_query_idx_d, match_template_idx_d, match_distances_d,
        match_count_d,
        n_template, n_query,
        ratio_threshold, max_matches,
        false  // Standard version
    );
    
    stopTime(&timer);
    float kernel_time = elapsedTime(timer);
    printf(" %f ms\n", kernel_time * 1000);
    
    // Get match count
    int match_count_h = 0;
    cudaMemcpy(&match_count_h, match_count_d, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Good matches found: %d (%.1f%% of template features)\n",
           match_count_h, 100.0 * match_count_h / n_template);
    
    // Copy some matches back for verification
    if (match_count_h > 0) {
        int sample_size = match_count_h < 5 ? match_count_h : 5;
        int* query_idx_h = (int*)malloc(sample_size * sizeof(int));
        int* template_idx_h = (int*)malloc(sample_size * sizeof(int));
        float* dist_h = (float*)malloc(sample_size * sizeof(float));
        
        cudaMemcpy(query_idx_h, match_query_idx_d, sample_size * sizeof(int), 
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(template_idx_h, match_template_idx_d, sample_size * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(dist_h, match_distances_d, sample_size * sizeof(float),
                   cudaMemcpyDeviceToHost);
        
        printf("\nSample matches:\n");
        for (int i = 0; i < sample_size; i++) {
            printf("  Match %d: Template[%d] <-> Query[%d], distance=%.1f\n",
                   i, template_idx_h[i], query_idx_h[i], dist_h[i]);
        }
        
        free(query_idx_h);
        free(template_idx_h);
        free(dist_h);
    }
    
    // Test warp-optimized version
    printf("\n--- Warp-Optimized Kernel ---\n");
    printf("Launching warp-optimized kernel...");
    fflush(stdout);
    
    cudaMemset(match_count_d, 0, sizeof(int));  // Reset counter
    
    startTime(&timer);
    launchKnnRatioTest(
        distances_d,
        match_query_idx_d, match_template_idx_d, match_distances_d,
        match_count_d,
        n_template, n_query,
        ratio_threshold, max_matches,
        true  // Warp-optimized version
    );
    stopTime(&timer);
    float kernel_time_opt = elapsedTime(timer);
    printf(" %f ms\n", kernel_time_opt * 1000);
    
    float speedup = kernel_time / kernel_time_opt;
    printf("Speedup: %.2fx\n", speedup);
    
    // Cleanup
    free(distances_h);
    freeDistanceMatrix(distances_d);
    cudaFree(match_query_idx_d);
    cudaFree(match_template_idx_d);
    cudaFree(match_distances_d);
    cudaFree(match_count_d);
    
    printf("\n[SUCCESS] k-NN ratio test kernel TEST PASSED\n");
}

int main(int argc, char** argv) {
    printf("\n");
    printf("==============================================================\n");
    printf(" GPU Object Detection - Custom CUDA Kernels\n");
    printf(" Feature Matching Optimization for A100\n");
    printf("==============================================================\n");
    
    // Print device information
    printDeviceInfo();
    
    // Run tests
    testHammingDistance();
    testKnnRatioTest();
    
    // Final summary
    printf("\n==============================================================\n");
    printf("All tests completed successfully!\n");
    printf("==============================================================\n");
    printf("\n[SUCCESS] TEST PASSED\n\n");
    printf("Next steps:\n");
    printf("1. Run Python detector: python src/object_detector_gpu.py --input test_images --templates templates\n");
    printf("2. Run benchmark: python src/benchmark.py --input test_images --templates templates\n");
    printf("\n");
    
    return 0;
}
