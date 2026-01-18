/******************************************************************************
 * GPU Object Detection - Support Implementation
 * CUDA utility functions for feature-based object detection
 * 
 * Author: Group U
 ******************************************************************************/

#include "support.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/******************************************************************************
 * Timer Functions
 ******************************************************************************/

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

/******************************************************************************
 * Descriptor Memory Management (Host)
 ******************************************************************************/

Descriptors allocateDescriptors(unsigned int count, unsigned int dim) {
    Descriptors desc;
    desc.count = count;
    desc.dim = dim;
    desc.data = (unsigned char*) malloc(count * dim * sizeof(unsigned char));
    
    if (desc.data == NULL) {
        FATAL("Unable to allocate host memory for descriptors");
    }
    
    return desc;
}

void freeDescriptors(Descriptors desc) {
    if (desc.data != NULL) {
        free(desc.data);
        desc.data = NULL;
    }
}

/******************************************************************************
 * Descriptor Memory Management (Device)
 ******************************************************************************/

Descriptors allocateDeviceDescriptors(unsigned int count, unsigned int dim) {
    Descriptors desc;
    desc.count = count;
    desc.dim = dim;
    
    cudaError_t cuda_ret = cudaMalloc((void**)&desc.data, count * dim * sizeof(unsigned char));
    if (cuda_ret != cudaSuccess) {
        FATAL("Unable to allocate device memory for descriptors");
    }
    
    return desc;
}

void freeDeviceDescriptors(Descriptors desc) {
    if (desc.data != NULL) {
        cudaFree(desc.data);
        desc.data = NULL;
    }
}

void copyToDeviceDescriptors(Descriptors dst, Descriptors src) {
    if (dst.count != src.count || dst.dim != src.dim) {
        FATAL("Descriptor dimension mismatch in copy");
    }
    
    size_t size = src.count * src.dim * sizeof(unsigned char);
    cudaError_t cuda_ret = cudaMemcpy(dst.data, src.data, size, cudaMemcpyHostToDevice);
    
    if (cuda_ret != cudaSuccess) {
        FATAL("Unable to copy descriptors to device");
    }
}

void copyFromDeviceDescriptors(Descriptors dst, Descriptors src) {
    if (dst.count != src.count || dst.dim != src.dim) {
        FATAL("Descriptor dimension mismatch in copy");
    }
    
    size_t size = src.count * src.dim * sizeof(unsigned char);
    cudaError_t cuda_ret = cudaMemcpy(dst.data, src.data, size, cudaMemcpyDeviceToHost);
    
    if (cuda_ret != cudaSuccess) {
        FATAL("Unable to copy descriptors from device");
    }
}

/******************************************************************************
 * Distance Matrix Management
 ******************************************************************************/

float* allocateDistanceMatrix(unsigned int rows, unsigned int cols) {
    float* mat;
    size_t size = rows * cols * sizeof(float);
    
    cudaError_t cuda_ret = cudaMalloc((void**)&mat, size);
    if (cuda_ret != cudaSuccess) {
        FATAL("Unable to allocate device memory for distance matrix");
    }
    
    return mat;
}

void freeDistanceMatrix(float* mat) {
    if (mat != NULL) {
        cudaFree(mat);
    }
}
