/******************************************************************************
 * GPU Object Detection - Support Header
 * CUDA utilities for feature-based object detection
 * 
 * Author: Group U (Muhammad Zahid, Sadikshya Satyal, Ayodeji Ibrahim, Ashir Kulshreshtha)
 * Target: NVIDIA A100 (sm_80)
 ******************************************************************************/

#ifndef __SUPPORT_H__
#define __SUPPORT_H__

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Timer structure for performance measurement
typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

// Descriptor structure for ORB features
typedef struct {
    unsigned int count;        // Number of descriptors
    unsigned int dim;          // Descriptor dimension (32 bytes for ORB)
    unsigned char* data;       // Descriptor data (host or device)
} Descriptors;

// Match structure
typedef struct {
    int queryIdx;              // Index in query descriptors
    int trainIdx;              // Index in train descriptors  
    float distance;            // Hamming distance
} Match;

// Detection result structure
typedef struct {
    int objectId;              // Object identifier (0=ball, 1=book)
    int matchCount;            // Number of good matches
    int inlierCount;           // Number of RANSAC inliers
    float confidence;          // Detection confidence
    float corners[8];          // Bounding box corners (4 x,y pairs)
} Detection;

// Function declarations

// Timer functions
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);

// Memory management
Descriptors allocateDescriptors(unsigned int count, unsigned int dim);
Descriptors allocateDeviceDescriptors(unsigned int count, unsigned int dim);
void copyToDeviceDescriptors(Descriptors dst, Descriptors src);
void copyFromDeviceDescriptors(Descriptors dst, Descriptors src);
void freeDescriptors(Descriptors desc);
void freeDeviceDescriptors(Descriptors desc);

// Distance computation
float* allocateDistanceMatrix(unsigned int rows, unsigned int cols);
void freeDistanceMatrix(float* mat);

// CUDA error checking
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Fatal error macro
#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)

// Endianness check
#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

#endif // __SUPPORT_H__
