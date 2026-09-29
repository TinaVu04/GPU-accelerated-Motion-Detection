/*
 * kernels/kernels.cuh
 *
 * Shared constants, macros, and device-function declarations used by every
 * kernel in this project.  Include this at the top of each .cu file.
 *
 * Nothing in here allocates memory or launches kernels — it is purely
 * declarations and compile-time constants.
 */

#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// ─── Block dimensions ─────────────────────────────────────────────────────────
// 32×8 = 256 threads per block.
//   • BLOCK_W=32 aligns with warp size → coalesced global memory reads
//   • BLOCK_H=8  keeps register pressure low while hiding memory latency
#define BLOCK_W  32
#define BLOCK_H   8

// ─── Gaussian blur limits ────────────────────────────────────────────────────
#define MAX_RADIUS      16
#define MAX_KERNEL_SIZE (2 * MAX_RADIUS + 1)

// ─── Error-checking macro ────────────────────────────────────────────────────
// Wraps any CUDA runtime call.  On failure it prints file/line and exits.
// Usage:  CUDA_CHECK( cudaMalloc(...) );
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ─── Constant memory for Gaussian weights ────────────────────────────────────
// Declared here (extern) so all translation units share the same symbol.
// Defined once in gaussian_blur.cu.
extern __constant__ float d_kernel[MAX_KERNEL_SIZE];

// ─── Host helper — build + upload 1-D Gaussian weights ───────────────────────
// Called once per sigma change (not every frame).
// Defined in gaussian_blur.cu, used by pipeline.cu.
void buildAndUploadKernel(float sigma, int radius);
