/*
 * kernels/gaussian_blur.cu
 *
 * Separable Gaussian blur — the most complex kernel in the pipeline.
 *
 * WHY SEPARABLE?
 *   A 2-D Gaussian G(x,y) = G(x)·G(y), so we can split one 2-D convolution
 *   into two 1-D passes:
 *     Pass 1 (horizontal): convolve each row with a 1-D Gaussian
 *     Pass 2 (vertical):   convolve each column of that result
 *
 *   Cost comparison for radius r:
 *     Naïve 2-D:       (2r+1)² multiply-adds per pixel
 *     Separable 1-D×2: 2·(2r+1) multiply-adds per pixel
 *   At r=10 (our default): 441 → 42.  ~10× fewer operations.
 *
 * WHY SHARED MEMORY?
 *   Each output pixel reads (2r+1) input pixels.  Without shared memory,
 *   neighbouring threads re-read the same global memory locations — wasting
 *   bandwidth.  We instead load a tile (+ halo) into fast shared memory once,
 *   then all threads in the block read from there.
 *
 *   Shared memory latency: ~4 cycles
 *   Global  memory latency: ~400 cycles  (without L2 hit)
 *
 * WHY CONSTANT MEMORY FOR WEIGHTS?
 *   All threads in a warp read the same weight at the same time.
 *   Constant memory broadcasts one value to all 32 threads in a single cycle.
 */

#include "kernels.cuh"

// ─── Definition of the constant-memory symbol declared in kernels.cuh ────────
__constant__ float d_kernel[MAX_KERNEL_SIZE];

// ─────────────────────────────────────────────────────────────────────────────
// buildAndUploadKernel
//
// Computes a normalised 1-D Gaussian on the CPU and uploads it to the
// __constant__ d_kernel array.  Call this once whenever sigma changes.
// ─────────────────────────────────────────────────────────────────────────────
void buildAndUploadKernel(float sigma, int radius)
{
    float h_kernel[MAX_KERNEL_SIZE];
    float sum = 0.0f;

    for (int i = -radius; i <= radius; ++i) {
        float v = expf(-0.5f * (float)(i * i) / (sigma * sigma));
        h_kernel[i + radius] = v;
        sum += v;
    }
    // Normalise so weights sum to 1 — preserves image brightness
    for (int i = 0; i < 2 * radius + 1; ++i)
        h_kernel[i] /= sum;

    CUDA_CHECK(cudaMemcpyToSymbol(
        d_kernel, h_kernel, (2 * radius + 1) * sizeof(float)));
}

// ─────────────────────────────────────────────────────────────────────────────
// gaussianBlurH  —  Horizontal pass
//
// Shared memory layout (one row slice of the block):
//
//   smem index:  0 ........... radius-1 | radius ........ radius+BLOCK_W-1 | radius+BLOCK_W ... smem_w-1
//                [ left apron          ] [ core tile                       ] [ right apron      ]
//
// The apron cells hold pixels from neighbouring blocks so that threads near
// the tile edge can read their full kernel neighbourhood from shared memory
// rather than going back to global memory.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void gaussianBlurH(
        const float* __restrict__ src,
        float*       __restrict__ dst,
        int width, int height, int radius)
{
    extern __shared__ float smem[];               // (BLOCK_W + 2*radius) * BLOCK_H floats
    const int smem_w = BLOCK_W + 2 * radius;

    const int gx = blockIdx.x * BLOCK_W + threadIdx.x;   // global x
    const int gy = blockIdx.y * BLOCK_H + threadIdx.y;   // global y

    // smem column for this thread's "owned" pixel (shifted right by apron)
    const int sx_core = threadIdx.x + radius;
    const int sy      = threadIdx.y;

    // Clamp pixel coordinates to image bounds (border-replicate padding)
    auto clampX = [&](int x) { return max(0, min(x, width  - 1)); };
    auto clampY = [&](int y) { return max(0, min(y, height - 1)); };

    // ── 1. Load the core tile pixel ───────────────────────────────────────
    if (gy < height)
        smem[sy * smem_w + sx_core] = src[clampY(gy) * width + clampX(gx)];

    // ── 2. Load left apron ────────────────────────────────────────────────
    // Only threads whose local x < radius participate.
    if (threadIdx.x < radius && gy < height) {
        int halo_gx = blockIdx.x * BLOCK_W + threadIdx.x - radius;
        smem[sy * smem_w + threadIdx.x] =
            src[clampY(gy) * width + clampX(halo_gx)];
    }

    // ── 3. Load right apron ───────────────────────────────────────────────
    // Only threads whose local x >= (BLOCK_W - radius) participate.
    if (threadIdx.x >= (BLOCK_W - radius) && gy < height) {
        int halo_sx = sx_core + radius;
        int halo_gx = gx + radius;
        smem[sy * smem_w + halo_sx] =
            src[clampY(gy) * width + clampX(halo_gx)];
    }

    // ── 4. Synchronise — all shared loads must be visible before anyone reads
    __syncthreads();

    // ── 5. Convolve along the row ─────────────────────────────────────────
    if (gx < width && gy < height) {
        float acc = 0.0f;
        for (int k = -radius; k <= radius; ++k)
            acc += smem[sy * smem_w + sx_core + k] * d_kernel[k + radius];
        dst[gy * width + gx] = acc;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// gaussianBlurV  —  Vertical pass
//
// Mirrors gaussianBlurH but operates along columns.
//
// Shared memory layout (one column slice of the block):
//
//   smem row:  0 ... radius-1   ← top apron
//              radius ... radius+BLOCK_H-1  ← core tile
//              radius+BLOCK_H ... smem_h-1  ← bottom apron
// ─────────────────────────────────────────────────────────────────────────────
__global__ void gaussianBlurV(
        const float* __restrict__ src,
        float*       __restrict__ dst,
        int width, int height, int radius)
{
    extern __shared__ float smem[];               // BLOCK_W * (BLOCK_H + 2*radius) floats

    const int gx = blockIdx.x * BLOCK_W + threadIdx.x;
    const int gy = blockIdx.y * BLOCK_H + threadIdx.y;

    const int sy_core = threadIdx.y + radius;
    const int sx      = threadIdx.x;

    auto clampX = [&](int x) { return max(0, min(x, width  - 1)); };
    auto clampY = [&](int y) { return max(0, min(y, height - 1)); };

    // ── 1. Core tile ──────────────────────────────────────────────────────
    if (gx < width)
        smem[sy_core * BLOCK_W + sx] = src[clampY(gy) * width + clampX(gx)];

    // ── 2. Top apron ──────────────────────────────────────────────────────
    if (threadIdx.y < radius && gx < width) {
        int halo_gy = blockIdx.y * BLOCK_H + threadIdx.y - radius;
        smem[threadIdx.y * BLOCK_W + sx] =
            src[clampY(halo_gy) * width + clampX(gx)];
    }

    // ── 3. Bottom apron ───────────────────────────────────────────────────
    if (threadIdx.y >= (BLOCK_H - radius) && gx < width) {
        int halo_sy = sy_core + radius;
        int halo_gy = gy + radius;
        smem[halo_sy * BLOCK_W + sx] =
            src[clampY(halo_gy) * width + clampX(gx)];
    }

    __syncthreads();

    // ── 4. Convolve along the column ──────────────────────────────────────
    if (gx < width && gy < height) {
        float acc = 0.0f;
        for (int k = -radius; k <= radius; ++k)
            acc += smem[(sy_core + k) * BLOCK_W + sx] * d_kernel[k + radius];
        dst[gy * width + gx] = acc;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// launchGaussianBlur  —  public entry point used by pipeline.cu
//
// Accepts device pointers that are already allocated and resident on the GPU.
// No host↔device copies happen here — the caller owns memory lifetime.
//
// Parameters:
//   d_src   — input  float32 image, values in [0, 1],  device memory
//   d_tmp   — scratch float32 buffer (same size),      device memory
//   d_dst   — output float32 image,                    device memory
//   width, height — image dimensions in pixels
//   radius  — blur radius (sigma already uploaded via buildAndUploadKernel)
// ─────────────────────────────────────────────────────────────────────────────
void launchGaussianBlur(
        const float* d_src,
        float*       d_tmp,
        float*       d_dst,
        int width, int height, int radius)
{
    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((width  + BLOCK_W - 1) / BLOCK_W,
              (height + BLOCK_H - 1) / BLOCK_H);

    // Dynamic shared memory size for each pass
    size_t smem_h = (size_t)(BLOCK_W + 2*radius) * BLOCK_H * sizeof(float);
    size_t smem_v = (size_t) BLOCK_W * (BLOCK_H + 2*radius) * sizeof(float);

    gaussianBlurH<<<grid, block, smem_h>>>(d_src, d_tmp, width, height, radius);
    gaussianBlurV<<<grid, block, smem_v>>>(d_tmp, d_dst, width, height, radius);
    // No sync here — pipeline.cu inserts a single sync after all three kernels
}
