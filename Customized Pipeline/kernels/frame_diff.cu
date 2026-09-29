/*
 * kernels/frame_diff.cu
 *
 * Absolute frame difference — Stage 3 of the motion detection pipeline.
 *
 * WHAT IT DOES:
 *   diff[i] = |blurred_current[i] - blurred_previous[i]|
 *
 *   Pixels that changed between frames become bright; static background → black.
 *   We run this on the blurred frames (not raw) so sensor noise doesn't create
 *   false motion detections.
 *
 * WHY THIS IS TRIVIALLY PARALLEL:
 *   Every output pixel depends only on the two corresponding input pixels —
 *   no neighbourhood, no shared memory needed.  Every thread does one subtraction
 *   and one abs().  This is a "perfectly parallel" / "embarrassingly parallel"
 *   problem: ideal GPU work with 100% occupancy and zero synchronisation cost.
 *
 * MEMORY ACCESS PATTERN:
 *   Two coalesced reads + one coalesced write per thread.  At 1920×1080 float32,
 *   that's ~25 MB touched — well within L2 cache on modern GPUs for repeated
 *   per-frame calls.
 */

#include "kernels.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// frameDiffKernel
//
// Each thread handles exactly one pixel.  Grid is sized to cover the full image.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void frameDiffKernel(
        const float* __restrict__ current,    // blurred current frame  [0,1]
        const float* __restrict__ previous,   // blurred previous frame [0,1]
        float*       __restrict__ diff,       // output: absolute difference
        int width, int height)
{
    // 1-D thread index covering the entire image as a flat array
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;    // guard out-of-bounds threads

    const int idx = y * width + x;
    diff[idx] = fabsf(current[idx] - previous[idx]);
}

// ─────────────────────────────────────────────────────────────────────────────
// launchFrameDiff  —  public entry point used by pipeline.cu
// ─────────────────────────────────────────────────────────────────────────────
void launchFrameDiff(
        const float* d_current,
        const float* d_previous,
        float*       d_diff,
        int width, int height)
{
    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((width  + BLOCK_W - 1) / BLOCK_W,
              (height + BLOCK_H - 1) / BLOCK_H);

    frameDiffKernel<<<grid, block>>>(d_current, d_previous, d_diff, width, height);
}
