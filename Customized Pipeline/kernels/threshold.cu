/*
 * kernels/threshold.cu
 *
 * Binary threshold — Stage 4 of the motion detection pipeline.
 *
 * WHAT IT DOES:
 *   thresh[i] = (diff[i] > threshold) ? 255 : 0
 *
 *   Converts the floating-point difference image into a binary mask:
 *     white pixel (255) = motion detected here
 *     black pixel (0)   = no motion / background
 *
 *   The threshold value (default 25/255 ≈ 0.098 in [0,1] space) filters out
 *   small lighting fluctuations while preserving genuine movement.
 *
 * OUTPUT FORMAT:
 *   uint8 — matching what cv2.findContours expects on the CPU side.
 *   The pipeline converts float32 diff → uint8 mask in this single kernel,
 *   avoiding a separate conversion step.
 *
 * NOTE ON WARP DIVERGENCE:
 *   The ternary (diff > thresh ? 255 : 0) causes all threads in a warp to
 *   take potentially different branches.  This is unavoidable for a threshold
 *   operation, but the cost is negligible — both branches are single-cycle
 *   assignments with no memory access.
 */

#include "kernels.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// thresholdKernel
//
// Reads float32 diff values, writes uint8 binary mask.
// threshold_val is normalised to [0,1] — same space as the diff image.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void thresholdKernel(
        const float*         __restrict__ diff,       // float32 diff image [0,1]
        unsigned char*       __restrict__ mask,       // output uint8 binary mask
        int width, int height,
        float threshold_val)                          // e.g. 25.0/255.0
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const int idx = y * width + x;
    mask[idx] = (diff[idx] > threshold_val) ? 255 : 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// launchThreshold  —  public entry point used by pipeline.cu
// ─────────────────────────────────────────────────────────────────────────────
void launchThreshold(
        const float*   d_diff,
        unsigned char* d_mask,
        int width, int height,
        float threshold_val)
{
    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((width  + BLOCK_W - 1) / BLOCK_W,
              (height + BLOCK_H - 1) / BLOCK_H);

    thresholdKernel<<<grid, block>>>(d_diff, d_mask, width, height, threshold_val);
}
