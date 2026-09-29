/*
 * pipeline.cu
 *
 * Motion detection GPU pipeline — Stage 2 through Stage 4.
 *
 * RESPONSIBILITIES:
 *   • Allocate and own all device (GPU) memory for the pipeline lifetime
 *   • Call buildAndUploadKernel once to set up Gaussian weights
 *   • Each frame: launch blur → diff → threshold with no host↔device copies
 *     between stages
 *   • After all three kernels, do a single D→H copy of the uint8 mask
 *     (the only output the CPU needs for Stage 5 contour detection)
 *   • Expose a plain C API so Python/ctypes can call it without C++ name mangling
 *
 * MEMORY LAYOUT (all float32 unless noted):
 *
 *   d_frame_f32   — current raw grayscale frame, uploaded from CPU each frame
 *   d_blur_cur    — current frame after Gaussian blur
 *   d_blur_tmp    — scratch buffer for the horizontal blur pass
 *   d_blur_prev   — previous blurred frame (stays on device between frames)
 *   d_diff        — absolute difference image
 *   d_mask        — uint8 binary threshold mask  ← only buffer copied to CPU
 *
 * DATA FLOW PER FRAME:
 *
 *   CPU (gray uint8)
 *       │  H→D  (one upload per frame)
 *       ▼
 *   d_frame_f32  →[gaussianBlurH]→ d_blur_tmp →[gaussianBlurV]→ d_blur_cur
 *                                                                    │
 *   d_blur_prev ──────────────────────────────────────────────────[frameDiff]
 *                                                                    │
 *                                                               d_diff
 *                                                                    │
 *                                                            [threshold]
 *                                                                    │
 *                                                               d_mask
 *                                                                    │  D→H
 *                                                                    ▼
 *                                                           CPU (mask uint8)
 *       d_blur_cur ──swap──► d_blur_prev   (pointer swap, zero cost)
 */

#include "kernels.cuh"
#include "pipeline.h"
#include <cmath>    // ceilf

// Forward declarations of kernel launchers defined in kernels/*.cu
void launchGaussianBlur(const float*, float*, float*, int, int, int);
void launchFrameDiff   (const float*, const float*, float*, int, int);
void launchThreshold   (const float*, unsigned char*, int, int, float);

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline state — allocated once in pipeline_init, freed in pipeline_destroy
// ─────────────────────────────────────────────────────────────────────────────
struct Pipeline {
    int   width, height;
    int   blur_radius;
    float threshold_val;

    float*         d_frame_f32;   // H→D landing buffer (converted from uint8)
    float*         d_blur_cur;    // blur output for current frame
    float*         d_blur_tmp;    // H-pass scratch
    float*         d_blur_prev;   // blur output from previous frame
    float*         d_diff;        // |cur - prev|
    unsigned char* d_mask;        // binary threshold mask (uint8)

    bool first_frame;             // skip diff on frame 0 (no previous frame yet)

    // CUDA events for per-stage timing
    cudaEvent_t ev_start, ev_post_blur, ev_post_diff, ev_post_thresh;
};

// ─────────────────────────────────────────────────────────────────────────────
// pipeline_init
// ─────────────────────────────────────────────────────────────────────────────
extern "C" Pipeline* pipeline_init(
        int   width,
        int   height,
        float sigma,
        float threshold_val)
{
    Pipeline* p = new Pipeline();
    p->width         = width;
    p->height        = height;
    p->blur_radius   = (int)ceilf(3.0f * sigma);
    p->threshold_val = threshold_val / 255.0f;   // normalise to [0,1]
    p->first_frame   = true;

    if (p->blur_radius > MAX_RADIUS) {
        fprintf(stderr, "pipeline_init: sigma=%.1f → radius=%d > MAX_RADIUS=%d\n",
                sigma, p->blur_radius, MAX_RADIUS);
        delete p;
        return nullptr;
    }

    const size_t float_bytes = (size_t)width * height * sizeof(float);
    const size_t mask_bytes  = (size_t)width * height * sizeof(unsigned char);

    CUDA_CHECK(cudaMalloc(&p->d_frame_f32, float_bytes));
    CUDA_CHECK(cudaMalloc(&p->d_blur_cur,  float_bytes));
    CUDA_CHECK(cudaMalloc(&p->d_blur_tmp,  float_bytes));
    CUDA_CHECK(cudaMalloc(&p->d_blur_prev, float_bytes));
    CUDA_CHECK(cudaMalloc(&p->d_diff,      float_bytes));
    CUDA_CHECK(cudaMalloc(&p->d_mask,      mask_bytes));

    // Upload Gaussian weights to __constant__ memory once
    buildAndUploadKernel(sigma, p->blur_radius);

    CUDA_CHECK(cudaEventCreate(&p->ev_start));
    CUDA_CHECK(cudaEventCreate(&p->ev_post_blur));
    CUDA_CHECK(cudaEventCreate(&p->ev_post_diff));
    CUDA_CHECK(cudaEventCreate(&p->ev_post_thresh));

    return p;
}

// ─────────────────────────────────────────────────────────────────────────────
// pipeline_process_frame
//
// Called once per camera frame.
//
// h_gray_u8    — CPU grayscale frame (uint8, width×height bytes)
// h_mask_out   — CPU buffer to receive the binary mask (uint8, width×height)
// timing_ms    — optional float[4]: blur_ms, diff_ms, thresh_ms, total_ms
//
// Returns 0 on success, -1 if it's the first frame (no diff available yet).
// ─────────────────────────────────────────────────────────────────────────────
extern "C" int pipeline_process_frame(
        Pipeline*            p,
        const unsigned char* h_gray_u8,
        unsigned char*       h_mask_out,
        float*               timing_ms)
{
    const int W = p->width, H = p->height;
    const size_t float_bytes = (size_t)W * H * sizeof(float);
    const size_t mask_bytes  = (size_t)W * H * sizeof(unsigned char);

    // ── 1. Upload grayscale frame to device, convert uint8 → float32 [0,1] ──
    // We upload as uint8 then convert on-device to avoid a large float copy.
    // For simplicity here we do it on the host; a real project could add a
    // small conversion kernel to overlap this with the previous frame's work.
    {
        // Temporary host float buffer — stack-allocate only for small frames;
        // for production use a pinned host buffer allocated in pipeline_init.
        float* h_tmp = new float[(size_t)W * H];
        for (int i = 0; i < W * H; ++i)
            h_tmp[i] = h_gray_u8[i] / 255.0f;
        CUDA_CHECK(cudaMemcpy(p->d_frame_f32, h_tmp, float_bytes,
                              cudaMemcpyHostToDevice));
        delete[] h_tmp;
    }

    CUDA_CHECK(cudaEventRecord(p->ev_start));

    // ── Stage 2: Gaussian blur ────────────────────────────────────────────
    launchGaussianBlur(
        p->d_frame_f32,
        p->d_blur_tmp,    // H-pass scratch
        p->d_blur_cur,    // final blur output
        W, H, p->blur_radius);

    CUDA_CHECK(cudaEventRecord(p->ev_post_blur));

    // First frame: no previous frame to diff against
    if (p->first_frame) {
        CUDA_CHECK(cudaMemcpy(p->d_blur_prev, p->d_blur_cur, float_bytes,
                              cudaMemcpyDeviceToDevice));
        p->first_frame = false;
        // Zero the mask so the caller gets a clean black image
        CUDA_CHECK(cudaMemset(p->d_mask, 0, mask_bytes));
        return -1;
    }

    // ── Stage 3: Frame difference ─────────────────────────────────────────
    launchFrameDiff(p->d_blur_cur, p->d_blur_prev, p->d_diff, W, H);
    CUDA_CHECK(cudaEventRecord(p->ev_post_diff));

    // ── Stage 4: Threshold ────────────────────────────────────────────────
    launchThreshold(p->d_diff, p->d_mask, W, H, p->threshold_val);
    CUDA_CHECK(cudaEventRecord(p->ev_post_thresh));

    // ── Synchronise + collect timing ──────────────────────────────────────
    CUDA_CHECK(cudaEventSynchronize(p->ev_post_thresh));

    if (timing_ms) {
        cudaEventElapsedTime(&timing_ms[0], p->ev_start,      p->ev_post_blur);
        cudaEventElapsedTime(&timing_ms[1], p->ev_post_blur,  p->ev_post_diff);
        cudaEventElapsedTime(&timing_ms[2], p->ev_post_diff,  p->ev_post_thresh);
        cudaEventElapsedTime(&timing_ms[3], p->ev_start,      p->ev_post_thresh);
    }

    // ── D→H: copy only the mask (uint8, ~2 MB at 1080p) ──────────────────
    CUDA_CHECK(cudaMemcpy(h_mask_out, p->d_mask, mask_bytes,
                          cudaMemcpyDeviceToHost));

    // ── Swap cur ↔ prev for next frame (pointer swap — zero cost) ─────────
    float* tmp     = p->d_blur_prev;
    p->d_blur_prev = p->d_blur_cur;
    p->d_blur_cur  = tmp;

    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// pipeline_destroy  —  free all device memory and CUDA objects
// ─────────────────────────────────────────────────────────────────────────────
extern "C" void pipeline_destroy(Pipeline* p)
{
    if (!p) return;
    cudaFree(p->d_frame_f32);
    cudaFree(p->d_blur_cur);
    cudaFree(p->d_blur_tmp);
    cudaFree(p->d_blur_prev);
    cudaFree(p->d_diff);
    cudaFree(p->d_mask);
    cudaEventDestroy(p->ev_start);
    cudaEventDestroy(p->ev_post_blur);
    cudaEventDestroy(p->ev_post_diff);
    cudaEventDestroy(p->ev_post_thresh);
    delete p;
}
