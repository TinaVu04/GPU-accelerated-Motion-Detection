/*
 * pipeline.h
 *
 * C-linkage API for the CUDA motion detection pipeline.
 * Python loads this via ctypes — no C++ name mangling allowed here.
 *
 * Usage from Python:
 *   import ctypes
 *   lib = ctypes.CDLL("./pipeline.so")
 *   # see motion_detect.py for full setup
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle — Python holds this as a void* / c_void_p
typedef struct Pipeline Pipeline;

/*
 * pipeline_init
 *
 * Allocates all device memory and uploads Gaussian weights.
 * Call once before the capture loop.
 *
 *   width, height    — frame dimensions in pixels
 *   sigma            — Gaussian blur standard deviation (try 3.0)
 *   threshold_val    — binary threshold in [0, 255] space (try 25.0)
 *
 * Returns a heap-allocated Pipeline handle, or NULL on error.
 */
Pipeline* pipeline_init(int width, int height, float sigma, float threshold_val);

/*
 * pipeline_process_frame
 *
 * Runs stages 2-4 on one grayscale frame.  No host↔device copies happen
 * between stages — everything stays on the GPU until the final mask download.
 *
 *   p            — handle from pipeline_init
 *   h_gray_u8    — CPU grayscale frame, width*height uint8 bytes (row-major)
 *   h_mask_out   — CPU output buffer, width*height uint8 bytes
 *   timing_ms    — optional float[4]: [blur_ms, diff_ms, thresh_ms, total_ms]
 *                  pass NULL to skip timing
 *
 * Returns  0 on success
 *         -1 on the first frame (no previous frame; mask is zeroed)
 */
int pipeline_process_frame(Pipeline*            p,
                           const unsigned char* h_gray_u8,
                           unsigned char*       h_mask_out,
                           float*               timing_ms);

/*
 * pipeline_destroy
 *
 * Frees all device memory and CUDA objects.  Call after the capture loop.
 */
void pipeline_destroy(Pipeline* p);

#ifdef __cplusplus
}
#endif
