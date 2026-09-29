"""
motion_detect.py

Motion detection using the pure-CUDA pipeline (stages 2-4 on GPU).

Pipeline stages:
  1. Grayscale        — CPU  (cv2.cvtColor, cheap, runs during memcpy overlap)
  2. Gaussian blur    — GPU  (custom kernel: kernels/gaussian_blur.cu)
  3. Frame difference — GPU  (custom kernel: kernels/frame_diff.cu)
  4. Threshold        — GPU  (custom kernel: kernels/threshold.cu)
  5. Contours         — CPU  (cv2.findContours — no GPU equivalent in OpenCV)

Stages 2-4 share device memory with zero host<->device copies between them.
Only one D->H copy happens per frame: the uint8 mask for stage 5.

Build the shared library first:
  nvcc -O2 -arch=sm_86 --shared -Xcompiler -fPIC \\
       -o pipeline.so pipeline.cu               \\
          kernels/gaussian_blur.cu              \\
          kernels/frame_diff.cu                 \\
          kernels/threshold.cu

Then run:
  python motion_detect.py
"""

import ctypes
import os
import time

import cv2
import numpy as np

# ─── Load the compiled pipeline shared library ────────────────────────────────
_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "pipeline.so"))

# pipeline_init(width, height, sigma, threshold_val) -> void*
_lib.pipeline_init.restype  = ctypes.c_void_p
_lib.pipeline_init.argtypes = [ctypes.c_int, ctypes.c_int,
                                ctypes.c_float, ctypes.c_float]

# pipeline_process_frame(handle, gray_u8*, mask_out*, timing_ms*) -> int
_lib.pipeline_process_frame.restype  = ctypes.c_int
_lib.pipeline_process_frame.argtypes = [
    ctypes.c_void_p,                             # pipeline handle
    ctypes.POINTER(ctypes.c_uint8),              # input grayscale frame
    ctypes.POINTER(ctypes.c_uint8),              # output mask
    ctypes.POINTER(ctypes.c_float),              # timing[4] or NULL
]

# pipeline_destroy(handle) -> void
_lib.pipeline_destroy.restype  = None
_lib.pipeline_destroy.argtypes = [ctypes.c_void_p]


# ─── Benchmark storage ────────────────────────────────────────────────────────
stage_times = {
    "grayscale":  [],
    "blur":       [],   # GPU time reported by CUDA events
    "frame_diff": [],   # GPU time reported by CUDA events
    "threshold":  [],   # GPU time reported by CUDA events
    "contours":   [],
}

# ─── Camera setup ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
ret, probe = cap.read()
if not ret:
    raise RuntimeError("Cannot open camera")

H, W = probe.shape[:2]
print(f"Camera: {W}×{H}")

# ─── Allocate pipeline (once) ─────────────────────────────────────────────────
SIGMA         = 3.0    # Gaussian standard deviation
THRESHOLD_VAL = 25.0   # binary threshold in [0, 255] space

handle = _lib.pipeline_init(W, H, SIGMA, THRESHOLD_VAL)
if not handle:
    raise RuntimeError("pipeline_init failed — check CUDA availability")

# Pre-allocate numpy buffers so we're not reallocating each frame
mask_buf   = np.empty((H, W), dtype=np.uint8)
timing_buf = (ctypes.c_float * 4)()

fps_start   = time.perf_counter()
frame_count = 0

print("Running CUDA pipeline... press Q to quit")

# Re-queue the probe frame so we don't skip frame 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ── Stage 1: Grayscale (CPU) ──────────────────────────────────────────
    t = time.perf_counter()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    stage_times["grayscale"].append((time.perf_counter() - t) * 1000)

    # ── Stages 2-4: GPU pipeline ──────────────────────────────────────────
    # Pass contiguous numpy array pointer directly — no copy into ctypes buffer
    gray_ptr = gray.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    mask_ptr = mask_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    result = _lib.pipeline_process_frame(handle, gray_ptr, mask_ptr, timing_buf)

    if result == 0:
        # timing_buf = [blur_ms, diff_ms, thresh_ms, total_ms]
        stage_times["blur"].append(timing_buf[0])
        stage_times["frame_diff"].append(timing_buf[1])
        stage_times["threshold"].append(timing_buf[2])

        # ── Stage 5: Contours (CPU — OpenCV has no GPU findContours) ─────
        t = time.perf_counter()
        contours, _ = cv2.findContours(mask_buf, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        stage_times["contours"].append((time.perf_counter() - t) * 1000)

    frame_count += 1
    elapsed = time.perf_counter() - fps_start
    fps = frame_count / elapsed

    cv2.putText(frame, f"CUDA FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
    cv2.imshow("CUDA Motion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('Q'):
        break

# ─── Cleanup ──────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
_lib.pipeline_destroy(handle)

# ─── Print benchmark results ──────────────────────────────────────────────────
total_elapsed = time.perf_counter() - fps_start
print("\n=== CUDA PIPELINE RESULTS ===")
print("  (blur/diff/thresh times are GPU-measured via CUDA events)")
total_avg = 0.0
for stage, times in stage_times.items():
    if times:
        avg_ms = float(np.mean(times))
        total_avg += avg_ms
        print(f"  {stage:<15} {avg_ms:.4f} ms")

print(f"  {'TOTAL':<15} {total_avg:.4f} ms")
print(f"  {'FPS':<15} {frame_count / total_elapsed:.1f}")
