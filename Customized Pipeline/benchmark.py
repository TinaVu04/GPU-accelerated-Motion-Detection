"""
benchmark.py

Side-by-side timing comparison:
  A) CPU baseline  — cv2.GaussianBlur + cv2.absdiff + cv2.threshold
  B) CUDA pipeline — custom kernels (blur + diff + threshold on GPU)

Runs N frames from the camera (or a synthetic image if no camera),
then prints a table you can paste directly into your report.

Usage:
  python benchmark.py [--frames 300] [--no-camera]
"""

import argparse
import ctypes
import os
import time

import cv2
import numpy as np

# ─── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--frames",    type=int,  default=300,
                    help="Number of frames to benchmark (default: 300)")
parser.add_argument("--no-camera", action="store_true",
                    help="Use synthetic frames instead of live camera")
args = parser.parse_args()

# ─── Load pipeline shared library ────────────────────────────────────────────
_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "pipeline.so"))

_lib.pipeline_init.restype  = ctypes.c_void_p
_lib.pipeline_init.argtypes = [ctypes.c_int, ctypes.c_int,
                                ctypes.c_float, ctypes.c_float]

_lib.pipeline_process_frame.restype  = ctypes.c_int
_lib.pipeline_process_frame.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_float),
]

_lib.pipeline_destroy.restype  = None
_lib.pipeline_destroy.argtypes = [ctypes.c_void_p]

# ─── Frame source ─────────────────────────────────────────────────────────────
if args.no_camera:
    cap = None
    # Synthetic 1080p frames: random noise to simulate worst-case diff
    W, H = 1920, 1080
    def next_frame():
        return np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
    print(f"Using synthetic {W}×{H} frames (random noise)")
else:
    cap = cv2.VideoCapture(0)
    ret, probe = cap.read()
    if not ret:
        raise RuntimeError("Cannot open camera — try --no-camera")
    H, W = probe.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    def next_frame():
        ret, f = cap.read()
        return f if ret else None
    print(f"Using live camera: {W}×{H}")

N = args.frames
SIGMA         = 3.0
THRESHOLD_VAL = 25.0

# ─── Storage ──────────────────────────────────────────────────────────────────
cpu_times  = {"blur": [], "diff": [], "thresh": [], "total": []}
cuda_times = {"blur": [], "diff": [], "thresh": [], "total": []}

# ─────────────────────────────────────────────────────────────────────────────
# A) CPU BASELINE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nRunning CPU baseline ({N} frames)...")
prev_gray = None

for i in range(N):
    frame = next_frame()
    if frame is None:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    t = time.perf_counter()
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    blur_ms = (time.perf_counter() - t) * 1000

    if prev_gray is None:
        prev_gray = blurred
        continue

    t = time.perf_counter()
    diff = cv2.absdiff(prev_gray, blurred)
    diff_ms = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    _, thresh = cv2.threshold(diff, int(THRESHOLD_VAL), 255, cv2.THRESH_BINARY)
    thresh_ms = (time.perf_counter() - t) * 1000

    cpu_times["blur"].append(blur_ms)
    cpu_times["diff"].append(diff_ms)
    cpu_times["thresh"].append(thresh_ms)
    cpu_times["total"].append(blur_ms + diff_ms + thresh_ms)

    prev_gray = blurred

# ─────────────────────────────────────────────────────────────────────────────
# B) CUDA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
print(f"Running CUDA pipeline ({N} frames)...")

handle    = _lib.pipeline_init(W, H, SIGMA, THRESHOLD_VAL)
mask_buf  = np.empty((H, W), dtype=np.uint8)
timing    = (ctypes.c_float * 4)()

# Reset frame source
if cap:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

for i in range(N):
    frame = next_frame()
    if frame is None:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray_ptr = gray.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    mask_ptr = mask_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    result = _lib.pipeline_process_frame(handle, gray_ptr, mask_ptr, timing)

    if result == 0:
        cuda_times["blur"].append(timing[0])
        cuda_times["diff"].append(timing[1])
        cuda_times["thresh"].append(timing[2])
        cuda_times["total"].append(timing[3])

_lib.pipeline_destroy(handle)
if cap:
    cap.release()

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────
def avg(lst):
    return float(np.mean(lst)) if lst else 0.0

def speedup(cpu_ms, cuda_ms):
    return cpu_ms / cuda_ms if cuda_ms > 0 else float("inf")

print(f"\n{'='*62}")
print(f"  BENCHMARK RESULTS  ({W}×{H}  |  {N} frames  |  sigma={SIGMA})")
print(f"{'='*62}")
print(f"  {'Stage':<12}  {'CPU (ms)':>10}  {'CUDA (ms)':>10}  {'Speedup':>10}")
print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")

for stage in ("blur", "diff", "thresh", "total"):
    c = avg(cpu_times[stage])
    g = avg(cuda_times[stage])
    s = speedup(c, g)
    label = stage if stage != "total" else "TOTAL"
    print(f"  {label:<12}  {c:>10.4f}  {g:>10.4f}  {s:>9.2f}x")

print(f"{'='*62}")
print()
print("  Note: CUDA times are GPU-measured (CUDA events).")
print("  CPU times are wall-clock (time.perf_counter).")
print("  Neither includes grayscale conversion or contour finding,")
print("  making this a direct comparison of stages 2-4 only.")
