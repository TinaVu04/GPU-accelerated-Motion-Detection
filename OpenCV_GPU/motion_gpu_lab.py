import cv2
import numpy as np
import time

# Enable OpenCL
cv2.ocl.setUseOpenCL(True)

width, height = 640, 480
num_frames = 200


stage_times = {
    "upload":     [],
    "grayscale":  [],
    "blur":       [],
    "frame_diff": [],
    "threshold":  [],
    "contours":   []
}


print("Running GPU (OpenCL) pipeline... press Q on the video window to quit")

# Warming up GPU
dummy = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
dummy_gpu = cv2.UMat(dummy)
for _ in range (10):
    cv2.GaussianBlur(dummy_gpu, (21,21), 0)
print("done")

prev_gray = None
for i in range (num_frames):
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    # STAGE 0 — Upload to GPU (host → device transfer)
    t = time.perf_counter()
    frame_gpu = cv2.UMat(frame)
    stage_times["upload"].append(time.perf_counter() - t)

    # STAGE 1 — Grayscale on GPU
    t = time.perf_counter()
    gray = cv2.cvtColor(frame_gpu, cv2.COLOR_BGR2GRAY)
    stage_times["grayscale"].append(time.perf_counter() - t)

    # STAGE 2 — Gaussian Blur on GPU
    t = time.perf_counter()
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    stage_times["blur"].append(time.perf_counter() - t)

    if prev_gray is None:
        prev_gray = blurred
        continue

    # STAGE 3 — Frame Difference on GPU
    t = time.perf_counter()
    diff = cv2.absdiff(prev_gray, blurred)
    stage_times["frame_diff"].append(time.perf_counter() - t)

    # STAGE 4 — Threshold on GPU
    t = time.perf_counter()
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    stage_times["threshold"].append(time.perf_counter() - t)

    # STAGE 5 — Download + Contours on CPU (contours aren't GPU-friendly)
    t = time.perf_counter()
    thresh_cpu = thresh.get()  # device → host transfer
    contours, _ = cv2.findContours(thresh_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_cpu = frame_gpu.get()
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame_cpu, (x, y), (x+w, y+h), (0, 255, 0), 2)
    stage_times["contours"].append(time.perf_counter() - t)

    prev_gray = blurred

# ── Print benchmark results ────────────────────────────────────
print("\n=== GPU (OpenCL) (CADE MACHINE) RESULTS ===")
total_avg = 0
for stage, times in stage_times.items():
    if times:
        avg_ms = np.mean(times) * 1000
        total_avg += avg_ms
        print(f"  {stage:<15} {avg_ms:.4f} ms")

print(f"  {'TOTAL':<15} {total_avg:.4f} ms")
print(f"  {'FPS':<15} {1000 / total_avg:.1f} FPS theoretical")

# ── Comparison ────────────────────────────────────────────────
print("\n=== CPU NUMBERS HERE TO COMPARE ===")
cpu = {
  "grayscale":       0.5729,
  "blur":            0.7380,
  "frame_diff":      0.2297,
  "threshold":       0.0568, 
  "contours":        0.4198
}
print(f"\n{'Stage':<15} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
print("-" * 50)
for stage in ['grayscale', 'blur', 'frame_diff', 'threshold', 'contours']:
    times = stage_times.get(stage, [])
    if time:
        gpu_avg = np.mean(times) * 1000
        cpu_avg = cpu.get(stage, 0)
        speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 0
        print(f"  {stage:<13} {cpu_avg:<12.4f} {gpu_avg:<12.4f} {speedup:<.2f}x")
