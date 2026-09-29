import cv2
import numpy as np
import time

# Enable OpenCL
cv2.ocl.setUseOpenCL(True)

stage_times = {
    "upload":     [],
    "grayscale":  [],
    "blur":       [],
    "frame_diff": [],
    "threshold":  [],
    "contours":   []
}

cap = cv2.VideoCapture(0)
prev_gray = None
fps_start = time.perf_counter()
frame_count = 0

print("Running GPU (OpenCL) pipeline... press Q on the video window to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

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
    frame_count += 1

    elapsed = time.perf_counter() - fps_start
    fps = frame_count / elapsed
    cv2.putText(frame_cpu, f"GPU FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("GPU Motion Detection", frame_cpu)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ── Print benchmark results ────────────────────────────────────
print("\n=== GPU (OpenCL) RESULTS ===")
total_avg = 0
for stage, times in stage_times.items():
    if times:
        avg_ms = np.mean(times) * 1000
        total_avg += avg_ms
        print(f"  {stage:<15} {avg_ms:.4f} ms")

print(f"  {'TOTAL':<15} {total_avg:.4f} ms")
print(f"  {'FPS':<15} {frame_count / (time.perf_counter() - fps_start):.1f}")

# ── Comparison ────────────────────────────────────────────────
print("\n=== PASTE YOUR CPU NUMBERS HERE TO COMPARE ===")
cpu = {
    "grayscale":  0.1956,
    "blur":       0.8597,
    "frame_diff": 0.1006,
    "threshold":  0.0449,
    "contours":   0.3414,
}
print(f"\n{'Stage':<15} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
print("-" * 50)
for stage, times in stage_times.items():
    if stage == "upload" or not times:
        continue
    gpu_avg = np.mean(times) * 1000
    cpu_avg = cpu.get(stage, 0)
    speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 0
    print(f"  {stage:<13} {cpu_avg:<12.4f} {gpu_avg:<12.4f} {speedup:<10.2f}x")
