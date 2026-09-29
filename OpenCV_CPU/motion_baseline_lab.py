import cv2
import numpy as np
import time

width, height = 640, 480
num_frames = 200

# ── Benchmark storage ──────────────────────────────────────────
stage_times = {
    "grayscale": [],
    "blur": [],
    "frame_diff": [],
    "threshold": [],
    "contours": []
}


print("Running CPU baseline... press Q to quit")

prev_gray = None
print("running CPU baseline synthetic frames")

for i in range(num_frames):
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    # STAGE 1 — Grayscale
    t = time.perf_counter()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    stage_times["grayscale"].append(time.perf_counter() - t)

    # STAGE 2 — Gaussian Blur
    t = time.perf_counter()
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    stage_times["blur"].append(time.perf_counter() - t)

    if prev_gray is None:
        prev_gray = blurred
        continue

    # STAGE 3 — Frame Difference
    t = time.perf_counter()
    diff = cv2.absdiff(prev_gray, blurred)
    stage_times["frame_diff"].append(time.perf_counter() - t)

    # STAGE 4 — Threshold
    t = time.perf_counter()
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    stage_times["threshold"].append(time.perf_counter() - t)

    # STAGE 5 — Contours
    t = time.perf_counter()
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    stage_times["contours"].append(time.perf_counter() - t)

    prev_gray = blurred


# ── Print benchmark results ────────────────────────────────────
print("\n=== CPU CADE LAB MACHINE BASELINE RESULTS ===")
total_avg = 0
for stage, times in stage_times.items():
    if times:
        avg_ms = np.mean(times) * 1000
        total_avg += avg_ms
        print(f"  {stage:<15} {avg_ms:.4f} ms")

print(f"  {'TOTAL':<15} {total_avg:.4f} ms")
print(f"  {'FPS':<15} {1000 / total_avg:.1f} FPS theoretical")
