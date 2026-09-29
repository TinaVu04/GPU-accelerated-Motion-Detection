import cv2
import numpy as np
import time

# ── Benchmark storage ──────────────────────────────────────────
stage_times = {
    "grayscale": [],
    "blur": [],
    "frame_diff": [],
    "threshold": [],
    "contours": []
}

cap = cv2.VideoCapture(0)
prev_gray = None
fps_start = time.perf_counter()
frame_count = 0

print("Running CPU baseline... press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

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
    frame_count += 1

    # FPS display
    elapsed = time.perf_counter() - fps_start
    fps = frame_count / elapsed
    cv2.putText(frame, f"CPU FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("CPU Motion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()

# ── Print benchmark results ────────────────────────────────────
print("\n=== CPU BASELINE RESULTS ===")
total_avg = 0
for stage, times in stage_times.items():
    if times:
        avg_ms = np.mean(times) * 1000
        total_avg += avg_ms
        print(f"  {stage:<15} {avg_ms:.4f} ms")

print(f"  {'TOTAL':<15} {total_avg:.4f} ms")
print(f"  {'FPS':<15} {frame_count / (time.perf_counter() - fps_start):.1f}")
