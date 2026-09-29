"""
walkthrough.py

Plain-English walkthrough of the CUDA motion detection pipeline.
No camera needed — uses two synthetic frames to show what each stage
does and why it exists.

Run:
  python walkthrough.py
"""

import numpy as np
import cv2

print("=" * 60)
print("  CUDA Motion Detection Pipeline — Step by Step")
print("=" * 60)

# ── Two synthetic frames ──────────────────────────────────────────
# Frame A: static grey background  (no motion)
# Frame B: same background + white rectangle  (simulated movement)
W, H = 320, 240
frame_a = np.full((H, W, 3), 80, dtype=np.uint8)
frame_b = np.full((H, W, 3), 80, dtype=np.uint8)
cv2.rectangle(frame_b, (100, 80), (200, 160), (255, 255, 255), -1)

print("\n  We have two frames from the camera:")
print(f"    Frame A — static background, shape {frame_a.shape}")
print(f"    Frame B — same background + a moving object, shape {frame_b.shape}")
print("  Goal: find where the object moved.\n")

# ─────────────────────────────────────────────────────────────────
print("─" * 60)
print("  STAGE 1 — Grayscale  [CPU]")
print("─" * 60)
print("""
  The raw camera frame has 3 colour channels (Blue, Green, Red).
  Motion detection only cares about brightness changes, not colour.
  So we collapse 3 channels → 1 channel.

  This also means 3x less data for the GPU to process later.
""")

gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

print(f"  Before: {frame_b.shape}  (H x W x 3 channels)")
print(f"  After:  {gray_b.shape}  (H x W x 1 channel)")
print(f"  Centre pixel (BGR): {frame_b[H//2, W//2]}  →  (grey): {gray_b[H//2, W//2]}")

# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STAGE 2 — Gaussian Blur  [GPU — gaussian_blur.cu]")
print("─" * 60)
print("""
  Camera sensors are noisy. Even on a perfectly still scene, pixels
  flicker slightly between frames. If we diff the raw frames, that
  noise shows up as false motion everywhere.

  Gaussian blur averages each pixel with its neighbours using a
  bell-curve weighting — close neighbours matter more than far ones.
  This smooths out flicker so only real movement survives the diff.

  On the GPU this uses shared memory tiling:
    - Each thread block loads a tile of pixels into fast shared memory
    - Neighbouring threads share the loaded data instead of re-reading
      from slow global memory
    - Two 1D passes (horizontal then vertical) instead of one 2D pass
      → 10x fewer multiply-adds for radius 10
""")

blurred_a = cv2.GaussianBlur(gray_a, (21, 21), 3)
blurred_b = cv2.GaussianBlur(gray_b, (21, 21), 3)

# show how a noisy pixel gets smoothed
noisy_pixel  = int(gray_b[H//2, W//2])
smooth_pixel = int(blurred_b[H//2, W//2])
print(f"  Centre pixel before blur: {noisy_pixel}")
print(f"  Centre pixel after blur:  {smooth_pixel}  (averaged with neighbours)")

# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STAGE 3 — Frame Difference  [GPU — frame_diff.cu]")
print("─" * 60)
print("""
  Subtract the previous blurred frame from the current blurred frame,
  pixel by pixel, and take the absolute value:

      diff[i] = |current[i] - previous[i]|

  Static background pixels cancel out → near zero.
  Moving object pixels changed significantly → bright.

  Every pixel is independent so this is embarrassingly parallel —
  no shared memory needed, just one subtraction per thread.
""")

diff = cv2.absdiff(blurred_a, blurred_b)

bg_pixel     = int(diff[10, 10])       # background region
motion_pixel = int(diff[H//2, W//2])  # inside the moving rectangle
print(f"  Background pixel diff:      {bg_pixel}   (near zero = no motion)")
print(f"  Moving object pixel diff:   {motion_pixel}  (large = motion detected)")

# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STAGE 4 — Threshold  [GPU — threshold.cu]")
print("─" * 60)
print("""
  The diff image is still greyscale — lots of small values from tiny
  lighting changes, large values from real motion. We snap it to
  black or white with a threshold of 25/255:

      pixel > 25  →  255 (white, motion)
      pixel ≤ 25  →  0   (black, background)

  This removes ambiguity and gives us a clean binary mask.
  The output is uint8, matching what findContours expects on the CPU.

  GPU note: this causes warp divergence — some threads go one branch,
  some go the other. It's unavoidable here but the branches are trivial
  (a single assignment) so the cost is negligible.
""")

_, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

white_pixels = int(np.sum(thresh == 255))
total_pixels = W * H
print(f"  Threshold value: 25")
print(f"  White pixels (motion): {white_pixels} / {total_pixels}  "
      f"({100*white_pixels/total_pixels:.1f}% of frame)")

# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STAGE 5 — Contours  [CPU — motion_detect.py]")
print("─" * 60)
print("""
  The threshold mask is a blob of white pixels. Contour finding traces
  the outlines of those blobs and lets us:
    - Draw bounding boxes around moving objects
    - Filter out tiny blobs that are probably still noise (area < 500)
    - Count objects, measure their size, track them

  This stays on the CPU because OpenCV has no GPU findContours,
  and the algorithm is inherently sequential — each step in the
  boundary trace depends on where the previous step ended up.
  It's also fast: we're only tracing edges of a few blobs, not
  processing every pixel.
""")

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
real = [c for c in contours if cv2.contourArea(c) > 500]

print(f"  Total contours found:          {len(contours)}")
print(f"  After area filter (> 500 px):  {len(real)}")
for i, c in enumerate(real):
    x, y, w, h = cv2.boundingRect(c)
    print(f"    Object {i+1}: bounding box x={x} y={y} w={w} h={h}  "
          f"area={cv2.contourArea(c):.0f}px")

# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print("""
  Stage 1  Grayscale    CPU   Simplify — drop colour, 3x less data
  Stage 2  Blur         GPU   Denoise — smooth flicker before diff
  Stage 3  Frame diff   GPU   Detect — find what changed
  Stage 4  Threshold    GPU   Clean — snap grey to black/white mask
  Stage 5  Contours     CPU   Interpret — blobs → bounding boxes

  Stages 2-4 share GPU device memory with zero host<->device copies
  between them. Only two transfers happen per frame:
    • One upload  (grayscale frame → GPU)
    • One download (uint8 mask → CPU for contours)
""")
