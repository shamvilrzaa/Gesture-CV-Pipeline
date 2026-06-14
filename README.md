# Gesture-Driven Computer Vision Pipeline

A real-time computer vision system that combines live image processing, RGB histogram analysis, and AI-powered hand gesture recognition into a single 1280x720 30fps virtual camera stream. Built for a Computer Vision university module at Technische Hochschule Ingolstadt.

---

## What it does

Show a hand gesture to your webcam and the system overlays a looped video reaction in real time. While streaming, you can cycle through image filters, toggle histogram equalization, and inspect per-channel image statistics — all live, all keyboard-controlled.

### Supported gestures

| Gesture | Trigger |
|---|---|
| Thumbs up | Thumb extended, all fingers curled |
| Peace | Index and middle fingers extended |
| Open palm | All five fingers extended |
| Fist | All fingers curled, thumb tucked |
| Hang loose | All four fingers extended, thumb tucked |
| Pointing up | Index finger only extended |

---

## Architecture

```
Webcam (index 1, 1280x720 @ 30fps)
    │
    ▼
VirtualCamera capture loop
    │
    ├─ Histogram equalization (CDF-based, per-channel)
    ├─ Linear transform (alpha/beta contrast and brightness)
    ├─ Image filter (Sobel / Gaussian Blur / Sharpen / Gabor / None)
    │
    ├─ MediaPipe Hand Landmarker (21 landmarks, float16 model)
    │       └─ Finger state classifier (PIP joint geometry)
    │               └─ Temporal smoother (majority vote, 5-frame window)
    │                       └─ Video overlay renderer (alpha-blended MP4 loop)
    │
    ├─ Live RGB histogram (Numba JIT-compiled, blitted to frame buffer)
    ├─ Image stats overlay (mean, std, min, max, mode, entropy per channel)
    └─ HUD controls overlay
    │
    ▼
pyvirtualcam output + OpenCV preview window
```

---

## Key technical decisions

| Decision | Why |
|---|---|
| Numba @njit for histogram | Looping over 1M+ pixels per frame in Python causes severe lag. JIT compilation brings it to near-C speed. |
| PIP joint geometry for finger detection | More stable than MCP-based detection when the hand tilts. Checks if fingertip Y < PIP Y (higher on screen = extended). |
| 5-frame majority vote smoothing | Eliminates gesture flickering from single-frame misclassifications without adding perceptible latency. |
| Alpha-blended MP4 overlay | Video loops reset on gesture change, allowing continuous animated reactions without state leakage. |
| Matplotlib blitting for histogram | Redraws only changed artists (bars), not the full figure — makes live histogram viable at 30fps. |
| CDF-based histogram equalization | Stretches the contrast distribution across all 256 values per channel independently, preserving colour balance. |

---

## Keyboard controls

| Key | Action |
|---|---|
| F | Cycle through filters (None / Sobel / Gaussian Blur / Sharpen / Gabor) |
| L | Toggle linear transform (contrast + brightness adjustment) |
| E | Toggle histogram equalization |
| S | Toggle image statistics overlay (mean, std, min, max, mode, entropy) |
| M | Toggle gesture and meme overlay |
| K | Toggle hand landmark skeleton |
| Q | Quit |

---

## Setup

### Requirements

- Python 3.10 or higher
- A webcam (built-in or external)
- macOS or Linux (pyvirtualcam has platform-specific dependencies)

### macOS — install virtual camera driver first

```bash
# Install OBS virtual camera (required by pyvirtualcam on Mac)
brew install obs
# Then open OBS once and enable the virtual camera
```

### Install Python dependencies

```bash
git clone https://github.com/shamvilrzaa/gesture-cv-pipeline
cd gesture-cv-pipeline

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Run

```bash
python run.py
```

> **Note:** On newer MacBooks (M-series), the built-in camera maps to device index `1` instead of `0`. If your camera is not detected, edit `run.py` and change `vc.capture_cv_video(1, ...)` to `vc.capture_cv_video(0, ...)`.

---

## Project structure

```
gesture-cv-pipeline/
├── run.py              # Main entry point and processing loop
├── basics.py           # Image processing, filters, stats, gesture detection
├── capturing.py        # VirtualCamera class and frame capture
├── overlays.py         # Histogram plotting and text overlay utilities
├── requirements.txt    # Python dependencies
├── hand_landmarker.task  # Pre-trained MediaPipe model weights (auto-downloaded)
└── memes/              # Video assets for gesture overlays
    ├── thumbs_up.mp4
    ├── peace.mp4
    ├── open_palm.mp4
    ├── fist.mp4
    ├── hang_loose.mp4
    └── pointing_up.mp4
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Computer vision | OpenCV 4.7 |
| Hand tracking | MediaPipe Hand Landmarker (float16, 21 landmarks) |
| Performance optimisation | Numba JIT (@njit) |
| Live plotting | Matplotlib with blitting |
| Virtual camera output | pyvirtualcam |
| Video overlay | OpenCV VideoCapture with alpha blending |

---

## What I learned

- JIT compilation with Numba makes pixel-level operations viable in real-time Python — without it, the histogram alone dropped the stream below 5fps.
- Geometric finger-state detection using PIP joints is significantly more reliable than MCP-based heuristics when the hand is not held flat toward the camera.
- Temporal smoothing over a short window (5 frames) is enough to eliminate gesture flickering without adding noticeable delay.
- Alpha blending with pre-multiplied RGBA channels avoids colour fringing at transparent edges.
- Matplotlib blitting (rendering only changed artists) is the key to overlaying a live histogram without blocking the frame loop.

---

## Authors

Developed as part of the Computer Vision module at Technische Hochschule Ingolstadt.

Team: syr6882, abr5414, mut7291
