# Intersection traffic dashboard

Live vehicle counts and red-light timing for the Zlín street camera, using YOLOv8 on Apple Metal.

Streamlink is tried first. This camera does not publish a playlist that Streamlink's YouTube plugin can open, so the app then uses yt-dlp's Android client to fetch the live HLS playlist and decodes it to frames. OpenCV draws the boxes, the counting line, and measures the red lamp.

## Setup (macOS)

From the project folder:

```bash
cd /Users/morpheus/project_camera
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The app opens in the browser. Press **Start**. The first run downloads `yolov8n.pt` and compiles Metal kernels, so the first frames are slower.

This Mac is Apple Silicon. PyTorch uses `device="mps"` when Metal is available. An Intel Mac falls back to CPU and the dashboard shows that device.

## What to adjust

- **Counting line** — default sits on the near approach, under the crosswalk. Move it onto the lane you want to count.
- **Count direction** — `down` is toward the bottom of the frame, `up` is toward the intersection, `both` counts either way. Each tracked vehicle is counted once.
- **Traffic light box** — default covers the signal near the middle of the intersection, facing the near approach. Shrink it onto one lamp, then turn on **Show red mask** and raise or lower **Red pixel ratio** until the mask lights up only on red.

## Project layout

```
app.py                          Streamlit dashboard
requirements.txt
traffic_dashboard/config.py     Defaults and shared stats
traffic_dashboard/capture.py    streamlink + OpenCV capture
traffic_dashboard/detector.py   YOLOv8 tracking on MPS
traffic_dashboard/counter.py    Virtual line crossing
traffic_dashboard/traffic_light.py  HSV red detection and phase timing
traffic_dashboard/annotate.py   Boxes, line, and signal overlay
traffic_dashboard/pipeline.py   Background processing loop
```
