# 🚦 Traffic Analyzer

> Real-time vehicle detection, heatmap generation, and traffic density analysis — powered by YOLOv8.

Point it at a traffic video, and it tells you exactly how many vehicles are on screen at any moment, where they're clustered, and whether traffic is light, moderate, or jammed — with charts and a heatmap to back it up.

---

## 🎯 What It Does

This tool processes traffic footage frame-by-frame using a YOLOv8 model, detects vehicles in each frame, accumulates their positions into a spatial heatmap, and produces a full statistical breakdown of traffic density over time.

**Input:** a traffic video (`.mp4`)  
**Output:** annotated live feed + heatmap image + density charts + CSV data

---

## ✨ Features

- 🚗 Detects cars, buses, trucks, and motorcycles using **YOLOv8m**
- 🔲 Draws bounding boxes and labels on each detected vehicle in real time
- 🌡 Generates a **spatial heatmap** of vehicle positions across the full video
- 🖼 Overlays the heatmap onto a real video frame for geographic context
- 📈 Plots vehicle count over time as a line chart
- 📊 Classifies each frame as **low / medium / high** traffic using IQR-based thresholds
- 💾 Exports all frame-level data to `traffic_data.csv`
- 🎥 Supports 7 different video inputs — select by number at runtime

---

## 🛠 Tech Stack

| | |
|---|---|
| Language | Python 3 |
| Detection model | YOLOv8m (Ultralytics) |
| Computer vision | OpenCV |
| Data analysis | Pandas, NumPy |
| Visualisation | Matplotlib |

---

## 📂 Project Structure

```
Traffic-Analyzer/
├── main.py                   # Core script
├── traffic(1).mp4            # Input videos (1–7)
├── traffic(2).mp4
├── ...
├── heatmap.png               # Generated: raw heatmap
├── heatmap_overlay.png       # Generated: heatmap blended onto video frame
├── traffic_over_time.png     # Generated: vehicle count line chart
├── density_distribution.png  # Generated: traffic density bar chart
├── traffic_data.csv          # Generated: per-frame data export
└── requirements.txt
```

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/withaarav/Traffic-Analyzer.git
cd Traffic-Analyzer

# Install dependencies
pip install ultralytics opencv-python pandas matplotlib numpy

# Run
python main.py
```

You'll be prompted to enter a video number (1–7). The model (`yolov8m.pt`) downloads automatically on first run via Ultralytics.

Or use pycharm to clone the repository if this shows error.

---

## 💡 How It Works

### Detection
Each frame is resized to 1280×720 and passed through YOLOv8m. Detections with confidence > 0.25 and class in `[car, bus, truck, motorcycle]` are counted and drawn with bounding boxes.

### Heatmap
Vehicle centre points `(cx, cy)` are accumulated into a float32 array across all frames. After processing, it's normalised, colourised with `COLORMAP_JET`, and blended onto a sample video frame at 0.5 opacity — showing exactly where traffic concentrates on the road.

### Traffic classification
Per-frame vehicle counts are classified using IQR thresholds computed from the full video:
- ≤ 25th percentile → **low traffic**
- 25th–75th percentile → **medium traffic**
- \> 75th percentile → **high traffic**

Adaptive thresholds mean the labels are always meaningful relative to each video, not hardcoded guesses.

---

## 📊 Sample Output

```
Total frames analyzed: 847
Average vehicles per frame: 6.43
Peak vehicle count: 14
Lowest vehicle count: 1

Traffic density breakdown:
medium traffic    421
low traffic       213
high traffic      213
```

**Files generated after each run:**

| File | Description |
|---|---|
| `heatmap.png` | Pure heatmap of vehicle positions |
| `heatmap_overlay.png` | Heatmap blended over real video frame |
| `traffic_over_time.png` | Vehicle count vs. time (line chart) |
| `density_distribution.png` | Low / medium / high frame distribution |
| `traffic_data.csv` | Per-frame: timestamp, frame number, count, density label |

---

## 🧠 What I Learned

- Running inference with a pre-trained YOLO model and parsing bounding box outputs
- Building a spatial heatmap from object centroids using NumPy float accumulators
- Using IQR-based adaptive thresholds instead of hardcoded density cutoffs
- Blending CV outputs onto video frames with `cv2.addWeighted`
- Structuring a full data pipeline: detection → aggregation → analysis → export

---

## 🔮 What's Next

- [ ] Vehicle tracking across frames (SORT / DeepSORT) to eliminate double-counting
- [ ] Per-lane vehicle counting using configurable zone lines
- [ ] Live webcam / RTSP stream support
- [ ] Real-time dashboard with Streamlit
- [ ] Speed estimation from frame-to-frame displacement

---

## 📬 Contact

Made by [Aarav Porwal](https://github.com/withaarav) · with.aarav@gmail.com
