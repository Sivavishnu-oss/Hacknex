# 🎬 HNX26EPS05 — Multi-Stream Video Intelligence Platform

An intelligent video surveillance and search platform built with **YOLO26** (`yolo26n.pt`) and **OpenCLIP**. Upload or select recorded CCTV video streams, search with a natural-language command bar, and automatically fetch the matching **photos and video timestamps**.

---

## 🌟 Key Features

- **Video Ingestion & Custom Upload**: Insert any custom recorded video (`.mp4`, `.mov`, `.avi`, `.mkv`) or choose from sample CCTV feeds (Gate, Parking, Lobby).
- **YOLO26 Object Detection & Tracking**: Leverages `yolo26n.pt` to detect persons, vehicles (cars, buses, trucks, bikes), and items (bags, backpacks) with bounding boxes and confidence metrics.
- **Open-Vocabulary Semantic Search (OpenCLIP ViT-B-32)**: Direct multimodal search matching free-form queries (actions, colors, scene attributes).
- **Interactive Command Bar**: Natural language query input (e.g., *"Find the car"*, *"Show me a person"*, *"Find the bicycle"*, *"White vehicle in parking"*).
- **Evidence Photos & Timestamp Extraction**: Returns labeled evidence snapshots with precise timecodes (e.g., `00:46` / `46.0s`).
- **Jump-to-Timestamp Playback**: Watch the exact moment in the video directly from the search result.
- **Cross-Camera Trajectory Analysis**: Reconstructs object movement paths across multiple cameras.
- **SQLite Database**: Structured storage for detections, camera metadata, and embeddings.

---

## 🏗️ Architecture & Pipeline

```text
               Video Input (.mp4 / .mov / uploaded)
                                 │
                                 ▼
                     OpenCV Frame Sampler (0.5s - 1.0s)
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
   YOLO26 Detector                              OpenCLIP ViT-B-32
  (Boxes, Objects, Conf)                      (Visual Embeddings)
         │                                               │
         └───────────────────────┬───────────────────────┘
                                 ▼
                 Dominant Color & Evidence Extractor
                                 │
                                 ▼
                      SQLite Database Storage
                                 │
                                 ▼
                 Natural Language Command Search
              ("Find a car", "Person near the gate")
                                 │
                                 ▼
               Ranked Evidence Photos + Timings + Jump-to-Video
```

---

## 📁 Repository Structure

```text
Hacknex/
├── app.py                         # Streamlit web dashboard with command bar and video upload
├── video_intelligence_engine.py   # Core YOLO26 + OpenCLIP indexing and retrieval engine
├── video_processor.py             # OpenCV frame extractor & YOLO detection pipeline
├── retrieval.py                   # Hybrid CLIP & SQL ranking with cross-camera tracking
├── query_parser.py                # Natural language intent & filter extractor
├── color_analyzer.py              # Dominant color analysis (HSV classification)
├── database.py                    # SQLite schema and query methods
├── ai_explainer.py                # Factual AI response layer
├── detection.py                   # Headless batch video indexing CLI
├── generate_sample_videos.py      # Sample CCTV footage generator
├── requirements.txt               # Python package dependencies
├── videos/                        # CCTV footage storage
│   ├── gate.mp4
│   ├── parking.mp4
│   └── lobby.mp4
├── evidence/                      # Extracted visual evidence frames & crops
└── database/
    └── cctv_intelligence.db       # SQLite indexed detections
```

---

## 🚀 Quickstart Guide & How to Run

### 1. Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- Virtual environment (`venv` or `uv`)
- Hardware acceleration supported (Apple Silicon MPS, NVIDIA CUDA, or CPU)

### 2. Environment Setup

#### Option A: Using standard Python `venv`
```bash
# Navigate to the project root directory
cd Hacknex

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate       # On macOS/Linux
# .venv\Scripts\activate        # On Windows

# Install required dependencies
pip install -r requirements.txt
```

#### Option B: Using `uv` (Faster)
```bash
cd Hacknex
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

### 3. Running the Project

#### Step 1: (Optional) Generate Sample CCTV Videos
If you do not have sample footage in the `videos/` directory or want synthetic test feeds (Gate, Parking, Lobby):
```bash
python generate_sample_videos.py
```

#### Step 2: (Optional) Pre-Index CCTV Footage via CLI
To batch-process and index the sample videos into the SQLite database before opening the dashboard:
```bash
python detection.py --sample-rate 0.5 --conf 0.25
```
*Options:*
- `--sample-rate`: Sampling interval in seconds (default: `0.5`).
- `--conf`: YOLO confidence threshold (default: `0.25`).

#### Step 3: Run the Streamlit Web Application
To launch the interactive dashboard:
```bash
streamlit run app.py
```
After starting, open your browser and navigate to:
👉 run in the localhost

---

## 🔍 How to Use the System

1. **Insert / Select Video**:
   - In the sidebar, select **"Upload My Own Video"** and drop any video file, or pick a sample camera feed (Gate, Parking, Lobby).
2. **Process Video**:
   - Click **"🚀 Process Video with YOLO26"** to scan the footage, detect objects, extract evidence frames, and compute visual embeddings.
3. **Search What You Say**:
   - Type in the command bar what you want to find:
     - `Find a car`
     - `Show me a person`
     - `Where is the bicycle?`
     - `Find white vehicle`
   - Click **"🔍 Search Video"** or select a quick suggestion chip.
4. **Inspect Evidence & Play Timing**:
   - View the photo evidence with confidence and detected class.
   - Expand the **▶️ Play from [Timestamp]** button to jump directly to that point in the video.

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Programming Language** | Python 3.11 |
| **Object Detection** | YOLO26 (`yolo26n.pt`) |
| **Multimodal Embeddings** | OpenCLIP (`ViT-B-32`, LAION-2B) |
| **Video Processing** | OpenCV (`cv2`) |
| **Database** | SQLite3 |
| **Frontend UI** | Streamlit |
| **Acceleration** | Apple Silicon MPS / CUDA |

---

## 📜 License
Developed for **HNX26EPS05 Multi-Stream Video Intelligence**.
