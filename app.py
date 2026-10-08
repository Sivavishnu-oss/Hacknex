import streamlit as st
import os
import shutil
import tempfile
from PIL import Image

from video_intelligence_engine import VideoIntelligenceEngine
from database import init_db, query_detections, get_unique_cameras, DEFAULT_DB_PATH
from ai_explainer import generate_llm_explanation

# Page setup
st.set_page_config(
    page_title="YOLO26 Multi-Stream Video Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #0f172a 100%);
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    .command-bar {
        background: rgba(30, 41, 59, 0.7);
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 6px;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.3);
    }
    .result-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(59, 130, 246, 0.35);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
    }
    .result-card:hover {
        border-color: #38bdf8;
    }
    .badge-time {
        background: #0284c7;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 13px;
        display: inline-block;
    }
    .badge-obj {
        background: #1e293b;
        color: #38bdf8;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 12px;
        border: 1px solid #334155;
        display: inline-block;
        margin-left: 6px;
    }
    .badge-score {
        background: #14532d;
        color: #86efac;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
        margin-left: 6px;
    }
    .ai-banner {
        background: linear-gradient(90deg, rgba(30, 58, 138, 0.4) 0%, rgba(88, 28, 135, 0.4) 100%);
        border-left: 4px solid #38bdf8;
        padding: 14px 18px;
        border-radius: 8px;
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "engine" not in st.session_state:
    with st.spinner("Initializing YOLO26 and OpenCLIP multimodal model..."):
        st.session_state["engine"] = VideoIntelligenceEngine(yolo_model="yolo26n.pt", clip_model="ViT-B-32")

if "active_catalog" not in st.session_state:
    st.session_state["active_catalog"] = []

if "active_video_path" not in st.session_state:
    st.session_state["active_video_path"] = "videos/gate.mp4" if os.path.exists("videos/gate.mp4") else None

if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""

engine = st.session_state["engine"]

# Top Header
st.title("🎬 Multi-Stream Video Intelligence with YOLO26 & CLIP")
st.caption("Upload or select any video, describe what you are looking for in natural language, and instantly fetch matching photos with precise timings.")

# Sidebar: Video Selection / Insertion
st.sidebar.header("📥 Video Input")
video_source_mode = st.sidebar.radio("Choose Video Source:", ["Upload My Own Video", "Select Sample CCTV Camera Footage"])

uploaded_file = None
selected_sample = None

if video_source_mode == "Upload My Own Video":
    uploaded_file = st.sidebar.file_uploader("Upload video file (MP4, MOV, AVI, MKV)", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_file is not None:
        temp_dir = os.path.join(tempfile.gettempdir(), "hacknex_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        saved_video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(saved_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # If new video file uploaded
        if st.session_state.get("active_video_path") != saved_video_path:
            st.session_state["active_video_path"] = saved_video_path
            st.session_state["active_catalog"] = []

else:
    sample_options = {
        "Gate Camera (Vehicles, Pedestrians, Bicycles)": "videos/gate.mp4",
        "Parking Camera (Cars, Bus, Luggage)": "videos/parking.mp4",
        "Lobby Camera (Reception, Walking People)": "videos/lobby.mp4"
    }
    selected_sample = st.sidebar.selectbox("Choose Sample Feed:", list(sample_options.keys()))
    sample_path = sample_options[selected_sample]
    if os.path.exists(sample_path) and st.session_state.get("active_video_path") != sample_path:
        st.session_state["active_video_path"] = sample_path
        st.session_state["active_catalog"] = []

# Video processing parameters
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Detection Parameters")
sample_rate = st.sidebar.slider("Sampling Interval (seconds)", min_value=0.2, max_value=2.0, value=0.5, step=0.1)
yolo_conf = st.sidebar.slider("YOLO26 Confidence Threshold", min_value=0.15, max_value=0.9, value=0.25, step=0.05)

# Primary Video Display & Status Bar
current_video = st.session_state.get("active_video_path")
col_vid, col_info = st.columns([1.2, 1])

with col_vid:
    if current_video and os.path.exists(current_video):
        st.video(current_video)
        st.caption(f"Currently loaded video: `{os.path.basename(current_video)}`")
    else:
        st.info("Please upload a video or select a sample stream to begin.")

with col_info:
    st.markdown("### 📊 Video Index Status")
    num_indexed = len(st.session_state["active_catalog"])

    if num_indexed == 0:
        st.warning("Video is not indexed yet. Click **Process Video with YOLO26** below to scan and extract detectable objects and visual embeddings.")
        if current_video and os.path.exists(current_video):
            if st.button("🚀 Process Video with YOLO26", type="primary", use_container_width=True):
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def update_progress(pct, msg):
                    progress_bar.progress(pct)
                    status_text.text(msg)

                ev_out_dir = os.path.join("evidence", "uploaded_" + os.path.splitext(os.path.basename(current_video))[0])
                with st.spinner("Processing frames through YOLO26 & CLIP..."):
                    cat = engine.process_and_index_video(
                        video_path=current_video,
                        output_dir=ev_out_dir,
                        sample_interval_sec=sample_rate,
                        conf_threshold=yolo_conf,
                        progress_callback=update_progress
                    )
                    st.session_state["active_catalog"] = cat

                st.success(f"Video indexed successfully! Captured {len(cat)} detections and keyframes.")
                st.rerun()
    else:
        st.success(f"Video indexed! **{num_indexed} visual events & keyframes** ready for natural language query.")
        reindex = st.button("🔄 Re-process Video", use_container_width=True)
        if reindex:
            st.session_state["active_catalog"] = []
            st.rerun()

st.markdown("---")

# Natural Language Command Bar / Search Box
st.subheader("🔎 Command Bar: Describe What You Want to Find")
st.write("Type anything in normal language (e.g. *'Find the car'*, *'Show me a person'*, *'Where is the bicycle?'*, *'A person in black shirt'*, *'White vehicle'*):")

col_search, col_btn = st.columns([4, 1])
with col_search:
    user_query = st.text_input(
        "Search command",
        value=st.session_state.get("last_query", "Find a car"),
        placeholder="Enter search command (e.g., 'Find the car', 'Show person walking', 'A white truck')...",
        label_visibility="collapsed"
    )
with col_btn:
    execute_search = st.button("🔍 Search Video", type="primary", use_container_width=True)

# Quick Suggestion Chips
chip_col1, chip_col2, chip_col3, chip_col4 = st.columns(4)
with chip_col1:
    if st.button("🚗 'Find a car'", use_container_width=True):
        user_query = "Find a car"
        execute_search = True
with chip_col2:
    if st.button("🚶 'Find a person'", use_container_width=True):
        user_query = "Find a person"
        execute_search = True
with chip_col3:
    if st.button("🚲 'Show bicycle'", use_container_width=True):
        user_query = "Show bicycle"
        execute_search = True
with chip_col4:
    if st.button("⚪ 'Find white vehicle'", use_container_width=True):
        user_query = "Find white vehicle"
        execute_search = True

# Execute Search & Render Matching Photos and Timings
if (execute_search or user_query) and len(st.session_state["active_catalog"]) > 0:
    st.session_state["last_query"] = user_query
    
    with st.spinner(f"Scanning video for: '{user_query}'..."):
        results = engine.search_video(
            user_query=user_query,
            catalog=st.session_state["active_catalog"],
            top_k=9
        )

    # AI Summary
    if results:
        top_hit = results[0]
        st.markdown(f"""
        <div class="ai-banner">
            <h4 style="margin: 0 0 6px 0; color: #38bdf8;">🎯 AI Detection Result</h4>
            <p style="margin: 0; font-size: 15px;">
                Found target <b>{top_hit['object']}</b> at timestamp <b>{top_hit['timestamp_str']}</b> ({top_hit['timestamp_sec']}s) with high confidence. 
                Identified <b>{len(results)} distinct matching moments</b> in this video.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(f"No clear occurrences matching '{user_query}' found in this video.")

    st.subheader(f"📸 Detected Photos & Video Timings ({len(results)} matches)")

    # Grid of Photos with Timings
    if results:
        cols = st.columns(3)
        for idx, item in enumerate(results):
            col = cols[idx % 3]
            with col:
                st.markdown(f"""
                <div class="result-card">
                    <div>
                        <span class="badge-time">⏱️ {item['timestamp_str']}</span>
                        <span class="badge-obj">{item['object'].upper()}</span>
                        <span class="badge-score">Match: {int(item['match_score']*100)}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Photo Evidence
                if os.path.exists(item["evidence_path"]):
                    img = Image.open(item["evidence_path"])
                    st.image(img, use_container_width=True, caption=f"Timestamp: {item['timestamp_str']} ({item['timestamp_sec']}s)")
                else:
                    st.caption(f"Photo: {item['evidence_path']}")

                # Play directly from this timestamp
                if current_video and os.path.exists(current_video):
                    with st.expander(f"▶️ Play from {item['timestamp_str']}"):
                        st.video(current_video, start_time=int(item["timestamp_sec"]))

elif len(st.session_state["active_catalog"]) == 0 and current_video:
    st.info("💡 To search what you say, click **'Process Video with YOLO26'** above first!")
