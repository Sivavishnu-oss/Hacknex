import streamlit as st
import os
import shutil
import tempfile
import pandas as pd
from PIL import Image

from video_intelligence_engine import VideoIntelligenceEngine
from database import (
    init_db,
    query_detections,
    get_unique_cameras,
    get_detections_for_camera,
    get_all_cameras_meta,
    DEFAULT_DB_PATH
)
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

if "show_database" not in st.session_state:
    st.session_state["show_database"] = False

engine = st.session_state["engine"]

# Derive current video name/camera identifier
def get_current_camera_name(vpath):
    if not vpath:
        return "Unknown"
    base = os.path.splitext(os.path.basename(vpath))[0]
    return base.capitalize()

current_camera_name = get_current_camera_name(st.session_state.get("active_video_path"))

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
            st.session_state["show_database"] = False

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
        st.session_state["show_database"] = False

# Quick toggle also in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🗄️ Database Inspector")
db_sidebar_toggle = st.sidebar.checkbox(
    "📂 View Video Database",
    value=st.session_state["show_database"],
    key="sidebar_db_toggle",
    help="Click to inspect all detections, metadata, and timestamps stored in SQLite for this video"
)
st.session_state["show_database"] = db_sidebar_toggle

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
        st.caption(f"Currently loaded video: `{os.path.basename(current_video)}` (Camera: **{current_camera_name}**)")
    else:
        st.info("Please upload a video or select a sample stream to begin.")

with col_info:
    st.markdown("### 📊 Video Index Status")
    num_indexed = len(st.session_state["active_catalog"])

    if num_indexed == 0:
        st.warning("Video is not indexed yet. Click **Process Video with YOLO26** below to scan and extract detectable objects and visual embeddings into the database.")
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
                        progress_callback=update_progress,
                        camera_name=current_camera_name
                    )
                    st.session_state["active_catalog"] = cat

                st.success(f"Video indexed & saved to database! Captured {len(cat)} detections and keyframes.")
                st.rerun()

        # Even if not indexed in current session, allow user to check if DB has records for this camera
        if st.button("🗄️ Click to See Video Database", use_container_width=True):
            st.session_state["show_database"] = not st.session_state["show_database"]
            st.rerun()

    else:
        st.success(f"Video indexed! **{num_indexed} visual events & keyframes** saved in SQLite database.")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            reindex = st.button("🔄 Re-process Video", use_container_width=True)
            if reindex:
                st.session_state["active_catalog"] = []
                st.rerun()
        with col_btn2:
            db_label = "❌ Close Database" if st.session_state["show_database"] else "🗄️ Click & See Database"
            if st.button(db_label, type="secondary", use_container_width=True):
                st.session_state["show_database"] = not st.session_state["show_database"]
                st.rerun()

# =========================================================================
# 🗄️ DATABASE VIEWER SECTION (CLICK & SEE THE DATABASE)
# =========================================================================
if st.session_state.get("show_database", False):
    st.markdown("---")
    db_header_col, db_close_col = st.columns([5, 1])
    with db_header_col:
        st.subheader(f"🗄️ Database Records for Video: `{os.path.basename(current_video) if current_video else 'N/A'}`")
        st.caption("Inspect all SQLite detections, coordinates, labels, confidence scores, and video metadata.")
    with db_close_col:
        if st.button("✖ Close Database", use_container_width=True):
            st.session_state["show_database"] = False
            st.rerun()

    # Query detections for this video/camera
    records = get_detections_for_camera(current_camera_name)
    
    # Also check if user has active in-memory catalog
    catalog_items = [c for c in st.session_state.get("active_catalog", []) if c.get("type") == "detection"]
    
    # Display quick summary metrics
    total_db_dets = len(records)
    total_catalog = len(catalog_items)
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("DB Detections", total_db_dets)
    with metric_col2:
        st.metric("Session Indexed Detections", total_catalog)
    with metric_col3:
        unique_classes = len(set(r["object_class"] for r in records)) if records else (len(set(c["object"] for c in catalog_items)) if catalog_items else 0)
        st.metric("Unique Classes", unique_classes)
    with metric_col4:
        st.metric("Storage DB", os.path.basename(DEFAULT_DB_PATH))

    # Tabs for detailed inspection
    db_tab1, db_tab2, db_tab3 = st.tabs(["📋 Detections Table", "🔍 Filter & Inspect", "📊 Video Metadata"])

    with db_tab1:
        if records:
            df_rows = []
            for r in records:
                df_rows.append({
                    "ID": r.get("id"),
                    "Camera": r.get("camera"),
                    "Timestamp": r.get("timestamp_str"),
                    "Time (s)": round(r.get("timestamp_sec", 0), 2),
                    "Class": r.get("object_class"),
                    "Confidence": f"{float(r.get('confidence', 0)):.2%}",
                    "Color": r.get("color_hint") or "N/A",
                    "BBox [x1, y1, x2, y2]": str(r.get("bbox", [])),
                    "Evidence Path": r.get("evidence_path")
                })
            df = pd.DataFrame(df_rows)
            st.dataframe(df, use_container_width=True, height=350)
            
            # Download options
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Database Records as CSV",
                data=csv_data,
                file_name=f"{current_camera_name}_detections.csv",
                mime="text/csv"
            )
        elif catalog_items:
            st.info("Showing in-memory catalog (unsaved or current session):")
            df_rows = []
            for idx, c in enumerate(catalog_items):
                df_rows.append({
                    "Index": idx + 1,
                    "Camera": current_camera_name,
                    "Timestamp": c.get("timestamp_str"),
                    "Time (s)": c.get("timestamp_sec"),
                    "Class": c.get("object"),
                    "Confidence": f"{float(c.get('confidence', 0)):.2%}",
                    "Color": c.get("color") or "N/A",
                    "BBox": str(c.get("bbox", [])),
                    "Evidence Path": c.get("evidence_path")
                })
            df = pd.DataFrame(df_rows)
            st.dataframe(df, use_container_width=True, height=350)
        else:
            st.warning(f"No database records found for camera/video '{current_camera_name}'. Please click '🚀 Process Video with YOLO26' above to populate the database.")

    with db_tab2:
        if records or catalog_items:
            source_data = records if records else [
                {"object_class": c["object"], "confidence": c["confidence"], "timestamp_sec": c["timestamp_sec"], "timestamp_str": c["timestamp_str"], "color_hint": c.get("color"), "bbox": c.get("bbox"), "evidence_path": c.get("evidence_path")}
                for c in catalog_items
            ]
            
            all_classes = sorted(list(set(r.get("object_class") for r in source_data if r.get("object_class"))))
            filt_col1, filt_col2 = st.columns(2)
            with filt_col1:
                selected_cls = st.selectbox("Filter by Object Class:", ["All"] + all_classes)
            with filt_col2:
                min_conf = st.slider("Filter by Minimum Confidence:", 0.0, 1.0, 0.25, 0.05)
                
            filtered = [
                r for r in source_data
                if (selected_cls == "All" or r.get("object_class") == selected_cls) and float(r.get("confidence", 0)) >= min_conf
            ]
            st.write(f"Showing **{len(filtered)}** filtered detections:")
            
            if filtered:
                # Show snapshot previews for top filtered entries
                preview_cols = st.columns(min(4, max(1, len(filtered))))
                for idx, f_item in enumerate(filtered[:4]):
                    with preview_cols[idx % len(preview_cols)]:
                        ev_path = f_item.get("evidence_path", "")
                        if ev_path and os.path.exists(ev_path):
                            st.image(ev_path, caption=f"{f_item.get('object_class')} @ {f_item.get('timestamp_str')}")
                        else:
                            st.caption(f"{f_item.get('object_class')} @ {f_item.get('timestamp_str')}")

    with db_tab3:
        all_cams = get_all_cameras_meta()
        if all_cams:
            cam_df = pd.DataFrame(all_cams)
            st.write("All registered video streams in SQLite database:")
            st.dataframe(cam_df, use_container_width=True)
        else:
            st.info("No registered cameras yet in metadata table.")

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
