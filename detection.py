import os
import argparse
from database import init_db, DEFAULT_DB_PATH
from video_processor import VideoProcessor

DEFAULT_VIDEOS = [
    ("videos/gate.mp4", "Gate"),
    ("videos/parking.mp4", "Parking"),
    ("videos/lobby.mp4", "Lobby")
]

def run_indexing(sample_rate: float = 0.5, conf: float = 0.25):
    print("=" * 60)
    print("Initializing Multi-Stream Video Intelligence Pipeline")
    print("=" * 60)

    # Initialize SQLite database
    init_db(DEFAULT_DB_PATH)

    processor = VideoProcessor(
        model_name="yolo26n.pt",
        sample_interval_sec=sample_rate,
        db_path=DEFAULT_DB_PATH
    )

    total_detections = 0
    for video_path, camera_name in DEFAULT_VIDEOS:
        if os.path.exists(video_path):
            print(f"\nProcessing camera footage: {camera_name} ({video_path})...")
            dets = processor.process_video(
                video_path=video_path,
                camera_name=camera_name,
                conf_threshold=conf
            )
            total_detections += len(dets)
        else:
            print(f"Warning: Video not found at {video_path}")

    print("\n" + "=" * 60)
    print(f"Indexing complete! Logged {total_detections} evidence detections.")
    print(f"Database saved to: {DEFAULT_DB_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and Index CCTV Videos")
    parser.add_argument("--sample-rate", type=float, default=0.5, help="Frame sampling interval in seconds")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    args = parser.parse_args()

    run_indexing(sample_rate=args.sample_rate, conf=args.conf)
