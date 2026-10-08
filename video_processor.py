import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from database import insert_detection, save_camera, DEFAULT_DB_PATH
from color_analyzer import extract_dominant_color

# Target classes to index for CCTV surveillance
INTERESTED_CLASSES = {
    "person", "car", "bus", "truck", "motorcycle", "bicycle", "backpack", "suitcase", "handbag"
}

class VideoProcessor:
    def __init__(self, model_name: str = "yolo26n.pt", sample_interval_sec: float = 0.5, db_path: str = DEFAULT_DB_PATH):
        self.model_name = model_name
        self.sample_interval_sec = sample_interval_sec
        self.db_path = db_path
        self.model = None
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        self._init_detector()

    def _init_detector(self):
        """Loads YOLO detector model."""
        try:
            from ultralytics import YOLO
            print(f"Loading YOLO model: {self.model_name}...")
            self.model = YOLO(self.model_name)
            print("YOLO model loaded successfully.")
        except Exception as e:
            print(f"Warning: YOLO not loaded or error: {e}")

    def _init_clip(self):
        """Lazy load CLIP model for embedding visual evidence."""
        if self.clip_model is not None:
            return
        try:
            import open_clip
            import torch
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            print(f"Loading OpenCLIP (ViT-B-32) on {device}...")
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
            model = model.to(device)
            model.eval()
            self.clip_model = model
            self.clip_preprocess = preprocess
            self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
            self.clip_device = device
            print("OpenCLIP model loaded successfully.")
        except Exception as e:
            print(f"Warning: OpenCLIP could not be loaded: {e}")

    def compute_image_embedding(self, image_bgr: np.ndarray) -> Optional[bytes]:
        """Computes CLIP embedding vector for an image or object crop."""
        self._init_clip()
        if self.clip_model is None:
            return None
        try:
            import torch
            from PIL import Image
            img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            img_tensor = self.clip_preprocess(pil_img).unsqueeze(0).to(self.clip_device)
            with torch.no_grad():
                features = self.clip_model.encode_image(img_tensor)
                features /= features.norm(dim=-1, keepdim=True)
                vector_np = features.cpu().numpy().astype(np.float32).flatten()
                return vector_np.tobytes()
        except Exception as e:
            print(f"Error computing CLIP embedding: {e}")
            return None

    def process_video(
        self,
        video_path: str,
        camera_name: str,
        evidence_dir: str = "evidence",
        conf_threshold: float = 0.35,
        track_objects: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Processes a CCTV video file:
        - Extracts frames at `sample_interval_sec` intervals
        - Runs YOLO object detection / tracking
        - Crops detected entities & saves evidence frames
        - Analyzes dominant attributes (color)
        - Computes multimodal CLIP embeddings
        - Persists detections into SQLite
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cam_dir = os.path.join(evidence_dir, camera_name.lower().replace(" ", "_"))
        os.makedirs(cam_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps

        # Register camera metadata
        save_camera(
            camera_id=camera_name.lower().replace(" ", "_"),
            camera_name=camera_name,
            video_path=video_path,
            fps=fps,
            duration_sec=duration_sec,
            resolution=f"{width}x{height}",
            db_path=self.db_path
        )

        frame_step = max(1, int(fps * self.sample_interval_sec))
        current_frame_idx = 0
        detections_collected = []

        print(f"Processing camera '{camera_name}' ({video_path}) - {duration_sec:.1f}s, sampling every {self.sample_interval_sec}s...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame_idx % frame_step == 0:
                timestamp_sec = current_frame_idx / fps
                mins = int(timestamp_sec // 60)
                secs = int(timestamp_sec % 60)
                timestamp_str = f"{mins:02d}:{secs:02d}"

                # Run YOLO tracking/inference
                if self.model is not None:
                    if track_objects:
                        results = self.model.track(frame, persist=True, verbose=False, conf=conf_threshold)
                    else:
                        results = self.model(frame, verbose=False, conf=conf_threshold)

                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            cls_name = self.model.names[cls_id]
                            conf = float(box.conf[0])

                            # Standardize bag / classes
                            normalized_cls = "bag" if cls_name in ["backpack", "suitcase", "handbag"] else cls_name

                            # Only record objects of interest
                            if normalized_cls not in INTERESTED_CLASSES and cls_name not in INTERESTED_CLASSES:
                                continue

                            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                            x1, y1, x2, y2 = max(0, xyxy[0]), max(0, xyxy[1]), min(width, xyxy[2]), min(height, xyxy[3])
                            track_id = int(box.id[0]) if box.id is not None else None

                            # Evidence frame generation with bounding box & timestamp overlay
                            evidence_frame = frame.copy()
                            cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label_tag = f"{normalized_cls.upper()} {conf:.2f} [{timestamp_str}]"
                            cv2.putText(evidence_frame, label_tag, (x1, max(20, y1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.putText(evidence_frame, f"CAM: {camera_name} | {timestamp_str}", (15, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                            evidence_filename = f"{mins:02d}_{secs:02d}_{normalized_cls}_{track_id or 'x'}.jpg"
                            evidence_file_path = os.path.join(cam_dir, evidence_filename)
                            cv2.imwrite(evidence_file_path, evidence_frame)

                            # Dominant color extraction on the cropped object
                            crop_obj = frame[y1:y2, x1:x2]
                            dominant_color = extract_dominant_color(crop_obj)

                            # Multimodal feature embedding (CLIP)
                            embedding_bytes = self.compute_image_embedding(crop_obj if crop_obj.size > 0 else frame)

                            # Save to Database
                            det_id = insert_detection(
                                camera=camera_name,
                                timestamp_sec=timestamp_sec,
                                timestamp_str=timestamp_str,
                                object_class=normalized_cls,
                                confidence=conf,
                                bbox=[x1, y1, x2, y2],
                                evidence_path=evidence_file_path,
                                color_hint=dominant_color,
                                embedding=embedding_bytes,
                                track_id=track_id,
                                db_path=self.db_path
                            )

                            detections_collected.append({
                                "id": det_id,
                                "camera": camera_name,
                                "timestamp_str": timestamp_str,
                                "object": normalized_cls,
                                "confidence": conf,
                                "color": dominant_color,
                                "evidence": evidence_file_path
                            })

            current_frame_idx += 1

        cap.release()
        print(f"Finished processing '{camera_name}'. Total detections logged: {len(detections_collected)}")
        return detections_collected
