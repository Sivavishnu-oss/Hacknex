import os
import cv2
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Any, Optional, Callable
from color_analyzer import extract_dominant_color

class VideoIntelligenceEngine:
    """
    Combined YOLO26 + OpenCLIP Engine for uploaded videos:
    1. Runs YOLO26 for precise object localization, bounding boxes, labels, and tracking.
    2. Runs OpenCLIP (ViT-B-32) for open-vocabulary zero-shot understanding (e.g. actions, clothes, colors).
    3. Analyzes dominant colors of entities.
    4. Enables search: type what you want in natural language -> retrieves matching photo evidence and precise timing.
    """
    def __init__(self, yolo_model: str = "yolo26n.pt", clip_model: str = "ViT-B-32"):
        from ultralytics import YOLO
        import open_clip

        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading YOLO26 ({yolo_model})...")
        self.yolo = YOLO(yolo_model)

        print(f"Loading OpenCLIP ({clip_model}) on {self.device}...")
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(clip_model, pretrained="laion2b_s34b_b79k")
        self.clip = self.clip.to(self.device)
        self.clip.eval()
        self.tokenizer = open_clip.get_tokenizer(clip_model)
        print("Engine ready with YOLO26 + OpenCLIP!")

    def process_and_index_video(
        self,
        video_path: str,
        output_dir: str,
        sample_interval_sec: float = 0.5,
        conf_threshold: float = 0.25,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        camera_name: Optional[str] = None,
        db_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Processes any video uploaded by the user:
        - Samples frames at sample_interval_sec
        - Runs YOLO26 detection on frames
        - Saves annotated evidence images
        - Computes multimodal CLIP embeddings for both objects and whole scenes
        - Stores detections and video metadata into SQLite database
        - Returns structured index of detections and visual keyframes
        """
        import database

        if camera_name is None:
            camera_name = os.path.splitext(os.path.basename(video_path))[0]

        actual_db_path = db_path or database.DEFAULT_DB_PATH
        database.init_db(actual_db_path)

        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps
        frame_step = max(1, int(fps * sample_interval_sec))

        # Save or update camera metadata
        database.delete_camera_records(camera_name, db_path=actual_db_path)
        database.save_camera(
            camera_id=camera_name.lower().replace(" ", "_"),
            camera_name=camera_name,
            video_path=video_path,
            fps=float(fps),
            duration_sec=float(duration_sec),
            resolution=f"{width}x{height}",
            db_path=actual_db_path
        )

        catalog = []
        frame_idx = 0

        # Batch tensors for efficient CLIP encoding
        pending_crops = []
        pending_items = []

        def flush_crops():
            if not pending_crops:
                return
            tensors = torch.stack(pending_crops).to(self.device)
            with torch.no_grad():
                feats = self.clip.encode_image(tensors)
                feats /= feats.norm(dim=-1, keepdim=True)
                feats_np = feats.cpu().numpy().astype(np.float32)

            db_batch = []
            for i, meta in enumerate(pending_items):
                meta["embedding"] = feats_np[i]
                catalog.append(meta)

                # Store YOLO detections into database
                if meta.get("type") == "detection":
                    db_batch.append({
                        "camera": camera_name,
                        "timestamp_sec": meta["timestamp_sec"],
                        "timestamp_str": meta["timestamp_str"],
                        "object_class": meta["object"],
                        "confidence": meta["confidence"],
                        "bbox": meta.get("bbox", []),
                        "color_hint": meta.get("color"),
                        "evidence_path": meta["evidence_path"],
                        "embedding": feats_np[i].tobytes(),
                        "track_id": None
                    })

            if db_batch:
                database.insert_detections_batch(db_batch, db_path=actual_db_path)

            pending_crops.clear()
            pending_items.clear()

        batch_size = 16

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                t_sec = frame_idx / fps
                mins = int(t_sec // 60)
                secs = int(t_sec % 60)
                time_str = f"{mins:02d}:{secs:02d}"

                # 1. Run YOLO26 Detection
                results = self.yolo(frame, conf=conf_threshold, verbose=False)
                has_yolo_detections = False

                for r in results:
                    boxes = r.boxes
                    if len(boxes) > 0:
                        has_yolo_detections = True

                    for b_idx, box in enumerate(boxes):
                        cls_id = int(box.cls[0])
                        cls_name = self.yolo.names[cls_id]
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                        x1, y1, x2, y2 = max(0, xyxy[0]), max(0, xyxy[1]), min(frame.shape[1], xyxy[2]), min(frame.shape[0], xyxy[3])

                        # Make annotated evidence photo
                        annotated = frame.copy()
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 60), 2)
                        label = f"{cls_name.upper()} {conf:.2f} [{time_str}]"
                        cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 60), 2)
                        cv2.putText(annotated, f"TIME: {time_str}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                        ev_filename = f"ev_{int(t_sec*100):06d}_{mins:02d}_{secs:02d}_{cls_name}_{b_idx}.jpg"
                        ev_path = os.path.join(output_dir, ev_filename)
                        cv2.imwrite(ev_path, annotated)

                        # Color extraction on object
                        crop_bgr = frame[y1:y2, x1:x2]
                        color_hint = extract_dominant_color(crop_bgr) if crop_bgr.size > 0 else "unknown"

                        # Crop for CLIP encoding
                        target_for_clip = crop_bgr if (crop_bgr.shape[0] > 15 and crop_bgr.shape[1] > 15) else frame
                        crop_rgb = cv2.cvtColor(target_for_clip, cv2.COLOR_BGR2RGB)
                        pil_crop = Image.fromarray(crop_rgb)
                        tensor = self.preprocess(pil_crop)

                        pending_crops.append(tensor)
                        pending_items.append({
                            "timestamp_sec": round(t_sec, 2),
                            "timestamp_str": time_str,
                            "type": "detection",
                            "object": cls_name.lower(),
                            "confidence": round(conf, 3),
                            "color": color_hint,
                            "bbox": [x1, y1, x2, y2],
                            "evidence_path": ev_path
                        })

                        if len(pending_crops) >= batch_size:
                            flush_crops()

                # Also record whole-frame scene keyframe so search matches broader scene descriptions
                # (e.g., "red car on the road", "empty room", "people standing together")
                full_ev_filename = f"scene_{int(t_sec*100):06d}_{mins:02d}_{secs:02d}.jpg"
                full_ev_path = os.path.join(output_dir, full_ev_filename)
                
                # Annotate timestamp on full scene
                scene_frame = frame.copy()
                cv2.putText(scene_frame, f"TIMESTAMP: {time_str}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imwrite(full_ev_path, scene_frame)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                pending_crops.append(self.preprocess(pil_frame))
                pending_items.append({
                    "timestamp_sec": round(t_sec, 2),
                    "timestamp_str": time_str,
                    "type": "scene",
                    "object": "scene",
                    "confidence": 0.5,
                    "color": "",
                    "bbox": [0, 0, frame.shape[1], frame.shape[0]],
                    "evidence_path": full_ev_path
                })

                if len(pending_crops) >= batch_size:
                    flush_crops()

                if progress_callback and duration_sec > 0:
                    pct = min(1.0, t_sec / duration_sec)
                    progress_callback(pct, f"Processing with YOLO26 & CLIP at {time_str}...")

            frame_idx += 1

        flush_crops()
        cap.release()
        return catalog

    def search_video(
        self,
        user_query: str,
        catalog: List[Dict[str, Any]],
        top_k: int = 12
    ) -> List[Dict[str, Any]]:
        """
        Finds what the user asked for in the indexed video:
        - Calculates semantic match score using OpenCLIP text-image embedding
        - Boosts matches using YOLO26 detected category & attributes
        - Groups and deduplicates timestamps
        - Returns ranked photos, timing, and explanations
        """
        if not catalog or not user_query.strip():
            return []

        q = user_query.lower().strip()

        # Compute query vector
        text_tokens = self.tokenizer([user_query]).to(self.device)
        with torch.no_grad():
            text_feat = self.clip.encode_text(text_tokens)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
            q_vec = text_feat.cpu().numpy().astype(np.float32).flatten()

        scored = []
        for item in catalog:
            emb = item.get("embedding")
            if emb is None:
                continue

            # Base semantic similarity from CLIP (dot product of unit vectors)
            sim = float(np.dot(q_vec, emb))

            # Bonus for exact YOLO26 object label presence in user query
            obj_name = item.get("object", "")
            bonus = 0.0
            if obj_name != "scene" and obj_name in q:
                bonus += 0.22

            # Bonus for color mention matching
            color = item.get("color", "")
            if color and color != "unknown" and color in q:
                bonus += 0.15

            total_score = sim + bonus

            scored.append({
                "timestamp_sec": item["timestamp_sec"],
                "timestamp_str": item["timestamp_str"],
                "object": obj_name,
                "confidence": item["confidence"],
                "color": item.get("color", ""),
                "evidence_path": item["evidence_path"],
                "match_score": round(total_score, 3),
                "clip_sim": round(sim, 3),
                "type": item["type"]
            })

        # Sort descending by match score
        scored.sort(key=lambda x: x["match_score"], reverse=True)

        # Cluster/deduplicate results within +/- 1.5 seconds so user gets distinct moments
        deduped = []
        for candidate in scored:
            too_close = any(abs(candidate["timestamp_sec"] - chosen["timestamp_sec"]) <= 1.2 for chosen in deduped)
            if not too_close:
                deduped.append(candidate)
            if len(deduped) >= top_k:
                break

        return deduped
