import os
import cv2
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Any, Callable, Optional

class ZeroShotVideoSearcher:
    """
    Direct zero-shot video semantic search engine powered by OpenCLIP (ViT-B-32).
    Allows indexing ANY uploaded video with dense frame sampling, and searching for
    ANY open-vocabulary prompt (e.g. 'a red sports car', 'person opening a door',
    'dog running on grass', 'someone looking at camera').
    """
    def __init__(self, model_name: str = 'ViT-B-32', pretrained: str = 'laion2b_s34b_b79k'):
        import open_clip
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading ZeroShot CLIP ({model_name}) on device: {self.device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        print("ZeroShot CLIP ready.")

    def extract_and_embed_video(
        self,
        video_path: str,
        output_frames_dir: str,
        sample_fps: float = 1.0,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Samples frames from video (e.g. 1 frame every second), saves keyframe images,
        and computes normalized CLIP embedding vectors for every frame.
        """
        os.makedirs(output_frames_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / video_fps

        step = max(1, int(video_fps / sample_fps))
        frame_idx = 0
        indexed_items = []

        batch_frames_tensors = []
        batch_metadata = []

        def flush_batch():
            if not batch_frames_tensors:
                return
            tensors = torch.stack(batch_frames_tensors).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(tensors)
                features /= features.norm(dim=-1, keepdim=True)
                features_np = features.cpu().numpy().astype(np.float32)

            for i, meta in enumerate(batch_metadata):
                meta["embedding"] = features_np[i]
                indexed_items.append(meta)

            batch_frames_tensors.clear()
            batch_metadata.clear()

        batch_size = 16

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                t_sec = frame_idx / video_fps
                mins = int(t_sec // 60)
                secs = int(t_sec % 60)
                time_str = f"{mins:02d}:{secs:02d}"

                frame_name = f"frame_{int(t_sec*100):06d}_{mins:02d}_{secs:02d}.jpg"
                frame_path = os.path.join(output_frames_dir, frame_name)

                # Save evidence image to disk
                cv2.imwrite(frame_path, frame)

                # Prepare tensor for CLIP
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                img_tensor = self.preprocess(pil_img)

                batch_frames_tensors.append(img_tensor)
                batch_metadata.append({
                    "timestamp_sec": round(t_sec, 2),
                    "timestamp_str": time_str,
                    "frame_path": frame_path,
                    "frame_index": frame_idx
                })

                if len(batch_frames_tensors) >= batch_size:
                    flush_batch()

                if progress_callback and duration_sec > 0:
                    progress_callback(min(1.0, t_sec / duration_sec), f"Analyzing video at {time_str}...")

            frame_idx += 1

        flush_batch()
        cap.release()
        return indexed_items

    def search(
        self,
        query: str,
        indexed_items: List[Dict[str, Any]],
        top_k: int = 8,
        min_threshold: float = 0.18
    ) -> List[Dict[str, Any]]:
        """
        Takes open-ended natural language query, encodes it into the CLIP embedding space,
        and returns best matching video timestamps and frames ordered by similarity score.
        """
        if not indexed_items or not query.strip():
            return []

        text_tokens = self.tokenizer([query.strip()]).to(self.device)
        with torch.no_grad():
            text_feat = self.model.encode_text(text_tokens)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
            text_vec = text_feat.cpu().numpy().astype(np.float32).flatten()

        scored_results = []
        for item in indexed_items:
            emb = item.get("embedding")
            if emb is not None:
                sim = float(np.dot(text_vec, emb))
                # Only keep candidates above minimal similarity
                if sim >= min_threshold:
                    scored_results.append({
                        "timestamp_sec": item["timestamp_sec"],
                        "timestamp_str": item["timestamp_str"],
                        "frame_path": item["frame_path"],
                        "similarity": round(sim, 4),
                        "confidence_percent": int(max(0.0, min(1.0, (sim - 0.15) / 0.25)) * 100)
                    })

        # Sort descending by similarity
        scored_results.sort(key=lambda x: x["similarity"], reverse=True)

        # Deduplicate results that occur within 1.5 seconds of each other
        deduped = []
        for r in scored_results:
            is_close = any(abs(r["timestamp_sec"] - ex["timestamp_sec"]) <= 1.5 for ex in deduped)
            if not is_close:
                deduped.append(r)
            if len(deduped) >= top_k:
                break

        return deduped
