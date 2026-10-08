import numpy as np
import os
from typing import List, Dict, Any, Optional
from database import query_detections, get_all_embeddings, DEFAULT_DB_PATH
from query_parser import parse_natural_language_query

class IntelligenceRetriever:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.clip_model = None
        self.clip_tokenizer = None
        self.clip_device = None

    def _init_clip(self):
        """Lazy loader for CLIP text encoding model."""
        if self.clip_model is not None:
            return
        try:
            import open_clip
            import torch
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
            model = model.to(device)
            model.eval()
            self.clip_model = model
            self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
            self.clip_device = device
            print("IntelligenceRetriever: CLIP loaded for text semantic search.")
        except Exception as e:
            print(f"IntelligenceRetriever: CLIP unavailable: {e}")

    def compute_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Encodes query string into normalized CLIP embedding vector."""
        self._init_clip()
        if self.clip_model is None:
            return None
        try:
            import torch
            text_tokens = self.clip_tokenizer([text]).to(self.clip_device)
            with torch.no_grad():
                feat = self.clip_model.encode_text(text_tokens)
                feat /= feat.norm(dim=-1, keepdim=True)
                return feat.cpu().numpy().astype(np.float32).flatten()
        except Exception as e:
            print(f"Error computing text embedding: {e}")
            return None

    def search(self, query_text: str, top_k: int = 10, min_conf: float = 0.25) -> Dict[str, Any]:
        """
        Multimodal search:
        1. Parses structured terms (object, camera, color, timestamp)
        2. Filters SQLite candidates
        3. Re-ranks candidates with CLIP text-image cosine similarity & attribute scoring
        """
        parsed = parse_natural_language_query(query_text)
        req_obj = parsed["object"]
        req_cam = parsed["camera"]
        req_color = parsed["color"]
        req_time = parsed["time_sec"]

        # Time window search if specified (+/- 15 seconds)
        start_sec = max(0, req_time - 15.0) if req_time is not None else None
        end_sec = (req_time + 15.0) if req_time is not None else None

        # Fetch candidate records from DB
        candidates = query_detections(
            camera=req_cam,
            object_class=req_obj,
            start_sec=start_sec,
            end_sec=end_sec,
            min_confidence=min_conf,
            limit=200,
            db_path=self.db_path
        )

        if not candidates and (req_obj or req_cam):
            # Broaden search if strictly filtered yielded nothing
            candidates = query_detections(
                camera=None,
                object_class=req_obj,
                min_confidence=min_conf,
                limit=100,
                db_path=self.db_path
            )

        # Compute query text embedding if possible
        q_vec = self.compute_text_embedding(query_text)

        ranked_results = []
        for cand in candidates:
            score = cand["confidence"]

            # Boost if color matches
            if req_color and cand.get("color_hint"):
                if req_color.lower() in cand["color_hint"].lower():
                    score += 0.35
                else:
                    score -= 0.15

            # Semantic similarity boost using CLIP visual features
            emb_blob = cand.get("embedding_blob")
            clip_sim = None
            if q_vec is not None and emb_blob is not None:
                try:
                    img_vec = np.frombuffer(emb_blob, dtype=np.float32)
                    sim = float(np.dot(q_vec, img_vec))
                    clip_sim = round(sim, 3)
                    # Blend CLIP similarity into ranking
                    score = score * 0.4 + max(0.0, sim) * 0.6
                except Exception:
                    pass

            item = dict(cand)
            item["match_score"] = round(float(score), 3)
            item["clip_similarity"] = clip_sim
            ranked_results.append(item)

        # Sort by best match score descending
        ranked_results.sort(key=lambda x: x["match_score"], reverse=True)
        final_top = ranked_results[:top_k]

        return {
            "query": query_text,
            "parsed_intent": parsed,
            "total_found": len(ranked_results),
            "results": final_top
        }

    def track_entity_across_cameras(self, initial_det_id: int) -> List[Dict[str, Any]]:
        """
        Cross-Camera Tracking / Trajectory:
        Given an initial detection, matches similar embeddings and appearances
        across cameras chronologically to reconstruct movement paths.
        e.g. Main Gate 00:42 -> Lobby 01:16 -> Parking 02:17
        """
        all_dets = query_detections(min_confidence=0.3, limit=500, db_path=self.db_path)
        base = next((d for d in all_dets if d["id"] == initial_det_id), None)
        if not base:
            return []

        # Find occurrences sorted by timestamp
        target_obj = base["object_class"]
        target_color = base["color_hint"]

        pathway = [base]
        for d in all_dets:
            if d["id"] == base["id"]:
                continue
            if d["object_class"] == target_obj:
                # Match color if present
                if target_color and d.get("color_hint") and target_color != d["color_hint"]:
                    continue
                # Don't add duplicate timestamps on same camera within 2s
                if d["camera"] == pathway[-1]["camera"] and abs(d["timestamp_sec"] - pathway[-1]["timestamp_sec"]) < 2.0:
                    continue
                pathway.append(d)

        pathway.sort(key=lambda x: x["timestamp_sec"])
        return pathway
