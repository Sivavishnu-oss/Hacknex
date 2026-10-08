import sqlite3
import json
import os
from typing import List, Dict, Any, Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "database", "cctv_intelligence.db")

def init_db(db_path: str = DEFAULT_DB_PATH):
    """Initializes SQLite database with necessary tables and indexes."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Detections table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera TEXT NOT NULL,
            timestamp_sec REAL NOT NULL,
            timestamp_str TEXT NOT NULL,
            object_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            bbox_json TEXT NOT NULL,
            color_hint TEXT,
            evidence_path TEXT NOT NULL,
            embedding_blob BLOB,
            track_id INTEGER
        )
    """)

    # Camera metadata
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cameras (
            camera_id TEXT PRIMARY KEY,
            camera_name TEXT NOT NULL,
            video_path TEXT NOT NULL,
            fps REAL,
            duration_sec REAL,
            resolution TEXT
        )
    """)

    # Indexes for fast lookup
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_det_camera ON detections(camera)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_det_object ON detections(object_class)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_det_time ON detections(timestamp_sec)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_det_track ON detections(track_id)")

    conn.commit()
    conn.close()

def save_camera(camera_id: str, camera_name: str, video_path: str, fps: float, duration_sec: float, resolution: str, db_path: str = DEFAULT_DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO cameras (camera_id, camera_name, video_path, fps, duration_sec, resolution)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (camera_id, camera_name, video_path, fps, duration_sec, resolution))
    conn.commit()
    conn.close()

def insert_detection(
    camera: str,
    timestamp_sec: float,
    timestamp_str: str,
    object_class: str,
    confidence: float,
    bbox: List[int],
    evidence_path: str,
    color_hint: Optional[str] = None,
    embedding: Optional[bytes] = None,
    track_id: Optional[int] = None,
    db_path: str = DEFAULT_DB_PATH
) -> int:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO detections (
            camera, timestamp_sec, timestamp_str, object_class, 
            confidence, bbox_json, color_hint, evidence_path, 
            embedding_blob, track_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        camera,
        timestamp_sec,
        timestamp_str,
        object_class.lower(),
        float(confidence),
        json.dumps(bbox),
        color_hint,
        evidence_path,
        embedding,
        track_id
    ))
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id

def insert_detections_batch(detections: List[Dict[str, Any]], db_path: str = DEFAULT_DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    records = [
        (
            d["camera"],
            d["timestamp_sec"],
            d["timestamp_str"],
            d["object_class"].lower(),
            float(d["confidence"]),
            json.dumps(d.get("bbox", [])),
            d.get("color_hint"),
            d["evidence_path"],
            d.get("embedding"),
            d.get("track_id")
        )
        for d in detections
    ]
    cursor.executemany("""
        INSERT INTO detections (
            camera, timestamp_sec, timestamp_str, object_class, 
            confidence, bbox_json, color_hint, evidence_path, 
            embedding_blob, track_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)
    conn.commit()
    conn.close()

def query_detections(
    camera: Optional[str] = None,
    object_class: Optional[str] = None,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    min_confidence: float = 0.25,
    limit: int = 100,
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    conditions = ["confidence >= ?"]
    params = [min_confidence]

    if camera and camera.lower() != "all":
        conditions.append("LOWER(camera) = ?")
        params.append(camera.lower())

    if object_class and object_class.lower() != "all":
        conditions.append("LOWER(object_class) = ?")
        params.append(object_class.lower())

    if start_sec is not None:
        conditions.append("timestamp_sec >= ?")
        params.append(start_sec)

    if end_sec is not None:
        conditions.append("timestamp_sec <= ?")
        params.append(end_sec)

    sql = f"""
        SELECT id, camera, timestamp_sec, timestamp_str, object_class, 
               confidence, bbox_json, color_hint, evidence_path, track_id
        FROM detections
        WHERE {" AND ".join(conditions)}
        ORDER BY timestamp_sec ASC, confidence DESC
        LIMIT ?
    """
    params.append(limit)

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    results = []
    for row in rows:
        item = dict(row)
        item["bbox"] = json.loads(item["bbox_json"]) if item["bbox_json"] else []
        results.append(item)
    conn.close()
    return results

def get_all_embeddings(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, camera, timestamp_sec, timestamp_str, object_class, 
               confidence, evidence_path, color_hint, embedding_blob, track_id
        FROM detections
        WHERE embedding_blob IS NOT NULL
    """)
    rows = cursor.fetchall()
    results = [dict(r) for r in rows]
    conn.close()
    return results

def get_unique_cameras(db_path: str = DEFAULT_DB_PATH) -> List[str]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT camera FROM detections ORDER BY camera")
    cameras = [row[0] for row in cursor.fetchall()]
    conn.close()
    return cameras

def delete_camera_records(camera_name: str, db_path: str = DEFAULT_DB_PATH):
    """Deletes existing detections and camera metadata for a given camera / video name."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detections WHERE LOWER(camera) = ?", (camera_name.lower(),))
    cursor.execute("DELETE FROM cameras WHERE LOWER(camera_id) = ? OR LOWER(camera_name) = ?", (camera_name.lower(), camera_name.lower()))
    conn.commit()
    conn.close()

def get_detections_for_camera(camera_name: str, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Returns all detections recorded for a specific camera/video."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, camera, timestamp_sec, timestamp_str, object_class,
               confidence, bbox_json, color_hint, evidence_path, track_id
        FROM detections
        WHERE LOWER(camera) = ?
        ORDER BY timestamp_sec ASC, confidence DESC
    """, (camera_name.lower(),))
    rows = cursor.fetchall()
    results = []
    for r in rows:
        item = dict(r)
        item["bbox"] = json.loads(item["bbox_json"]) if item["bbox_json"] else []
        results.append(item)
    conn.close()
    return results

def get_unique_objects(db_path: str = DEFAULT_DB_PATH) -> List[str]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT object_class FROM detections ORDER BY object_class")
    objs = [row[0] for row in cursor.fetchall()]
    conn.close()
    return objs

def get_all_cameras_meta(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cameras ORDER BY camera_id")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

if __name__ == "__main__":
    init_db()
    print("Database schema successfully initialized.")

