import re
from typing import Dict, Any, Optional

# Supported known objects in standard CCTV / COCO domain
KNOWN_OBJECTS = {
    "car": ["car", "cars", "automobile", "automobiles", "vehicle", "vehicles", "sedan", "suv"],
    "person": ["person", "people", "man", "woman", "human", "guy", "individual", "pedestrian", "someone"],
    "truck": ["truck", "trucks", "pickup", "lorry"],
    "bus": ["bus", "buses"],
    "motorcycle": ["motorcycle", "motorcycles", "bike", "motorbike", "scooter"],
    "bicycle": ["bicycle", "bicycles", "cycle"],
    "bag": ["bag", "backpack", "luggage", "suitcase", "handbag", "purse"]
}

# Supported cameras and aliases
KNOWN_CAMERAS = {
    "Gate": ["gate", "main gate", "entrance", "entry", "front gate", "main entrance"],
    "Parking": ["parking", "parking lot", "garage", "car park", "parking area"],
    "Lobby": ["lobby", "reception", "waiting area", "hall", "corridor"]
}

# Common colors
KNOWN_COLORS = ["red", "blue", "green", "black", "white", "silver", "yellow", "gray", "grey", "orange", "dark"]

def parse_natural_language_query(query: str) -> Dict[str, Any]:
    """
    Parses natural language query into structured search terms:
    - target_object: detected category (e.g. car, person)
    - target_camera: camera location filter if specified (e.g. Gate, Parking, Lobby)
    - color: attribute like red, white, black
    - time_spec: approximate time in seconds if mentioned
    - raw_query: original string for semantic CLIP comparison
    """
    q = query.lower().strip()

    detected_object: Optional[str] = None
    for canonical, synonyms in KNOWN_OBJECTS.items():
        for syn in synonyms:
            # Word boundary match
            if re.search(r'\b' + re.escape(syn) + r'\b', q):
                detected_object = canonical
                break
        if detected_object:
            break

    detected_camera: Optional[str] = None
    for canonical, synonyms in KNOWN_CAMERAS.items():
        for syn in synonyms:
            if re.search(r'\b' + re.escape(syn) + r'\b', q):
                detected_camera = canonical
                break
        if detected_camera:
            break

    detected_color: Optional[str] = None
    for color in KNOWN_COLORS:
        if re.search(r'\b' + re.escape(color) + r'\b', q):
            detected_color = color
            break

    # Check for timestamps e.g. "at 00:42", "at 42 seconds", "1:15"
    time_sec: Optional[float] = None
    time_match = re.search(r'\b(\d{1,2}):(\d{2})\b', q)
    if time_match:
        mins, secs = int(time_match.group(1)), int(time_match.group(2))
        time_sec = float(mins * 60 + secs)
    else:
        sec_match = re.search(r'\b(\d+)\s*(?:sec|second|seconds|s)\b', q)
        if sec_match:
            time_sec = float(sec_match.group(1))

    return {
        "raw_query": query,
        "object": detected_object,
        "camera": detected_camera,
        "color": detected_color,
        "time_sec": time_sec
    }

if __name__ == "__main__":
    test_queries = [
        "Find the red car that entered the main gate.",
        "Was there a car in the parking area?",
        "Find a person in the lobby.",
        "Find cars near the gate at 00:42",
        "Show me a white truck"
    ]
    for tq in test_queries:
        print(f"'{tq}' -> {parse_natural_language_query(tq)}")
