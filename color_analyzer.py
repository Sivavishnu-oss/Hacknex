import numpy as np
from typing import List, Tuple, Dict, Any
from PIL import Image

# Supported colors mapped to HSV range conditions
COLOR_RANGES = {
    "red": [
        ((0, 70, 50), (10, 255, 255)),
        ((170, 70, 50), (180, 255, 255))
    ],
    "blue": [
        ((100, 70, 50), (135, 255, 255))
    ],
    "green": [
        ((35, 70, 50), (85, 255, 255))
    ],
    "yellow": [
        ((20, 70, 70), (35, 255, 255))
    ],
    "orange": [
        ((10, 100, 100), (20, 255, 255))
    ],
    "white": [
        ((0, 0, 190), (180, 40, 255))
    ],
    "black": [
        ((0, 0, 0), (180, 255, 50))
    ],
    "silver": [
        ((0, 0, 100), (180, 30, 200))
    ]
}

def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Converts RGB (0-255) to OpenCV HSV scale: H in [0, 180], S in [0, 255], V in [0, 255]."""
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    cmax = max(r_norm, g_norm, b_norm)
    cmin = min(r_norm, g_norm, b_norm)
    delta = cmax - cmin

    # Hue
    if delta == 0:
        h = 0
    elif cmax == r_norm:
        h = 60 * (((g_norm - b_norm) / delta) % 6)
    elif cmax == g_norm:
        h = 60 * (((b_norm - r_norm) / delta) + 2)
    else:
        h = 60 * (((r_norm - g_norm) / delta) + 4)
    h_cv = (h / 2.0) % 180

    # Saturation
    s_cv = 0 if cmax == 0 else (delta / cmax) * 255

    # Value
    v_cv = cmax * 255

    return (h_cv, s_cv, v_cv)

def extract_dominant_color(img_crop: np.ndarray) -> str:
    """
    Extracts dominant color from an image crop (numpy array RGB or BGR).
    Returns color name like 'red', 'white', 'black', 'blue', etc.
    """
    if img_crop is None or img_crop.size == 0:
        return "unknown"

    # Sample central region of the bounding box to ignore background edges
    h, w = img_crop.shape[:2]
    if h > 8 and w > 8:
        crop_center = img_crop[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    else:
        crop_center = img_crop

    pixels = crop_center.reshape(-1, 3)
    if len(pixels) > 500:
        indices = np.random.choice(len(pixels), 500, replace=False)
        pixels = pixels[indices]

    color_scores = {c: 0 for c in COLOR_RANGES}

    for p in pixels:
        # Assuming RGB input
        h_val, s_val, v_val = rgb_to_hsv(int(p[0]), int(p[1]), int(p[2]))
        for color_name, ranges in COLOR_RANGES.items():
            matched = False
            for (low, high) in ranges:
                if low[0] <= h_val <= high[0] and low[1] <= s_val <= high[1] and low[2] <= v_val <= high[2]:
                    matched = True
                    break
            if matched:
                color_scores[color_name] += 1

    best_color = max(color_scores, key=color_scores.get)
    if color_scores[best_color] > 0.15 * len(pixels):
        return best_color
    return "gray"
