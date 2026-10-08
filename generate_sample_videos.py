import cv2
import numpy as np
import os

def create_synthetic_cctv_video(output_path: str, location: str, duration_sec: int = 15, fps: int = 25):
    """
    Generates a realistic CCTV camera video with timecode overlay, camera branding,
    and distinct simulated objects (red car, white car, pedestrians, backpacks)
    to guarantee immediate, end-to-end testing of detection, indexing, and cross-camera tracking.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 640, 480
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Generating video for camera '{location}' -> {output_path} ({duration_sec}s)...")

    for frame_idx in range(total_frames):
        # Create CCTV background
        t = frame_idx / fps
        mins = int(t // 60)
        secs = int(t % 60)
        time_str = f"{mins:02d}:{secs:02d}"

        # Base scene rendering
        img = np.zeros((height, width, 3), dtype=np.uint8)

        if location == "Gate":
            # Roadway & Gate structure
            img[:] = (60, 60, 60) # Asphalt road
            cv2.rectangle(img, (0, 0), (width, 140), (120, 100, 80), -1) # Sky/barrier
            cv2.line(img, (0, 300), (width, 300), (255, 255, 255), 2) # Lane mark
            cv2.putText(img, "MAIN GATE ENTRANCE", (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

            # Red car enters from left to right between 00:02 and 00:08 (peaks at 00:04 ~ 00:06)
            if 1.0 <= t <= 9.0:
                progress = (t - 1.0) / 8.0
                car_x = int(progress * (width + 160) - 80)
                car_y = 280
                # Red car body (BGR: Blue=30, Green=30, Red=220)
                cv2.rectangle(img, (car_x, car_y), (car_x + 130, car_y + 55), (30, 30, 220), -1)
                # Roof
                cv2.rectangle(img, (car_x + 25, car_y - 25), (car_x + 95, car_y), (30, 30, 180), -1)
                # Windows
                cv2.rectangle(img, (car_x + 35, car_y - 20), (car_x + 85, car_y - 2), (200, 220, 240), -1)
                # Wheels
                cv2.circle(img, (car_x + 25, car_y + 55), 14, (20, 20, 20), -1)
                cv2.circle(img, (car_x + 105, car_y + 55), 14, (20, 20, 20), -1)

            # Pedestrian walks by around 00:07 - 00:13
            if 6.0 <= t <= 14.0:
                p_progress = (t - 6.0) / 8.0
                px = int(width - p_progress * (width + 60))
                py = 180
                # Head
                cv2.circle(img, (px + 15, py), 12, (200, 180, 150), -1)
                # Torso (blue shirt)
                cv2.rectangle(img, (px, py + 12), (px + 30, py + 65), (180, 50, 40), -1)
                # Legs
                cv2.line(img, (px + 8, py + 65), (px + 5, py + 110), (40, 40, 40), 4)
                cv2.line(img, (px + 22, py + 65), (px + 25, py + 110), (40, 40, 40), 4)

        elif location == "Parking":
            # Parking lot bays
            img[:] = (70, 70, 75)
            for bay_x in [100, 250, 400, 550]:
                cv2.line(img, (bay_x, 150), (bay_x, 420), (255, 255, 255), 2)
            cv2.putText(img, "PARKING LOT ZONE B", (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

            # Parked White Car
            cv2.rectangle(img, (120, 220), (230, 360), (230, 230, 230), -1)
            cv2.rectangle(img, (135, 260), (215, 320), (80, 80, 80), -1)

            # The Red Car arrives at parking later (t=06s to t=14s)
            if 5.0 <= t <= 14.0:
                progress = (t - 5.0) / 9.0
                car_x = int(progress * 300 + 260)
                car_y = 220
                cv2.rectangle(img, (car_x, car_y), (car_x + 120, car_y + 130), (30, 30, 220), -1)
                cv2.rectangle(img, (car_x + 15, car_y + 30), (car_x + 105, car_y + 85), (60, 60, 60), -1)

        elif location == "Lobby":
            # Lobby interior floor & walls
            img[:] = (180, 170, 160)
            cv2.rectangle(img, (0, 0), (width, 160), (140, 130, 120), -1)
            cv2.rectangle(img, (180, 160), (460, 220), (100, 70, 50), -1) # Reception desk
            cv2.putText(img, "RECEPTION DESK", (240, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Person walking across lobby with bag (t=03s to 12s)
            if 2.0 <= t <= 13.0:
                progress = (t - 2.0) / 11.0
                px = int(progress * (width + 60) - 30)
                py = 220
                # Head
                cv2.circle(img, (px + 20, py), 15, (190, 170, 140), -1)
                # Torso (Dark green jacket)
                cv2.rectangle(img, (px, py + 15), (px + 40, py + 80), (40, 120, 50), -1)
                # Bag / backpack on person
                cv2.rectangle(img, (px - 15, py + 25), (px + 5, py + 65), (20, 20, 20), -1)
                # Legs
                cv2.line(img, (px + 10, py + 80), (px + 8, py + 140), (30, 30, 30), 5)
                cv2.line(img, (px + 30, py + 80), (px + 32, py + 140), (30, 30, 30), 5)

        # CCTV timestamp and camera OSD overlay
        osd_text = f"CAM: {location.upper()} | REC [{time_str}] | 25 FPS"
        cv2.rectangle(img, (10, 10), (380, 42), (0, 0, 0), -1)
        cv2.putText(img, osd_text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.circle(img, (width - 25, 25), 8, (0, 0, 255), -1) # Red recording dot

        out.write(img)

    out.release()
    print(f"Generated video for {location}: {output_path}")

if __name__ == "__main__":
    create_synthetic_cctv_video("videos/gate.mp4", "Gate", duration_sec=15)
    create_synthetic_cctv_video("videos/parking.mp4", "Parking", duration_sec=15)
    create_synthetic_cctv_video("videos/lobby.mp4", "Lobby", duration_sec=15)
