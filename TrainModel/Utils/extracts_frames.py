import cv2
import os
from pathlib import Path

def extract_frames_from_video(video_path, output_dir, step=5, max_frames=None):
    video_path = Path(video_path)
    output_subdir = Path(output_dir)
    output_subdir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_id = 0
    saved_id = 0
    saved_paths = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % step == 0:
            filename = f"{saved_id:05d}.jpg"
            output_path = output_subdir / filename
            cv2.imwrite(str(output_path), frame)
            saved_paths.append(str(output_path))
            saved_id += 1
            if max_frames is not None and saved_id >= max_frames:
                break

        frame_id += 1

    cap.release()
    return saved_paths


