"""
Stage 0: Video Frame Extraction
Extracts frames at a given framerate from a raw video file for photogrammetry.
"""

import argparse
import cv2
from pathlib import Path


def extract_frames(video_path, save_dir, target_fps=8.0):
    video_path = Path(video_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found at: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = max(1, round(source_fps / target_fps))

    frame_idx = 0
    saved_idx = 0

    print(f"Extracting frames from {video_path.name} ({source_fps:.1f} fps, {total_frames} total frames)...")
    print(f"Target sampling rate: {target_fps} fps (every {frame_interval} frames)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_path = save_dir / f"frame_{saved_idx:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_idx += 1
        frame_idx += 1

    cap.release()
    print(f"Extracted {saved_idx} frames -> {save_dir}")
    return saved_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video for 3D reconstruction.")
    parser.add_argument("--video", default="data/raw_videos/cv-4.mp4", help="Path to input video file")
    parser.add_argument("--output", default="data/frames/cv-4", help="Output directory for extracted frames")
    parser.add_argument("--fps", type=float, default=2.0, help="Target FPS sampling rate (e.g. 6.0 - 10.0)")
    args = parser.parse_args()

    extract_frames(args.video, args.output, target_fps=args.fps)