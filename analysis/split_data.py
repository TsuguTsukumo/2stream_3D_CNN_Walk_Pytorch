import os
import subprocess
from pathlib import Path
import cv2

input_root = "/workspace/data/Cross_Validation/ex_20250116_lat_organized"
output_root = "/workspace/data/analysis/ex_20250116_lat_organaized_split"

def get_video_duration(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = int(total_frames // fps)
    cap.release()
    return duration

def split_video_per_second(input_video_path, output_dir):
    filename = input_video_path.stem  
    duration = get_video_duration(input_video_path)

    if duration == 0:
        return

    os.makedirs(output_dir, exist_ok=True)
    for i in range(duration):
        output_filename = f"{filename}_{i+1}-{duration}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_video_path),
            "-ss", str(i),
            "-t", "1",
            "-c", "copy",
            output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def process_dataset(input_root, output_root):
    for dirpath, dirnames, filenames in os.walk(input_root):
        for file in filenames:
            if file.endswith(".mp4"):
                input_video_path = Path(os.path.join(dirpath, file))
                rel_path = input_video_path.parent.relative_to(input_root)
                output_dir = Path(output_root) / rel_path
                split_video_per_second(input_video_path, output_dir)

process_dataset(input_root, output_root)
