import os
import random
import shutil
from moviepy.editor import VideoFileClip
from sklearn.model_selection import KFold

def get_video_metadata(folder_path):
    videos = []
    total = 0
    counter = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('full_ap.mp4', 'full_lat.mp4')):
                file_path = os.path.join(root, file)
                try:
                    with VideoFileClip(file_path) as video:
                        duration = video.duration
                        disease = os.path.basename(os.path.dirname(root))
                        date = os.path.basename(root)
                        videos.append({
                            "file_path": file_path,
                            "duration": duration,
                            "disease": disease,
                            "date": date,
                            "type": file  # full_ap or full_lat
                        })
                        total += video.duration
                        counter += 1
                        print("number:", counter)
                        print("duration:", video.duration)
                        print("total:", total)
                except Exception as e:
                    print(f"Could not process {file_path}: {e}")
    return videos

def save_split_videos(videos, output_folder, n_splits=5):
    random.shuffle(videos)  # シャッフルで偏り防止
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(videos)):
        train_videos = [videos[i] for i in train_idx]
        val_videos = [videos[i] for i in val_idx]
        
        fold_output_path = os.path.join(output_folder, f'fold{fold}')
        train_output_path = os.path.join(fold_output_path, 'train')
        val_output_path = os.path.join(fold_output_path, 'val')
        
        for path in [train_output_path, val_output_path]:
            os.makedirs(path, exist_ok=True)
        
        print(f"\nSaving Fold {fold}:")
        save_videos(train_videos, train_output_path, "Train")
        save_videos(val_videos, val_output_path, "Validation")

def save_videos(video_list, destination, set_type):
    total_duration = 0
    for video in video_list:
        total_duration += video['duration']
        dest_folder = os.path.join(destination, video['disease'], video['date'])
        os.makedirs(dest_folder, exist_ok=True)
        shutil.copy(video['file_path'], dest_folder)
    print(f"{set_type} set: {len(video_list)} videos, Total duration: {total_duration:.2f} seconds")

# フォルダパスを指定
input_folder_path = "/workspace/data/data/new_data_selected_cropped copy/ASD_not/serch"
output_folder_path = "/path/to/output/folder"

get_video_metadata(input_folder_path)

