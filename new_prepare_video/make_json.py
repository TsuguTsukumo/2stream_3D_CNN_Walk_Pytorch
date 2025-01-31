import os
import cv2
import json

def get_video_duration_and_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # フレームレート（FPS）
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # フレーム数
    duration = frame_count / fps  # 動画の合計時間（秒）
    cap.release()
    return duration, fps

def generate_dataset_json(root_dir):
    dataset = {}

    # 疾患カテゴリのディレクトリを取得
    for disease in ["ASD", "Normal"]:
        disease_dir = os.path.join(root_dir, disease)
        disease_data = {}

        # 日付ごとに処理
        for date in os.listdir(disease_dir):
            date_dir = os.path.join(disease_dir, date)
            if os.path.isdir(date_dir):
                total_segment_number = 0
                total_time = 0
                fps = None

                # 各セグメントごとに処理
                for segment in os.listdir(date_dir):
                    if segment.endswith(".mp4"):  # mp4ファイルを全て対象に
                        total_segment_number += 1
                        video_path = os.path.join(date_dir, segment)
                        duration, segment_fps = get_video_duration_and_fps(video_path)
                        total_time += duration
                        # 最初の動画のFPSを記録
                        if fps is None:
                            fps = segment_fps

                disease_data[date] = {
                    "total_segment_number": total_segment_number,
                    "total_time": total_time,
                    "fps": fps
                }

        dataset[disease] = disease_data

    return dataset



# データセットのパスを指定
root_dir = "/workspace/data/Video/Segment_video_ASDandNormal_ap"
dataset_json = generate_dataset_json(root_dir)

# JSONをファイルに保存
with open("/workspace/data/Video/Segment_video_ASDandNormal/dataset_info.json", "w", encoding="utf-8") as f:
    json.dump(dataset_json, f, ensure_ascii=False, indent=4)
