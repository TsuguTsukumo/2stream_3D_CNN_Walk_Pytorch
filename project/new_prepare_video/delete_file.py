import os
import cv2
from pathlib import Path
import shutil

def process_videos(input_dir, output_dir):
    # 入力フォルダをスキャン
    for date_folder in os.listdir(input_dir):
        date_path = os.path.join(input_dir, date_folder)
        print(date_path)
        if not os.path.isdir(date_path):
            continue

        output_date_path = os.path.join(output_dir, date_folder)
        os.makedirs(output_date_path, exist_ok=True)

        segments = sorted([seg for seg in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, seg))])
        output_segment_index = 0  # 出力用の連番インデックス

        for segment_folder in segments:
            segment_path = os.path.join(date_path, segment_folder)
            video_files = sorted([f for f in os.listdir(segment_path) if f.endswith(".mp4")])

            if len(video_files) != 2:
                print(f"動画が2つ揃っていないセグメントをスキップ: {segment_path}")
                continue

            # 最初の動画のフレーム数を取得
            video_path = os.path.join(segment_path, video_files[0])
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # フレーム数が50以下の場合はスキップ
            if frame_count <= 50:
                print(f"フレーム数が50以下のためセグメントをスキップ: {segment_path}")
                continue

            # 出力用のセグメントフォルダを作成
            output_segment_index += 1
            output_segment_folder = f"segment{output_segment_index}"
            output_segment_path = os.path.join(output_date_path, output_segment_folder)
            os.makedirs(output_segment_path, exist_ok=True)

            # 動画を処理して出力フォルダに保存
            for video_file in video_files:
                input_video_path = os.path.join(segment_path, video_file)
                output_video_path = os.path.join(output_segment_path, video_file)

                cap = cv2.VideoCapture(input_video_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

                frame_index = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 後ろから数フレームをスキップ
                    if frame_index < frame_count - 10:
                        out.write(frame)
                    frame_index += 1

                cap.release()
                out.release()

            print(f"セグメントを処理しました: {segment_path} -> {output_segment_path}")

# 使用例
input_dir = "/workspace/data/TEST_Normal_9"  # 元の動画フォルダ
output_dir = "/workspace/data/TEST_Normal_9_selected"  # 処理後の保存先
process_videos(input_dir, output_dir)