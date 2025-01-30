import cv2
import os

def split_video_by_time(input_folder, output_folder, split_duration=2):
    # 入力フォルダ内の日付フォルダを走査
    for date_folder in os.listdir(input_folder):
        date_path = os.path.join(input_folder, date_folder)
        print("AAA:", date_path)
        if os.path.isdir(date_path):
            # 日付フォルダ内の各動画を処理
            for video_filename in os.listdir(date_path):
                if video_filename.endswith('.mp4'):
                    print("BBB")
                    # 動画の名前と拡張子を分割
                    video_name = os.path.splitext(video_filename)[0]
                    video_path = os.path.join(date_path, video_filename)

                    # 出力フォルダの作成
                    output_video_folder = os.path.join(output_folder, date_folder, video_name)
                    if not os.path.exists(output_video_folder):
                        os.makedirs(output_video_folder)

                    # 動画を開く
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_interval = int(fps * split_duration)  # 2秒ごとのフレーム数
                    count = 0
                    part = 0

                    print(f"Processing {video_filename} in {date_folder}")

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # 動画を2秒ごとに分割して保存
                        if count % frame_interval == 0:
                            if part > 0:
                                out.release()  # 前の動画を閉じる

                            output_filename = f"{date_folder}_{video_name}_{part}.mp4"
                            output_path = os.path.join(output_video_folder, output_filename)
                            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))
                            part += 1

                        out.write(frame)
                        count += 1

                    # 最後の動画を閉じる
                    out.release()
                    cap.release()

# 使い方
input_folder     = '/workspace/data/analysis'  # 入力動画が保存されているフォルダ
output_folder    = '/workspace/data/analysis'  # 分割された動画を保存するフォルダ
split_video_by_time(input_folder, output_folder)
