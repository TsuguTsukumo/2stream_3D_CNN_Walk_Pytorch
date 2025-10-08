import cv2
import os
import glob
from ultralytics import YOLO
import numpy as np

# --- パス設定 ---
# 入力データが格納されているルートディレクトリ
INPUT_DIR = '/workspace/data/new_data_selected'

# 処理したいディレクトリ名のリスト
TARGET_DIRS = ['ASD', 'DHS', 'LCS', 'HipOA']

# 出力ディレクトリを指定
OUTPUT_DIR = '/workspace/data/cropped_direction'

# --- YOLOv8モデルのロード ---
model = YOLO('yolov8n.pt')

# --- 処理のための変数初期化 ---
monitoring_frames = 10
stable_walk_frames = 10
MIN_VIDEO_DURATION_SEC = 2
TARGET_SIZE = 256
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
walk_count = 0

# 出力ディレクトリが存在しない場合は作成
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 動画ペアの検索と処理 ---
print("入力ディレクトリを検索中です...")

video_pairs = []
for root, dirs, files in os.walk(INPUT_DIR):
    # ターゲットディレクトリがパスに含まれているかチェック
    # この部分を修正し、より正確なパスを抽出
    relative_path_parts = os.path.relpath(root, INPUT_DIR).split(os.sep)
    
    if any(target_dir in relative_path_parts for target_dir in TARGET_DIRS) and 'full_ap.mp4' in files and 'full_lat.mp4' in files:
        full_ap_path = os.path.join(root, 'full_ap.mp4')
        full_lat_path = os.path.join(root, 'full_lat.mp4')
        video_pairs.append((full_ap_path, full_lat_path, root))
            
print(f"合計 {len(video_pairs)} 組の動画ペアが見つかりました。")

if not video_pairs:
    print("エラー: 指定されたディレクトリで動画ペアが見つかりませんでした。")
    print("TARGET_DIRS リストが正しいか、およびディレクトリ構造が正しいことを確認してください。")
    exit()

for front_video_path, side_video_path, original_root_dir in video_pairs:
    print(f"\n--- 動画ペアの処理を開始: ---")
    print(f"側面動画: {side_video_path}")
    print(f"正面動画: {front_video_path}")

    # --- 処理のためのローカル変数初期化 ---
    side_center_x_history = []
    side_previous_direction = "Unknown"
    side_current_stable_direction = "Unknown"
    side_stable_direction_count = 0
    side_frame_buffer = []
    front_frame_buffer = []
    is_saving_section = False
    
    side_cap = cv2.VideoCapture(side_video_path)
    front_cap = cv2.VideoCapture(front_video_path)

    if not side_cap.isOpened() or not front_cap.isOpened():
        print("エラー: ファイルが開けませんでした。スキップします。")
        continue

    fps = side_cap.get(cv2.CAP_PROP_FPS)
    min_frames_to_save = int(fps * MIN_VIDEO_DURATION_SEC)
    side_width = int(side_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    side_height = int(side_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    front_width = int(front_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    front_height = int(front_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 0
    while side_cap.isOpened() and front_cap.isOpened():
        ret_side, side_frame = side_cap.read()
        ret_front, front_frame = front_cap.read()
        if not ret_side or not ret_front:
            break
        frame_count += 1
        
        # --- 側面動画の処理 ---
        side_results = model(side_frame, verbose=False)
        side_current_x_center = None
        side_person_detected = False
        side_person_crop = None
        for result in side_results:
            boxes = result.boxes
            if boxes:
                for box in boxes:
                    if int(box.cls) == 0 and box.conf > 0.4:
                        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                        h, w = y2 - y1, x2 - x1
                        if w > h or (h * w) < (side_height * side_width * 0.005):
                            continue
                        side_current_x_center = (x1 + x2) / 2
                        side_person_detected = True
                        person_roi = side_frame[y1:y2, x1:x2]
                        new_h = TARGET_SIZE
                        new_w = int(w * (new_h / h))
                        resized_person = cv2.resize(person_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        side_person_crop = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
                        x_offset = int((TARGET_SIZE - new_w) / 2)
                        side_person_crop[:, x_offset:x_offset+new_w] = resized_person
                        break
            if side_person_detected:
                break

        # --- 正面動画の処理 ---
        front_results = model(front_frame, verbose=False)
        front_person_detected = False
        front_person_crop = None
        min_x_pos = float('inf')
        leftmost_box = None
        for result in front_results:
            boxes = result.boxes
            if boxes:
                for box in boxes:
                    if int(box.cls) == 0 and box.conf > 0.4:
                        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                        if (x2 - x1) * (y2 - y1) > (front_height * front_width * 0.005):
                            if x1 < min_x_pos:
                                min_x_pos = x1
                                leftmost_box = box
        if leftmost_box:
            x1, y1, x2, y2 = leftmost_box.xyxy[0].int().tolist()
            h, w = y2 - y1, x2 - x1
            if w <= h:
                front_person_detected = True
                person_roi = front_frame[y1:y2, x1:x2]
                new_h = TARGET_SIZE
                new_w = int(w * (new_h / h))
                resized_person = cv2.resize(person_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
                front_person_crop = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
                x_offset = int((TARGET_SIZE - new_w) / 2)
                front_person_crop[:, x_offset:x_offset+new_w] = resized_person

        # --- 側面動画の方向判定 ---
        side_direction = "Unknown"
        if side_person_detected:
            side_center_x_history.append(side_current_x_center)
            if len(side_center_x_history) > monitoring_frames:
                side_center_x_history.pop(0)
            if len(side_center_x_history) >= 2:
                first_x = side_center_x_history[0]
                last_x = side_center_x_history[-1]
                if last_x < first_x - 5:
                    side_direction = "Left"
                elif last_x > first_x + 5:
                    side_direction = "Right"
                else:
                    side_direction = "Standing/No significant movement"
        
        # --- 安定した歩行と同期区間の検出 ---
        if side_direction == side_previous_direction:
            side_stable_direction_count += 1
        else:
            side_stable_direction_count = 0
        
        if side_stable_direction_count == stable_walk_frames and side_direction in ["Left", "Right"] and not is_saving_section:
            print(f"フレーム {frame_count}: 安定した {side_direction} 方向への歩行を検出しました。バッファリングを開始します。")
            is_saving_section = True
            side_current_stable_direction = side_direction

        if is_saving_section:
            end_of_section = (side_direction != side_current_stable_direction) or (not side_person_detected) or (not front_person_detected)
            if not end_of_section:
                if side_person_crop is not None and front_person_crop is not None:
                    side_frame_buffer.append(side_person_crop)
                    front_frame_buffer.append(front_person_crop)
            else:
                print(f"フレーム {frame_count}: 歩行区間の終わりを検出しました。長さは {len(side_frame_buffer)} フレームです。")
                if len(side_frame_buffer) >= min_frames_to_save:
                    walk_count += 1
                    save_dir_name = f"walk_{side_current_stable_direction}_{str(walk_count).zfill(3)}"
                    # 出力パスを再構築
                    relative_path = os.path.relpath(original_root_dir, INPUT_DIR)
                    final_output_dir = os.path.join(OUTPUT_DIR, relative_path, save_dir_name)
                    os.makedirs(final_output_dir, exist_ok=True)
                    print(f"動画を保存します: {final_output_dir}")
                    out_side = cv2.VideoWriter(os.path.join(final_output_dir, 'side.mp4'), fourcc, fps, (TARGET_SIZE, TARGET_SIZE))
                    out_front = cv2.VideoWriter(os.path.join(final_output_dir, 'front.mp4'), fourcc, fps, (TARGET_SIZE, TARGET_SIZE))
                    for i in range(len(side_frame_buffer)):
                        out_side.write(side_frame_buffer[i])
                        out_front.write(front_frame_buffer[i])
                    out_side.release()
                    out_front.release()
                else:
                    print("動画が短すぎるため、保存しません。")
                side_frame_buffer = []
                front_frame_buffer = []
                is_saving_section = False
                side_current_stable_direction = "Unknown"
        
        if side_person_detected:
            side_previous_direction = side_direction
        else:
            side_previous_direction = "Unknown"
            side_stable_direction_count = 0
            
    if frame_count % 100 == 0:
        print(f"フレーム {frame_count} 処理中...")

    if is_saving_section and len(side_frame_buffer) >= min_frames_to_save:
        walk_count += 1
        save_dir_name = f"walk_{side_current_stable_direction}_{str(walk_count).zfill(3)}"
        relative_path = os.path.relpath(original_root_dir, INPUT_DIR)
        final_output_dir = os.path.join(OUTPUT_DIR, relative_path, save_dir_name)
        os.makedirs(final_output_dir, exist_ok=True)
        print(f"動画の最後に到達しました。動画を保存します: {final_output_dir}")
        out_side = cv2.VideoWriter(os.path.join(final_output_dir, 'side.mp4'), fourcc, fps, (TARGET_SIZE, TARGET_SIZE))
        out_front = cv2.VideoWriter(os.path.join(final_output_dir, 'front.mp4'), fourcc, fps, (TARGET_SIZE, TARGET_SIZE))
        for i in range(len(side_frame_buffer)):
            out_side.write(side_frame_buffer[i])
            out_front.write(front_frame_buffer[i])
        out_side.release()
        out_front.release()
    else:
        print("動画の最後に到達しました。短い区間のため保存しません。")

    side_cap.release()
    front_cap.release()
    print("--- 処理完了 ---")

print("すべての動画ペアの処理が完了しました。")