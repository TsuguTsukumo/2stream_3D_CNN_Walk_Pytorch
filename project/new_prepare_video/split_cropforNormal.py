import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

def detect_person(model, frame, conf_threshold=0.5):
    results = model(frame)
    detections = []
    for result in results[0].boxes:
        if int(result.cls[0]) == 0:  # Person class
            x1, y1, x2, y2 = result.xyxy[0].tolist()
            conf = result.conf[0].item()
            if conf >= conf_threshold:  # 信頼度が閾値を超えた場合のみ追加
                detections.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
    return detections

def get_bbox_center(bbox):
    x1, y1, x2, y2, _ = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def crop_and_resize(frame, bbox, target_size=(512, 512)):
    x1, y1, x2, y2, _ = bbox
    cropped = frame[y1:y2, x1:x2]

    # 画像のアスペクト比を維持してリサイズ
    height, width = cropped.shape[:2]
    aspect_ratio = width / height
    new_height = target_size[1]
    new_width = int(aspect_ratio * new_height)

    # リサイズ幅がターゲット幅を超えないように調整
    if new_width > target_size[0]:
        new_width = target_size[0]
        new_height = target_size[1]

    # リサイズ
    resized = cv2.resize(cropped, (new_width, new_height))

    # 黒い背景に合わせて512x512に
    final_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    x_offset = (target_size[0] - new_width) // 2
    final_frame[:, x_offset:x_offset + new_width] = resized

    return final_frame

def split_video_by_direction(input_lat_path, input_ap_path, output_dir, model, min_frames=50):
    cap_lat = cv2.VideoCapture(input_lat_path)
    cap_ap = cv2.VideoCapture(input_ap_path)

    fps = int(cap_lat.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap_lat.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_lat.get(cv2.CAP_PROP_FRAME_HEIGHT))
    segment_index = 0

    direction_history = []  # 保持する方向履歴
    direction = None  # 現在の進行方向
    writer_lat, writer_ap = None, None
    frame_count = 0  # セグメント内のフレーム数

    frame_index = 0
    while cap_lat.isOpened() and cap_ap.isOpened():
        ret_lat, frame_lat = cap_lat.read()
        ret_ap, frame_ap = cap_ap.read()
        if not ret_lat or not ret_ap:
            break

        # 側面のターゲットのバウンディングボックスの検出
        detections_lat = detect_person(model, frame_lat)
        if not detections_lat:
            continue
        target_bbox_lat = detections_lat[0]  # 側面で最初に検出されたターゲットを使用
        frame_lat_cropped = crop_and_resize(frame_lat, target_bbox_lat)

        # 正面のターゲットのバウンディングボックスの検出
        detections_ap = detect_person(model, frame_ap)
        if not detections_ap:
            continue

        # 正面動画で人物が複数人検出された場合、一番左の人物をターゲット
        target_bbox_ap = min(detections_ap, key=lambda bbox: bbox[0])  # x1が最小のバウンディングボックスを選択
        frame_ap_cropped = crop_and_resize(frame_ap, target_bbox_ap)

        # 方向履歴の更新
        direction_history.append(target_bbox_lat[0])  # x1を使用して方向を決定
        if len(direction_history) > 10:
            direction_history.pop(0)

        if len(direction_history) == 10:
            avg_direction = direction_history[-1] - direction_history[0]
            new_direction = "right" if avg_direction > 0 else "left"

            if direction != new_direction:
                if writer_lat and writer_ap:
                    # セグメント内のフレーム数が条件を満たしているか確認
                    if frame_count >= min_frames:
                        writer_lat.release()
                        writer_ap.release()

                    else:
                        # フレーム数が少ない場合、セグメントをスキップ
                        print(f"Skipping segment {segment_index} due to insufficient frames")
                
                # 新しいセグメントを開始
                segment_folder = os.path.join(output_dir, f'segment{segment_index}')
                os.makedirs(segment_folder, exist_ok=True)

                lat_segment_path = os.path.join(segment_folder, f'lat_{segment_index}.mp4')
                ap_segment_path = os.path.join(segment_folder, f'ap_{segment_index}.mp4')

                writer_lat = cv2.VideoWriter(
                    lat_segment_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (512, 512)
                )
                writer_ap = cv2.VideoWriter(
                    ap_segment_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (512, 512)
                )
                segment_index += 1
                direction = new_direction
                frame_count = 0  # 新しいセグメントが始まるのでフレーム数をリセット

        # フレームの保存
        if writer_lat and writer_ap:
            writer_lat.write(frame_lat_cropped)
            writer_ap.write(frame_ap_cropped)
            frame_count += 1  # セグメント内のフレーム数をカウント

        frame_index += 1

    # 後処理
    if writer_lat and writer_ap:
        # セグメント内のフレーム数が条件を満たしている場合のみ保存
        if frame_count >= min_frames:
            writer_lat.release()
            writer_ap.release()
        else:
            print(f"Skipping final segment {segment_index} due to insufficient frames")

    cap_lat.release()
    cap_ap.release()


def process_videos(input_dir, output_dir, gpu_id=0):
    torch.cuda.set_device(gpu_id)
    
    model = YOLO("yolo11n.pt")
    model.to('cuda')

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file == "full_lat.mp4":
                input_lat_path = os.path.join(root, file)
                input_ap_path = os.path.join(root, "full_ap.mp4")
                date_folder = os.path.basename(root)
                output_path = os.path.join(output_dir, date_folder)
                os.makedirs(output_path, exist_ok=True)
                split_video_by_direction(input_lat_path, input_ap_path, output_path, model)

# 入力と出力ディレクトリの設定
input_dir = "/workspace/data/new_data_selected/non_ASD/Normal"
output_dir = "/workspace/data/TEST_Normal_9"
process_videos(input_dir, output_dir, gpu_id=0)
