import os
import cv2
import torch
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n.pt')
model.to('cuda')

# 検出結果からバウンディングボックスを取得し、人物のみを対象にする関数
def get_bounding_box(results, view='front', conf_threshold=0.7):
    target_bbox = None
    max_conf = -1  # side用の信頼度管理変数
    print(f"検出数: {len(results)}")

    for det in results:
        bbox = det.xyxy.cpu().numpy() if isinstance(det.xyxy, torch.Tensor) else det.xyxy

        if det.cls == 0 and det.conf >= conf_threshold:
            print(f"人物検出 - バウンディングボックス: {bbox}, 信頼度: {det.conf}")

            if view == 'front':
                # frontでは左端に最も近い人物を選択
                if target_bbox is None or bbox[0][0] < target_bbox[0][0]:
                    target_bbox = bbox
            elif view == 'side':
                # sideでは最も信頼度の高い人物を選択
                if det.conf > max_conf:
                    target_bbox = bbox
                    max_conf = det.conf

    if target_bbox is None:
        print(f"{view} で人物が検出されませんでした。")
    
    return target_bbox

# アスペクト比を維持しながら512×512のフレームにリサイズし、余白を黒で埋める関数
def resize_with_aspect_ratio(frame, target_size=512):
    h, w = frame.shape[:2]
    aspect_ratio = w / h

    if aspect_ratio > 1:
        new_w = target_size
        new_h = int(target_size / aspect_ratio)
    else:
        new_h = target_size
        new_w = int(target_size * aspect_ratio)

    resized_frame = cv2.resize(frame, (new_w, new_h))

    output_frame = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    output_frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame

    return output_frame

# 各サブフォルダ内の2つの動画を処理する関数
def process_videos(input_folder):
    front_video_path = os.path.join(input_folder, 'full_ap.mp4')
    side_video_path = os.path.join(input_folder, 'full_lat.mp4')

    out_folder = '/workspace/data/data/new_data_selected_cropped/non_ASD'
    output_folder = os.path.join(out_folder, 'Normal')
    os.makedirs(output_folder, exist_ok=True)
    print(os.path.basename(input_folder))

    front_output_path   = os.path.join(output_folder, os.path.basename(input_folder))
    side_output_path    = os.path.join(output_folder, os.path.basename(input_folder))
    os.makedirs(front_output_path, exist_ok=True)
    os.makedirs(side_output_path, exist_ok=True)
    print(front_output_path)
    front_output_path   = os.path.join(front_output_path, 'full_ap.mp4')
    side_output_path    = os.path.join(side_output_path, 'full_lat.mp4')
    print(front_output_path)

    front_cap = cv2.VideoCapture(front_video_path)
    side_cap = cv2.VideoCapture(side_video_path)

    fps = int(front_cap.get(cv2.CAP_PROP_FPS))

    front_out = cv2.VideoWriter(front_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (512, 512))
    side_out = cv2.VideoWriter(side_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (512, 512))

    while True:
        ret1, front_frame = front_cap.read()
        ret2, side_frame = side_cap.read()

        if not ret1 or not ret2:
            break

        front_results_list = model(front_frame)
        side_results_list = model(side_frame)

        front_results = front_results_list[0]
        side_results = side_results_list[0]

        front_bbox = get_bounding_box(front_results.boxes, view='front', conf_threshold=0.5)
        side_bbox = get_bounding_box(side_results.boxes, view='side', conf_threshold=0.5)

        if front_bbox is not None and side_bbox is not None:
            try:
                front_x1 = max(0, int(front_bbox[0][0]))
                front_y1 = max(0, int(front_bbox[0][1]))
                front_x2 = min(front_frame.shape[1], int(front_bbox[0][2]))
                front_y2 = min(front_frame.shape[0], int(front_bbox[0][3]))

                side_x1 = max(0, int(side_bbox[0][0]))
                side_y1 = max(0, int(side_bbox[0][1]))
                side_x2 = min(side_frame.shape[1], int(side_bbox[0][2]))
                side_y2 = min(side_frame.shape[0], int(side_bbox[0][3]))

                front_crop = front_frame[front_y1:front_y2, front_x1:front_x2]
                side_crop = side_frame[side_y1:side_y2, side_x1:side_x2]

                front_resized = resize_with_aspect_ratio(front_crop, target_size=512)
                side_resized = resize_with_aspect_ratio(side_crop, target_size=512)

                front_out.write(front_resized)
                side_out.write(side_resized)

            except Exception as e:
                print(f"切り抜きエラー: {e}")

        else:
            print("対象者がどちらかの動画に検出されなかったため、フレームをスキップします。")

    front_cap.release()
    side_cap.release()
    front_out.release()
    side_out.release()

# 指定フォルダ内のすべてのサブフォルダに対して処理を行う関数
def process_all_videos_in_folder(root_folder):
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_path}")
            process_videos(folder_path)

# 実行
root_folder = '/workspace/data/data/new_data_selected/non_ASD/Normal'
process_all_videos_in_folder(root_folder)