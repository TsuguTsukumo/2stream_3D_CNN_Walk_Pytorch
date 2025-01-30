import os
import cv2
from ultralytics import YOLO

def detect_person(model, frame, conf_threshold=0.5):
    results = model(frame)
    detections = []
    for result in results[0].boxes:
        if int(result.cls[0]) == 0 and result.conf[0].item() > conf_threshold:  # Person class
            x1, y1, x2, y2 = result.xyxy[0].tolist()
            conf = result.conf[0].item()
            detections.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
    return detections

def get_bbox_center(bbox):
    x1, y1, x2, y2, _ = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def split_video_by_direction(input_lat_path, input_ap_path, output_dir, model):
    cap_lat = cv2.VideoCapture(input_lat_path)
    cap_ap = cv2.VideoCapture(input_ap_path)

    fps = int(cap_lat.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap_lat.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_lat.get(cv2.CAP_PROP_FRAME_HEIGHT))
    segment_index = 0

    direction_history = []  # 保持する方向履歴
    direction = None  # 現在の進行方向
    writer_lat, writer_ap = None, None

    frame_index = 0
    last_detection_frame = -20  # 最初の20フレームは無視
    while cap_lat.isOpened() and cap_ap.isOpened():
        ret_lat, frame_lat = cap_lat.read()
        ret_ap, frame_ap = cap_ap.read()
        if not ret_lat or not ret_ap:
            break

        # 人物検出
        detections = detect_person(model, frame_lat)
        if not detections:
            # ターゲットが検出されていない場合はスキップ
            continue

        # 最初の検出から20フレームはスキップ
        if frame_index - last_detection_frame < 20:
            frame_index += 1
            continue

        # ターゲットの中心点を取得
        target_bbox = detections[0]  # 最初に検出された人物をターゲットとする
        center_x, center_y = get_bbox_center(target_bbox)

        # 方向履歴の更新
        direction_history.append(center_x)
        if len(direction_history) > 10:
            direction_history.pop(0)

        # 平均方向の計算
        if len(direction_history) == 10:
            avg_direction = direction_history[-1] - direction_history[0]
            new_direction = "right" if avg_direction > 0 else "left"

            if direction != new_direction:
                # 新しいセグメントを開始
                if writer_lat and writer_ap and frame_index - last_detection_frame > 50:
                    writer_lat.release()
                    writer_ap.release()
                segment_folder = os.path.join(output_dir, f'segment{segment_index}')
                os.makedirs(segment_folder, exist_ok=True)

                lat_segment_path = os.path.join(segment_folder, f'lat_{segment_index}.mp4')
                ap_segment_path = os.path.join(segment_folder, f'ap_{segment_index}.mp4')

                writer_lat = cv2.VideoWriter(
                    lat_segment_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (frame_width, frame_height)
                )
                writer_ap = cv2.VideoWriter(
                    ap_segment_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (frame_width, frame_height)
                )
                segment_index += 1
                direction = new_direction
                last_detection_frame = frame_index

        # フレームの保存
        if writer_lat and writer_ap:
            writer_lat.write(frame_lat)
            writer_ap.write(frame_ap)

        frame_index += 1

    # 後処理
    if writer_lat and writer_ap:
        writer_lat.release()
        writer_ap.release()
    cap_lat.release()
    cap_ap.release()

def process_videos(input_dir, output_dir):
    model = YOLO("yolo11n.pt")  # 適切なモデルを選択
    model.to('cuda')

    # 疾患Aおよび疾患Bのサブディレクトリを処理
    for disease in os.listdir(input_dir):
        disease_dir = os.path.join(input_dir, disease)

        # 疾患ディレクトリ内の各日付フォルダを処理
        if os.path.isdir(disease_dir):
            for date_folder in os.listdir(disease_dir):
                date_folder_path = os.path.join(disease_dir, date_folder)

                # 各日付フォルダ内の動画ファイルを処理
                if os.path.isdir(date_folder_path):
                    input_lat_path = os.path.join(date_folder_path, "full_lat.mp4")
                    input_ap_path = os.path.join(date_folder_path, "full_ap.mp4")
                    
                    # 正常にファイルが存在するか確認
                    if os.path.exists(input_lat_path) and os.path.exists(input_ap_path):
                        # 出力ディレクトリに疾患ごとのサブフォルダを作成
                        disease_output_dir = os.path.join(output_dir, disease)
                        os.makedirs(disease_output_dir, exist_ok=True)

                        # 出力先のディレクトリに日付ごとのサブフォルダを作成
                        output_path = os.path.join(disease_output_dir, date_folder)
                        os.makedirs(output_path, exist_ok=True)

                        # 動画分割処理を実行
                        split_video_by_direction(input_lat_path, input_ap_path, output_path, model)

# 入力と出力ディレクトリの設定
input_dir = "/workspace/data/data/new_data_selected/non_ASD"
output_dir = "/workspace/data/data/TEST10"
process_videos(input_dir, output_dir)
