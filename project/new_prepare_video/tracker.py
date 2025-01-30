import cv2
from ultralytics import YOLO
import os
import json

# YOLOv8モデルをロード
model = YOLO('yolov8n.pt')
model.to('cuda')
print(f"Using device: {model.device}")

# 動画ファイルを指定してトラッキングを実行
results = model.track(
    source='/workspace/project/project/new_prepare_video/data/full_ap.mp4',
    save=False,
    show=False,
    classes=[0],  # 人のみを対象
    tracker='bytetrack.yaml',
    conf=0.7
)

# トラッキング結果から人物IDごとのバウンディングボックスを保持
frame_counts = {}
bbox_storage = {}  # {person_id: [list of bboxes across frames]}

for result in results:
    for det in result.boxes.data:
        person_id = int(det[4].item())
        bbox = det[:4].cpu().numpy()  # バウンディングボックス座標 (x1, y1, x2, y2)
        
        if person_id in frame_counts:
            frame_counts[person_id] += 1
            bbox_storage[person_id].append(bbox)
        else:
            frame_counts[person_id] = 1
            bbox_storage[person_id] = [bbox]

# 最も長く映っている人物のIDを特定
most_visible_id = max(frame_counts, key=frame_counts.get)
print(f"最も長く映っている人物のID: {most_visible_id}")

# 元の動画を再読み込み
video_path = '/workspace/project/project/new_prepare_video/sample.mp4'
cap = cv2.VideoCapture(video_path)

# 出力動画の保存設定 (切り抜き後の解像度を最初のフレームから取得)
first_bbox = bbox_storage[most_visible_id][0]
x1, y1, x2, y2 = map(int, first_bbox)
cropped_width = x2 - x1
cropped_height = y2 - y1

output_path = '/workspace/project/project/new_prepare_video/sample_test2.mp4'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, fourcc, fps, (cropped_width, cropped_height))

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_idx < len(bbox_storage[most_visible_id]):
        x1, y1, x2, y2 = map(int, bbox_storage[most_visible_id][frame_idx])
        cropped_frame = frame[y1:y2, x1:x2]
        
        # リサイズは不要だが、確認のためサイズを統一
        cropped_frame = cv2.resize(cropped_frame, (cropped_width, cropped_height))
        
        out.write(cropped_frame)
    
    frame_idx += 1

cap.release()
out.release()

print(f"切り抜き動画を保存しました: {output_path}")
