import cv2
import torch
from tool.darknet2pytorch import Darknet
from tool.torch_utils import do_detect
import os

def detect_person_and_crop(video_path, output_path, model, confidence_threshold=0.5, nms_threshold=0.4, use_cuda=False):
    # モデルを適切なデバイスに移動
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    cap = cv2.VideoCapture(video_path)
    
    # ビデオの幅と高さを取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # ビデオ
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' または 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # フレームを適切なデバイスに転送
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device)

        # 人物検出
        boxes = do_detect(model, img, confidence_threshold, nms_threshold, use_cuda=use_cuda)
        
        person_detected = False

        for box in boxes:
            cls_id = int(box[6])
            if cls_id == 0:  # 'person' クラスのIDが0と仮定
                x1 = int(box[0] * width)
                y1 = int(box[1] * height)
                x2 = int(box[2] * width)
                y2 = int(box[3] * height)
                bbox_height = y2 - y1
                
                # 全身が映っているかを判定
                if bbox_height / height <= 0.98:
                    # フレームを切り出し
                    person_frame = frame[y1:y2, x1:x2]
                    frame[y1:y2, x1:x2] = person_frame  # フレームの該当部分を更新
                    person_detected = True
                    break  # 一度検出されれば他の検出結果は無視

        if not person_detected:
            # 人物が検出されていない場合もフレームを追加
            print(f"Frame {frame_index} - No person detected, skipped.")

        # フレームをビデオに書き込み
        out.write(frame)

        # 進行状況を表示
        progress = (frame_index + 1) / total_frames * 100
        print(f"Processing frame {frame_index + 1}/{total_frames} ({progress:.2f}%)")

        frame_index += 1

    cap.release()
    out.release()

# YOLOv4モデルのロード
cfg_file = "./cfg/yolov4.cfg"
weight_file = "yolov4.weights"
model = Darknet(cfg_file)
model.load_weights(weight_file)
model.eval()

# ビデオの出力パス
output_path = "output_video.mp4"

# ビデオの処理
detect_person_and_crop("full_ap.mp4", output_path, model, use_cuda=True)
