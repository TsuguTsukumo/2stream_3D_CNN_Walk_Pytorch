# ==============================================================================
# 概要
# ==============================================================================
# このスクリプトは、Gradioを使用してWeb UIを構築し、
# 歩行動画からASD（自閉スペクトラム症）の傾向を分類するデモアプリケーションです。
#
# 主な機能：
# 1. 分類用の学習済みモデルとHydra設定ファイルを読み込みます。
# 2. アップロードされた動画を1秒ごとのクリップに分割します。
# 3. 各クリップから8フレームを抽出し、YOLOv8で人物を検出・切り出します。
# 4. 各クリップに対して分類推論を行い、結果を一覧で表示します。
# 5. Grad-CAMを用いて、モデルが判断の根拠とした領域を可視化します。
#
# 必要なファイル：
# - yolov8n.pt: YOLOv8の学習済みモデルファイル。
# - model.ckpt: 学習済みのPytorch分類モデルファイル。
# - make_model.py: `single`クラスが定義されたモデル定義ファイル。
# - 対応するconfig.yaml: モデル学習時の設定ファイル（自動で検出されます）。
#
# 実行方法：
# 1. ターミナルで `pip install -r requirements.txt` を実行して、
#    必要なライブラリをインストールします。
# 2. `python app.py` で実行してGradioサーバーを起動します。
# 3. 表示されたURL（例: http://127.0.0.1:7860）にブラウザでアクセスします。
# ==============================================================================

import gradio as gr
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os
from collections import OrderedDict
from types import SimpleNamespace
from make_model import single
import yaml
from pathlib import Path
from torchvision import transforms

# 【追加】Grad-CAM関連のライブラリをインポート
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==========
# デバイス設定 (GPU or CPU)
# ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")


# ==========
# モデル構築
# ==========
def build_model(config_path: Path):
    """
    Hydraの設定ファイル(config.yaml)を読み込み、
    その設定に基づいてモデルのインスタンスを生成して返します。
    """
    def dict_to_sns(d):
        if not isinstance(d, dict):
            return d
        return SimpleNamespace(**{k: dict_to_sns(v) for k, v in d.items()})

    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        print(f"設定ファイルを読み込みました: {config_path}")
        hparams = dict_to_sns(config_dict)
        model = single(hparams)
        return model
    except FileNotFoundError:
        print(f"エラー: 設定ファイル '{config_path}' が見つかりませんでした。")
        return None
    except Exception as e:
        print(f"設定ファイルの読み込みまたはモデルの構築中にエラーが発生しました: {e}")
        return None

# ==========
# Grad-CAM Reshape Transform
# ==========
def reshape_transform(tensor):
    # 5Dテンソル (B, T, C, H, W) を 4Dテンソル (B*T, C, H, W) に変換
    if len(tensor.shape) == 5:
        b, t, c, h, w = tensor.size()
        result = tensor.reshape(b * t, c, h, w)
        return result
    return tensor

# ==========
# モデル読み込み
# ==========
model_path = "/workspace/project/logs/resnet/single/2025-10-03/05-53-53/fold0/version_0/checkpoints/11-1.75-0.6574.ckpt"
clf_model = None
target_layers = None

if not os.path.exists(model_path):
    print(f"警告: モデルファイル '{model_path}' が見つかりません。")
else:
    try:
        p = Path(model_path)
        log_dir = p.parent.parent.parent.parent
        config_path = log_dir / ".hydra" / "config.yaml"

        print(f"'{model_path}' から分類モデルを読み込んでいます...")
        clf_model = build_model(config_path)

        if clf_model is not None:
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get('state_dict', checkpoint)

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace("model.", "", 1)
                new_state_dict[name] = v
            
            clf_model.load_state_dict(new_state_dict, strict=False)
            clf_model = clf_model.to(device)
            clf_model.eval()
            print("✅ 分類モデルの読み込みと重みの適用に成功しました。")

            # 【修正】Grad-CAMのターゲットレイヤーをより具体的に指定
            target_layers = [clf_model.resnet_model.layer4[-1]]

    except Exception as e:
        print("="*60)
        print("❌ 分類モデルの読み込み中に致命的なエラーが発生しました。")
        print(f"エラー内容: {e}")
        print("="*60)
        clf_model = None

# ==========
# YOLOモデル（人物検出用）
# ==========
try:
    print("YOLOv8モデルを読み込んでいます...")
    yolo_model = YOLO("yolov8n.pt")
    print("✅ YOLOv8モデルの読み込みに成功しました。")
except Exception as e:
    print(f"❌ YOLOモデルの読み込み中にエラーが発生しました: {e}")
    yolo_model = None


# ==========
# 画像前処理
# ==========
def preprocess_frame(frame):
    if yolo_model is None: return None
    results = yolo_model(frame, conf=0.15, classes=[0])
    if len(results[0].boxes) == 0: return None
    boxes = results[0].boxes.xyxy.cpu().numpy()
    largest_box = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    x1, y1, x2, y2 = map(int, largest_box)
    crop = frame[y1:y2, x1:x2]
    if crop.shape[0] == 0 or crop.shape[1] == 0: return None
    target_size = 224
    h0, w0 = crop.shape[:2]
    scale = target_size / max(h0, w0)
    new_w, new_h = int(w0 * scale), int(h0 * scale)
    resized = cv2.resize(crop, (new_w, new_h))
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

# ==========
# 実行パイプライン
# ==========
def run_pipeline(video_file, progress=gr.Progress()):
    if yolo_model is None or clf_model is None:
        return [], [], "❌ モデルが正しく読み込まれていないため、処理を実行できません。"

    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
        print("警告: 動画のFPSが取得できませんでした。30FPSとして処理を続行します。")

    all_heatmaps = []
    all_frames_raw = []
    all_results_text = []
    
    frame_number = 0
    clip_duration_frames = int(fps)
    uniform_temporal_subsample_num = 8

    while True:
        frames_for_clip = []
        for _ in range(clip_duration_frames):
            ret, frame = cap.read()
            if not ret: break
            frames_for_clip.append(frame)
        
        if len(frames_for_clip) < clip_duration_frames:
            break
            
        current_second = frame_number / fps
        progress(frame_number / cap.get(cv2.CAP_PROP_FRAME_COUNT), desc=f"{current_second:.1f}秒地点を処理中...")

        num_frames_in_clip = len(frames_for_clip)
        indices = np.linspace(0, num_frames_in_clip - 1, num=uniform_temporal_subsample_num, dtype=int)
        sampled_frames = [frames_for_clip[i] for i in indices]

        frames_raw_clip = []
        for frame in sampled_frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = preprocess_frame(frame_rgb)
            if processed is not None:
                frames_raw_clip.append(processed)
        
        if len(frames_raw_clip) < uniform_temporal_subsample_num:
            frame_number += len(frames_for_clip)
            continue

        transform = transforms.Compose([transforms.ToTensor()])
        imgs_tensor = torch.stack([transform(f) for f in frames_raw_clip])
        imgs_tensor = imgs_tensor.to(torch.float32).to(device)
        imgs_tensor = imgs_tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = clf_model(imgs_tensor)
        
        mean_logit = outputs.mean().item()
        pred = 1 if mean_logit > 0 else 0
        prob_for_display = 1 / (1 + np.exp(-mean_logit))
        label = "ASDの傾向あり" if pred == 1 else "ASDの傾向なし"
        result_line = f"- {current_second:.1f} - {current_second + 1:.1f}秒: {label} (スコア: {prob_for_display:.4f})"
        all_results_text.append(result_line)
        
        # --- Grad-CAM計算 ---
        with torch.enable_grad():
            cam = GradCAM(model=clf_model.resnet_model, target_layers=target_layers) # Reshape transformは不要
            targets = [ClassifierOutputTarget(0) for _ in range(outputs.size(0))]
            
            # 5Dテンソルを4Dに変換してCAMに入力
            B, T, C, H, W = imgs_tensor.shape
            input_tensor_4d = imgs_tensor.view(B * T, C, H, W)
            
            # 【修正】平滑化オプションを削除
            grayscale_cam = cam(input_tensor=input_tensor_4d, targets=targets)

        original_images_for_cam = imgs_tensor.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
        
        for img_np, cam_map in zip(original_images_for_cam, grayscale_cam):
            resized_cam_map = cv2.resize(cam_map, (img_np.shape[1], img_np.shape[0]))
            visualization = show_cam_on_image(img_np, resized_cam_map, use_rgb=True)
            all_heatmaps.append(visualization)
        
        all_frames_raw.extend(frames_raw_clip)
        frame_number += len(frames_for_clip)

    cap.release()

    if not all_results_text:
        return [], [], "❌ 動画から人物を検出できなかったか、処理できるクリップがありませんでした。"

    final_result_text = "\n".join(all_results_text)
    return all_heatmaps, all_frames_raw, final_result_text

# ==========
# Gradio UI
# ==========
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🧠 ASD分類モデル 推論デモ (1秒ごと)")

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="歩行動画をアップロード")
            result_output = gr.Textbox(label="分類結果 (1秒ごと)", interactive=False, lines=8)
            submit_btn = gr.Button("実行", variant="primary")
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Grad-CAM可視化"):
                    gallery_output = gr.Gallery(label="Grad-CAM", columns=8, height="auto", object_fit="contain")
                with gr.TabItem("前処理後フレーム"):
                    preprocessed_output = gr.Gallery(label="前処理後フレーム", columns=8, height="auto", object_fit="contain")

    submit_btn.click(
        fn=run_pipeline,
        inputs=video_input,
        outputs=[gallery_output, preprocessed_output, result_output]
    )
    
    gr.Markdown("--- \n **注意:** このデモは研究目的で作成されたものであり、医学的診断に代わるものではありません。")

if __name__ == "__main__":
    demo.launch()

