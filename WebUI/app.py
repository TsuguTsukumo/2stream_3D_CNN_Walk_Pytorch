# ==============================================================================
# 概要
# ==============================================================================
# このスクリプトは、Gradioを使用してWeb UIを構築し、
# 正面・側面の歩行動画からASD傾向を分類するデモアプリケーションです。
#
# 主な機能：
# 1. ユーザーが選択したモデル（Single/Fusion）の学習済みモデルを動的に読み込みます。
# 2. 動画を1秒ごとのクリップに分割し、YOLOv8で人物を検出・切り出します。
# 3. 各クリップに対して分類推論を行います。
# 4. 全クリップの結果を統合し、動画全体のサマリーをタブに表示します。
# 5. 最もスコアが高かった区間と、注目部位の傾向をイラストで可視化します。
# 6. Singleモデル選択時には、Grad-CAMで判断根拠を可視化します。
#
# 必要なファイル：
# - yolov8n.pt: YOLOv8の学習済みモデルファイル。
# - 各種モデルの.ckptファイル: MODEL_PATHSにパスを設定してください。
# - make_model.py: 各種モデルクラスが定義されたモデル定義ファイル。
# ==============================================================================

import gradio as gr
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os
from collections import OrderedDict, Counter
from types import SimpleNamespace
from make_model import single, early_fusion, late_fusion, slow_fusion
import yaml
from pathlib import Path
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==========
# 【重要】モデルのパス設定
# ==========
MODEL_PATHS = {
    "single_front": "/workspace/project/logs/resnet/single/2025-10-03/05-53-53/fold0/version_0/checkpoints/11-1.75-0.6574.ckpt",
    "single_side": "/workspace/logs/resnet/single/2025-10-06/04-29-41/fold1/version_0/checkpoints/12-0.96-0.7715.ckpt",
    "early_fusion": "/workspace/project/logs/resnet/early_fusion/2025-02-03/05-15-18/fold2/version_0/checkpoints/16-0.60-0.7234.ckpt",
    "late_fusion": "/workspace/logs/resnet/late_fusion/2025-10-07/05-01-13/fold4/version_0/checkpoints/1-0.72-0.6206.ckpt",
    "slow_fusion": "/workspace/project/logs/resnet/slow_fusion/2025-02-03/05-15-18/fold0/version_0/checkpoints/8-0.94-0.5779.ckpt",
}
# ==========
# デバイス設定 (GPU or CPU)
# ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")


# ==========
# モデル構築
# ==========
def build_model(config_path: Path):
    def dict_to_sns(d):
        if not isinstance(d, dict): return d
        return SimpleNamespace(**{k: dict_to_sns(v) for k, v in d.items()})

    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        hparams = dict_to_sns(config_dict)
        model_type_from_config = hparams.train.experiment

        if model_type_from_config == 'single':
            model = single(hparams)
        elif model_type_from_config == 'early_fusion':
            model = early_fusion(hparams)
        elif model_type_from_config == 'late_fusion':
            model = late_fusion(hparams)
        elif model_type_from_config == 'slow_fusion':
            model = slow_fusion(hparams)
        else:
            raise ValueError(f"不明なモデルタイプ: {model_type_from_config}")
        return model
    except Exception as e:
        print(f"モデル構築中にエラー: {e}")
        return None

def load_model(model_type):
    model_path = MODEL_PATHS.get(model_type)
    if not model_path or not os.path.exists(model_path):
        print(f"警告: '{model_type}'のモデルパスが見つかりません。")
        return None
    
    try:
        p = Path(model_path)
        log_dir = p.parent.parent.parent.parent
        config_path = log_dir / ".hydra" / "config.yaml"

        model = build_model(config_path)
        if model is None: raise ValueError("モデルのインスタンス化に失敗。")

        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("model.", "", 1)
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(device)
        model.eval()
        print(f"✅ '{model_type}'モデルの読み込みに成功しました。")
        return model
    except Exception as e:
        print(f"❌ '{model_type}'モデル読み込み中にエラー: {e}")
        return None

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
def preprocess_frame(frame, crop_mode):
    if yolo_model is None: return None, None
    results = yolo_model(frame, conf=0.15, classes=[0])
    if len(results[0].boxes) == 0: return None, None
    boxes = results[0].boxes.xyxy.cpu().numpy()
    largest_box = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    x1, y1, x2, y2 = map(int, largest_box)

    if crop_mode == "下半身":
        y1 = y1 + (y2 - y1) // 2
        
    crop = frame[y1:y2, x1:x2]
    if crop.shape[0] == 0 or crop.shape[1] == 0: return None, None
    target_size = 224
    h0, w0 = crop.shape[:2]
    scale = target_size / max(h0, w0)
    new_w, new_h = int(w0 * scale), int(h0 * scale)
    resized = cv2.resize(crop, (new_w, new_h))
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    bbox_on_canvas = (x_offset, y_offset, new_w, new_h)
    return canvas, bbox_on_canvas

# ==========
# Grad-CAM生成と部位特定
# ==========
def get_heatmaps_and_focus(cam_model, input_tensor_5d, bboxes_on_canvas):
    if cam_model is None: 
        return [np.zeros((224, 224, 3), dtype=np.uint8)] * 8, "N/A", {"頭部": 0, "胴体": 0, "脚部": 0}
    
    heatmaps_clip = []
    clip_sums = {"頭部": 0.0, "胴体": 0.0, "脚部": 0.0}

    with torch.enable_grad():
        cam_target_model = cam_model.resnet_model
        target_layers = [cam_target_model.layer4[-1]]
        cam = GradCAM(model=cam_target_model, target_layers=target_layers)
        B, T, C, H, W = input_tensor_5d.shape
        input_tensor_4d = input_tensor_5d.view(B * T, C, H, W)
        targets = [ClassifierOutputTarget(0)] * (B * T)
        grayscale_cam = cam(input_tensor=input_tensor_4d, targets=targets)
    
    original_images_for_cam = input_tensor_5d.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
    
    for i, (img_np, cam_map) in enumerate(zip(original_images_for_cam, grayscale_cam)):
        resized_cam_map = cv2.resize(cam_map, (img_np.shape[1], img_np.shape[0]))
        visualization = show_cam_on_image(img_np, resized_cam_map, use_rgb=True)
        heatmaps_clip.append(visualization)
        
        bbox_x, bbox_y, bbox_w, bbox_h = map(int, bboxes_on_canvas[i])
        if bbox_h > 0:
            head_end_y = bbox_y + int(bbox_h * 0.25)
            torso_end_y = bbox_y + int(bbox_h * 0.6)
            legs_end_y = bbox_y + bbox_h
            
            clip_sums["頭部"] += np.sum(resized_cam_map[bbox_y:head_end_y, bbox_x:bbox_x+bbox_w])
            clip_sums["胴体"] += np.sum(resized_cam_map[head_end_y:torso_end_y, bbox_x:bbox_x+bbox_w])
            clip_sums["脚部"] += np.sum(resized_cam_map[torso_end_y:legs_end_y, bbox_x:bbox_x+bbox_w])

    focus_part = max(clip_sums, key=clip_sums.get) if sum(clip_sums.values()) > 0 else "不明"

    return heatmaps_clip, focus_part, clip_sums

# ==========
# 注目部位イラスト生成
# ==========
def create_focus_illustration_html(focus_part, view_type, is_asd):
    highlight_color = "#E53E3E" if is_asd else "#D3D3D3"
    default_color = "#D3D3D3"

    fills = {"head": default_color, "torso": default_color, "legs": default_color}
    if focus_part == "頭部": fills["head"] = highlight_color
    elif focus_part == "胴体": fills["torso"] = highlight_color
    elif focus_part == "脚部": fills["legs"] = highlight_color

    if view_type == "front":
        svg_code = f"""<svg width="150" height="400" viewBox="0 0 150 400" xmlns="http://www.w3.org/2000/svg">
            <style>.body {{ stroke: #555; stroke-width:2; }}</style><title>正面図</title>
            <g id="figure-front">
                <circle id="head-front" class="body" cx="75" cy="50" r="35" fill="{fills['head']}"/>
                <rect id="torso-front" class="body" x="40" y="90" width="70" height="120" rx="10" fill="{fills['torso']}"/>
                <rect id="arm-front-left" class="body" x="15" y="95" width="20" height="110" rx="10" fill="{fills['torso']}"/>
                <rect id="arm-front-right" class="body" x="115" y="95" width="20" height="110" rx="10" fill="{fills['torso']}"/>
                <rect id="legs-front-left" class="body" x="40" y="215" width="32" height="150" rx="10" fill="{fills['legs']}"/>
                <rect id="legs-front-right" class="body" x="78" y="215" width="32" height="150" rx="10" fill="{fills['legs']}"/>
            </g>
            <text x="75" y="390" font-family="sans-serif" font-size="20" text-anchor="middle">正面</text></svg>"""
    else: # side view
        # viewBoxを広げ、全体を中央に配置するよう調整
        svg_code = f"""<svg width="180" height="400" viewBox="0 0 180 400" xmlns="http://www.w3.org/2000/svg">
            <style>.body {{ stroke: #555; stroke-width:2; }}</style><title>側面図</title>
            <g id="figure-side" transform="translate(20, 0)"> <!-- 全体をさらに右にずらす --><!-- Far limbs (opacity reduced) --><rect id="arm-side-far" class="body" x="65" y="95" width="20" height="110" rx="10" transform="rotate(30, 75, 100)" fill="{fills['torso']}" opacity="0.7"/>
                <rect id="leg-side-far" class="body" x="63" y="210" width="25" height="150" rx="10" transform="rotate(-20, 75, 215)" fill="{fills['legs']}" opacity="0.7"/>

                <!-- Body --><circle id="head-side" class="body" cx="75" cy="50" r="35" fill="{fills['head']}"/>
                <rect id="torso-side" class="body" x="60" y="90" width="50" height="120" rx="10" fill="{fills['torso']}"/>
                
                <!-- Near limbs --><rect id="arm-side-near" class="body" x="65" y="95" width="20" height="110" rx="10" transform="rotate(-40, 75, 100)" fill="{fills['torso']}"/>
                <rect id="leg-side-near" class="body" x="63" y="210" width="25" height="150" rx="10" transform="rotate(30, 75, 215)" fill="{fills['legs']}"/>
            </g>
            <text x="75" y="390" font-family="sans-serif" font-size="20" text-anchor="middle">側面</text></svg>"""
    return f"<div style='text-align:center;'>{svg_code}</div>"

# ==========
# 実行パイプライン
# ==========
def run_pipeline(model_type, video_front, video_side, crop_mode, progress=gr.Progress()):
    def make_error_return(message):
        return ([], [], [], [], 0, None, None, message, None, None,
                gr.update(interactive=False), gr.update(interactive=False), "", gr.update())

    progress(0, desc=f"'{model_type}'モデルを読み込み中...")
    clf_model = load_model(model_type)
    if clf_model is None: return make_error_return(f"❌ '{model_type}'のモデル読み込みに失敗しました。")

    is_fusion, is_single = "fusion" in model_type, "single" in model_type
    single_front_cam_model, single_side_cam_model = None, None
    if is_fusion:
        progress(0.1, desc="CAM用Singleモデルを読み込み中...")
        single_front_cam_model, single_side_cam_model = load_model("single_front"), load_model("single_side")

    cap_front, cap_side = None, None
    if is_fusion or "front" in model_type:
        if not video_front: return make_error_return("❌ 正面動画が必要です。")
        cap_front = cv2.VideoCapture(video_front)
    if is_fusion or "side" in model_type:
        if not video_side: return make_error_return("❌ 側面動画が必要です。")
        cap_side = cv2.VideoCapture(video_side)

    fps = cap_front.get(cv2.CAP_PROP_FPS) if cap_front else cap_side.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    
    total_frames = float('inf')
    if cap_front: total_frames = min(total_frames, cap_front.get(cv2.CAP_PROP_FRAME_COUNT))
    if cap_side: total_frames = min(total_frames, cap_side.get(cv2.CAP_PROP_FRAME_COUNT))

    all_composite_images, time_interval_labels, all_scores, all_focus_parts = [], [], [], []
    all_clip_sums_front, all_clip_sums_side = [], []
    frame_number, clip_duration_frames, uniform_temporal_subsample_num = 0, int(fps), 8

    while True:
        frames_for_clip_front, frames_for_clip_side = [], []
        for _ in range(clip_duration_frames):
            ret_front, ret_side = True, True
            if cap_front: ret_front, frame_front = cap_front.read()
            if cap_side: ret_side, frame_side = cap_side.read()
            if not ret_front or not ret_side: break
            if cap_front: frames_for_clip_front.append(frame_front)
            if cap_side: frames_for_clip_side.append(frame_side)
        
        if (cap_front and len(frames_for_clip_front) < clip_duration_frames) or \
           (cap_side and len(frames_for_clip_side) < clip_duration_frames):
            break
            
        current_second = frame_number / fps
        progress(frame_number / total_frames, desc=f"{current_second:.1f}秒地点を処理中...")

        def sample_and_process_clip(clip_frames, crop_mode):
            indices = np.linspace(0, len(clip_frames) - 1, num=uniform_temporal_subsample_num, dtype=int)
            processed_clip, bboxes = [], []
            for i in indices:
                frame, bbox = preprocess_frame(cv2.cvtColor(clip_frames[i], cv2.COLOR_BGR2RGB), crop_mode)
                if frame is not None:
                    processed_clip.append(frame)
                    bboxes.append(bbox)
            return processed_clip, bboxes

        frames_raw_clip_front, bboxes_front = sample_and_process_clip(frames_for_clip_front, crop_mode) if cap_front else ([], [])
        frames_raw_clip_side, bboxes_side = sample_and_process_clip(frames_for_clip_side, crop_mode) if cap_side else ([], [])

        if (cap_front and len(frames_raw_clip_front) < uniform_temporal_subsample_num) or \
           (cap_side and len(frames_raw_clip_side) < uniform_temporal_subsample_num):
            frame_number += clip_duration_frames
            continue

        transform = transforms.Compose([transforms.ToTensor()])
        imgs_tensor_front = torch.stack([transform(f) for f in frames_raw_clip_front]).to(torch.float32).to(device).unsqueeze(0) if cap_front else None
        imgs_tensor_side = torch.stack([transform(f) for f in frames_raw_clip_side]).to(torch.float32).to(device).unsqueeze(0) if cap_side else None

        with torch.no_grad():
            if is_fusion: outputs = clf_model(imgs_tensor_front, imgs_tensor_side)
            elif "front" in model_type: outputs = clf_model(imgs_tensor_front)
            else: outputs = clf_model(imgs_tensor_side)
        
        logit = outputs.mean().item()
        prob_for_display = 1 / (1 + np.exp(-logit))
        interval_label, _ = f"{current_second:.1f}-{current_second + 1:.1f}s", time_interval_labels.append(f"{current_second:.1f}-{current_second + 1:.1f}s")
        all_scores.append(prob_for_display)
        
        if is_single:
            cam_model, input_tensor, bboxes, frames_for_display = (clf_model, imgs_tensor_front, bboxes_front, frames_raw_clip_front) if "front" in model_type else (clf_model, imgs_tensor_side, bboxes_side, frames_raw_clip_side)
            heatmaps_clip, focus_part, clip_sums = get_heatmaps_and_focus(cam_model, input_tensor, bboxes)
            all_focus_parts.append(focus_part)
            if "front" in model_type: all_clip_sums_front.append(clip_sums)
            else: all_clip_sums_side.append(clip_sums)
            top_row, bottom_row = cv2.hconcat(frames_for_display), cv2.hconcat(heatmaps_clip)
            composite_image = cv2.vconcat([top_row, bottom_row])
        elif is_fusion:
            heatmaps_front, focus_front, clip_sums_front = get_heatmaps_and_focus(single_front_cam_model, imgs_tensor_front, bboxes_front)
            heatmaps_side, focus_side, clip_sums_side = get_heatmaps_and_focus(single_side_cam_model, imgs_tensor_side, bboxes_side)
            focus_part = f"正面:{focus_front}, 側面:{focus_side}"
            all_focus_parts.append(focus_part)
            all_clip_sums_front.append(clip_sums_front)
            all_clip_sums_side.append(clip_sums_side)
            row1, row2, row3, row4 = cv2.hconcat(frames_raw_clip_front), cv2.hconcat(heatmaps_front), cv2.hconcat(frames_raw_clip_side), cv2.hconcat(heatmaps_side)
            composite_image = cv2.vconcat([row1, row2, row3, row4])
        
        all_composite_images.append(composite_image)
        frame_number += clip_duration_frames

    if cap_front: cap_front.release()
    if cap_side: cap_side.release()

    if not all_scores: return make_error_return("❌ 処理できるクリップがありませんでした。")
    
    overall_score = np.mean(all_scores)
    overall_label, label_color = ("ASDの傾向あり", "#E53E3E") if overall_score > 0.5 else ("ASDの傾向なし", "#38A169")
    most_abnormal_index = np.argmax(all_scores)
    most_abnormal_image = all_composite_images[most_abnormal_index]

    def get_focus_summary_and_part(clip_sums_list):
        if not clip_sums_list: return "N/A", "N/A"
        total_sums = {"頭部": 0.0, "胴体": 0.0, "脚部": 0.0}
        for clip_sums in clip_sums_list:
            for part, value in clip_sums.items(): total_sums[part] += value
        if sum(total_sums.values()) == 0: return "N/A", "N/A"
        most_common_part = max(total_sums, key=total_sums.get)
        summary = f"**{most_common_part}**"
        if overall_score > 0.5: summary = f"<span style='color:{label_color};'>{summary}</span>"
        return summary, most_common_part

    focus_summary_front, most_common_front = get_focus_summary_and_part(all_clip_sums_front)
    focus_summary_side, most_common_side = get_focus_summary_and_part(all_clip_sums_side)

    summary_markdown = f"""<div style='text-align:center; padding: 10px; border-radius: 5px; background-color: #f9f9f9;'>
        <p style='margin-bottom: 5px;'>最終判定</p><h2 style='color:{label_color}; margin-top: 0px;'>{overall_label}</h2><hr>
        <div style='text-align:left;'>
        **平均スコア:** {overall_score:.4f}<br>
        **主な注目部位 (正面):** {focus_summary_front}<br>
        **主な注目部位 (側面):** {focus_summary_side}<br><br>
        **【最も顕著な区間】**<br>
        **区間:** {time_interval_labels[most_abnormal_index]}<br>
        **スコア:** {all_scores[most_abnormal_index]:.4f}</div></div>"""

    display_label_text = f"区間: {time_interval_labels[0]} | スコア: {all_scores[0]:.4f} | 注目部位: {all_focus_parts[0]}"
    image_label = "上段: 入力, 下段: CAM" if is_single else "1,3段目: 入力(正,側), 2,4段目: CAM(正,側)"
    
    return (all_composite_images, time_interval_labels, all_scores, all_focus_parts, 0, all_composite_images[0],
            most_abnormal_image, summary_markdown, 
            create_focus_illustration_html(most_common_front, "front", overall_score > 0.5),
            create_focus_illustration_html(most_common_side, "side", overall_score > 0.5),
            gr.update(interactive=False), gr.update(interactive=len(all_composite_images) > 1), display_label_text, gr.update(label=image_label))

def navigate_images(current_index, direction, all_images, all_intervals, all_scores, all_focus_parts):
    if not all_images: return 0, None, gr.update(interactive=False), gr.update(interactive=False), ""
    max_index, new_index = len(all_images) - 1, max(0, min(current_index + direction, len(all_images) - 1))
    image, prev_btn, next_btn = all_images[new_index], gr.update(interactive=new_index > 0), gr.update(interactive=new_index < max_index)
    label = f"区間: {all_intervals[new_index]} | スコア: {all_scores[new_index]:.4f} | 注目部位: {all_focus_parts[new_index]}"
    return new_index, image, prev_btn, next_btn, label

# ==========
# Gradio UI
# ==========
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🧠 ASD分類モデル 推論デモ")

    image_state, interval_state, scores_state, focus_parts_state, current_index_state = gr.State([]), gr.State([]), gr.State([]), gr.State([]), gr.State(0)

    with gr.Row():
        with gr.Column(scale=1):
            model_type_input = gr.Radio(list(MODEL_PATHS.keys()), value="early_fusion", label="モデルを選択")
            video_input_front = gr.Video(label="正面動画 (Single/Fusionで利用)")
            video_input_side = gr.Video(label="側面動画 (Single/Fusionで利用)")
            crop_mode_input = gr.Radio(["全身", "下半身"], value="全身", label="切り出し範囲")
            submit_btn = gr.Button("実行", variant="primary")

        with gr.Column(scale=2):
            with gr.Tabs(selected=0) as tabs: # デフォルトを全体サマリー(id=0)に設定
                with gr.TabItem("全体サマリー", id=0):
                    with gr.Row():
                        summary_output = gr.Markdown()
                        with gr.Row():
                            illustration_front = gr.HTML()
                            illustration_side = gr.HTML()
                    abnormal_image_output = gr.Image(label="最も顕著な区間の可視化", interactive=False)
                with gr.TabItem("インタラクティブ表示", id=1):
                    with gr.Row():
                        prev_btn = gr.Button("前の区間", interactive=False)
                        display_label = gr.Textbox(label="表示中の情報", interactive=False, text_align="center")
                        next_btn = gr.Button("次の区間", interactive=False)
                    image_output = gr.Image(label="秒ごとの可視化結果", interactive=False)

    submit_btn.click(
        fn=run_pipeline,
        inputs=[model_type_input, video_input_front, video_input_side, crop_mode_input],
        outputs=[image_state, interval_state, scores_state, focus_parts_state, current_index_state, image_output,
                 abnormal_image_output, summary_output, illustration_front, illustration_side,
                 prev_btn, next_btn, display_label, image_output]
    )
    
    prev_btn.click(
        fn=lambda idx, imgs, intervals, scores, parts: navigate_images(idx, -1, imgs, intervals, scores, parts),
        inputs=[current_index_state, image_state, interval_state, scores_state, focus_parts_state],
        outputs=[current_index_state, image_output, prev_btn, next_btn, display_label]
    )

    next_btn.click(
        fn=lambda idx, imgs, intervals, scores, parts: navigate_images(idx, 1, imgs, intervals, scores, parts),
        inputs=[current_index_state, image_state, interval_state, scores_state, focus_parts_state],
        outputs=[current_index_state, image_output, prev_btn, next_btn, display_label]
    )
    
    gr.Markdown("--- \n **注意:** このデモは研究目的で作成されたものであり、医学的診断に代わるものではありません。")

if __name__ == "__main__":
    demo.launch()