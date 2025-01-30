import os
from moviepy.editor import VideoFileClip, clips_array

# データセットのルートディレクトリ
root_dir = "/workspace/data/data/Segment_video"  # 実際のパスに置き換えてください

# 疾患Aと疾患Bのディレクトリを処理
diseases = ['ASD']

for disease in diseases:
    disease_dir = os.path.join(root_dir, disease)
    
    # 日付ごとに処理
    for date_dir in os.listdir(disease_dir):
        date_path = os.path.join(disease_dir, date_dir)
        
        # 出力先ディレクトリ（疾患 -> 日付）
        output_date_dir = os.path.join("/workspace/data/data/Combined_video", disease, date_dir)
        os.makedirs(output_date_dir, exist_ok=True)
        
        # セグメントごとに処理
        for segment_dir in os.listdir(date_path):
            segment_path = os.path.join(date_path, segment_dir)
            
            # ap.mp4 と lat.mp4 を読み込む
            video1_path = os.path.join(segment_path, 'ap.mp4')
            video2_path = os.path.join(segment_path, 'lat.mp4')
            
            if os.path.exists(video1_path) and os.path.exists(video2_path):
                # 動画を読み込み、リサイズ
                clip1 = VideoFileClip(video1_path).resize((256, 512))
                clip2 = VideoFileClip(video2_path).resize((256, 512))

                # 動画を横に並べる
                combined = clips_array([[clip1, clip2]]).resize((512, 512))
                
                # 出力パス設定（疾患 -> 日付 -> segmentX_combined.mp4）
                output_path = os.path.join(output_date_dir, f"{segment_dir}_combined.mp4")
                
                # 結果を保存
                combined.write_videofile(output_path, codec="libx264", fps=int(clip1.fps))
