import os
import shutil

def restructure_videos(inp_dir, out_ap_dir, out_lat_dir):
    # 入力ディレクトリを探索
    for category in os.listdir(inp_dir):
        category_path = os.path.join(inp_dir, category)
        if not os.path.isdir(category_path):
            continue
        
        # 出力ディレクトリのカテゴリフォルダを作成
        ap_category_out = os.path.join(out_ap_dir, category)
        lat_category_out = os.path.join(out_lat_dir, category)
        os.makedirs(ap_category_out, exist_ok=True)
        os.makedirs(lat_category_out, exist_ok=True)
        
        for date_folder in os.listdir(category_path):
            date_path = os.path.join(category_path, date_folder)
            if not os.path.isdir(date_path):
                continue
            
            # 出力ディレクトリの日付フォルダを作成
            ap_date_out = os.path.join(ap_category_out, date_folder)
            lat_date_out = os.path.join(lat_category_out, date_folder)
            os.makedirs(ap_date_out, exist_ok=True)
            os.makedirs(lat_date_out, exist_ok=True)
            
            for segment_folder in os.listdir(date_path):
                segment_path = os.path.join(date_path, segment_folder)
                if not os.path.isdir(segment_path):
                    continue
                
                # ファイルパスを生成
                ap_file = os.path.join(segment_path, "ap.mp4")
                lat_file = os.path.join(segment_path, "lat.mp4")
                
                # 出力ファイルパスを生成
                ap_out_file = os.path.join(ap_date_out, f"{segment_folder}.mp4")
                lat_out_file = os.path.join(lat_date_out, f"{segment_folder}.mp4")
                
                # ファイルをコピー
                if os.path.exists(ap_file):
                    shutil.copy2(ap_file, ap_out_file)
                if os.path.exists(lat_file):
                    shutil.copy2(lat_file, lat_out_file)

# ディレクトリ構造の指定
inp_dir = "/workspace/data/Video/Segment_video_ASDandNormal"  # 入力ディレクトリ
out_ap_dir = "/workspace/data/Video/Segment_video_ASDandNormal_ap"  # ap.mp4の出力ディレクトリ
out_lat_dir = "/workspace/data/Video/Segment_video_ASDandNormal_lat"  # lat.mp4の出力ディレクトリ

# 関数の実行
restructure_videos(inp_dir, out_ap_dir, out_lat_dir)
