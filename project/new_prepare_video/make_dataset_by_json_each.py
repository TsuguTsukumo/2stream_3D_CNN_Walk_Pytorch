import os
import shutil
import json

# 入力ディレクトリと出力ディレクトリ
root_dir = "/workspace/data/Video/Segment_video"
output_root_dir_ap = "/workspace/data/output_dir_ap"  # ap用の出力ディレクトリ
output_root_dir_lat = "/workspace/data/output_dir_lat"  # lat用の出力ディレクトリ

# JSONファイルのパス
split_json_file_path = "/workspace/data/Combined_video/split_results.json"


# JSONファイルを読み込み
with open(split_json_file_path, "r") as f:
    splits = json.load(f)

# ファイルをコピーする関数
def copy_files_for_split(split_data, split_type, fold_number):
    for person_data in split_data:
        category_name = person_data["category"]
        person_id = person_data["person_id"]

        # 元のデータセットのパス
        original_path = os.path.join(root_dir, category_name, person_id)

        # 出力ディレクトリのパス
        output_fold_dir_ap = os.path.join(output_root_dir_ap, f"fold_{fold_number}", split_type, category_name, person_id)
        output_fold_dir_lat = os.path.join(output_root_dir_lat, f"fold_{fold_number}", split_type, category_name, person_id)

        os.makedirs(output_fold_dir_ap, exist_ok=True)
        os.makedirs(output_fold_dir_lat, exist_ok=True)

        # ap.mp4 と lat.mp4 をそれぞれコピー
        for video_file, output_dir in [("ap.mp4", output_fold_dir_ap), ("lat.mp4", output_fold_dir_lat)]:
            video_path = os.path.join(original_path, video_file)
            if os.path.exists(video_path):
                shutil.copy2(video_path, output_dir)
            else:
                print(f"警告: ファイルが見つかりません - {video_path}")

# 各foldに対してファイルをコピー
def create_folds():
    for split_name, split_data in splits.items():
        print(f"処理中: {split_name}")

        # Fold番号を取得
        fold_number = int(split_name.split('_')[-1])  # split_1 -> 1

        # トレーニングセットと検証セットのフォルダを作成
        for split_type in ["train", "val"]:
            split_type_data = split_data[split_type + "_data"]

            # JSONファイルの情報に基づいてファイルをコピー
            copy_files_for_split(split_type_data, split_type, fold_number)

    print("フォルダとデータのコピーが完了しました！")

# 実行
create_folds()

