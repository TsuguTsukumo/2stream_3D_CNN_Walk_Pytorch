import os
import shutil

def restructure_dataset(root_dir, out_dir):
    """
    入力データセットを再構成して、指定した形式で出力する。
    
    Args:
        root_dir (str): 入力データセットのルートディレクトリ
        out_dir (str): 再構成されたデータセットの出力ディレクトリ
    """
    try:
        # foldごとに処理
        for fold in os.listdir(root_dir):
            fold_path = os.path.join(root_dir, fold)

            if not os.path.isdir(fold_path):
                continue

            # train/val/test ディレクトリを処理
            for split in os.listdir(fold_path):
                split_path = os.path.join(fold_path, split)

                if not os.path.isdir(split_path):
                    continue

                # カテゴリごとに処理
                for category in os.listdir(split_path):
                    category_path = os.path.join(split_path, category)
                    
                    if not os.path.isdir(category_path):
                        continue

                    # 出力先カテゴリディレクトリを作成
                    category_out_path = os.path.join(out_dir, fold, split, category)
                    os.makedirs(category_out_path, exist_ok=True)

                    # 日付ディレクトリを処理
                    for date in os.listdir(category_path):
                        date_path = os.path.join(category_path, date)

                        if not os.path.isdir(date_path):
                            continue

                        # 動画ファイルをリネームしてコピー
                        video_files = [f for f in os.listdir(date_path) if f.endswith('.mp4')]

                        for i, video_file in enumerate(video_files):
                            try:
                                # 新しいファイル名を作成（カテゴリ名を先頭に追加）
                                file_basename, file_ext = os.path.splitext(video_file)
                                new_video_name = f"{category}_{date}_{i+1}.mp4"
                                new_video_path = os.path.join(category_out_path, new_video_name)

                                # ファイルを新しい場所にコピー
                                shutil.copy(os.path.join(date_path, video_file), new_video_path)
                            except Exception as e:
                                print(f"Error copying file {video_file} in {date_path}: {e}")

        print("データセットの再構成（コピー）が完了しました！")
    except Exception as e:
        print(f"Error processing dataset: {e}")



# 使用例
root_dir = "/workspace/data/Cross_Validation/ex_20250122_preset_lat" # 入力データセットのルートディレクトリ
out_dir = "/workspace/data/Cross_Validation/ex_20250122_pre_lat"    # 出力先ディレクトリ

restructure_dataset(root_dir, out_dir)
