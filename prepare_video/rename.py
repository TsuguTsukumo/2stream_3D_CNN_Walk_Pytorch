import os

def append_parent_dir_to_all_files(folder_path):
    # フォルダ内のすべてのファイルを取得
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            
            # 親ディレクトリ名を取得
            parent_dir = os.path.basename(os.path.dirname(file_path))
            
            # ファイル名と拡張子を分ける
            file_name, file_ext = os.path.splitext(file)
            
            # 新しいファイル名を作成
            new_file_name = f"{file_name}_{parent_dir}{file_ext}"
            
            # 新しいファイルパスを生成
            new_file_path = os.path.join(root, new_file_name)
            
            # ファイル名を変更
            os.rename(file_path, new_file_path)
            print(f"Renamed: {file_path} -> {new_file_path}")

# 使用例: フォルダパスを指定
folder_path = "/workspace/data/split_pad_dataset_512/raw/LCS"
append_parent_dir_to_all_files(folder_path)
