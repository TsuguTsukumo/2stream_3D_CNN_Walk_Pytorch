import os
import re

def rename_directories(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            old_path = os.path.join(dirpath, dirname)
            
            # リネーム処理: 'valid' -> 'val'
            if dirname == "valid":
                new_path = os.path.join(dirpath, "val")
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")

            # リネーム処理: 'fold_1' -> 'fold0', 'fold_2' -> 'fold1', ...
            match = re.match(r"fold_(\d+)", dirname)
            if match:
                new_name = f"fold{int(match.group(1)) - 1}"  # fold_1 -> fold0, fold_2 -> fold1
                new_path = os.path.join(dirpath, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")

            # リネーム処理: 'Normal' -> 'ASD_not'
            if dirname == "Normal":
                new_path = os.path.join(dirpath, "ASD_not")
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")


# 使用例
root_directory = "/workspace/data/Cross_Validation/ex_20250122_ap"  # ルートディレクトリを指定
rename_directories(root_directory)
