import os

def rename_files(root_dir):
    # ディレクトリ内の全てのファイルを再帰的にチェック
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # "lat_" または "ap_" で始まるファイルをターゲットにする
            if filename.startswith('lat_'):
                new_name = 'lat.mp4'
            elif filename.startswith('ap_'):
                new_name = 'ap.mp4'
            else:
                continue  # "lat_" または "ap_" で始まらないファイルは無視

            # ファイルのフルパス
            old_file = os.path.join(dirpath, filename)
            new_file = os.path.join(dirpath, new_name)
            
            # 名前が異なればリネームを実行
            if old_file != new_file:
                os.rename(old_file, new_file)
                print(f'Renamed: {old_file} -> {new_file}')

# 使用例
root_directory = '/workspace/data/Video/Segment_video/TEST_Normal_9_selected'  # データセットのルートディレクトリ
rename_files(root_directory)
