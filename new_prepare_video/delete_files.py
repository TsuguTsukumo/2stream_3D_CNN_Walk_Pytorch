import os

# 親ディレクトリのパスを指定
parent_directory = '/workspace/data/data/new_data_cropped_cross_validation/fold2'


# ディレクトリ内のすべてのフォルダとそのサブフォルダを再帰的に調べる
for root, dirs, files in os.walk(parent_directory):
    folder_name = os.path.basename(root)
    file_count = len(files)
    print(f'{folder_name} に含まれるファイル数: {file_count}')
