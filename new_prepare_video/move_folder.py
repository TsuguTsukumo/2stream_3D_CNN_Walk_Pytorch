import os
import shutil

# 元のデータセットディレクトリ
input_dir = '/workspace/data/Cross_Validation/ex_20250116_lat'

output_dir = '/workspace/data/Cross_Validation/ex_20250116_lat_organized'

# 新しいディレクトリ構造を作成
def create_new_structure():
    # foldの番号を取得
    folds = [fold for fold in os.listdir(input_dir) if fold.startswith('fold_')]

    for fold in folds:
        fold_path = os.path.join(input_dir, fold)
        
        # 出力先にfoldのディレクトリを作成
        output_fold_path = os.path.join(output_dir, fold)
        if not os.path.exists(output_fold_path):
            os.makedirs(output_fold_path)
        
        for phase in ['train', 'valid']:
            # 新しいディレクトリの作成
            new_fold_phase_path = os.path.join(output_fold_path, phase)
            if not os.path.exists(new_fold_phase_path):
                os.makedirs(new_fold_phase_path)

            # ASDとASD_notのディレクトリ作成
            for new_folder in ['ASD', 'ASD_not']:
                new_class_folder_path = os.path.join(new_fold_phase_path, new_folder)
                if not os.path.exists(new_class_folder_path):
                    os.makedirs(new_class_folder_path)

            # 各クラスの動画ファイルを新しいフォルダにコピー
            for class_name in ['ASD', 'DHS', 'LCS', 'HipOA']:
                class_folder = os.path.join(fold_path, phase, class_name)
                if os.path.exists(class_folder):
                    # 'ASD'の場合は'ASD'に、その他の疾患は'ASD_not'にコピー
                    if class_name == 'ASD':
                        target_folder = os.path.join(new_fold_phase_path, 'ASD')
                    else:
                        target_folder = os.path.join(new_fold_phase_path, 'ASD_not')

                    for video in os.listdir(class_folder):
                        src_video_path = os.path.join(class_folder, video)
                        if os.path.isfile(src_video_path):
                            shutil.copy(src_video_path, target_folder)

create_new_structure()
