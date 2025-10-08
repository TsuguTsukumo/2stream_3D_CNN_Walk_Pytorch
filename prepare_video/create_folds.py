import os
import re
from pathlib import Path
import sys
import shutil

# --- 設定項目 ---
# 1. 最初のスクリプトで生成されたセグメント化済みビデオが格納されているディレクトリ
PROCESSED_DIR = "/workspace/data/output_segmented"

# 2. train/valのFold構成を定義している学習用ディレクトリ
LEARNING_DIR = "/workspace/data/Cross_Validation/ex_20250116_ap_organized"

# 3. 新しくFold構造を作成する出力先ディレクトリの「ベース名」
# ★★★ 変更点 ★★★
# このベース名に "_front" と "_side" が自動的に追加されます
FINAL_OUTPUT_DIR_BASE = "/workspace/data/out_dir_ver2_classified"

# ファイルの配置方法: 'symlink' (推奨) または 'copy'
LINK_METHOD = 'symlink'
# ----------------

def get_fold_validation_dates(learning_dir: Path) -> dict:
    """
    学習用ディレクトリをスキャンし、各Foldのvalidationに使われる日付のセットを返す
    """
    print(f"学習用ディレクトリをスキャン中: {learning_dir}")
    fold_dates = {}
    
    fold_paths = sorted([d for d in learning_dir.iterdir() if d.is_dir() and d.name.startswith('fold')])
    if not fold_paths:
        print(f"エラー: '{learning_dir}'内に 'fold' で始まるディレクトリが見つかりません。")
        sys.exit(1)

    for fold_path in fold_paths:
        val_dir = fold_path / 'val'
        if not val_dir.is_dir():
            print(f"警告: {fold_path} 内に 'val' ディレクトリが見つかりません。スキップします。")
            continue

        print(f"  - {fold_path.name} のvalidation日付を収集中...")
        
        val_files = val_dir.rglob('*.mp4')
        date_set = set()
        for f in val_files:
            match = re.search(r'(\d{8})', f.name)
            if match:
                date_set.add(match.group(1))
        
        if date_set:
            fold_dates[fold_path.name] = date_set
            print(f"    -> {len(date_set)}個のユニークな日付を発見しました。")
        else:
            print(f"    -> validationファイルから日付を抽出できませんでした。")

    return fold_dates

def get_all_source_dates(processed_dir: Path) -> set:
    """
    セグメント化済みデータディレクトリから、存在するすべての日付のセットを返す
    """
    print(f"\n全ソースデータの日付をスキャン中: {processed_dir}")
    all_dates = set()
    
    source_files = processed_dir.rglob('*.mp4')
    for f in source_files:
        match = re.search(r'(\d{8})', f.name)
        if match:
            all_dates.add(match.group(1))
    
    print(f"-> {len(all_dates)}個のユニークな日付をソースデータ全体から発見しました。")
    return all_dates

def reorganize_by_fold_with_classification():
    """
    メイン処理: データをFold構造に再編成し、ASD/ASD_notに分類する
    """
    processed_path = Path(PROCESSED_DIR)
    learning_path = Path(LEARNING_DIR)

    if not processed_path.is_dir():
        print(f"エラー: セグメント化済みデータディレクトリが見つかりません: {processed_path}")
        return
    if not learning_path.is_dir():
        print(f"エラー: 学習用ディレクトリが見つかりません: {learning_path}")
        return

    fold_to_val_dates = get_fold_validation_dates(learning_path)
    if not fold_to_val_dates:
        print("エラー: 学習用ディレクトリから有効なFold情報を取得できませんでした。処理を中断します。")
        return

    all_source_dates = get_all_source_dates(processed_path)
    if not all_source_dates:
        print("エラー: ソースデータから日付情報を取得できませんでした。処理を中断します。")
        return
    
    # ★★★ 変更点: ここから ★★★
    # 'front' と 'side' を順番に処理するループ
    for view in ['front', 'side']:
        
        final_output_path = Path(f"{FINAL_OUTPUT_DIR_BASE}_{view}")
        source_view_path = processed_path / view

        if not source_view_path.is_dir():
            print(f"\n警告: ソースディレクトリ '{source_view_path}' が見つかりません。'{view}' の処理をスキップします。")
            continue

        print(f"\n--- '{view}' ビデオの再編成を開始します ---")
        print(f"出力先: {final_output_path}")

        source_files = list(source_view_path.rglob('*.mp4'))
        total_files = len(source_files)
        print(f"{total_files}個の '{view}' セグメントファイルを処理します...")

        for i, source_file in enumerate(source_files):
            if (i + 1) % 500 == 0:
                print(f"  進捗: {i + 1} / {total_files}")

            file_date_match = re.search(r'(\d{8})', source_file.name)
            if not file_date_match:
                continue
            
            file_date = file_date_match.group(1)

            if source_file.name.startswith('ASD_'):
                classification_subdir = 'ASD'
            else:
                classification_subdir = 'ASD_not'

            for fold_name, val_dates in fold_to_val_dates.items():
                train_dates = all_source_dates - val_dates
                
                base_dest_dir = None
                if file_date in val_dates:
                    base_dest_dir = final_output_path / fold_name / 'val'
                elif file_date in train_dates:
                    base_dest_dir = final_output_path / fold_name / 'train'
                
                if base_dest_dir:
                    final_dest_dir = base_dest_dir / classification_subdir
                    final_dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = final_dest_dir / source_file.name
                    
                    if dest_path.exists() or dest_path.is_symlink():
                        continue

                    try:
                        if LINK_METHOD == 'symlink':
                            os.symlink(source_file, dest_path)
                        elif LINK_METHOD == 'copy':
                            shutil.copy2(source_file, dest_path)
                    except Exception as e:
                        print(f"エラー: {dest_path} の作成に失敗しました。")
                        print(e)
    # ★★★ 変更点: ここまで ★★★

    print("\n--- 全ての処理が完了しました ---")

if __name__ == "__main__":
    reorganize_by_fold_with_classification()