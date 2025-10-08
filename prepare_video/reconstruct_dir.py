import os
import re
import subprocess
import shutil
from pathlib import Path

# --- 設定項目 ---
# 入力ディレクトリのパス（ユーザーの例に基づいています）
INPUT_DIR = "/workspace/data/cropped_direction"
# 出力ディレクトリのパス
OUTPUT_DIR = "/workspace/data/output_segmented"
# ----------------

def get_video_duration(video_path):
    """ffprobeを使用して動画の再生時間（秒）を取得する"""
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        return float(result.stdout)
    except FileNotFoundError:
        print("エラー: ffprobeコマンドが見つかりません。FFmpegがインストールされ、PATHが通っているか確認してください。")
        return None
    except subprocess.CalledProcessError as e:
        print(f"エラー: {video_path} の再生時間を取得できませんでした。")
        print(f"ffprobe stderr: {e.stderr}")
        return None
    except ValueError:
        print(f"エラー: {video_path} から有効な再生時間を取得できませんでした。")
        return None

def process_videos(input_dir, output_dir):
    """
    指定されたディレクトリ構造を再構成し、動画をセグメント化する
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    print(f"処理を開始します...")
    print(f"入力ディレクトリ: {input_path}")
    print(f"出力ディレクトリ: {output_path}")

    # 処理対象の動画ファイルを検索
    video_files = list(input_path.glob("**/*.mp4"))
    if not video_files:
        print("エラー: 入力ディレクトリ内にmp4ファイルが見つかりませんでした。")
        return
        
    print(f"{len(video_files)}個の動画ファイルを検出しました。")

    for video_file in video_files:
        try:
            # --- 1. パスから情報を抽出 ---
            # front.mp4 or side.mp4
            view = video_file.stem
            if view not in ["front", "side"]:
                print(f"スキップ: ファイル名が'front.mp4'または'side.mp4'ではありません: {video_file}")
                continue

            # walk_Left_1461 など
            trial_dir_name = video_file.parent.name
            # 日付 (e.g., 20171211)
            date = video_file.parent.parent.name
            # カテゴリ (e.g., ASD, DHS, etc.)
            category = video_file.parent.parent.parent.name
            
            # non-ASDの場合、さらに1つ上の階層がカテゴリ名になる
            if category == "non-ASD":
                category = video_file.parent.parent.parent.parent.name


            # 試行ディレクトリ名から向きと番号を抽出 (例: walk_Left_1461)
            match = re.search(r"_(Right|Left)_(\d+)", trial_dir_name, re.IGNORECASE)
            if not match:
                print(f"スキップ: ファイルの親ディレクトリ名から向きと番号を抽出できませんでした: {trial_dir_name}")
                continue
            direction = match.group(1).capitalize() # Right or Left に統一
            number = match.group(2)

            print(f"\n処理中: {video_file}")

            # --- 2. 動画の長さを取得し、セグメント化を計算 ---
            duration = get_video_duration(video_file)
            if duration is None or duration < 1.0:
                print(f"スキップ: 動画の長さが1秒未満か、取得に失敗しました: {video_file}")
                continue

            total_segments = int(duration)
            remainder = duration - total_segments
            start_offset = remainder / 2

            # --- 3. 出力ディレクトリとファイル名を準備 ---
            output_subdir = output_path / view / category
            output_subdir.mkdir(parents=True, exist_ok=True)

            base_filename = f"{category}_{direction}_{date}_{number}"
            
            print(f"動画長: {duration:.2f}秒, セグメント数: {total_segments}, 開始オフセット: {start_offset:.2f}秒")

            # --- 4. FFmpegでセグメント化を実行 ---
            for i in range(total_segments):
                segment_num = i + 1
                segment_start_time = start_offset + i
                
                output_filename = f"{base_filename}_{segment_num}-{total_segments}.mp4"
                output_filepath = output_subdir / output_filename
                
                ffmpeg_command = [
                    "ffmpeg",
                    "-ss", str(segment_start_time),
                    "-i", str(video_file),
                    "-t", "1",
                    "-c", "copy",
                    "-y", # 常に上書き
                    str(output_filepath)
                ]
                
                # -c copy が失敗した場合のフォールバック
                try:
                    subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except subprocess.CalledProcessError:
                    print(f"警告: '-c copy'でのセグメント化に失敗しました。再エンコードを試みます: {output_filename}")
                    ffmpeg_command_reencode = [
                        "ffmpeg",
                        "-ss", str(segment_start_time),
                        "-i", str(video_file),
                        "-t", "1",
                        "-y",
                        str(output_filepath)
                    ]
                    subprocess.run(ffmpeg_command_reencode, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                print(f"  -> 作成完了: {output_filepath.name}")

        except Exception as e:
            print(f"予期せぬエラーが発生しました: {video_file}")
            print(str(e))

    print("\nすべての処理が完了しました。")


if __name__ == "__main__":
    # スクリプトのメイン処理
    process_videos(INPUT_DIR, OUTPUT_DIR)