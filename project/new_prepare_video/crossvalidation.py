import json
from sklearn.model_selection import train_test_split

# JSONファイルのパス
json_file_path = "/workspace/data/Video/Segment_video_ASDandNormal/dataset_info.json"

# JSONデータを読み込む
with open(json_file_path, "r") as f:
    data = json.load(f)

# 出力用ディレクトリとファイル名
output_file_path = "/workspace/data/Video/Segment_video_ASDandNormal/split_results.json"

# 各カテゴリごとにトレーニング・検証セットを分割
def create_split(random_seed):
    train_data = []
    valid_data = []
    category_totals = {}  # 各カテゴリの合計時間と人数を記録する辞書

    for category, people in data.items():
        # カテゴリ内のデータをリスト化
        category_data = [
            {"category": category, "person_id": person_id, "total_time": info["total_time"]}
            for person_id, info in people.items()
        ]
        # 4:1の比率で分割 (ランダムシードを使用)
        category_train, category_valid = train_test_split(
            category_data, test_size=0.2, random_state=random_seed
        )
        train_data.extend(category_train)
        valid_data.extend(category_valid)
        # カテゴリごとの合計時間と人数を計算
        category_totals[category] = {
            "total_time": sum(item["total_time"] for item in category_data),
            "total_people": len(category_data),
        }

    # 合計時間と合計人数を計算
    def calculate_totals(data):
        total_time = sum(item["total_time"] for item in data)
        total_people = len(data)
        return total_time, total_people

    train_total_time, train_total_people = calculate_totals(train_data)
    valid_total_time, valid_total_people = calculate_totals(valid_data)

    return {
        "train_data": train_data,
        "valid_data": valid_data,
        "train_total_time": train_total_time,
        "train_total_people": train_total_people,
        "valid_total_time": valid_total_time,
        "valid_total_people": valid_total_people,
        "category_totals": category_totals,
    }

# 5つの異なる分割パターンを作成
splits = []
for i in range(1, 6):
    splits.append(create_split(random_seed=i))

# 結果をJSONファイルとして保存
output_data = {f"Split_{idx+1}": split for idx, split in enumerate(splits)}
with open(output_file_path, "w") as f:
    json.dump(output_data, f, indent=4)

# 保存完了メッセージ
print(f"分割結果を {output_file_path} に保存しました！")
