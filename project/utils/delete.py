import os
import random
import shutil
from pathlib import Path

def thin_out_synchronized_datasets(inp_ap_dir, inp_lat_dir, out_ap_dir, out_lat_dir, ratio, seed=42):
    random.seed(seed)
    inp_ap_dir = Path(inp_ap_dir)
    inp_lat_dir = Path(inp_lat_dir)
    out_ap_dir = Path(out_ap_dir)
    out_lat_dir = Path(out_lat_dir)

    for fold_dir in inp_ap_dir.glob('fold*'):
        for split in ['train', 'val']:
            for cls in ['ASD', 'ASD_not']:
                src_ap = fold_dir / split / cls
                src_lat = inp_lat_dir / fold_dir.name / split / cls

                dest_ap = out_ap_dir / fold_dir.name / split / cls
                dest_lat = out_lat_dir / fold_dir.name / split / cls
                dest_ap.mkdir(parents=True, exist_ok=True)
                dest_lat.mkdir(parents=True, exist_ok=True)

                files_ap = sorted(src_ap.glob('*.mp4'))
                if not files_ap:
                    continue

                if split == 'train':
                    selected_files = random.sample(files_ap, int(len(files_ap) * ratio))
                else:  # val: すべて使用
                    selected_files = files_ap

                for f_ap in selected_files:
                    f_lat = src_lat / f_ap.name
                    if f_lat.exists():
                        shutil.copy(f_ap, dest_ap / f_ap.name)
                        shutil.copy(f_lat, dest_lat / f_lat.name)

                print(f"[{fold_dir.name}/{split}/{cls}] {len(files_ap)} -> {len(selected_files)} files copied.")

thin_out_synchronized_datasets(
    inp_ap_dir= "/workspace/data/Cross_Validation/ex_20250116_ap_organized",
    inp_lat_dir="/workspace/data/Cross_Validation/ex_20250116_lat_organized",
    out_ap_dir="/workspace/data/Cross_Validation/40/ap",
    out_lat_dir="/workspace/data/Cross_Validation/40/lat",
    ratio=0.4
)
