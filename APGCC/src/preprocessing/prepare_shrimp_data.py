"""
prepare_shrimp_data.py
======================
Chuẩn bị dataset tôm thẻ để tương thích với codebase APGCC gốc.

Mục tiêu:
  - Đọc ảnh .jpg từ images/ và nhãn "X Y" từ labels/
  - Chia train/val/test theo tỉ lệ cho trước
  - Tạo file .list mà APGCC sử dụng để load data

Cách chạy:
  python prepare_shrimp_data.py \
      --src_images ./images \
      --src_labels ./labels \
      --output_dir ./data/shrimp \
      --val_ratio 0.15 \
      --test_ratio 0.10
"""

import os
import sys
import shutil

# Đảm bảo working directory luôn là thư mục gốc
project_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(project_dir) in ["src", "core", "preprocessing", "utils"]:
    project_dir = os.path.dirname(project_dir)
os.chdir(project_dir)
sys.path.insert(0, os.path.join(project_dir, "apgcc"))
sys.path.insert(0, project_dir)

import random
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare shrimp dataset for APGCC")
    parser.add_argument("--src_images", required=True, help="Thư mục chứa ảnh .jpg gốc")
    parser.add_argument("--src_labels", required=True, help="Thư mục chứa nhãn .txt gốc")
    parser.add_argument("--output_dir", required=True, help="Thư mục output dataset")
    parser.add_argument("--val_ratio",  type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def verify_label(label_path):
    """Kiểm tra label có hợp lệ không (mỗi dòng = 2 số thực)."""
    with open(label_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Line {i+1} in {label_path}: expected 'X Y', got '{line}'")
        float(parts[0]); float(parts[1])   # sẽ raise nếu không phải số
    return len(lines)   # trả về số lượng tôm trong ảnh


def prepare_split(image_paths, label_paths, split_dir, list_path):
    """Copy ảnh và nhãn vào split_dir, ghi file .list."""
    os.makedirs(split_dir, exist_ok=True)
    with open(list_path, "w") as flist:
        for img_path, lbl_path in zip(image_paths, label_paths):
            img_name = os.path.basename(img_path)
            lbl_name = os.path.basename(lbl_path)
            dst_img = os.path.join(split_dir, img_name)
            dst_lbl = os.path.join(split_dir, lbl_name)
            shutil.copy2(img_path, dst_img)
            shutil.copy2(lbl_path, dst_lbl)
            # APGCC đọc list theo format: đường dẫn tuyệt đối / tương đối tới ảnh
            flist.write(dst_img + "\n")
    print(f"  → {len(image_paths)} mẫu → {split_dir}")


def main():
    args = parse_args()
    random.seed(args.seed)

    src_images = Path(args.src_images)
    src_labels = Path(args.src_labels)
    out = Path(args.output_dir)

    # --- Thu thập tất cả cặp (ảnh, nhãn) hợp lệ ---
    pairs = []
    missing_labels = []
    for img_file in sorted(src_images.glob("*.jpg")):
        stem = img_file.stem
        lbl_file = src_labels / f"{stem}.txt"
        if not lbl_file.exists():
            missing_labels.append(str(img_file))
            continue
        try:
            count = verify_label(str(lbl_file))
            pairs.append((str(img_file), str(lbl_file), count))
        except ValueError as e:
            print(f"[WARN] Bỏ qua {img_file.name}: {e}")

    if missing_labels:
        print(f"[WARN] {len(missing_labels)} ảnh không có nhãn tương ứng:")
        for p in missing_labels[:5]:
            print(f"       {p}")

    # Thống kê nhanh
    counts = [c for _, _, c in pairs]
    print(f"\n[INFO] Tổng mẫu hợp lệ: {len(pairs)}")
    print(f"[INFO] Số tôm/ảnh — min:{min(counts)}, max:{max(counts)}, "
          f"mean:{sum(counts)/len(counts):.1f}")

    # --- Shuffle và chia ---
    random.shuffle(pairs)
    n = len(pairs)
    n_test = max(1, int(n * args.test_ratio))
    n_val  = max(1, int(n * args.val_ratio))
    n_train = n - n_val - n_test

    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:n_train + n_val]
    test_pairs  = pairs[n_train + n_val:]

    print(f"\n[INFO] Chia split: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")

    # --- Copy vào output_dir ---
    for split_name, split_pairs in [("train", train_pairs),
                                     ("val",   val_pairs),
                                     ("test",  test_pairs)]:
        imgs = [p[0] for p in split_pairs]
        lbls = [p[1] for p in split_pairs]
        split_dir  = out / split_name
        list_path  = out / f"{split_name}.list"
        prepare_split(imgs, lbls, str(split_dir), str(list_path))

    print(f"\n[DONE] Dataset đã sẵn sàng tại: {out.resolve()}")
    print(f"       Sử dụng các file list:")
    print(f"         DATA_ROOT: {out.resolve()}")
    print(f"         TRAIN_LIST: {out / 'train.list'}")
    print(f"         VAL_LIST:   {out / 'val.list'}")
    print(f"         TEST_LIST:  {out / 'test.list'}")


if __name__ == "__main__":
    main()
