"""
evaluate.py – Đánh giá MAE / MAPE và Inference Time trên tập Validation
=========================================================================
Cách dùng:
    python src/evaluate.py
    python src/evaluate.py --model runs/segment/train/weights/best.pt
    python src/evaluate.py --model runs/segment/train/weights/best.pt --conf 0.5 --device cpu
"""

import os
import sys
import time
import argparse
import glob
import numpy as np

# ── Đảm bảo working directory luôn là thư mục gốc YOLOv8 ─────────────────────
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_dir)

from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────────────────────
def get_ground_truth(img_path: str) -> int:
    """
    Đọc file label YOLO tương ứng và trả về số đối tượng ground-truth.
    Quy ước đường dẫn: .../images/val/xxx.jpg → .../labels/val/xxx.txt
    """
    lbl_path = img_path.replace("images", "labels", 1)
    lbl_path = os.path.splitext(lbl_path)[0] + ".txt"

    if not os.path.exists(lbl_path):
        return -1  # -1 = không có nhãn

    with open(lbl_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return len(lines)


# ──────────────────────────────────────────────────────────────────────────────
def evaluate(model_path: str, val_img_dir: str, conf: float, device: str):
    print(f"\n{'='*60}")
    print(f"  YOLOv8 Counting Evaluation  –  Val set")
    print(f"{'='*60}")
    print(f"  Model  : {model_path}")
    print(f"  Val dir: {val_img_dir}")
    print(f"  Conf   : {conf}   |   Device: {device}")
    print(f"{'='*60}\n")

    # ── 1. Load model ─────────────────────────────────────────────────────────
    print("[INFO] Đang nạp model...")
    model = YOLO(model_path)

    # ── 2. Lấy danh sách ảnh val ──────────────────────────────────────────────
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(os.path.join(val_img_dir, ext)))
    img_paths.sort()

    if not img_paths:
        print(f"[ERROR] Không tìm thấy ảnh nào trong '{val_img_dir}'")
        sys.exit(1)

    print(f"[INFO] Tìm thấy {len(img_paths)} ảnh trong tập val.\n")

    # ── 3. Inference từng ảnh & thu thập kết quả ──────────────────────────────
    gt_counts   = []
    pred_counts = []
    inf_times   = []   # ms / ảnh
    skipped     = 0

    print(f"{'No.':<6} {'File':<30} {'GT':>5} {'Pred':>6} {'|Error|':>8} {'Time(ms)':>10}")
    print("-" * 70)

    for idx, img_path in enumerate(img_paths, 1):
        gt = get_ground_truth(img_path)
        if gt == -1:
            skipped += 1
            continue  # bỏ qua ảnh không có nhãn

        # Đo thời gian inference thuần (không tính I/O đọc ảnh)
        t_start = time.perf_counter()
        results = model.predict(
            source=img_path,
            conf=conf,
            device=device,
            verbose=False,
            show=False,
        )
        t_end = time.perf_counter()

        inf_ms  = (t_end - t_start) * 1000
        result  = results[0]
        pred    = len(result.masks) if result.masks is not None else 0
        abs_err = abs(pred - gt)

        gt_counts.append(gt)
        pred_counts.append(pred)
        inf_times.append(inf_ms)

        fname = os.path.basename(img_path)[:28]
        print(f"{idx:<6} {fname:<30} {gt:>5} {pred:>6} {abs_err:>8} {inf_ms:>9.1f}")

    print("-" * 70)

    if not gt_counts:
        print("[WARN] Không có ảnh nào có file nhãn tương ứng. Kiểm tra lại thư mục labels/val.")
        sys.exit(1)

    # ── 4. Tính các chỉ số ────────────────────────────────────────────────────
    gt_arr   = np.array(gt_counts,   dtype=float)
    pred_arr = np.array(pred_counts, dtype=float)
    err_arr  = np.abs(pred_arr - gt_arr)

    mae  = np.mean(err_arr)
    # MAPE: bỏ qua ảnh có gt = 0 để tránh chia-cho-0
    mask_nonzero = gt_arr > 0
    if mask_nonzero.sum() > 0:
        mape = np.mean(err_arr[mask_nonzero] / gt_arr[mask_nonzero]) * 100
    else:
        mape = float("nan")

    avg_inf = np.mean(inf_times)
    med_inf = np.median(inf_times)
    min_inf = np.min(inf_times)
    max_inf = np.max(inf_times)
    fps     = 1000.0 / avg_inf if avg_inf > 0 else 0

    n_valid = len(gt_counts)
    n_exact = int(np.sum(err_arr == 0))

    # ── 5. In kết quả tổng hợp ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  KẾT QUẢ ĐÁNH GIÁ  ({n_valid}/{len(img_paths)} ảnh có nhãn)")
    print(f"{'='*60}")
    print(f"  Counting Metrics")
    print(f"  {'MAE':<25}: {mae:.4f}")
    print(f"  {'MAPE':<25}: {mape:.2f} %")
    print(f"  {'Exact match (|err|=0)':<25}: {n_exact}/{n_valid} ({100*n_exact/n_valid:.1f} %)")
    print(f"\n  Inference Time")
    print(f"  {'Mean':<25}: {avg_inf:.2f} ms  ({fps:.1f} FPS)")
    print(f"  {'Median':<25}: {med_inf:.2f} ms")
    print(f"  {'Min':<25}: {min_inf:.2f} ms")
    print(f"  {'Max':<25}: {max_inf:.2f} ms")
    if skipped:
        print(f"\n  [WARN] Bỏ qua {skipped} ảnh không tìm thấy file nhãn tương ứng.")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Đánh giá MAE / MAPE / Inference Time của YOLOv8 trên tập Validation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runs/segment/train/weights/best.pt",
        help="Đường dẫn tới file .pt đã train (mặc định: runs/segment/train/weights/best.pt)",
    )
    parser.add_argument(
        "--val",
        type=str,
        default="dataset/images/val",
        help="Thư mục chứa ảnh val (mặc định: dataset/images/val)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Ngưỡng confidence (mặc định: 0.5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device để chạy: '0' (GPU đầu tiên), 'cpu' (mặc định: '0')",
    )

    args = parser.parse_args()
    evaluate(
        model_path  = args.model,
        val_img_dir = args.val,
        conf        = args.conf,
        device      = args.device,
    )
