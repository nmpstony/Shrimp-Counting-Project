"""
evaluate_shrimp.py
==================
Script đánh giá mô hình đã train và inference trên ảnh mới.

Cách dùng:

  # Đánh giá trên tập test
  python evaluate_shrimp.py \
      --config configs/SHRIMP_train.yml \
      --weight outputs/shrimp/best_model.pth \
      --split test

  # Inference trên ảnh bất kỳ
  python evaluate_shrimp.py \
      --config configs/SHRIMP_train.yml \
      --weight outputs/shrimp/best_model.pth \
      --image path/to/shrimp_image.jpg \
      --visualize
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T

# === Thêm APGCC vào path và set working dir = gốc ===
project_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(project_dir) in ["src", "core", "preprocessing", "utils"]:
    project_dir = os.path.dirname(project_dir)
os.chdir(project_dir)

sys.path.insert(0, os.path.join(project_dir, "apgcc"))
sys.path.insert(0, project_dir)

# Import từ code gốc APGCC
from apgcc.models import build_model           # VGG16 + IFI + APG heads

try:
    from apgcc.config import cfg as default_cfg
    from apgcc.config import update_config
    HAS_CONFIG_MODULE = True
except ImportError:
    HAS_CONFIG_MODULE = False

# Import custom dataset
from datasets.shrimp_dataset import ShrimpDataset, shrimp_collate_fn
from torch.utils.data import DataLoader

def load_config(config_path):
    """Load YAML config thủ công nếu APGCC không có config module."""
    import yaml
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg

# ============================================================
# NMS dựa trên khoảng cách (loại bỏ điểm trùng lặp)
# ============================================================

def distance_nms(points, scores, min_dist=20.0):
    """
    Giữ lại điểm có score cao nhất trong vùng bán kính min_dist.
    """
    if len(points) == 0:
        return points, scores

    # Sort theo score giảm dần
    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        idx = order[0]
        keep.append(idx)
        
        if len(order) == 1:
            break
            
        # Tính khoảng cách từ điểm hiện tại tới các điểm còn lại
        current_point = points[idx]
        other_points = points[order[1:]]
        
        dists = np.sqrt(np.sum((other_points - current_point) ** 2, axis=1))
        
        # Chỉ giữ lại các điểm nằm ngoài bán kính min_dist
        inds = np.where(dists > min_dist)[0]
        order = order[inds + 1]

    return points[keep], scores[keep]


# ============================================================
# Inference trên một ảnh
# ============================================================

def inference_single_image(model, img_path, device, threshold=0.5):
    """
    Chạy inference trên một ảnh và trả về tọa độ tôm dự đoán.

    Returns:
        pred_points: np.ndarray [N, 2] — tọa độ (x, y) của từng con tôm
        scores     : np.ndarray [N]    — confidence score
        count      : int               — số con tôm đếm được
    """
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])

    # Load ảnh
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h0, w0  = img_rgb.shape[:2]

    # Preprocess
    img_pil    = Image.fromarray(img_rgb)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)  # [1, 3, H, W]

    # Pad ảnh thành bội số của 16
    import torch.nn.functional as F
    _h, _w = img_tensor.shape[2:]
    pad_h = (16 - _h % 16) % 16
    pad_w = (16 - _w % 16) % 16
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h))

    # Forward
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)

    pred_logits = outputs["pred_logits"][0]   # [num_queries, 2]
    pred_points = outputs["pred_points"][0]   # [num_queries, 2]

    scores = pred_logits.softmax(dim=-1)[:, 1].cpu().numpy()  # [num_queries]
    points = pred_points.cpu().numpy()                        # [num_queries, 2]

    # Filter theo threshold
    mask  = scores > threshold
    filtered_points = points[mask]
    filtered_scores = scores[mask]

    # Áp dụng Distance NMS để lọc bỏ các điểm chồng lấn
    final_points, final_scores = distance_nms(filtered_points, filtered_scores, min_dist=50.0)

    return final_points, final_scores, len(final_points)


def visualize_predictions(img_path, pred_points, gt_points=None,
                           save_path=None, threshold=0.5):
    """
    Vẽ kết quả dự đoán lên ảnh.
    - Chấm đỏ: dự đoán
    - Chấm xanh: ground truth (nếu có)
    """
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    # Vẽ GT points (xanh lá)
    if gt_points is not None and len(gt_points) > 0:
        for (x, y) in gt_points:
            cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Vẽ pred points (đỏ)
    pred_points = pred_points.reshape(-1, 2) if len(pred_points) > 0 else []
    for (x, y) in pred_points:
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Label số lượng
    gt_text   = f"GT: {len(gt_points)}"  if gt_points is not None else ""
    pred_text = f"Pred: {len(pred_points)}"
    cv2.putText(img, pred_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    if gt_text:
        cv2.putText(img, gt_text,   (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    if save_path:
        cv2.imwrite(str(save_path), img)
        print(f"  Saved: {save_path}")
    else:
        cv2.imshow("APGCC Shrimp Prediction", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img


# ============================================================
# Evaluate trên dataset
# ============================================================

def evaluate_dataset(model, dataloader, device, threshold, output_dir=None):
    """
    Đánh giá MAE, RMSE trên toàn bộ tập val/test.
    In kết quả chi tiết từng ảnh.
    """
    model.eval()

    results  = []
    pred_all = []
    gt_all   = []

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)

        # Pad ảnh thành bội số của 16
        import torch.nn.functional as F
        _h, _w = images.shape[2:]
        pad_h = (16 - _h % 16) % 16
        pad_w = (16 - _w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            images = F.pad(images, (0, pad_w, 0, pad_h))

        with torch.no_grad():
            outputs = model(images)

        for b in range(len(targets)):
            gt_count  = len(targets[b]["point"])
            gt_all.append(gt_count)

            scores      = outputs["pred_logits"][b].softmax(dim=-1)[:, 1].cpu().numpy()
            pts         = outputs["pred_points"][b].cpu().numpy()
            
            mask = scores > threshold
            filtered_pts = pts[mask]
            filtered_scores = scores[mask]
            
            # Áp dụng NMS
            final_pts, _ = distance_nms(filtered_pts, filtered_scores, min_dist=50.0)
            pred_count = len(final_pts)
            pred_all.append(pred_count)

            error = abs(pred_count - gt_count)
            results.append({
                "batch": batch_idx,
                "gt":    gt_count,
                "pred":  pred_count,
                "error": error,
            })
            print(f"  Image {batch_idx*dataloader.batch_size + b + 1:4d}: "
                  f"GT={gt_count:3d}  Pred={pred_count:3d}  |Error|={error:3d}")

    # Tổng kết
    errors  = [r["error"] for r in results]
    sq_errs = [(r["pred"] - r["gt"])**2 for r in results]

    mae  = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(sq_errs)))

    print(f"\n{'='*50}")
    print(f"  TOTAL IMAGES : {len(results)}")
    print(f"  MAE          : {mae:.4f}")
    print(f"  RMSE         : {rmse:.4f}")
    print(f"  Max Error    : {max(errors)}")
    print(f"  Min Error    : {min(errors)}")
    print(f"{'='*50}\n")

    # Lưu kết quả JSON
    if output_dir:
        result_path = Path(output_dir) / "eval_results.json"
        with open(result_path, "w") as f:
            json.dump({"mae": mae, "rmse": rmse, "details": results}, f, indent=2)
        print(f"  Results saved: {result_path}")

    return mae, rmse


# ============================================================
# Main
# ============================================================

def main():
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/SHRIMP_train.yml")
    parser.add_argument("--weight",    required=True)
    parser.add_argument("--split",     default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--image",     default="",
                        help="Inference trên một ảnh cụ thể")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--device",    default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    from apgcc.config import cfg as apgcc_cfg
    apgcc_cfg.MODEL.ENCODER = cfg["MODEL"]["BACKBONE"]
    apgcc_cfg.MODEL.STRIDE = cfg["MODEL"]["STRIDE"]
    apgcc_cfg.MODEL.WEIGHT_DICT = {
        'loss_ce': cfg["TRAIN"]["LAMBDA1"],
        'loss_points': cfg["TRAIN"]["LAMBDA2"],
        'loss_aux': cfg["TRAIN"]["LAMBDA5"]
    }
    apgcc_cfg.MODEL.AUX_EN = False
    
    model = build_model(apgcc_cfg, training=False)
    ckpt = torch.load(args.weight, map_location=device)
    # Hỗ trợ cả format state_dict thuần và dict có 'model' key
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model = model.to(device)
    print(f"[INFO] Loaded checkpoint: {args.weight}")

    # --- Mode: Single image inference ---
    if args.image:
        print(f"\n[Inference] {args.image}")
        pred_points, scores, count = inference_single_image(
            model, args.image, device, args.threshold
        )
        print(f"  → Số tôm đếm được: {count}")
        print(f"  → Số tôm đếm được (threshold={args.threshold}): {count}")
        if count > 0:
            print(f"  → Toạ độ 5 con đầu tiên:\n{pred_points[:5]}")
        
        # Lấy top 5 scores thực tế cao nhất từ toàn bộ mảng scores
        all_scores = np.sort(scores)[::-1]
        print(f"  → Top-5 max scores (dù dưới threshold): {all_scores[:5]}")

        if args.visualize:
            # Load GT nếu có
            gt_path = os.path.splitext(args.image)[0] + ".txt"
            gt_pts  = None
            if os.path.exists(gt_path):
                gt_pts = []
                with open(gt_path) as f:
                    for line in f:
                        p = line.strip().split()
                        if len(p) == 2:
                            gt_pts.append([float(p[0]), float(p[1])])
                gt_pts = np.array(gt_pts)

            result_dir = Path("example") / "result"
            result_dir.mkdir(parents=True, exist_ok=True)
            save = result_dir / (Path(args.image).stem + "_pred.jpg")
            visualize_predictions(args.image, pred_points, gt_pts,
                                   save_path=str(save))
        return

    # --- Mode: Dataset evaluation ---
    data_root = Path(cfg["DATA_ROOT"])
    list_file  = data_root / cfg[f"{args.split.upper()}_LIST"]

    dataset = ShrimpDataset(
        list_file=str(list_file),
        crop_size=None,
        is_train=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=shrimp_collate_fn)

    print(f"\n[Eval] Split={args.split}, #images={len(dataset)}")
    evaluate_dataset(model, loader, device, args.threshold,
                     output_dir=cfg["OUTPUT_DIR"])


if __name__ == "__main__":
    main()
