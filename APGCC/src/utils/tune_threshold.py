import argparse
import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# === Đảm bảo đường dẫn module gốc APGCC hoạt động ===
project_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(project_dir) in ["src", "core", "preprocessing", "utils"]:
    project_dir = os.path.dirname(project_dir)
os.chdir(project_dir)

sys.path.insert(0, os.path.join(project_dir, "apgcc"))
sys.path.insert(0, project_dir)

from apgcc.config import cfg as apgcc_cfg
from apgcc.models import build_model
from datasets.shrimp_dataset import ShrimpDataset, shrimp_collate_fn
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser(description="Find optimal threshold for APGCC on Shrimp Dataset")
    parser.add_argument("--config", type=str, default="configs/SHRIMP_train.yml")
    parser.add_argument("--weight", type=str, default="outputs/shrimp/best_model.pth")
    parser.add_argument("--min_dist", type=float, default=50.0, help="NMS radius for removing duplicate points")
    return parser.parse_args()

def distance_nms(points, scores, min_dist=50.0):
    if len(points) == 0:
        return points, scores
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        idx = order[0]
        keep.append(idx)
        if len(order) == 1:
            break
        current_point = points[idx]
        other_points = points[order[1:]]
        dists = np.sqrt(np.sum((other_points - current_point) ** 2, axis=1))
        inds = np.where(dists > min_dist)[0]
        order = order[inds + 1]
    return points[keep], scores[keep]

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Cấu hình
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. Build Model
    apgcc_cfg.MODEL.ENCODER = cfg.get("MODEL", {}).get("ENCODER", "vgg16_bn")
    apgcc_cfg.MODEL.STRIDE = cfg["MODEL"]["STRIDE"]
    apgcc_cfg.MODEL.WEIGHT_DICT = {
        'loss_ce': cfg["TRAIN"]["LAMBDA1"],
        'loss_points': cfg["TRAIN"]["LAMBDA2"],
        'loss_aux': cfg["TRAIN"]["LAMBDA5"]
    }
    apgcc_cfg.MODEL.AUX_EN = False
    
    model = build_model(apgcc_cfg, training=False)

    # 3. Load Trọng số
    if not os.path.exists(args.weight):
        print(f"[!] Lỗi: Không tìm thấy file weight {args.weight}")
        return
        
    print(f"[*] Đang nạp trọng số từ {args.weight}...")
    # Xử lý cả dict có chứa key 'model' lẫn pure state dict
    state_dict = torch.load(args.weight, map_location=device)
    if "model" in state_dict:
        model.load_state_dict(state_dict["model"])
    else:
        model.load_state_dict(state_dict)
        
    model.to(device)
    model.eval()

    # 4. Load Dữ liệu Validation
    data_root = Path(cfg["DATA_ROOT"])
    val_dataset = ShrimpDataset(
        list_file=str(data_root / cfg["VAL_LIST"]),
        crop_size=None,
        is_train=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=shrimp_collate_fn,
    )

    print("\n" + "="*60)
    print(f"Bước 1/2: Nạp {len(val_dataset)} ảnh Validation nguyên gốc siêu nét (5 Megapixels)")
    print("Quá trình này tốn khoảng 3-6 phút vì chạy Full-Image qua VGG16 (dự đoán tất cả toạ độ)")
    print("="*60)

    # Chạy inference trên toàn bộ tập val đúng 1 CÚ DUY NHẤT để tiết kiệm 100 lần thời gian 
    cached_predictions = []
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(val_loader, desc="Inference mạng nơ-ron", ncols=80)):
            images = images.to(device)

            import torch.nn.functional as F
            h, w = images.shape[2:]
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            if pad_h > 0 or pad_w > 0:
                images = F.pad(images, (0, pad_w, 0, pad_h))

            outputs = model(images)
            pred_logits = outputs["pred_logits"]
            pred_points = outputs["pred_points"]
            
            for b in range(len(targets)):
                gt_count = len(targets[b]["point"])
                # Extract probabilities cho lớp 1 (Tôm)
                scores = pred_logits[b].softmax(dim=-1)[:, 1].cpu().numpy()
                pts = pred_points[b].cpu().numpy()
                
                cached_predictions.append({
                    "scores": scores,
                    "pts": pts,
                    "gt_count": gt_count
                })

    print(f"\n[DONE] Đã lưu cache {len(cached_predictions)} ảnh. Bắt đầu Sweep ngưỡng Threshold!")

    # 5. Quét qua hàng loạt các Threshold rất nhanh
    print("\n" + "="*50)
    print(f"Bước 2/2: Quét ngưỡng Threshold từ 0.05 đến 0.45 với NMS {args.min_dist}")
    print("="*50)

    thresholds = np.arange(0.05, 0.46, 0.01)
    
    best_mae = float("inf")
    best_thresh = 0.15
    best_rmse = float("inf")
    
    print(f"{'Threshold':<15} | {'MAE':<15} | {'RMSE':<15}")
    print("-" * 50)
    
    for thr in thresholds:
        pred_counts = []
        gt_counts = []
        
        for item in cached_predictions:
            scores = item["scores"]
            pts = item["pts"]
            gt_count = item["gt_count"]
            
            # 5.1 Chọn lọc điểm sương sương theo Threshold
            keep = scores > thr
            filt_pts = pts[keep]
            filt_scores = scores[keep]
            
            # 5.2 Xóa trùng lặp trên cùng 1 tôm bằng siêu tốc NMS
            final_pts, _ = distance_nms(filt_pts, filt_scores, min_dist=args.min_dist)
            
            pred_counts.append(len(final_pts))
            gt_counts.append(gt_count)

        # 5.3 Tính MAE và RMSE cho tập hợp Threshold này
        pred_counts = np.array(pred_counts)
        gt_counts   = np.array(gt_counts)
        
        mae  = np.mean(np.abs(pred_counts - gt_counts))
        rmse = np.sqrt(np.mean((pred_counts - gt_counts)**2))
        
        # Đánh dấu Threshold vĩ đại nhất
        marker = ""
        if mae < best_mae:
            best_mae = mae
            best_thresh = thr
            best_rmse = rmse
            marker = " <--- MỚI THE BEST!"
            
        print(f"{thr:<15.3f} | {mae:<15.3f} | {rmse:<15.3f} {marker}")

    print("\n" + "="*50)
    print(f"[HOÀN TẤT] THRESHOLD TỐI ƯU NHẤT LÀ: {best_thresh:.3f}")
    print(f"Bình quân mỗi bức ảnh (chứa khoảng 37 con) đếm sai đúng {best_mae:.3f} con (MAE)")
    print(f"Khuếch đại độ lệch RMSE: {best_rmse:.3f}")
    print("\nBạn hãy đổi tham số TEST.THRESHOLD thành mức tối ưu này trong configs/SHRIMP_train.yml nhé!")

if __name__ == "__main__":
    main()
