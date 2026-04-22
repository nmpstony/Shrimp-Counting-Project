"""
train_shrimp.py
===============
Script huấn luyện APGCC cho bài toán đếm tôm thẻ.

Cách dùng:
    # Training từ đầu
    python train_shrimp.py --config configs/SHRIMP_train.yml

    # Resume từ checkpoint
    python train_shrimp.py --config configs/SHRIMP_train.yml \
                           --resume outputs/shrimp/checkpoint_epoch50.pth

    # Training với VGG16 pretrained
    python train_shrimp.py --config configs/SHRIMP_train.yml \
                           --pretrained_backbone vgg16_bn

Lưu ý tích hợp với code gốc:
    Script này gọi lại các module gốc của APGCC (trong thư mục apgcc/),
    chỉ thay phần DataLoader bằng ShrimpDataset của bạn.
"""

import os
import sys
import time
import math
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

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


# ============================================================
# Config loader (fallback nếu không dùng APGCC config module)
# ============================================================

def load_config(config_path):
    """Load YAML config thủ công nếu APGCC không có config module."""
    import yaml
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


# ============================================================
# Metrics — MAE và RMSE (Paper Section 5.2)
# ============================================================

class AverageMeter:
    """Theo dõi giá trị trung bình và tổng tích lũy."""
    def __init__(self, name=""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def compute_counting_metrics(pred_counts, gt_counts):
    """
    Tính MAE và RMSE cho một epoch/batch.

    Args:
        pred_counts: list[float] — số tôm dự đoán mỗi ảnh
        gt_counts  : list[int]   — số tôm thật mỗi ảnh

    Returns:
        mae  : float — Mean Absolute Error
        rmse : float — Root Mean Square Error (tác giả gọi là MSE trong paper
               nhưng thực ra là RMSE: sqrt(mean(err^2)))
    """
    errors = [abs(p - g) for p, g in zip(pred_counts, gt_counts)]
    sq_err = [(p - g) ** 2 for p, g in zip(pred_counts, gt_counts)]
    mae  = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(sq_err)))
    return mae, rmse


# ============================================================
# Build optimizer — backbone lr nhỏ hơn head lr
# ============================================================

def build_optimizer(model, cfg):
    """
    Tách backbone params (lr thấp) và head params (lr cao).

    Lý do: VGG16 backbone đã pretrained trên ImageNet, cần fine-tune
    nhẹ nhàng (lr × 0.1) để không phá vỡ feature đã học.
    APG heads và IFI MLP học từ đầu → dùng lr đầy đủ.
    """
    lr        = cfg["TRAIN"]["LR"]
    backbone_lr = lr * cfg["TRAIN"]["BACKBONE_LR_RATIO"]
    wd        = cfg["TRAIN"]["WEIGHT_DECAY"]

    backbone_params = []
    head_params     = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Backbone params: thường có tên chứa 'backbone' hoặc 'encoder'
        if "backbone" in name or "encoder" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
        {"params": head_params,     "lr": lr,          "name": "head"},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)
    print(f"[Optimizer] AdamW — backbone_lr={backbone_lr:.2e}, head_lr={lr:.2e}, "
          f"weight_decay={wd}")
    return optimizer


# ============================================================
# Training loop một epoch
# ============================================================

def train_one_epoch(model, criterion, dataloader, optimizer, device, epoch, cfg, writer=None):
    model.train()
    criterion.train()

    loss_meter   = AverageMeter("Loss")
    loss_cls_m   = AverageMeter("L_cls")
    loss_pts_m   = AverageMeter("L_pts")

    log_interval = cfg["LOG"]["INTERVAL"]
    start_time   = time.time()

    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device)

        # Chuyển targets lên device
        targets_dev = []
        for tgt in targets:
            targets_dev.append({
                k: v.to(device) for k, v in tgt.items()
            })

        # --- Forward pass ---
        outputs = model(images)

        # --- Tính loss qua APGCC criterion ---
        loss_dict = criterion(outputs, targets_dev)
        weight_dict = criterion.weight_dict
        total_loss = sum(loss_dict[k] * weight_dict[k]
                         for k in loss_dict if k in weight_dict)

        # --- Backward ---
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        # --- Logging ---
        bs = images.size(0)
        loss_meter.update(total_loss.item(), bs)
        loss_cls_m.update(loss_dict.get("loss_ce", torch.tensor(0)).item(), bs)
        loss_pts_m.update(loss_dict.get("loss_points", torch.tensor(0)).item(), bs)

        if (i + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            logging.info(
                f"Epoch [{epoch}] Iter [{i+1}/{len(dataloader)}] "
                f"Loss={loss_meter.avg:.4f} "
                f"(ce={loss_cls_m.avg:.4f}, pts={loss_pts_m.avg:.4f}) "
                f"Time={elapsed:.1f}s"
            )

    # Ghi TensorBoard
    if writer is not None:
        writer.add_scalar("Train/Loss",    loss_meter.avg, epoch)
        writer.add_scalar("Train/L_ce",    loss_cls_m.avg, epoch)
        writer.add_scalar("Train/L_pts",   loss_pts_m.avg, epoch)

    return loss_meter.avg


# ============================================================
# Validation — tính MAE và RMSE
def distance_nms(points, scores, min_dist=20.0):
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

@torch.no_grad()
def evaluate(model, dataloader, device, cfg, epoch=0, writer=None):
    """
    Đánh giá mô hình trên tập val/test.
    """
    model.eval()

    threshold    = cfg["TEST"]["THRESHOLD"]
    pred_counts  = []
    gt_counts    = []
    image_errors = []

    for images, targets in dataloader:
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
            gt_count  = len(targets[b]["point"])
            gt_counts.append(gt_count)

            scores = pred_logits[b].softmax(dim=-1)[:, 1].cpu().numpy()
            pts = pred_points[b].cpu().numpy()
            
            mask = scores > threshold
            filtered_pts = pts[mask]
            filtered_scores = scores[mask]
            
            final_pts, _ = distance_nms(filtered_pts, filtered_scores, min_dist=50.0)
            pred_count = len(final_pts)
            pred_counts.append(pred_count)

            image_errors.append(abs(pred_count - gt_count))

    mae, rmse = compute_counting_metrics(pred_counts, gt_counts)

    logging.info(
        f"[Eval Epoch {epoch}] "
        f"MAE={mae:.2f}  RMSE={rmse:.2f}  "
        f"(#images={len(pred_counts)}, threshold={threshold})"
    )

    if writer is not None:
        writer.add_scalar("Val/MAE",  mae,  epoch)
        writer.add_scalar("Val/RMSE", rmse, epoch)

    return mae, rmse


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train APGCC for Shrimp Counting")
    parser.add_argument("--config",   default="configs/SHRIMP_train.yml")
    parser.add_argument("--resume",   default="", help="Checkpoint để resume")
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--pretrained_backbone", default="vgg16_bn",
                        choices=["vgg16_bn", "vgg16", "resnet50"])
    args = parser.parse_args()

    # --- Config ---
    cfg = load_config(args.config)

    # --- Output dir ---
    out_dir = Path(cfg["OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "train.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ]
    )

    # --- Device ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Datasets & DataLoaders ---
    data_root = Path(cfg["DATA_ROOT"])

    train_dataset = ShrimpDataset(
        list_file=str(data_root / cfg["TRAIN_LIST"]),
        crop_size=tuple(cfg["CROP_SIZE"]),
        is_train=True,
    )
    val_dataset = ShrimpDataset(
        list_file=str(data_root / cfg["VAL_LIST"]),
        crop_size=None,      # Không crop khi validate — dùng full ảnh
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["TRAIN"]["BATCH_SIZE"],
        shuffle=True,
        num_workers=cfg["NUM_WORKERS"],
        collate_fn=shrimp_collate_fn,
        pin_memory=True,
        drop_last=True,       # Tránh batch size = 1 ở cuối epoch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,         # Validate từng ảnh để đếm chính xác
        shuffle=False,
        num_workers=2,
        collate_fn=shrimp_collate_fn,
    )

    logging.info(f"Train: {len(train_dataset)} images, "
                 f"Val: {len(val_dataset)} images")

    # --- Build APGCC cfg object (easydict) từ base config ---
    from apgcc.config import cfg as apgcc_cfg
    # Override backbone theo argument
    apgcc_cfg.MODEL.ENCODER = args.pretrained_backbone
    
    # Đồng bộ thông số từ SHRIMP_train.yml sang apgcc_cfg
    apgcc_cfg.MODEL.STRIDE = cfg["MODEL"]["STRIDE"]
    apgcc_cfg.MODEL.WEIGHT_DICT = {
        'loss_ce': cfg["TRAIN"]["LAMBDA1"],
        'loss_points': cfg["TRAIN"]["LAMBDA2"],
        'loss_aux': cfg["TRAIN"]["LAMBDA5"]  # APGCC code cần key này để `del` mượt mà khi đổi AUX_EN
    }
    # Đồng bộ tham số cân bằng class (foreground vs background)
    apgcc_cfg.MODEL.EOS_COEF = cfg["TRAIN"].get("EOS_COEF", 0.05)
    apgcc_cfg.MODEL.AUX_EN = False
    
    # Training với criterion
    model, criterion = build_model(apgcc_cfg, training=True)
    model = model.to(device)
    criterion = criterion.to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model params: {param_count / 1e6:.2f}M")

    # --- Optimizer ---
    optimizer = build_optimizer(model, cfg)

    # --- LR Scheduler ---
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["TRAIN"]["LR_STEP"],
        gamma=cfg["TRAIN"]["LR_GAMMA"],
    )

    # --- Resume ---
    start_epoch = 1
    best_mae    = float("inf")

    if args.resume:
        logging.info(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_mae    = ckpt.get("best_mae", float("inf"))

    # --- TensorBoard ---
    writer = None
    if cfg["LOG"]["TENSORBOARD"]:
        writer = SummaryWriter(log_dir=str(out_dir / "tb_logs"))

    # ============================================================
    # Training loop chính
    # ============================================================
    logging.info("=" * 60)
    logging.info(f"Bắt đầu training: {cfg['TRAIN']['EPOCHS']} epochs")
    logging.info("=" * 60)

    for epoch in range(start_epoch, cfg["TRAIN"]["EPOCHS"] + 1):
        # --- Train ---
        train_loss = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch, cfg, writer
        )

        scheduler.step()

        # --- Validate ---
        if epoch % cfg["VAL"]["INTERVAL"] == 0 or epoch == cfg["TRAIN"]["EPOCHS"]:
            mae, rmse = evaluate(
                model, val_loader, device, cfg, epoch, writer
            )

            # In kết quả rõ ràng
            print(f"\n{'='*50}")
            print(f"  Epoch {epoch:4d}/{cfg['TRAIN']['EPOCHS']}  |  "
                  f"Loss: {train_loss:.4f}  |  "
                  f"MAE: {mae:.2f}  |  "
                  f"RMSE: {rmse:.2f}")
            print(f"{'='*50}\n")

            # --- Lưu best model ---
            is_best = mae < best_mae
            if is_best:
                best_mae = mae
                torch.save(model.state_dict(),
                           str(out_dir / "best_model.pth"))
                logging.info(f"  ✓ Lưu best model (MAE={best_mae:.2f})")

        # --- Checkpoint định kỳ ---
        if epoch % 20 == 0:
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_mae":  best_mae,
            }, str(out_dir / f"checkpoint_epoch{epoch:04d}.pth"))

    # --- Test cuối cùng ---
    logging.info("\n[DONE] Training hoàn tất!")
    logging.info(f"  Best VAL MAE: {best_mae:.2f}")
    logging.info(f"  Best model: {out_dir / 'best_model.pth'}")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
