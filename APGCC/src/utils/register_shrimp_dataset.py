"""
register_shrimp_dataset.py
===========================
Patch để đăng ký ShrimpDataset vào hệ thống registry của APGCC.

APGCC gốc dùng pattern:
    dataset = build_dataset(cfg.DATASET)   # 'SHHA', 'SHHB', 'NWPU', ...

File này mở rộng registry để hỗ trợ thêm 'shrimp'.

Cách dùng:
    # Import patch này TRƯỚC KHI gọi build_dataset
    import register_shrimp_dataset   # noqa

    # Sau đó dùng bình thường
    dataset = build_dataset('shrimp', list_file='...', is_train=True)

=== TẠI SAO CẦN PATCH NÀY? ===

Code gốc (apgcc/datasets/__init__.py) có dạng:

    DATASET_REGISTRY = {
        'SHHA': ShanghaiTechDataset,
        'SHHB': ShanghaiTechDataset,
        'NWPU': NWPUDataset,
        ...
    }

    def build_dataset(name, **kwargs):
        return DATASET_REGISTRY[name](**kwargs)

Nhãn SHHA/NWPU dùng .mat → class ShanghaiTechDataset gọi scipy.io.loadmat.
Nhãn tôm dùng .txt → cần ShrimpDataset.

Thay vì sửa code gốc (gây conflict khi pull), ta dùng monkey-patch.
"""

import sys
import os

# Đảm bảo working directory luôn là thư mục gốc
project_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(project_dir) in ["src", "core", "preprocessing", "utils"]:
    project_dir = os.path.dirname(project_dir)
os.chdir(project_dir)

# Thêm APGCC vào path
sys.path.insert(0, os.path.join(project_dir, "apgcc"))
sys.path.insert(0, project_dir)

# Import registry gốc
try:
    import apgcc.datasets as apgcc_datasets
    _registry_patched = False
except ImportError:
    apgcc_datasets = None
    _registry_patched = False

# Import dataset tôm
from datasets.shrimp_dataset import ShrimpDataset, shrimp_collate_fn


def patch_registry():
    """Thêm 'shrimp' vào DATASET_REGISTRY của APGCC."""
    global _registry_patched

    if _registry_patched:
        return

    if apgcc_datasets is None:
        print("[WARN] Không tìm thấy apgcc.datasets — bỏ qua patch registry")
        return

    # Tìm dict registry — tên có thể khác nhau tùy version
    registry_names = ["DATASET_REGISTRY", "_dataset_registry", "dataset_map"]
    for attr_name in registry_names:
        if hasattr(apgcc_datasets, attr_name):
            registry = getattr(apgcc_datasets, attr_name)
            registry["shrimp"] = ShrimpDataset
            registry["SHRIMP"] = ShrimpDataset   # case-insensitive
            print(f"[INFO] Đã patch '{attr_name}' với ShrimpDataset")
            _registry_patched = True
            return

    # Nếu không tìm thấy registry dạng dict, patch hàm build_dataset
    if hasattr(apgcc_datasets, "build_dataset"):
        _original_build = apgcc_datasets.build_dataset

        def patched_build_dataset(name, **kwargs):
            if name.lower() == "shrimp":
                return ShrimpDataset(**kwargs)
            return _original_build(name, **kwargs)

        apgcc_datasets.build_dataset = patched_build_dataset
        print("[INFO] Đã monkey-patch apgcc_datasets.build_dataset")
        _registry_patched = True
    else:
        print("[WARN] Không tìm thấy build_dataset trong apgcc.datasets")


# Tự động patch khi import
patch_registry()


# ============================================================
# Helper: tạo DataLoader trực tiếp không qua registry
# ============================================================

from torch.utils.data import DataLoader


def build_shrimp_loaders(cfg):
    """
    Tạo train/val/test DataLoader cho dataset tôm.
    Dùng trực tiếp thay vì đi qua APGCC registry nếu muốn đơn giản hơn.

    Args:
        cfg: dict load từ SHRIMP_train.yml

    Returns:
        train_loader, val_loader, test_loader
    """
    from pathlib import Path
    data_root = Path(cfg["DATA_ROOT"])

    train_ds = ShrimpDataset(
        list_file=str(data_root / cfg["TRAIN_LIST"]),
        crop_size=tuple(cfg["CROP_SIZE"]),
        is_train=True,
    )
    val_ds = ShrimpDataset(
        list_file=str(data_root / cfg["VAL_LIST"]),
        crop_size=None,
        is_train=False,
    )
    test_ds = ShrimpDataset(
        list_file=str(data_root / cfg["TEST_LIST"]),
        crop_size=None,
        is_train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["TRAIN"]["BATCH_SIZE"],
        shuffle=True,
        num_workers=cfg["NUM_WORKERS"],
        collate_fn=shrimp_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=shrimp_collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=shrimp_collate_fn,
    )

    return train_loader, val_loader, test_loader
