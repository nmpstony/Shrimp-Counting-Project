"""
datasets/shrimp_dataset.py
==========================
Custom Dataset Loader cho bài toán đếm tôm thẻ với APGCC.

=== TẠI SAO CẦN FILE NÀY? ===

Dataset gốc của APGCC (ShanghaiTech, NWPU) lưu nhãn dạng .mat (MATLAB),
load bằng scipy.io.loadmat và trả về dict {'image_info': ...}.
Dataset tôm của bạn lưu nhãn dạng .txt với format "X Y" mỗi dòng.

Cơ chế APG cần nhãn đầu vào là:
  target['point'] : Tensor [N, 2]  — N điểm tâm (x, y) của mỗi con tôm
                    Dùng để:
                    (1) Matching Hungarian: so khớp proposals với GT points
                    (2) Sinh Auxiliary Positive Points (Apos): offset ngẫu nhiên
                        trong bán kính n_pos quanh mỗi GT point → buộc model
                        nhận diện vùng gần GT là positive
                    (3) Sinh Auxiliary Negative Points (Aneg): offset ngẫu nhiên
                        trong vành khuyên [n_pos, n_neg] → buộc model reject
                        vùng xa GT
  target['labels']: Tensor [N]     — nhãn class (luôn = 0 vì chỉ có 1 class: tôm)

Không cần pseudo-box hay density map vì APGCC là pure point-based.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


# ============================================================
# Augmentation helpers
# ============================================================

def random_crop(image, points, crop_size):
    """
    Crop ngẫu nhiên ảnh và cập nhật tọa độ điểm tương ứng.
    Loại bỏ các điểm nằm ngoài vùng crop.
    """
    h, w = image.shape[:2]
    ch, cw = crop_size

    # Đảm bảo crop không lớn hơn ảnh
    ch = min(ch, h)
    cw = min(cw, w)

    top  = np.random.randint(0, h - ch + 1)
    left = np.random.randint(0, w - cw + 1)

    image_crop = image[top:top+ch, left:left+cw]

    if len(points) > 0:
        pts = points.copy()
        pts[:, 0] -= left   # x
        pts[:, 1] -= top    # y
        # Giữ lại điểm trong vùng crop
        mask = (pts[:, 0] >= 0) & (pts[:, 0] < cw) & \
               (pts[:, 1] >= 0) & (pts[:, 1] < ch)
        pts = pts[mask]
    else:
        pts = points

    return image_crop, pts


def random_flip(image, points):
    """Lật ngang ngẫu nhiên."""
    if np.random.rand() < 0.5:
        w = image.shape[1]
        image = image[:, ::-1].copy()
        if len(points) > 0:
            points = points.copy()
            points[:, 0] = w - 1 - points[:, 0]
    return image, points


# ============================================================
# Dataset class
# ============================================================

class ShrimpDataset(Dataset):
    """
    Dataset loader cho ảnh tôm thẻ.

    Cấu trúc thư mục expected:
      root/
        train/
          img_001.jpg  + img_001.txt
          img_002.jpg  + img_002.txt
        train.list   (mỗi dòng = đường dẫn tuyệt đối tới ảnh)

    Args:
        list_file  : đường dẫn tới file .list (train.list / val.list)
        crop_size  : (H, W) kích thước crop khi training. None = không crop.
        is_train   : True = dùng augmentation, False = inference mode
        img_size   : resize ảnh về kích thước cố định (None = giữ nguyên)
    """

    def __init__(self, list_file, crop_size=(512, 512),
                 is_train=True, img_size=None):
        self.is_train   = is_train
        self.crop_size  = crop_size
        self.img_size   = img_size

        # Đọc danh sách ảnh
        with open(list_file, "r") as f:
            self.img_paths = [line.strip() for line in f if line.strip()]

        # ImageNet normalization — giống code gốc APGCC
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225])
        ])

        print(f"[ShrimpDataset] Loaded {len(self.img_paths)} images "
              f"(is_train={is_train})")

    def __len__(self):
        return len(self.img_paths)

    def _load_label(self, img_path):
        """
        Load file nhãn .txt tương ứng với ảnh.

        Format nhãn của bạn:  "X Y"  (pixel coords, space-separated)
        Ví dụ:
            152.3 87.1
            310.0 200.5
            ...

        Trả về: np.ndarray shape [N, 2], dtype float32
                Nếu ảnh không có tôm nào → shape [0, 2]
        """
        # Suy ra đường dẫn nhãn từ đường dẫn ảnh
        label_path = os.path.splitext(img_path)[0] + ".txt"

        if not os.path.exists(label_path):
            # Không có nhãn → ảnh rỗng (0 con tôm)
            return np.zeros((0, 2), dtype=np.float32)

        points = []
        with open(label_path, "r") as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    print(f"[WARN] {label_path} line {line_no+1}: "
                          f"expected 2 values, got {len(parts)}")
                    continue
                x, y = float(parts[0]), float(parts[1])
                points.append([x, y])

        if len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        return np.array(points, dtype=np.float32)   # [N, 2]

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        # --- Load ảnh ---
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # HWC, RGB

        # --- Load nhãn ---
        points = self._load_label(img_path)   # [N, 2]

        # --- Optional: resize ---
        if self.img_size is not None:
            h0, w0 = image.shape[:2]
            image = cv2.resize(image, (self.img_size[1], self.img_size[0]))
            if len(points) > 0:
                scale_x = self.img_size[1] / w0
                scale_y = self.img_size[0] / h0
                points[:, 0] *= scale_x
                points[:, 1] *= scale_y

        # --- Augmentation (chỉ khi training) ---
        if self.is_train:
            if self.crop_size is not None:
                image, points = random_crop(image, points, self.crop_size)
            image, points = random_flip(image, points)

        # --- Chuyển ảnh thành Tensor ---
        image_pil = Image.fromarray(image)
        image_tensor = self.transform(image_pil)   # [3, H, W]

        # --- Build target dict — đây là format mà APGCC Loss yêu cầu ---
        #
        # APGCC/apgcc/models/APGCCNet.py -> forward() truyền target list
        # Mỗi phần tử target là dict với 2 key:
        #   'point'  : FloatTensor [N, 2]  — tọa độ (x, y) pixel
        #   'labels' : LongTensor  [N]     — class index (luôn 0 với tôm)
        #
        # APG sẽ tự động sinh Apos và Aneg từ 'point' trong quá trình forward:
        #   Apos_i = point_i + U(-n_pos, +n_pos)   → confidence target = 1
        #   Aneg_j = point_j + U(n_pos, n_neg) sign — confidence target = 0
        # (xem apgcc/models/APGCCNet.py: generate_auxiliary_points())
        #
        target = {
            "point": torch.as_tensor(points, dtype=torch.float32),   # [N, 2]
            "labels": torch.ones(len(points), dtype=torch.long),     # [N]
        }

        return image_tensor, target


# ============================================================
# Collate function — gộp batch với số lượng tôm khác nhau mỗi ảnh
# ============================================================

def shrimp_collate_fn(batch):
    """
    Custom collate function vì mỗi ảnh có số lượng điểm khác nhau.
    Không thể dùng default_collate (yêu cầu tensor cùng shape).

    Trả về:
        images : Tensor [B, 3, H, W]
        targets: list of dict — giữ nguyên, APGCC xử lý từng ảnh riêng
    """
    images  = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


# ============================================================
# Quick test — chạy: python datasets/shrimp_dataset.py
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--list", required=True, help="Đường dẫn tới train.list")
    args = parser.parse_args()

    dataset = ShrimpDataset(
        list_file=args.list,
        crop_size=(512, 512),
        is_train=True
    )

    print(f"\n[TEST] Tổng số mẫu: {len(dataset)}")

    img, tgt = dataset[0]
    print(f"[TEST] Image tensor shape: {img.shape}")
    print(f"[TEST] Points shape: {tgt['point'].shape}")
    print(f"[TEST] Labels shape: {tgt['labels'].shape}")
    print(f"[TEST] Số tôm trong ảnh đầu tiên: {len(tgt['point'])}")

    if len(tgt['point']) > 0:
        print(f"[TEST] Toạ độ 3 điểm đầu:\n{tgt['point'][:3]}")

    # Test DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2,
                        collate_fn=shrimp_collate_fn, shuffle=True)
    imgs, tgts = next(iter(loader))
    print(f"\n[TEST] Batch images: {imgs.shape}")
    print(f"[TEST] Batch points: {[t['point'].shape for t in tgts]}")
    print("\n✅ Dataset loader hoạt động bình thường!")
