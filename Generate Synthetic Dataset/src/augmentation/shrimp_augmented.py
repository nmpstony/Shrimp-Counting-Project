import os
import sys
import json
import random

# Đảm bảo working directory luôn là thư mục gốc
project_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(project_dir) in ["src", "preprocessing", "augmentation", "visualization"]:
    project_dir = os.path.dirname(project_dir)
os.chdir(project_dir)
sys.path.append(os.path.join(project_dir, "src", "augmentation"))

from PIL import Image
from tqdm import tqdm  # Thêm thư viện tqdm
from data_augmentation import augment_foreground

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
INPUT_DIR   = "images/backgrounds"

OUTPUT_DIR  = "images/backgrounds_augmented"

TARGET_COUNT = 250  # Tổng số lượng tôm sau khi nhân bản

def main():
    # Tạo thư mục đầu ra
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    orig_filenames = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"🚀 Bắt đầu nhân bản tôm: {len(orig_filenames)} ảnh gốc -> {TARGET_COUNT} biến thể.")

    # Sử dụng tqdm để quản lý vòng lặp
    for i in tqdm(range(TARGET_COUNT), desc="Đang nhân bản tôm", unit="ảnh"):
        # 1. Chọn ngẫu nhiên 1 con tôm gốc
        orig_name = random.choice(orig_filenames)
        orig_path = os.path.join(INPUT_DIR, orig_name)
        
        try:
            # 2. Đọc và áp dụng tăng cường (Màu sắc, ánh sáng...)
            img = Image.open(orig_path).convert("RGBA")
            img_aug = augment_foreground(img)
            
            # 3. Đặt tên file theo định dạng 0000.png tăng dần
            new_name = f"{i:04d}.png"
            out_path = os.path.join(OUTPUT_DIR, new_name)
            
            # 4. Lưu ảnh
            img_aug.save(out_path)

        except Exception as e:
            # Dùng tqdm.write để in lỗi mà không làm hỏng thanh tiến độ
            tqdm.write(f"⚠️ Lỗi tại index {i} (file {orig_name}): {e}")

    print("\n" + "="*50)
    print(f"✅ HOÀN TẤT! Đã lưu {TARGET_COUNT} ảnh tại: {OUTPUT_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()