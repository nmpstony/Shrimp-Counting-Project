"""
Unified Image Processing Script
1. Xóa phông nền (rembg)
2. Crop sát vùng tôm (PIL + numpy)
3. Lưu kết quả cuối cùng vào Images/foreground_results (không lưu file trung gian)
"""

import sys
import io
import os

# Đảm bảo working directory luôn là thư mục gốc
project_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(project_dir) in ["src", "preprocessing", "augmentation", "visualization"]:
    project_dir = os.path.dirname(project_dir)
os.chdir(project_dir)
sys.path.append(os.path.join(project_dir, "src", "augmentation"))

from pathlib import Path
from PIL import Image
import numpy as np
from rembg import remove

# ── Cấu hình đường dẫn ──────────────────────────────────────────────
INPUT_DIR = Path(__file__).parent / "images" / "foreground_raw"
OUTPUT_DIR = Path(__file__).parent / "images" / "foreground_processed"

# Padding (pixel) xung quanh bounding box để không cắt quá sát
PADDING = 5

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def remove_background_in_memory(input_path: Path) -> Image.Image:
    """Xóa phông nền của một ảnh và trả về PIL Image (RGBA)."""
    with open(input_path, "rb") as f:
        input_data = f.read()

    output_data = remove(input_data)
    
    # Chuyển output_data sang PIL Image RGBA
    img = Image.open(io.BytesIO(output_data)).convert("RGBA")
    return img


def crop_to_content(image: Image.Image, padding: int = PADDING) -> Image.Image:
    """
    Crop ảnh RGBA sát vùng có nội dung (pixel không trong suốt).
    Sử dụng kênh alpha để xác định bounding box.
    """
    alpha = np.array(image)[:, :, 3]  # Kênh alpha

    # Tìm các hàng và cột có pixel không trong suốt (alpha > 0)
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)

    if not rows.any() or not cols.any():
        # Ảnh hoàn toàn trong suốt → trả về nguyên gốc
        return image

    # Xác định bounding box
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # Thêm padding (đảm bảo không vượt biên)
    h, w = alpha.shape
    row_min = max(0, row_min - padding)
    row_max = min(h - 1, row_max + padding)
    col_min = max(0, col_min - padding)
    col_max = min(w - 1, col_max + padding)

    # Crop ảnh theo bounding box
    cropped = image.crop((col_min, row_min, col_max + 1, row_max + 1))
    return cropped


def main():
    # Tạo thư mục đầu ra nếu chưa tồn tại
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Lấy danh sách ảnh đầu vào
    image_files = sorted(
        f for f in INPUT_DIR.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    total = len(image_files)
    if total == 0:
        print(f"Không tìm thấy ảnh nào trong thư mục đầu vào: {INPUT_DIR}")
        sys.exit(1)

    print(f"Tìm thấy {total} ảnh trong: {INPUT_DIR}")
    print(f"Kết quả sẽ được lưu trực tiếp vào: {OUTPUT_DIR}")
    print(f"Padding: {PADDING}px")
    print("-" * 60)

    success_count = 0
    error_count = 0

    for idx, img_path in enumerate(image_files, start=1):
        # Đổi phần mở rộng sang .png để hỗ trợ nền trong suốt
        output_path = OUTPUT_DIR / (img_path.stem + ".png")

        # Bỏ qua nếu ảnh đã được xử lý
        if output_path.exists():
            print(f"[{idx}/{total}] ✓ Đã tồn tại, bỏ qua: {img_path.name}")
            success_count += 1
            continue

        try:
            print(f"[{idx}/{total}] Đang xử lý: {img_path.name} ...", end=" ", flush=True)
            
            # Step 1: Remove Background (in memory)
            img_rgba = remove_background_in_memory(img_path)
            
            # Step 2: Tight Crop (in memory)
            original_size = img_rgba.size
            final_img = crop_to_content(img_rgba, PADDING)
            new_size = final_img.size
            
            # Save final result
            final_img.save(output_path, format="PNG")
            
            print(f"OK ({original_size[0]}x{original_size[1]} → {new_size[0]}x{new_size[1]})")
            success_count += 1
            
        except Exception as e:
            print(f"LỖI: {e}")
            error_count += 1

    # Tổng kết
    print("-" * 60)
    print(f"Hoàn tất! Thành công: {success_count}/{total} | Lỗi: {error_count}/{total}")
    print(f"Thư mục đầu ra: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
