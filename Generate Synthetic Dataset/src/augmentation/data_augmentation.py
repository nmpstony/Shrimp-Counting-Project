import random
import os
import sys

# Đảm bảo working directory luôn là thư mục gốc
project_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(project_dir) in ["src", "preprocessing", "augmentation", "visualization"]:
    project_dir = os.path.dirname(project_dir)
os.chdir(project_dir)
sys.path.append(os.path.join(project_dir, "src", "augmentation"))

from PIL import Image, ImageEnhance

def augment_foreground(img: Image.Image) -> Image.Image:
    img_aug = img.copy()
    
    # 1. Màu sắc / Độ tươi (0.5 xám xịt -> 1.5 rực rỡ)
    if random.random() < 0.8:
        img_aug = ImageEnhance.Color(img_aug).enhance(random.uniform(0.3, 1.05))
        
    # 2. Độ sáng nội tại của con tôm (tôm sáng/tối khác nhau)
    if random.random() < 0.7:
        img_aug = ImageEnhance.Brightness(img_aug).enhance(random.uniform(0.6, 1.15))
        
    # 3. Độ tương phản vỏ tôm
    if random.random() < 0.7:
        img_aug = ImageEnhance.Contrast(img_aug).enhance(random.uniform(0.8, 1.15))
        
    # 4. Độ sắc nét chi tiết (râu, vảy, mép vỏ)
    if random.random() < 0.5:
        img_aug = ImageEnhance.Sharpness(img_aug).enhance(random.uniform(0.8, 2.0))
        
    return img_aug

def augment_background(img: Image.Image) -> Image.Image:
    """
    LẦN 2: Tăng cường toàn cục (Kênh RGB).
    Áp dụng lên bức ảnh HOÀN CHỈNH để mô phỏng môi trường ánh sáng xưởng.
    (Không áp dụng làm mờ để bảo toàn đặc trưng điểm).
    """
    img_aug = img.copy()

    # 1. Cân bằng trắng / Nhiệt độ màu môi trường (Mô phỏng đèn vàng/trắng)
    if random.random() < 0.6:
        img_aug = ImageEnhance.Color(img_aug).enhance(random.uniform(0.7, 1.3))

    # 2. Ánh sáng tổng thể của xưởng (chói nắng hoặc góc kẹt tối)
    if random.random() < 0.6:
        img_aug = ImageEnhance.Brightness(img_aug).enhance(random.uniform(0.8, 1.2))

    # 3. Độ tương phản của Camera (Mô phỏng các loại điện thoại/camera khác nhau)
    if random.random() < 0.6:
        img_aug = ImageEnhance.Contrast(img_aug).enhance(random.uniform(0.8, 1.2))

    return img_aug