import os
import sys
import cv2
import numpy as np

# Đảm bảo working directory luôn là thư mục gốc
project_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(project_dir) in ["src", "preprocessing", "augmentation", "visualization"]:
    project_dir = os.path.dirname(project_dir)
os.chdir(project_dir)
sys.path.append(os.path.join(project_dir, "src", "augmentation"))

import random
import matplotlib.pyplot as plt

def visualize_yolo_segmentation(image_path: str, label_path: str, output_path: str = None):
    """
    Đọc ảnh và nhãn YOLO Seg, vẽ mask bán trong suốt lên ảnh.
    """
    if not os.path.exists(image_path):
        print(f"Không tìm thấy ảnh: {image_path}")
        return
    if not os.path.exists(label_path):
        print(f"Không tìm thấy nhãn: {label_path}")
        return

    # 1. Đọc ảnh
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    # 2. Đọc file nhãn .txt
    with open(label_path, "r") as f:
        lines = f.readlines()

    # 3. Vẽ từng con tôm
    overlay = img.copy() # Lớp dùng để vẽ màu trong suốt
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7: # 1 class_id + ít nhất 3 điểm (6 tọa độ)
            continue
            
        class_id = int(parts[0])
        # Lấy mảng tọa độ (x1, y1, x2, y2...) và chuyển thành float
        norm_coords = np.array(parts[1:], dtype=np.float32)
        
        # Reshape thành mảng Nx2: [[x1, y1], [x2, y2], ...]
        points = norm_coords.reshape(-1, 2)
        
        # Giải chuẩn hóa (Denormalize): nhân với Width và Height của ảnh
        points[:, 0] = points[:, 0] * w
        points[:, 1] = points[:, 1] * h
        
        # Ép kiểu về số nguyên để OpenCV có thể vẽ
        points = points.astype(np.int32)

        # Tạo màu ngẫu nhiên cho từng con tôm (BGR)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Vẽ hình đa giác đặc (tô màu khối) lên lớp overlay
        cv2.fillPoly(overlay, [points], color)
        
        # Vẽ đường viền (contour) nét mảnh lên ảnh gốc cho sắc nét
        cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)

    # 4. Trộn lớp overlay với ảnh gốc để tạo hiệu ứng bán trong suốt (Alpha blending)
    alpha = 0.45  # Độ trong suốt của mask (0.0 -> 1.0)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # 5. Lưu hoặc Hiển thị
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Đã lưu ảnh visualize tại: {output_path}")
    else:
        # Nếu dùng Jupyter Notebook / Colab, dùng matplotlib để vẽ
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f"Visualized: {os.path.basename(image_path)}")
        plt.show()

# ==========================================
# CÁCH SỬ DỤNG
# ==========================================
if __name__ == "__main__":
    # Thay đổi đường dẫn tới thư mục dataset của bạn
    DATASET_DIR = "."#"datasets/Segments"
    
    IMG_DIR = os.path.join(DATASET_DIR, "images")
    LBL_DIR = os.path.join(DATASET_DIR, "labels")
    OUT_DIR = os.path.join(DATASET_DIR, "visualized") # Thư mục lưu ảnh đã vẽ
    
    os.makedirs(OUT_DIR, exist_ok=True)

    # Lấy danh sách 5 ảnh đầu tiên để test
    img_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])[:10]

    for img_name in img_files:
        base_name = os.path.splitext(img_name)[0]
        
        img_path = os.path.join(IMG_DIR, f"{base_name}.jpg")
        lbl_path = os.path.join(LBL_DIR, f"{base_name}.txt")
        out_path = os.path.join(OUT_DIR, f"{base_name}_vis.jpg")
        
        visualize_yolo_segmentation(img_path, lbl_path, out_path)