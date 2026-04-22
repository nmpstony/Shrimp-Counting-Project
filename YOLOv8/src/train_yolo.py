import os
import sys
from ultralytics import YOLO

# Đảm bảo working directory luôn là thư mục gốc YOLOv8
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_dir)

if __name__ == "__main__":
    # Khởi tạo model phân vùng (Segmentation)
    model = YOLO('yolov8n-seg.pt')

    # Huấn luyện
    results = model.train(
        data='dataset/shrimp.yaml',
        epochs=400,
        imgsz=640,
        batch=16,
        device=0, # dùng GPU đầu tiên
        workers=8, # số luồng CPU xử lý data
        cache=True # load sẵn toàn bộ 1000 ảnh vào RAM
    )