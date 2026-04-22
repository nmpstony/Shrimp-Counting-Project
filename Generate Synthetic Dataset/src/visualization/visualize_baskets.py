import cv2
import json
import os
import sys

# Đảm bảo working directory luôn là thư mục gốc
project_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(project_dir) in ["src", "preprocessing", "augmentation", "visualization"]:
    project_dir = os.path.dirname(project_dir)
os.chdir(project_dir)
sys.path.append(os.path.join(project_dir, "src", "augmentation"))


# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
JSON_FILE = "images/backgrounds/results.json"  # File JSON chứa thông số rổ của bạn
IMG_DIR = "images/backgrounds"             # Thư mục chứa ảnh rổ gốc
OUT_DIR = "images/backgrounds_check"     # Thư mục lưu ảnh sau khi đã vẽ viền

# Tạo thư mục đầu ra nếu chưa tồn tại
os.makedirs(OUT_DIR, exist_ok=True)

def draw_circles_from_json():
    # 1. Đọc dữ liệu từ file JSON
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Bắt đầu kiểm tra và vẽ viền cho {len(data)} ảnh...")

    # 2. Duyệt qua từng phần tử trong file JSON
    for img_name, info in data.items():
        cx, cy = info["center"]
        r = info["radius"]
        
        img_path = os.path.join(IMG_DIR, img_name)
        
        # 3. Đọc ảnh bằng OpenCV
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"❌ Không thể đọc ảnh: {img_path}")
            continue
            
        # 4. Vẽ tâm rổ (Chấm tròn ĐỎ, đặc ruột)
        # Bán kính chấm đỏ = 10, độ dày = -1 (tô kín)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), -1) 
        
        # 5. Vẽ đường viền rổ (Đường tròn XANH LÁ, rỗng ruột)
        # Bán kính = r từ JSON, độ dày đường viền = 5
        cv2.circle(img, (cx, cy), r, (0, 255, 0), 5)
        
        # --- (Tùy chọn) Vẽ thêm vòng tròn VÙNG AN TOÀN màu VÀNG ---
        # Đây là vùng 85% bán kính để rải tôm không bị trào ra mép
        safe_r = int(r * 0.85)
        cv2.circle(img, (cx, cy), safe_r, (0, 255, 255), 3)

        # 6. Lưu ảnh đã vẽ ra thư mục kiểm tra
        out_path = os.path.join(OUT_DIR, img_name)
        cv2.imwrite(out_path, img)
        print(f"✅ Đã vẽ xong: {img_name}")

    print(f"\n🎉 Hoàn tất! Hãy vào thư mục '{OUT_DIR}' để xem thành quả nhé.")

# Chạy hàm
draw_circles_from_json()