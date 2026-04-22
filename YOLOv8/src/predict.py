import os
import sys
import cv2
import argparse

# Đảm bảo working directory luôn là thư mục gốc YOLOv8
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_dir)

import tkinter as tk
from ultralytics import YOLO

def process_single_image(model, img_path, conf_thres=0.5, show_ui=True):
    """
    Dự đoán trên 1 ảnh. Tự động tương thích với cả ảnh có nhãn (Validation) 
    và ảnh mới hoàn toàn (Real-world testing).
    """
    if not os.path.exists(img_path):
        print(f"❌ Lỗi: Không tìm thấy ảnh tại '{img_path}'")
        return

    # 1. KIỂM TRA XEM CÓ FILE GROUND TRUTH KHÔNG
    lbl_path = img_path.replace("images", "labels").rsplit('.', 1)[0] + ".txt"
    has_gt = os.path.exists(lbl_path)
    gt_count = 0

    if has_gt:
        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            gt_count = len(lines)
    else:
        if show_ui:
            print(f"[INFO] Nhận diện đây là ảnh thực tế mới toanh (Không có Ground Truth).")

    # 2. DỰ ĐOÁN
    results = model.predict(source=img_path, conf=conf_thres, show=False, verbose=False)
    result = results[0]

    # 3. ĐẾM SỐ LƯỢNG PREDICT
    pred_count = len(result.masks) if result.masks is not None else 0
    annotated_img = result.plot()

    # 4. VẼ KẾT QUẢ LÊN ẢNH
    org = (20, 60)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5

    if has_gt:
        text = f"GT: {gt_count} | Pred: {pred_count}"
        text_color = (0, 255, 0) if gt_count == pred_count else (0, 0, 255)
    else:
        text = f"Count: {pred_count}"
        text_color = (0, 200, 255) 

    # Viền đen dày dặn (Shadow)
    cv2.putText(annotated_img, text, org, font, font_scale, (0, 0, 0), 6)
    # Chữ màu
    cv2.putText(annotated_img, text, org, font, font_scale, text_color, 3)

    # 5. LƯU ẢNH VÀO THƯ MỤC EXAMPLE/RESULT
    base_name = os.path.basename(img_path)
    result_dir = os.path.join("example", "result")
    os.makedirs(result_dir, exist_ok=True)
    out_filename = os.path.join(result_dir, f"{os.path.splitext(base_name)[0]}_result.jpg")
    
    cv2.imwrite(out_filename, annotated_img)
    if show_ui:
        print(f"✅ Đã lưu kết quả tại file: {out_filename}")

    # 6. HIỂN THỊ UI (NẾU CẦN)
    if show_ui:
        try:
            root = tk.Tk()
            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()
            root.destroy() 
        except:
            screen_w, screen_h = 1920, 1080 # Fallback
            
        window_name = f"Result: {base_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, screen_w // 2, screen_h // 2)

        cv2.imshow(window_name, annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script test YOLOv8-Seg (Hỗ trợ 1 ảnh HOẶC thư mục)")
    parser.add_argument("source", type=str, help="Đường dẫn tới file ảnh HOẶC thư mục chứa ảnh")
    parser.add_argument("--model", type=str, default="runs/segment/train/weights/best.pt", help="Đường dẫn tới file .pt của mô hình")
    parser.add_argument("--conf", type=float, default=0.5, help="Ngưỡng tự tin Confidence (mặc định 0.5)")

    args = parser.parse_args()
    
    print(f"[INFO] Đang nạp mô hình...")
    model = YOLO(args.model)

    if os.path.isfile(args.source):
        process_single_image(model, args.source, args.conf, show_ui=True)
    elif os.path.isdir(args.source):
        img_files = [f for f in os.listdir(args.source) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"[INFO] Tìm thấy {len(img_files)} ảnh trong thư mục '{args.source}'. Bắt đầu xử lý hàng loạt...")
        for i, img_name in enumerate(img_files):
            img_path = os.path.join(args.source, img_name)
            print(f"[{i+1}/{len(img_files)}] Đang phân tích: {img_name}")
            process_single_image(model, img_path, args.conf, show_ui=False)
        print(f"\n✅ Hoàn tất! Bạn có thể xem lại thành quả của {len(img_files)} ảnh tại thư mục 'example/result'.")
    else:
        print(f"❌ Lỗi: Không tìm thấy file hay thư mục '{args.source}'")