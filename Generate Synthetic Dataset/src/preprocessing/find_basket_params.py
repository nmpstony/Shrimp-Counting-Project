"""
detect_circle_v4.py
Xác định tâm và bán kính của rổ/thau tròn từ ảnh chụp trên xuống.

Cải tiến so với v3:
- Fix mask bị đảo ngược khi rổ chạm góc ảnh (invert nếu white > 50%)
- Dùng largest-contour fill thay vì raw threshold để tách vật thể
- Kết hợp thêm Canny edge trực tiếp trên vùng vành rổ để tăng robustness
- Output: lưu ảnh debug vào folder riêng + file results.json

Yêu cầu: pip install opencv-python numpy
"""

import cv2
import numpy as np
import sys
import os

# Đảm bảo working directory luôn là thư mục gốc
project_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(project_dir) in ["src", "preprocessing", "augmentation", "visualization"]:
    project_dir = os.path.dirname(project_dir)
os.chdir(project_dir)
sys.path.append(os.path.join(project_dir, "src", "augmentation"))

import random
import json


# ─────────────────────────────────────────────────────────────────────────────
# RANSAC fit circle
# ─────────────────────────────────────────────────────────────────────────────

def _fit_circle_3pts(p1, p2, p3):
    ax, ay = float(p1[0]), float(p1[1])
    bx, by = float(p2[0]), float(p2[1])
    cx, cy = float(p3[0]), float(p3[1])
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-6:
        return None
    ux = ((ax**2 + ay**2) * (by - cy) +
          (bx**2 + by**2) * (cy - ay) +
          (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) +
          (bx**2 + by**2) * (ax - cx) +
          (cx**2 + cy**2) * (bx - ax)) / d
    r = np.hypot(ax - ux, ay - uy)
    return ux, uy, r


def ransac_fit_circle(points, n_iter=3000, tol=12):
    if len(points) < 10:
        return None
    pts = points.astype(float)
    best_inliers = 0
    best_circle = None
    for _ in range(n_iter):
        idx = random.sample(range(len(pts)), 3)
        result = _fit_circle_3pts(pts[idx[0]], pts[idx[1]], pts[idx[2]])
        if result is None:
            continue
        ux, uy, r = result
        if r < 10:
            continue
        dists = np.abs(np.hypot(pts[:, 0] - ux, pts[:, 1] - uy) - r)
        inliers = np.sum(dists < tol)
        if inliers > best_inliers:
            best_inliers = inliers
            best_circle = (ux, uy, r)
    if best_circle is None:
        return None
    # Least-squares refine
    ux, uy, r = best_circle
    dists = np.abs(np.hypot(pts[:, 0] - ux, pts[:, 1] - uy) - r)
    inlier_pts = pts[dists < tol]
    if len(inlier_pts) >= 3:
        from numpy.linalg import lstsq
        A = np.column_stack([inlier_pts[:, 0], inlier_pts[:, 1],
                             np.ones(len(inlier_pts))])
        b = inlier_pts[:, 0]**2 + inlier_pts[:, 1]**2
        res, _, _, _ = lstsq(A, b, rcond=None)
        cx_r, cy_r = res[0] / 2, res[1] / 2
        r_r = np.sqrt(max(res[2] + cx_r**2 + cy_r**2, 0))
        if r_r > 10:
            return cx_r, cy_r, r_r
    return best_circle


# ─────────────────────────────────────────────────────────────────────────────
# Bước 1: Tách vật thể - pipeline mới
# ─────────────────────────────────────────────────────────────────────────────

def _get_object_mask(img_bgr):
    """
    Tách vật thể (rổ/thau) khỏi nền.
    
    Pipeline:
    1. GrabCut seed từ vùng trung tâm (rổ/thau luôn nằm ở giữa ảnh)
    2. Nếu GrabCut thất bại → threshold + largest contour fill
    3. Đảm bảo mask không bị invert (vật thể = trắng)
    """
    h, w = img_bgr.shape[:2]
    mask_gc = np.zeros((h, w), np.uint8)

    # ── Thử GrabCut với rect bao quanh 85% trung tâm ảnh ────────────────────
    margin_x = int(w * 0.08)
    margin_y = int(h * 0.08)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    try:
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(img_bgr, mask_gc, rect, bgd_model, fgd_model, 5,
                    cv2.GC_INIT_WITH_RECT)
        # GC_FGD=1, GC_PR_FGD=3 → foreground
        object_mask = np.where((mask_gc == 1) | (mask_gc == 3),
                               np.uint8(255), np.uint8(0))
    except Exception:
        object_mask = None

    # ── Kiểm tra GrabCut có ổn không (phải có ít nhất 10% pixel trắng) ──────
    if object_mask is None or np.mean(object_mask > 0) < 0.10:
        object_mask = _threshold_largest_contour(img_bgr)

    # ── Morphology để lấp lỗ rổ ─────────────────────────────────────────────
    k_sz = max(3, w // 20)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    return object_mask


def _threshold_largest_contour(img_bgr):
    """
    Fallback: threshold → lấy contour lớn nhất → fill.
    Tự động invert nếu cần.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Thử cả Otsu threshold và adaptive threshold, lấy cái nào tốt hơn
    _, thresh_otsu = cv2.threshold(blurred, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_adapt = cv2.adaptiveThreshold(blurred, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 51, 5)

    best_mask = None
    best_score = -1

    for thresh in [thresh_otsu, thresh_adapt, cv2.bitwise_not(thresh_otsu)]:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w // 20, w // 20))
        t = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=3)
        contours, _ = cv2.findContours(t, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        # Vật thể phải chiếm 20-90% diện tích ảnh
        ratio = area / (h * w)
        if not (0.20 < ratio < 0.90):
            continue
        # Ưu tiên contour gần tâm ảnh nhất
        M = cv2.moments(largest)
        if M["m00"] < 1:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        dist_to_center = np.hypot(cx - w / 2, cy - h / 2) / np.hypot(w, h)
        score = ratio - dist_to_center
        if score > best_score:
            best_score = score
            filled = np.zeros((h, w), np.uint8)
            cv2.drawContours(filled, [largest], -1, 255, -1)
            best_mask = filled

    if best_mask is None:
        # Last resort: vùng trung tâm 70%
        best_mask = np.zeros((h, w), np.uint8)
        cx, cy = w // 2, h // 2
        cv2.ellipse(best_mask, (cx, cy), (int(w * 0.35), int(h * 0.35)),
                    0, 0, 360, 255, -1)

    return best_mask


# ─────────────────────────────────────────────────────────────────────────────
# Bước 2: Centroid của mask
# ─────────────────────────────────────────────────────────────────────────────

def _mask_centroid(mask):
    M = cv2.moments(mask)
    if M["m00"] < 1:
        h, w = mask.shape
        return w // 2, h // 2
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


# ─────────────────────────────────────────────────────────────────────────────
# Bước 3: Quét tia + lọc outlier
# ─────────────────────────────────────────────────────────────────────────────

def _outer_edge_points_from(mask, origin_x, origin_y, n_angles=720):
    h, w = mask.shape
    max_r = int(np.hypot(w, h)) + 1
    pts = []
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    for a in angles:
        cos_a, sin_a = np.cos(a), np.sin(a)
        for r in range(max_r, 5, -1):
            x = int(origin_x + r * cos_a)
            y = int(origin_y + r * sin_a)
            if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
                pts.append([x, y])
                break
    return np.array(pts) if pts else None


def _filter_outliers(pts, origin_x, origin_y, sigma=1.8):
    if pts is None or len(pts) < 10:
        return pts
    radii = np.hypot(pts[:, 0] - origin_x, pts[:, 1] - origin_y)
    med = np.median(radii)
    mad = np.median(np.abs(radii - med))
    mad = max(mad, 5.0)
    threshold = sigma * mad * 1.4826
    keep = np.abs(radii - med) < threshold
    if keep.sum() < 0.6 * len(pts):
        keep = np.abs(radii - med) < (threshold * 1.5)
    return pts[keep] if keep.sum() >= 10 else pts


# ─────────────────────────────────────────────────────────────────────────────
# Main detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_basket_circle(image_path: str, debug: bool = False,
                          output_dir: str = None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[LỖI] Không thể đọc: {image_path}")
        return None

    h, w = img.shape[:2]
    max_dim = 800
    scale = min(max_dim / w, max_dim / h, 1.0)
    rw, rh = int(w * scale), int(h * scale)
    small = cv2.resize(img, (rw, rh))

    tol_px = max(8, int(min(rw, rh) * 0.018))
    circle = None
    mask = _get_object_mask(small)

    # ── Phương pháp A: mask → centroid → quét tia → lọc → RANSAC 2 vòng ────
    origin_x, origin_y = _mask_centroid(mask)
    edge_pts = _outer_edge_points_from(mask, origin_x, origin_y, n_angles=720)

    if edge_pts is not None and len(edge_pts) >= 10:
        # Vòng 1: RANSAC thô
        c_rough = ransac_fit_circle(edge_pts, n_iter=1500, tol=tol_px * 2)
        ref_x = int(c_rough[0]) if c_rough else origin_x
        ref_y = int(c_rough[1]) if c_rough else origin_y
        # Lọc outlier dựa trên tâm thô
        filtered = _filter_outliers(edge_pts, ref_x, ref_y, sigma=1.8)
        # Vòng 2: RANSAC tinh
        if filtered is not None and len(filtered) >= 10:
            circle = ransac_fit_circle(filtered, n_iter=3000, tol=tol_px)

    # ── Phương pháp B: Canny trực tiếp trên vùng vành + RANSAC ─────────────
    if circle is None:
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        edges = cv2.Canny(blurred, 20, 80)
        # Mask chỉ giữ vùng vành: hình khuyên từ 30% đến 55% min(rw,rh)
        ring_mask = np.zeros((rh, rw), np.uint8)
        cx_img, cy_img = rw // 2, rh // 2
        r_outer = int(min(rw, rh) * 0.55)
        r_inner = int(min(rw, rh) * 0.28)
        cv2.circle(ring_mask, (cx_img, cy_img), r_outer, 255, -1)
        cv2.circle(ring_mask, (cx_img, cy_img), r_inner,   0, -1)
        edges = cv2.bitwise_and(edges, ring_mask)
        ys, xs = np.where(edges > 0)
        if len(xs) >= 10:
            edge_pts_b = np.column_stack([xs, ys])
            filtered_b = _filter_outliers(edge_pts_b, origin_x, origin_y, sigma=2.0)
            circle = ransac_fit_circle(filtered_b, n_iter=3000, tol=tol_px)

    if circle is None:
        print(f"[CẢNH BÁO] Không tìm thấy: {image_path}")
        return None

    cx_s, cy_s, r_s = circle
    cx_orig = int(cx_s / scale)
    cy_orig = int(cy_s / scale)
    r_orig  = int(r_s  / scale)

    result = {
        "center_x":     cx_orig,
        "center_y":     cy_orig,
        "radius":       r_orig,
        "image_width":  w,
        "image_height": h,
    }

    if debug:
        _save_debug(img, cx_orig, cy_orig, r_orig, image_path,
                    mask_small=mask, scale=scale,
                    centroid=(int(origin_x / scale), int(origin_y / scale)),
                    output_dir=output_dir)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Lưu debug ra output_dir
# ─────────────────────────────────────────────────────────────────────────────

def _save_debug(img, cx, cy, r, src_path, mask_small=None, scale=1.0,
                centroid=None, output_dir=None):
    out = img.copy()
    cv2.circle(out, (cx, cy), r,  (0, 255, 0), 5)
    cv2.circle(out, (cx, cy), 12, (0, 0, 255), -1)
    if centroid:
        cv2.circle(out, centroid, 10, (0, 165, 255), -1)  # cam
    cv2.putText(out, f"C=({cx},{cy})  R={r}px",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 4)

    basename = os.path.splitext(os.path.basename(src_path))[0]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        debug_path = os.path.join(output_dir, f"{basename}_debug.jpg")
        mask_path  = os.path.join(output_dir, f"{basename}_mask.jpg")
    else:
        debug_path = os.path.splitext(src_path)[0] + "_debug.jpg"
        mask_path  = os.path.splitext(src_path)[0] + "_mask.jpg"

    cv2.imwrite(debug_path, out)
    if mask_small is not None:
        cv2.imwrite(mask_path, mask_small)
    print(f"  → {debug_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Xử lý hàng loạt → output_dir + results.json
# ─────────────────────────────────────────────────────────────────────────────

def process_folder(input_folder: str, output_folder: str, debug: bool = False):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted([
        f for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in exts
        and "_debug" not in f and "_mask" not in f
    ])

    if not files:
        print(f"Không tìm thấy ảnh nào trong: {input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    results = {}
    print(f"{'File':<30} {'Tâm (cx, cy)':<20} {'Bán kính':>10}  Kích thước")
    print("-" * 80)

    for fname in files:
        fpath = os.path.join(input_folder, fname)
        res = detect_basket_circle(fpath, debug=debug, output_dir=output_folder)
        if res:
            results[fname] = {
                "center": [res["center_x"], res["center_y"]],
                "radius": res["radius"],
            }
            print(f"{fname:<30} ({res['center_x']:>5}, {res['center_y']:>5})   "
                  f"{res['radius']:>8}px   "
                  f"{res['image_width']}×{res['image_height']}")
        else:
            # results[fname] = None  # Or skip if failed
            print(f"{fname:<30} *** Không phát hiện được ***")

    # Lưu JSON
    json_path = os.path.join(output_folder, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n✓ Đã lưu kết quả: {json_path}")
    print(f"✓ Tổng: {len(files)} ảnh, "
          f"{len(results)} thành công")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Cách dùng:")
        print()
        print("  # Xử lý 1 file:")
        print("  python detect_circle_v4.py <ảnh.jpg>")
        print("  python detect_circle_v4.py <ảnh.jpg> --debug")
        print()
        print("  # Xử lý cả folder (kết quả lưu vào backgrounds_check/):")
        print("  python detect_circle_v4.py images/backgrounds/ images/backgrounds_check/")
        print("  python detect_circle_v4.py images/backgrounds/ images/backgrounds_check/ --debug")
        sys.exit(0)

    debug = "--debug" in args
    positional = [a for a in args if not a.startswith("--")]

    if len(positional) == 2:
        # Folder mode
        process_folder(positional[0], positional[1], debug=debug)

    elif len(positional) == 1:
        target = positional[0]
        if os.path.isdir(target):
            # Folder mode với output mặc định
            out_dir = target.rstrip("/\\") + "_check"
            process_folder(target, out_dir, debug=debug)
        else:
            # Single file mode
            result = detect_basket_circle(target, debug=debug)
            if result:
                print(f"\nKết quả: {target}")
                print(f"  Tâm     : ({result['center_x']}, {result['center_y']})")
                print(f"  Bán kính: {result['radius']} px")
                print(f"  Ảnh     : {result['image_width']} × {result['image_height']} px")
    else:
        print("[LỖI] Sai cú pháp. Chạy không có tham số để xem hướng dẫn.")
        sys.exit(1)