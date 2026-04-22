"""
Cấu trúc thư mục yêu cầu:
    images/backgrounds/
        00.jpg, 01.jpg, ...      <- ảnh rổ chụp từ trên xuống
        baskets_params.json      <- {"00.jpg": {"center": [x,y], "radius": r}}
    images/shrimps/
        00.png, 01.png, ...      <- ảnh tôm RGBA đã tách nền
    dataset/
        images/                  <- output ảnh RGB .jpg
        labels/                  <- output nhãn .txt (mỗi dòng "X Y")
"""
import cv2
import os
import sys

# Đảm bảo working directory luôn là thư mục gốc Generate Synthetic Dataset
project_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(project_dir) == "src":
    project_dir = os.path.dirname(project_dir)
os.chdir(project_dir)

import json
import math
import random
import numpy as np
from PIL import Image
import concurrent.futures
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH TOÀN CỤC
# ─────────────────────────────────────────────────────────────────────────────
N_IMAGES          = 5         # số ảnh tổng hợp cần sinh
SAFE_RATIO        = 0.85         # R_safe = R_rổ * SAFE_RATIO  (tôm chỉ rải trong vùng này)
TARGET_SIZE_RATIO = 0.60         # cạnh dài tôm (sau scale) = R_rổ * tỉ lệ này
OCCLUSION_RATIO   = 0          # tỷ lệ che khuất tối đa cho phép (cả tôm mới lẫn tôm cũ)
SCALE_JITTER      = (0.88, 1.11) # nhiễu ngẫu nhiên nhân thêm vào scale_base
ALPHA_THRESHOLD   = 32           # ngưỡng alpha để coi pixel là "thịt tôm" (0–255)
                                 # = 32/255 ≈ 12.5%: loại bỏ viền mờ do nội suy khi rotate/resize
RETRY_JITTER      = 25           # số lần thử dịch chuyển nhỏ trước khi bỏ qua một vị trí

# Tham số xoắn ốc (Spiral Spreading)
SPIRAL_DR_RATIO   = 0.04         # bước tiến bán kính mỗi vòng = R_safe * tỉ lệ này
                                 # 0.04 → ~25 vòng/rổ; 0.10 → ~10 vòng (quá thưa)
SPIRAL_ANGLE_STEP = 15           # bước góc quét tối thiểu (độ) – giới hạn dưới của adaptive step
ANGLE_JITTER      = 10           # ± nhiễu góc quét (độ)
RADIUS_JITTER     = 0.08         # ± nhiễu bán kính tương đối (tỉ lệ r_current)

ROT_RANGE_CENTER  = 360          # biên xoay ở tâm: 0° ÷ 360° (hoàn toàn tự do)
ROT_RANGE_EDGE    = 60           # biên xoay ở rìa: tangent ± 30°

BG_DIR  = "images/backgrounds"
FG_DIR  = "images/foreground_augmented"
OUT_DIR = "datasets/Segments1"


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 1: TIỆN ÍCH TÍNH TOÁN
# ═════════════════════════════════════════════════════════════════════════════

def alpha_to_binary_mask(pil_rgba: "Image.Image") -> np.ndarray:
    """Trích kênh Alpha của ảnh RGBA → mảng bool 2D (H×W).
    pixel = True  nếu alpha > ALPHA_THRESHOLD  (thịt tôm thực sự)
    pixel = False nếu alpha ≤ ALPHA_THRESHOLD  (trong suốt / viền mờ do nội suy)
    """
    return np.array(pil_rgba)[:, :, 3] > ALPHA_THRESHOLD


def check_instance_collision(instance_map: np.ndarray,
                             occlusion_ratio: float,
                             shrimp_stats: dict,
                             new_mask: np.ndarray,
                             paste_x: int, paste_y: int,
                             canvas_w: int, canvas_h: int) -> bool:
    """
    Kiểm tra xem có thể dán tôm mới tại (paste_x, paste_y) mà không vi phạm
    ngưỡng che khuất không.

    Hai điều kiện phải đồng thời thỏa mãn:
      1. Tôm MỚI không bị che bởi tôm cũ quá occlusion_ratio diện tích của chính nó.
      2. Không tôm CŨ nào bị tôm mới che thêm đến mức phần lộ còn lại < (1 - occlusion_ratio).

    Trả về:
        True  → vi phạm, KHÔNG dán
        False → an toàn,  cho phép dán
    """
    sh, sw = new_mask.shape
    clip_x1, clip_y1 = max(0, paste_x), max(0, paste_y)
    clip_x2, clip_y2 = min(canvas_w, paste_x + sw), min(canvas_h, paste_y + sh)

    # Tôm hoàn toàn ra ngoài canvas → từ chối
    if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
        return True

    # Cắt vùng tương ứng trên instance_map và mask tôm mới
    map_cropped      = instance_map[clip_y1:clip_y2, clip_x1:clip_x2]
    new_mask_cropped = new_mask[clip_y1 - paste_y : clip_y2 - paste_y,
                                clip_x1 - paste_x : clip_x2 - paste_x]

    # Các pixel của instance_map mà mask tôm mới dẫm lên
    covered_pixels = map_cropped[new_mask_cropped]
    covered_ids    = covered_pixels[covered_pixels > 0]  # bỏ nền (ID = 0)

    if len(covered_ids) > 0:
        # Điều kiện 1: tôm mới đè lên tôm cũ không quá occlusion_ratio
        new_area = np.count_nonzero(new_mask_cropped)
        if len(covered_ids) / new_area >= occlusion_ratio:
            return True

        # Điều kiện 2: không tôm cũ nào bị che thêm đến mức < (1 - occlusion_ratio)
        unique_ids, loss_counts = np.unique(covered_ids, return_counts=True)
        for uid, loss in zip(unique_ids, loss_counts):
            stats = shrimp_stats[uid]
            simulated_remaining = stats['visible'] - loss
            if (simulated_remaining / stats['original']) < (1.0 - occlusion_ratio):
                return True

    return False


def stamp_instance_map(instance_map: np.ndarray,
                       shrimp_stats: dict,
                       new_mask: np.ndarray,
                       paste_x: int, paste_y: int,
                       canvas_w: int, canvas_h: int,
                       shrimp_id: int) -> None:
    """
    Ghi nhận tôm mới vào instance_map và cập nhật shrimp_stats.
    Gọi hàm này CHỈ SAU KHI check_instance_collision trả về False.

    Các bước:
      1. Trừ số pixel bị che khỏi 'visible' của từng tôm cũ bị dẫm lên.
      2. Ghi ID tôm mới vào các pixel tương ứng trên instance_map.
      3. Đăng ký tôm mới vào shrimp_stats với 'original' = toàn bộ mask
         (trước clip) để tránh sai lệch khi tôm nằm ở mép canvas.
    """
    sh, sw = new_mask.shape
    clip_x1, clip_y1 = max(0, paste_x), max(0, paste_y)
    clip_x2, clip_y2 = min(canvas_w, paste_x + sw), min(canvas_h, paste_y + sh)

    if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
        return

    map_cropped      = instance_map[clip_y1:clip_y2, clip_x1:clip_x2]
    new_mask_cropped = new_mask[clip_y1 - paste_y : clip_y2 - paste_y,
                                clip_x1 - paste_x : clip_x2 - paste_x]

    # Bước 1: cập nhật visible của các tôm cũ bị tôm mới đè lên
    covered_pixels = map_cropped[new_mask_cropped]
    covered_ids    = covered_pixels[covered_pixels > 0]
    if len(covered_ids) > 0:
        unique_ids, loss_counts = np.unique(covered_ids, return_counts=True)
        for uid, loss in zip(unique_ids, loss_counts):
            shrimp_stats[uid]['visible'] -= loss

    # Bước 2: đóng dấu ID tôm mới (boolean indexing ghi trực tiếp lên instance_map gốc)
    instance_map[clip_y1:clip_y2, clip_x1:clip_x2][new_mask_cropped] = shrimp_id

    # Bước 3: đăng ký tôm mới – 'original' dùng toàn bộ mask gốc (trước clip)
    area = int(np.count_nonzero(new_mask))
    shrimp_stats[shrimp_id] = {'original': area, 'visible': area}



def merge_contours_bridge(contours: list) -> np.ndarray:
    """Nối nhiều contour rời thành 1 polygon duy nhất bằng cầu nối (bridge).

    Khi một con tôm bị tôm khác đè cắt thành 2+ mảnh rời, mỗi mảnh là 1
    contour riêng.  Hàm này tìm cặp điểm gần nhất giữa các contour rồi
    "bắc cầu" nối chúng lại thành 1 polygon liền mạch, đảm bảo mỗi con tôm
    chỉ sinh 1 dòng label duy nhất -> giữ đúng count.

    Thuật toán:
      1. Sắp xếp contour theo diện tích giảm dần.
      2. Bắt đầu từ contour lớn nhất (merged).
      3. Với mỗi contour còn lại, tìm cặp điểm gần nhất giữa merged và
         contour đó (bằng broadcasting NumPy), rồi chèn contour mới vào
         merged tại vị trí cầu nối.

    Trả về: contour đã merge, shape (N, 1, 2), dtype int32.
    """
    if len(contours) == 1:
        return contours[0]

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    merged = contours[0].reshape(-1, 2).astype(np.float64)

    for contour in contours[1:]:
        other = contour.reshape(-1, 2).astype(np.float64)

        # Tìm cặp điểm gần nhất bằng broadcasting
        diff = merged[:, np.newaxis, :] - other[np.newaxis, :, :]
        dist_sq = (diff ** 2).sum(axis=2)
        flat_idx = dist_sq.argmin()
        best_i, best_j = np.unravel_index(flat_idx, dist_sq.shape)

        # Xoay other để bắt đầu từ điểm cầu nối (best_j)
        other_rolled = np.roll(other, -best_j, axis=0)

        # Nối: merged[0..best_i] -> cầu -> other (đi vòng) -> cầu -> merged[best_i..end]
        merged = np.vstack([
            merged[:best_i + 1],
            other_rolled,
            other_rolled[:1],
            merged[best_i:]
        ])

    return merged.astype(np.int32).reshape(-1, 1, 2)


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 2: NẠP DỮ LIỆU
# ═════════════════════════════════════════════════════════════════════════════

def load_backgrounds(bg_dir: str) -> list:
    """
    Nạp ảnh nền và thông số rổ từ baskets_params.json.
    Trả về: [{"path", "image": PIL RGB, "center": (cx,cy), "radius": r}, ...]
    """
    json_path = os.path.join(bg_dir, "baskets_params.json")
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    backgrounds = []
    for fname, info in meta.items():
        img_path = os.path.join(bg_dir, fname)
        if not os.path.exists(img_path):
            print(f"  [WARN] Không tìm thấy background: {img_path}")
            continue
        backgrounds.append({
            "path"  : img_path,
            "image" : Image.open(img_path).convert("RGB"),
            "center": tuple(info["center"]),
            "radius": info["radius"],
        })
    print(f"[INFO] Nạp {len(backgrounds)} ảnh background.")
    return backgrounds


def load_foregrounds(fg_dir: str) -> list:
    """
    Quét thư mục fg_dir và nạp đường dẫn tất cả ảnh PNG (Lazy Loading).
    Image.open() sẽ được gọi sau trong process_single_image khi con tôm
    đó được bốc ra dùng.
    Trả về: [{"path": str}, ...]
    """
    foregrounds = []
    for fname in sorted(os.listdir(fg_dir)):
        if not fname.lower().endswith(".png"):
            continue
        img_path = os.path.join(fg_dir, fname)
        foregrounds.append({"path": img_path})
    print(f"[INFO] Tìm thấy {len(foregrounds)} ảnh foreground (tôm) – Lazy Loading.")
    return foregrounds
    """
    Đọc metadata tôm từ shrimp_centers_aug.json (Lazy Loading).
    Chỉ lưu đường dẫn file và tọa độ tâm; nhành Image.open() sẽ được gọi
    sau trong process_single_image khi con tôm đó được bốc ra dùng.
    Trả về: [{"path": str, "center": (cx, cy)}, ...]
    """
    json_path = os.path.join(fg_dir, "shrimp_centers_aug.json")
    with open(json_path, "r", encoding="utf-8") as f:
        centers = json.load(f)

    foregrounds = []
    for fname, info in centers.items():
        img_path = os.path.join(fg_dir, fname)
        if not os.path.exists(img_path):
            print(f"  [WARN] Không tìm thấy foreground: {img_path}")
            continue
        foregrounds.append({
            "path"  : img_path
        })
    print(f"[INFO] Đọc metadata {len(foregrounds)} ảnh foreground (tôm) – Lazy Loading.")
    return foregrounds


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 3: PIPELINE CHÍNH – RẢI TÔM THEO XOẮN ỐC
# ═════════════════════════════════════════════════════════════════════════════

def place_shrimp_on_basket(canvas: Image.Image,
                           basket_cx: float, basket_cy: float,
                           basket_r: float,
                           foregrounds: list) -> list:
    """
    Rải tôm lên canvas theo xoắn ốc từ tâm ra rìa; kiểm soát che khuất.
    TRẢ VỀ `instance_map` (dùng để sinh nhãn Segmentation).
    """
    R_safe    = basket_r * SAFE_RATIO
    spiral_dr = R_safe * SPIRAL_DR_RATIO  # bước tiến bán kính, tỉ lệ với R_safe

    # Instance Map: mảng int32 cùng kích thước canvas.
    # Giá trị 0 = nền trống; giá trị k = pixel thuộc tôm thứ k.
    canvas_w, canvas_h = canvas.size
    instance_map    = np.zeros((canvas_h, canvas_w), dtype=np.int32)
    shrimp_stats    = {}   # {shrimp_id: {'original': int, 'visible': int}}
    current_shrimp_id = 1

    r_current        = spiral_dr  # bắt đầu từ vòng đầu tiên (không từ 0 để tránh tụ tâm)
    last_ring_failed = False       # dừng sớm khi 2 vòng ngoài cùng liên tiếp không đặt được

    while r_current <= R_safe:
        ring_placed = False

        # Bước góc thích nghi: arc = r·Δθ ≈ spiral_dr → Δθ = degrees(spiral_dr / r)
        # Giới hạn trong [SPIRAL_ANGLE_STEP, 90°] để không quá thưa / dày
        adaptive_step = math.degrees(spiral_dr / r_current)
        effective_step = max(SPIRAL_ANGLE_STEP, min(90.0, adaptive_step))

        angle = 0.0
        while angle < 360.0:

            # ── Tọa độ điểm rải với nhiễu góc và bán kính ──────────────
            angle_jittered = angle + random.uniform(-ANGLE_JITTER, ANGLE_JITTER)
            r_jittered     = max(0.0, r_current + random.uniform(
                                     -r_current * RADIUS_JITTER,
                                      r_current * RADIUS_JITTER))
            rad    = math.radians(angle_jittered)
            drop_x = basket_cx + r_jittered * math.cos(rad)
            drop_y = basket_cy + r_jittered * math.sin(rad)

            # Bỏ qua nếu điểm rải ra ngoài vùng an toàn
            if math.hypot(drop_x - basket_cx, drop_y - basket_cy) > R_safe:
                angle += effective_step
                continue

            # ── Chọn tôm ngẫu nhiên & Scale ────────────────────────────
            shrimp_data = random.choice(foregrounds)
            with Image.open(shrimp_data["path"]) as _f:
                shrimp_img = _f.convert("RGBA")
            orig_w, orig_h = shrimp_img.size
            # scale để cạnh dài nhất đạt TARGET_SIZE_RATIO * R_rổ, thêm jitter
            scale_final = (basket_r * TARGET_SIZE_RATIO / max(orig_w, orig_h)) \
                          * random.uniform(*SCALE_JITTER)

            new_w = max(1, int(orig_w * scale_final))
            new_h = max(1, int(orig_h * scale_final))
            shrimp_scaled = shrimp_img.resize((new_w, new_h), resample=Image.LANCZOS)

            # ── Xoay động theo vị trí trên vòng xoắn ──────────────────
            # t=0 (tâm) → xoay hoàn toàn tự do; t=1 (rìa) → nằm theo tiếp tuyến
            t_radius  = min(1.0, r_current / R_safe)
            rot_range = ROT_RANGE_CENTER * (1.0 - t_radius) + ROT_RANGE_EDGE * t_radius

            # Góc tiếp tuyến: vuông góc với vector tâm→điểm rải
            tangent_deg = math.degrees(math.atan2(drop_y - basket_cy,
                                                  drop_x - basket_cx)) + 90.0
            angle_rot = tangent_deg + random.uniform(-rot_range / 2.0, rot_range / 2.0)

            # PIL rotate(θ) = ngược chiều KĐH, expand=True để không cắt viền
            shrimp_rotated = shrimp_scaled.rotate(angle_rot, expand=True,
                                                  resample=Image.BICUBIC)
            rot_w, rot_h = shrimp_rotated.size

            # Binary mask tôm sau xoay (tính 1 lần, dùng lại cho mọi retry)
            shrimp_mask_bin = alpha_to_binary_mask(shrimp_rotated)

            # ── Retry Jitter: dò khe hở lân cận trước khi bỏ qua góc ──
            # Thử offset (0,0) trước, rồi RETRY_JITTER lần nhiễu ngẫu nhiên ±½·spiral_dr
            jitter_half = spiral_dr * 0.5
            offsets = [(0.0, 0.0)] + [
                (random.uniform(-jitter_half, jitter_half),
                 random.uniform(-jitter_half, jitter_half))
                for _ in range(RETRY_JITTER)
            ]

            for dx_retry, dy_retry in offsets:
                # Vị trí dán: ép tâm bounding box ≡ điểm rải (+ retry offset)
                paste_x = int(round(drop_x + dx_retry - rot_w / 2.0))
                paste_y = int(round(drop_y + dy_retry - rot_h / 2.0))

                collision = check_instance_collision(
                    instance_map=instance_map,
                    occlusion_ratio=OCCLUSION_RATIO,
                    shrimp_stats=shrimp_stats,
                    new_mask=shrimp_mask_bin,
                    paste_x=paste_x, paste_y=paste_y,
                    canvas_w=canvas_w, canvas_h=canvas_h
                )

                if not collision:
                    # Dán tôm lên canvas – dùng binary mask để đồng bộ với instance_map
                    # (tránh pixel viền mờ alpha thấp ghi đè ảnh mà không có trong mask)
                    paste_alpha = Image.fromarray(
                        (shrimp_mask_bin * 255).astype(np.uint8)
                    )
                    shrimp_paste = shrimp_rotated.copy()
                    shrimp_paste.putalpha(paste_alpha)
                    canvas.paste(shrimp_paste, (paste_x, paste_y),
                                 mask=paste_alpha)

                    # Cập nhật Instance Map & shrimp_stats
                    stamp_instance_map(
                        instance_map=instance_map,
                        shrimp_stats=shrimp_stats,
                        new_mask=shrimp_mask_bin,
                        paste_x=paste_x, paste_y=paste_y,
                        canvas_w=canvas_w, canvas_h=canvas_h,
                        shrimp_id=current_shrimp_id
                    )

                    current_shrimp_id += 1
                    ring_placed = True
                    break  # sang góc tiếp theo ngay khi tìm được khe hở

            angle += effective_step

        # Điều kiện dừng sớm: 2 vòng ngoài liên tiếp không đặt được tôm nào
        if r_current >= R_safe and not ring_placed:
            if last_ring_failed:
                break
            last_ring_failed = True
        else:
            last_ring_failed = False

        r_current += spiral_dr

    return instance_map


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 4: SINH N ẢNH – ĐA NHÂN CPU
# ═════════════════════════════════════════════════════════════════════════════

def process_single_image(idx, backgrounds, foregrounds, img_out_dir, label_out_dir):
    """Sinh 1 ảnh tổng hợp; được gọi song song bởi ProcessPoolExecutor."""
    bg_data  = random.choice(backgrounds)
    canvas   = bg_data["image"].copy().convert("RGBA")
    basket_cx, basket_cy = bg_data["center"]
    basket_r             = bg_data["radius"]
    canvas_w, canvas_h   = canvas.size
    
    instance_map = place_shrimp_on_basket(
        canvas=canvas,
        basket_cx=basket_cx, basket_cy=basket_cy,
        basket_r=basket_r,
        foregrounds=foregrounds
    )

    # ── TRÍCH XUẤT POLYGON TỪ INSTANCE MAP ──
    yolo_seg_lines = []
    unique_ids = np.unique(instance_map)
    
    for uid in unique_ids:
        if uid == 0: 
            continue # Bỏ qua ID=0 (nền trống)
            
        # Tạo binary mask cho từng con tôm cụ thể
        binary_mask = (instance_map == uid).astype(np.uint8) * 255

        # Morphological closing: lấp lỗ nhỏ bên trong thân tôm cong chữ C
        # Kernel size tỉ lệ với kích thước tôm (adaptive)
        coords = np.argwhere(binary_mask > 0)
        if len(coords) == 0:
            continue
        bbox_h = coords[:, 0].max() - coords[:, 0].min() + 1
        bbox_w = coords[:, 1].max() - coords[:, 1].min() + 1
        k_size = max(3, int(min(bbox_h, bbox_w) * 0.15)) | 1  # luôn lẻ
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # Tìm đường viền contour
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        # Lọc contour có diện tích đáng kể (>= 1% tổng diện tích)
        total_area = sum(cv2.contourArea(c) for c in contours)
        min_area = total_area * 0.01
        significant = [c for c in contours if cv2.contourArea(c) >= min_area]

        if not significant:
            continue

        # Bắc cầu (Bridge Merging): nối các mảnh rời thành 1 polygon duy nhất
        merged_contour = merge_contours_bridge(significant)

        # Giảm số lượng điểm polygon (Làm mượt để file txt không quá nặng)
        epsilon = 0.002 * cv2.arcLength(merged_contour, True)
        approx_polygon = cv2.approxPolyDP(merged_contour, epsilon, True)

        # Tiền xử lý tọa độ: Chuẩn hóa [0, 1] cho YOLO
        poly_points = []
        for point in approx_polygon:
            x, y = point[0]
            nx = round(x / canvas_w, 6)
            ny = round(y / canvas_h, 6)
            poly_points.extend([nx, ny])

        # YOLO yêu cầu đa giác phải có ít nhất 3 điểm (6 tọa độ x,y)
        if len(poly_points) >= 6:
            # class_id = 0
            line_str = "0 " + " ".join(map(str, poly_points))
            yolo_seg_lines.append(line_str)

    # ── Lưu file
    out_name = f"{idx:04d}"

    img_save_path = os.path.join(img_out_dir, f"{out_name}.jpg")
    canvas.convert("RGB").save(img_save_path, quality=95)

    lbl_save_path = os.path.join(label_out_dir, f"{out_name}.txt")
    with open(lbl_save_path, "w", encoding="utf-8") as f:
        for line in yolo_seg_lines:
            f.write(line + "\n")

    return len(yolo_seg_lines)


def generate_dataset_multicore(n_images: int, bg_dir: str, fg_dir: str, out_dir: str):
    """Phân phối n_images task xuống ProcessPoolExecutor (tất cả nhân CPU)."""
    img_out_dir   = os.path.join(out_dir, "images")
    label_out_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_out_dir,   exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    backgrounds = load_backgrounds(bg_dir)
    foregrounds = load_foregrounds(fg_dir)

    if not backgrounds or not foregrounds:
        raise RuntimeError("Thiếu dữ liệu background hoặc foreground!")

    print(f"[INFO] Bắt đầu sinh {n_images} ảnh bằng đa nhân CPU...")
    print(f"[INFO] Dataset sẽ lưu tại: {os.path.abspath(out_dir)}\n")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_single_image,
                            idx, backgrounds, foregrounds,
                            img_out_dir, label_out_dir)
            for idx in range(n_images)
        ]
        with tqdm(total=n_images, unit="ảnh", desc="Sinh dataset") as pbar:
            for future in concurrent.futures.as_completed(futures):
                n_shrimps = future.result()
                pbar.update(1)
                pbar.set_postfix(tôm=n_shrimps)

    print(f"\n[DONE] Dataset lưu tại: {os.path.abspath(out_dir)}")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    generate_dataset_multicore(
        n_images=N_IMAGES,
        bg_dir=BG_DIR,
        fg_dir=FG_DIR,
        out_dir=OUT_DIR
    )