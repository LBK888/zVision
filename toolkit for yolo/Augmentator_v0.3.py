'''
Augmentation by CCW, CW and flip rotations to increase the train data from Roboflow
suitable for microscopy or marco photography from top/bottom view images
version: 2025-05-22 v0.3

with test run function to preview the results.
* Only for Roboflow's YOLO keypoint format
* also cleans up the keypoints coords with >1 values, which seems useless for training
* tested on python 3.10.12
'''

import os
import cv2
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from tqdm import tqdm

# 取得main.py所在的目錄路徑
ROOT_DIR = Path(__file__).parent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===========================
# 變換公式：對於 normalized 標籤數值（假設圖像大小固定為 640x640）
# ---------------------------
def transform_bbox(cx, cy, w, h, transform):
    if transform == 'ccw':  # 逆時針 90 度
        new_cx = cy
        new_cy = 1 - cx
        new_w = h
        new_h = w
    elif transform == 'cw':  # 順時針 90 度
        new_cx = 1 - cy
        new_cy = cx
        new_w = h
        new_h = w
    elif transform == 'hflip':  # 水平翻轉
        new_cx = 1 - cx
        new_cy = cy
        new_w = w
        new_h = h
    elif transform == 'vflip':  # 垂直翻轉
        new_cx = cx
        new_cy = 1 - cy
        new_w = w
        new_h = h
    else:
        new_cx, new_cy, new_w, new_h = cx, cy, w, h
    return new_cx, new_cy, new_w, new_h

def transform_keypoint(kp_x, kp_y, transform):
    """
    修正後的 transform_keypoint()：
    先提取 kp_x、kp_y 的小數部分（即網格內偏移量），再依據翻轉/旋轉操作計算新小數部分。
    
    例如：
      - 水平翻轉： new_frac_x = 1 - frac_x, new_frac_y = frac_y
      - 順時針旋轉 90°： new_frac_x = frac_y, new_frac_y = 1 - frac_x
    """
    # 取出小數部分（網格內偏移量）
    frac_x = kp_x - math.floor(kp_x)
    frac_y = kp_y - math.floor(kp_y)
    if transform == 'ccw':  # 逆時針 90 度
        new_frac_x = frac_y
        new_frac_y = 1 - frac_x 
    elif transform == 'cw':  # 順時針 90 度
        new_frac_x = 1 - frac_y 
        new_frac_y = frac_x 
    elif transform == 'hflip':  # 水平翻轉
        new_frac_x = 1 - frac_x
        new_frac_y = frac_y
    elif transform == 'vflip':  # 垂直翻轉
        new_frac_x = frac_x
        new_frac_y = 1 - frac_y
    else:
        new_frac_x, new_frac_y = frac_x, frac_y
    return new_frac_x, new_frac_y

def transform_label_line(line, transform):
    """
    將一行 YOLOv8 keypoint 標籤資料進行變換，
    格式：class cx cy w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...
    """
    parts = line.strip().split()
    if not parts:
        return ""
    cls = parts[0]
    cx = float(parts[1])
    cy = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])
    new_cx, new_cy, new_w, new_h = transform_bbox(cx, cy, w, h, transform)
    new_parts = [cls, f"{new_cx:.18f}", f"{new_cy:.18f}", f"{new_w:.18f}", f"{new_h:.18f}"]
    num_kpts = (len(parts) - 5) // 3
    for i in range(num_kpts):
        kp_x = float(parts[5 + 3*i])
        kp_y = float(parts[5 + 3*i + 1])
        kp_v = parts[5 + 3*i + 2]  # 保持可見性標記
        new_kp_x, new_kp_y = transform_keypoint(kp_x, kp_y, transform)
        new_parts.append(f"{new_kp_x:.18f}")
        new_parts.append(f"{new_kp_y:.18f}")
        new_parts.append(kp_v)
    return " ".join(new_parts)

def transform_image(img, transform):
    """
    對影像進行相應變換：
      - 'cw': 順時針 90 度旋轉
      - 'ccw': 逆時針 90 度旋轉
      - 'hflip': 水平鏡像
      - 'vflip': 垂直鏡像
    """
    if transform == 'cw':
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif transform == 'ccw':
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif transform == 'hflip':
        return cv2.flip(img, 1)
    elif transform == 'vflip':
        return cv2.flip(img, 0)
    else:
        return img

def generate_new_filename(original_filename, transform):
    """
    產生新檔名：
      - 如果原檔名中包含 ".rf."，則將它替換成 ".{transform}."。
      - 否則在副檔名前插入 ".{transform}"。
    """
    if ".rf." in original_filename:
        return original_filename.replace(".rf.", f".{transform}.")
    else:
        base, ext = os.path.splitext(original_filename)
        return base + f".{transform}" + ext
    
    #s = original_filename + transform + str(random.random())
    #return hashlib.md5(s.encode()).hexdigest()

# ---------------------------
# 繪製標註：bounding box 與 keypoints
# ---------------------------
def draw_annotations(img, label_lines, image_size=640):
    """
    根據 label 檔的每行資料，在影像上畫出 bounding box 與關鍵點。
    使用規則：
      - class 0 ("Good_Fish") 以藍色框線 (BGR: (255,0,0))
      - class 1 ("bad_Fish") 以紫色框線 (BGR: (255,0,255))
      - 第一個關鍵點：紅色圓點 (BGR: (0,0,255))
      - 第二個關鍵點：綠色圓點 (BGR: (0,255,0))
    """
    img_draw = img.copy()
    for line in label_lines:
        parts = line.strip().split()
        if not parts:
            continue
        cls = int(parts[0])
        if cls == 0:
            box_color = (240, 0, 0)   # 藍色
        elif cls == 1:
            box_color = (240, 0, 240) # 紫色
        else:
            box_color = (0, 240, 0)
        cx = float(parts[1])
        cy = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        abs_cx = cx * image_size
        abs_cy = cy * image_size
        abs_w = w * image_size
        abs_h = h * image_size
        x1 = int(abs_cx - abs_w/2)
        y1 = int(abs_cy - abs_h/2)
        x2 = int(abs_cx + abs_w/2)
        y2 = int(abs_cy + abs_h/2)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), box_color, 2)
        num_kpts = (len(parts) - 5) // 3
        for i in range(num_kpts):
            kp_x = float(parts[5 + 3*i])
            kp_y = float(parts[5 + 3*i + 1])
            kp_x_frac = kp_x - math.floor(kp_x)
            kp_y_frac = kp_y - math.floor(kp_y)
            abs_kp_x = int(kp_x_frac * image_size)
            abs_kp_y = int(kp_y_frac * image_size)
            kp_color = (0, 0, 255) if i == 0 else (0, 255, 0)
            cv2.circle(img_draw, (abs_kp_x, abs_kp_y), 3, kp_color, -1)

    return img_draw

# ===========================
# 資料集處理主程式
# ---------------------------
def process_dataset(base_dirs, transforms_to_apply, test_mode=False):
    """
    若 test_mode 為 True，僅處理 test 資料夾中的第一組檔案，
    並對原圖依照所有變換做處理，再並排顯示檢查結果。
    否則遍歷各資料夾，對每組影像與標籤進行使用者指定的變換，
    並產生新的檔案（不覆蓋原檔）。
    """
    if test_mode:
        # 測試模式僅從 test 資料夾挑選第一組影像與標籤
        test_images_dir = os.path.join(ROOT_DIR, "valid", "images")
        test_labels_dir = os.path.join(ROOT_DIR, "valid", "labels")
        if not os.path.exists(test_images_dir) or not os.path.exists(test_labels_dir):
            logging.warning(f"Test folder {test_images_dir} or {test_labels_dir} does not exist, skipping test mode.")
            return
        files = os.listdir(test_images_dir)
        if not files:
            logging.warning("No image files in test folder!")
            return
        file_chosen = files[0]
        image_path = os.path.join(test_images_dir, file_chosen)
        label_path = os.path.join(test_labels_dir, os.path.splitext(file_chosen)[0] + ".txt")
        img = cv2.imread(image_path)
        if not os.path.exists(label_path):
            logging.warning(f"Label file not found: {label_path}")
            return
        with open(label_path, "r") as f:
            label_lines = f.readlines()
        # 產生原圖及各變換後影像與標籤
        transformed_imgs = [img]
        transformed_labels = [label_lines]
        for t in transforms_to_apply:
            t_img = transform_image(img, t)
            t_labels = [transform_label_line(line, t) for line in label_lines]
            transformed_imgs.append(t_img)
            transformed_labels.append(t_labels)
        # 繪製標註
        annotated_imgs = [draw_annotations(im, labs) for im, labs in zip(transformed_imgs, transformed_labels)]
        n = len(annotated_imgs)
        plt.figure(figsize=(32,8))
        titles = ["Original"] + [f"Aug: {t}" for t in transforms_to_apply]
        for i, (img_disp, title) in enumerate(zip(annotated_imgs, titles)):
            img_rgb = cv2.resize(img_disp, (1280,1280), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            plt.subplot(1, n, i+1)
            plt.imshow(img_rgb)
            plt.title(title)
            plt.axis("off")
        plt.show()
    else:
        # 遍歷每個 base 目錄：train, test, valid
        for base in base_dirs:
            images_dir = os.path.join(ROOT_DIR, base, "images")
            labels_dir = os.path.join(ROOT_DIR, base, "labels")
            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                logging.warning(f"Folder {images_dir} or {labels_dir} does not exist, skipping.")
                continue
            files = os.listdir(images_dir)
            for file in tqdm(files, desc=f"Processing {base}"):
                if not (file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")):
                    continue
                image_path = os.path.join(images_dir, file)
                label_path = os.path.join(labels_dir, os.path.splitext(file)[0] + ".txt")
                if not os.path.exists(label_path):
                    logging.warning(f"Label file does not exist: {label_path}")
                    continue
                img = cv2.imread(image_path)
                with open(label_path, "r") as f:
                    label_lines = f.readlines()

                # 去除 keypoints 的整數部分 **Roboflow的label檔案，kps帶有超過1的標記，似乎會導致train model被剃除
                new_label_lines = []
                for line in label_lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    new_parts = parts[:5]  # 保留 class, cx, cy, w, h
                    num_kpts = (len(parts) - 5) // 3
                    for i in range(num_kpts):
                        kp_x = float(parts[5 + 3*i])
                        kp_y = float(parts[5 + 3*i + 1])
                        kp_v = parts[5 + 3*i + 2]
                        # 只保留小數部分
                        new_kp_x = kp_x - math.floor(kp_x)
                        new_kp_y = kp_y - math.floor(kp_y)
                        new_parts.append(f"{new_kp_x:.18f}")
                        new_parts.append(f"{new_kp_y:.18f}")
                        new_parts.append(kp_v)
                    new_label_lines.append(" ".join(new_parts))      
                # 覆蓋原有的標籤檔案
                with open(label_path, "w") as f:
                    for line in new_label_lines:
                        f.write(line + "\n")
                    logging.info(f"Fixed keypoints in: {label_path}")
                # 對每個指定變換進行處理
                for t in transforms_to_apply:
                    t_img = transform_image(img, t)
                    t_labels = [transform_label_line(line, t) for line in label_lines]
                    new_base = generate_new_filename(file, t)
                    new_image_path = os.path.join(images_dir, new_base + ".jpg")
                    new_label_path = os.path.join(labels_dir, new_base + ".txt")
                    cv2.imwrite(new_image_path, t_img)
                    with open(new_label_path, "w") as f_out:
                        for line in t_labels:
                            f_out.write(line + "\n")
                    logging.info(f"Saved: {new_image_path} and {new_label_path}")

# ===========================
# 主程式入口
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="使用多種變換增強 YOLOv8 keypoint 資料集")
    parser.add_argument("--mode", choices=["test", "process"], default="test",
                        help="測試模式只處理一組檔案並顯示結果；process 則遍歷所有檔案")
    parser.add_argument("--transforms", type=str, default="cw,ccw,hflip,vflip",
                        help="用逗號分隔的變換選項，選項：cw, ccw, hflip, vflip")
    args = parser.parse_args()
    transforms_to_apply = [t.strip() for t in args.transforms.split(",") if t.strip()]
    base_dirs = ["train", "test", "valid"]
    process_dataset(base_dirs, transforms_to_apply, test_mode=(args.mode=="test"))

if __name__ == "__main__":
    main()
