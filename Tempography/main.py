"""
Tempography 影片時間套色處理工具 - v0.1b - 2025-05-28

功能說明:
- 支援批次處理資料夾內多部影片，或單一影片
- 可手動或自動指定 ROI
- 自動計算背景（支援 mode/median/average）
- 去除背景後，對所有 frame 疊圖並套用 rainbow colormap
- 結果自動儲存為 PNG 圖片

安裝需求:
- Python 3.7+
- opencv-python
- numpy
- scipy

安裝方式:
pip install opencv-python numpy scipy

使用方法:
# 處理資料夾內所有影片，手動選 ROI
python main.py --folder <影片資料夾> --show

# 自動沿用同一 ROI 處理所有影片
python main.py --folder <影片資料夾> --roi 100,100,300,300

# 只處理單一影片
python main.py --file <影片路徑> --roi 100,100,300,300

參數說明:
--folder      影片資料夾路徑
--file        指定單一影片路徑
--roi         指定 ROI (格式: x,y,w,h)
--bg_method   背景計算方法 (mode/medi/avg)
--max_bg_frames 背景計算最多取幾張 frame
--stack_method 疊圖方式 (mean/median)
--show        顯示結果
--save_dir    結果儲存資料夾
"""

import cv2
import numpy as np
import statistics as stats
import os
import glob
import argparse
import tkinter as tk
from tkinter import filedialog
from scipy import stats as scipy_stats
from tqdm import tqdm


def select_roi(frame, window_title='Select Region, then press ENTER', filename=None):
    """
    讓使用者從指定的幀中選擇感興趣區域（ROI）。
    若 filename 不為 None，會在左上角顯示檔案名稱。
    """
    display_frame = frame.copy()
    if filename:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display_frame, filename, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, frame.shape[1] // 2, frame.shape[0] // 2)
    cv2.imshow(window_title, display_frame)
    rect_box = cv2.selectROI(window_title, display_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    return rect_box


def calculate_background(video_cap, bg_method='mode', max_frames=129, frame_skip=10):
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= frame_skip:
        raise ValueError('影片長度不足')
    # 均勻抽樣 frame index
    sample_indices = np.linspace(frame_skip, total_frames-1, min(max_frames, total_frames-frame_skip), dtype=int)
    frames = []
    for idx in tqdm(sample_indices, desc='背景計算/抽樣frame'):
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = video_cap.read()
        if not ret:
            continue
        frames.append(frame)
    if not frames:
        raise ValueError('無法抽取背景 frame')
    frames_np = np.stack(frames, axis=0)
    if bg_method == 'medi':
        print('使用中位數median模式計算背景中...')
        background = np.median(frames_np, axis=0)
    elif bg_method == 'mode':
        # 用 scipy.stats.mode 處理多維陣列
        print('使用眾數mode模式計算背景中，本步驟運算時間較長...')
        background = scipy_stats.mode(frames_np, axis=0, keepdims=True)[0][0]
    else:  # avg
        print('使用平均mean模式計算背景中...')
        background = np.mean(frames_np, axis=0)
    return background.astype(np.uint8)


def remove_background_frm(frame, background):
    return cv2.absdiff(frame, background)


def apply_colormap_and_stack(frames, colormap=cv2.COLORMAP_RAINBOW, stack_method='max'):
    colored_frames = [cv2.applyColorMap(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), colormap) for f in tqdm(frames, desc='套用colormap')]
    stack_arr = np.stack(colored_frames, axis=0)
    if stack_method == 'max':
        stacked = np.max(stack_arr, axis=0)
    elif stack_method == 'median':
        stacked = np.median(stack_arr, axis=0)
    else:
        stacked = np.mean(stack_arr, axis=0)
    return stacked.astype(np.uint8)


def get_rainbow_color(idx, total):
    # HSV 色環分布，H: 0~179 (OpenCV)，S:255, V:255
    h = int(179 * idx / max(total-1, 1))
    color = cv2.cvtColor(np.uint8([[[h, 255, 255]]]), cv2.COLOR_HSV2BGR)[0,0]
    return tuple(int(x) for x in color)  # (B, G, R)


def colorize_gray(gray_img, color):
    # gray_img: 2D, color: (B, G, R)
    color_img = np.zeros((*gray_img.shape, 3), dtype=np.uint8)
    for c in range(3):
        color_img[..., c] = (gray_img.astype(np.float32) * (color[c]/255)).astype(np.uint8)
    return color_img


def select_roi_frame(video_path, filename=None, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = frame_skip if total_frames > frame_skip else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print(f"無法讀取影片: {video_path}")
        return None
    return select_roi(frame, filename=filename)


def process_video(video_path, roi=None, bg_method='mode', max_bg_frames=500, show_result=False, save_dir=None, stack_method='max', frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法開啟影片: {video_path}")
        return
    total_frames_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 只處理 frame_skip 之後的 frame
    if total_frames_all <= frame_skip:
        print(f"影片長度不足，無法跳過 {frame_skip} 幀: {video_path}")
        cap.release()
        return
    total_frames = total_frames_all - frame_skip
    if roi is None:
        roi = select_roi_frame(video_path, filename=os.path.basename(video_path), frame_skip=frame_skip)
        if roi is None:
            cap.release()
            return
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_skip)
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        print(f"無法讀取影片: {video_path}")
        cap.release()
        return
    x, y, w, h = roi
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_skip)
    n_bg_frames = min(max(int(total_frames * 0.1), 50), max_bg_frames)
    try:
        background = calculate_background(cap, bg_method=bg_method, max_frames=n_bg_frames, frame_skip=frame_skip)  
    except Exception as e:
        print(f"背景計算失敗: {e}")
        return
    # 儲存背景圖
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(video_path))[0]
        bg_out_path = os.path.join(save_dir, f"{base}_background.png")
        result, encoded_bg = cv2.imencode('.png', background)
        if result:
            encoded_bg.tofile(bg_out_path)
            print(f"已儲存背景圖: {bg_out_path}")
        else:
            print(f"儲存背景圖失敗: {bg_out_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_skip)
    stacked = None
    stack_list = [] if stack_method == 'median' else None
    count = 0
    # 建立去背影片 writer
    fg_video_path = None
    fg_writer = None
    if save_dir:
        fg_video_path = os.path.join(save_dir, f"{base}_fg.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        fg_size = (w, h)
        fg_writer = cv2.VideoWriter(fg_video_path, fourcc, fps, fg_size)
    for i in tqdm(range(total_frames), desc='移除背景+colormap'):
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"frame 讀取失敗於 frame {i+frame_skip}，跳過剩餘 frame")
                break
            fg = remove_background_frm(frame, background)
            fg_roi = fg[y:y+h, x:x+w]
            # 寫入去背影片
            if fg_writer is not None:
                fg_writer.write(fg_roi)
            gray = cv2.cvtColor(fg_roi, cv2.COLOR_BGR2GRAY)
            color = get_rainbow_color(i, total_frames)
            colored = colorize_gray(gray, color)
            if stack_method == 'median':
                stack_list.append(colored)
            elif stack_method == 'max':
                if stacked is None:
                    stacked = colored.astype(np.float32)
                else:
                    stacked = np.maximum(stacked, colored.astype(np.float32))
            else:
                if stacked is None:
                    stacked = np.zeros_like(colored, dtype=np.float64)
                stacked += colored.astype(np.float64)
            count += 1
        except Exception as e:
            print(f"frame 讀取異常於 frame {i+frame_skip}: {e}")
            break
    cap.release()
    if fg_writer is not None:
        fg_writer.release()
        print(f"已儲存去背影片: {fg_video_path}")
    if count == 0 or (stack_method == 'median' and not stack_list):
        print(f"無法取得前景 frame: {video_path}")
        return
    if stack_method == 'median':
        stacked = np.median(np.stack(stack_list, axis=0), axis=0)
    elif stack_method == 'max':
        pass  # 已經是最大值
    else:
        stacked /= count
    stacked = stacked.astype(np.uint8)
    if save_dir:
        out_path = os.path.join(save_dir, f"{base}_stacked.png")
        ext = os.path.splitext(out_path)[1]
        result, encoded_img = cv2.imencode(ext, stacked)
        if result:
            encoded_img.tofile(out_path)
            print(f"已儲存: {out_path}")
        else:
            print(f"儲存失敗: {out_path}")
    if show_result:
        cv2.imshow('Stacked Result', stacked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return stacked


def parse_roi(roi_str):
    try:
        x, y, w, h = map(int, roi_str.split(','))
        return (x, y, w, h)
    except:
        raise argparse.ArgumentTypeError('ROI 格式需為 x,y,w,h')


def main():
    parser = argparse.ArgumentParser(description='Tempography 影片時間套色處理')
    parser.add_argument('--folder', type=str, help='影片資料夾路徑')
    parser.add_argument('--file', type=str, help='單一影片路徑')
    parser.add_argument('--roi', type=str, help='指定 ROI (格式: x,y,w,h)')
    parser.add_argument('--bg_method', type=str, default='mode', choices=['mode','medi','avg'], help='背景計算方法')
    parser.add_argument('--max_bg_frames', type=int, default=150, help='背景計算最多取幾張frame')
    parser.add_argument('--stack_method', type=str, default='max', choices=['max','mean','median'], help='疊圖方式')
    parser.add_argument('--frame_skip', type=int, default=10000, help='前面要跳過的frame數')
    parser.add_argument('--show', action='store_true', help='顯示結果')
    parser.add_argument('--save_dir', type=str, default=None, help='結果儲存資料夾')
    args = parser.parse_args()

    if not args.folder and not args.file:
        print('未指定 --folder 或 --file，請選擇影片資料夾...')
        root = tk.Tk()
        root.withdraw()
        folder_selected = filedialog.askdirectory(title='請選擇影片資料夾')
        if not folder_selected:
            print('未選擇資料夾，程式結束')
            return
        args.folder = folder_selected
    roi = parse_roi(args.roi) if args.roi else None
    video_files = []
    if args.file:
        video_files = [args.file]
    else:
        exts = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        for ext in exts:
            video_files.extend(glob.glob(os.path.join(args.folder, ext)))
    if not video_files:
        print('找不到影片檔案')
        return
    if not args.save_dir:
        base_folder = args.folder if args.folder else os.path.dirname(args.file)
        args.save_dir = os.path.join(base_folder, 'output')
    rois = {}
    if not roi:
        for video_path in video_files:
            roi_val = select_roi_frame(video_path, filename=os.path.basename(video_path), frame_skip=args.frame_skip)
            if roi_val is not None:
                rois[video_path] = roi_val
    for video_path in video_files:
        print(f"處理: {video_path}")
        this_roi = roi if roi else rois.get(video_path)
        process_video(video_path, roi=this_roi, bg_method=args.bg_method, max_bg_frames=args.max_bg_frames, show_result=args.show, save_dir=args.save_dir, stack_method=args.stack_method, frame_skip=args.frame_skip)

if __name__ == '__main__':
    main()