'''
Vid snapper, v2025-02-26
This program will process all videos in the folder, and evenly snapshot frames for Roboflow annotation. 
*Roboflow does not supper .avi
'''

import cv2
import os
import numpy as np


def sanitize_filename(name):
    illegal_chars = r'\/:*?"<>|'
    for ch in illegal_chars:
        name = name.replace(ch, '')
    return name.strip()


def extract_frames(video_path,frameN=20,skip=50):
    # 取得影片檔名（不含副檔名）作為資料夾名稱
    video_name = sanitize_filename(os.path.splitext(os.path.basename(video_path))[0].strip())
    
    # 建立對應的資料夾
    output_dir = os.path.join(os.path.dirname(video_path), video_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 讀取影片
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 如果影片總幀數少於100幀，則跳過處理
    if total_frames <= skip*2+frameN*2:
        print(f"影片 {video_name} 幀數太少，跳過處理")
        cap.release()
        return
    
    # 計算要擷取的幀的位置
    # 去除前後50幀後的有效幀數範圍
    valid_start = skip
    valid_end = total_frames - skip
    valid_frames = valid_end - valid_start
    
    # 計算20個均勻分布的幀位置
    frame_indices = np.linspace(valid_start, valid_end-1, frameN, dtype=int)
    

    # 擷取並儲存指定的幀
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            output_path = os.path.join(output_dir, f'frame_{i+1:03d}.jpg')
            ret2, buffer = cv2.imencode('.jpg', frame)
            if ret2:
                with open(output_path, 'wb') as f:
                    f.write(buffer)
                print(f"已儲存 {output_path}")
            else:
                print(f"編碼失敗：{output_path}")
        else:
            print(f"讀取第 {frame_idx} 幀失敗")
    
    cap.release()


def main():
    # 取得目前執行檔案所在的資料夾路徑
    main_folder = os.path.dirname(os.path.realpath(__file__))+'/'
    
    # 支援的影片格式
    video_extensions = ['.mp4', '.avi', '.mov']
    video_files = []
    
    # 搜尋指定目錄下所有支援的影片檔
    for file in os.listdir(main_folder):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(main_folder, file))
    
    if not video_files:
        print("未找到任何影片檔")
        return
    
    # 處理每個影片
    for video_path in video_files:
        print(f"\n處理影片: {video_path}")
        extract_frames(video_path)
    
    print("\n所有影片處理完成！")

if __name__ == "__main__":
    main()