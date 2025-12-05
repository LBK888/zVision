'''
Long term time-lapse multi-camera recording, 2025-12-02 v0.3

For DFRobot FIT0729 (8 Megapixels USB Camera)

Max Effective Resolution: 3264(H) X 2448(V)  8 Megapixel
Image Output Format: JPEG 
Supported Resolution and Frame Rate: 
                 3264X2448 @ 15fps  / 2592X1944@ 15fps
                 1920x1080 @ 30fps  / 1600X1200@ 30fps
                 1280X720 @ 30fps  / 960X540@ 30fps
                 800X600 @ 30fps   / 640X480@ 30fps

'''

import cv2
import time
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
from shapely import geometry

### USER Settings ###
# 設定相機數目
camera_no=2
# 設定時間間隔（10 分鐘 = 10）
interval_min = 5
#顯示視窗大小 (30% = 0.3)
Window_size_r=0.3
# 設定相機啟動後所需的自動對焦/曝光時間 (秒)
cam_pause=2.5
# 設定錄影解析度
width, height = 2592, 1944
# 設定曝光值 (可選)
exposure_value = 1  # 曝光值通常在 -7 到 7 之間

# plate format (6-well or other)
plate_rows=2
plate_cols=3

#root path
rootdir=os.path.dirname(os.path.realpath(__file__))+'/'
#folder
folder=rootdir
#folder='D:/Webcam_recodring/'
# Load a model
#model = YOLO('yolov8n-seg.pt')  # load an official model
model = YOLO(folder+'v7t17best-L.pt')  # load a custom model

### Not Settings ###
interval = interval_min * 60
# Set Window size to 30％
window_width = int(width * Window_size_r)
window_height = int(height * Window_size_r)

def set_resolution(cap, width, height):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def set_exposure(cap, exposure_value):
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)

def split_frame(frame, rows, cols):
    """將圖片分割成 rows x cols 的網格"""
    h, w = frame.shape[:2]
    well_h, well_w = h // rows, w // cols
    wells = []
    for i in range(rows):
        for j in range(cols):
            well = frame[i*well_h:(i+1)*well_h, j*well_w:(j+1)*well_w]
            offset_x = j * well_w  # 計算 x 方向偏移
            offset_y = i * well_h  # 計算 y 方向偏移
            wells.append({
                "image": well, 
                "row": i+1, 
                "col": j+1,
                "offset_x": offset_x,
                "offset_y": offset_y
            })
    return wells

def combine_frames(frames, rows, cols):
    """將多個圖片合併成一個網格"""
    well_h, well_w = frames[0].shape[:2]
    combined = np.zeros((well_h * rows, well_w * cols, 3), dtype=np.uint8)
    idx = 0
    for i in range(rows):
        for j in range(cols):
            combined[i*well_h:(i+1)*well_h, j*well_w:(j+1)*well_w] = frames[idx]
            idx += 1
    return combined



def main():
    # 記錄上次拍攝時間
    last_capture_time = 0
    
    while True:
        # 當前時間
        current_time = time.time()
        current_datetime = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')

        # 如果是第一次執行(last_capture_time=0)或已經超過間隔時間，則進行拍攝
        if last_capture_time == 0 or (current_time - last_capture_time) >= interval:
            last_capture_time = current_time
            
            for camID in range(camera_no):
                # 選擇 webcam
                cap = cv2.VideoCapture(camID)
                set_resolution(cap, width, height)
                #set_exposure(cap, exposure_value)

                # 檢查 webcam 是否成功打開
                if not cap.isOpened():
                    print(f'Error: Could not open webcam ID{camID}.')
                    continue

                # 讀取一個 frame
                time.sleep(cam_pause)   #等待自動對焦與自動曝光
                ret, frame = cap.read()

                if not ret:
                    print(f'Error: Failed to grab frame @ time={current_time} @ camID={camID}')
                    cap.release()
                    continue

                # 分割圖片
                wells = split_frame(frame, plate_rows, plate_cols)
                all_detection_data = []
                annotated_wells = []

                # 對每個分割區域進行預測
                for well in wells:
                    results = model(well["image"])
                    
                    # 處理檢測結果
                    if results[0].masks is not None:
                        masks_np = results[0].masks.xy
                        
                        for i, (mask, box) in enumerate(zip(masks_np, results[0].boxes)):
                            class_name = results[0].names[int(box.cls)]
                            
                            # 檢查 mask 座標數量是否足夠 (至少需要 3 個點才能構成多邊形，使用者建議檢查是否少於 4 個)
                            if len(mask) < 4:
                                continue

                            polygon_mask = geometry.Polygon(mask)
                            if not polygon_mask.is_valid:
                                polygon_mask = polygon_mask.buffer(0.01)
                            area = polygon_mask.area
                            
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            # 加入 offset 到座標值
                            x1 += well["offset_x"]
                            x2 += well["offset_x"]
                            y1 += well["offset_y"]
                            y2 += well["offset_y"]
                            confidence = float(box.conf[0])
                            
                            detection_data = {
                                '時間戳記': current_time,
                                '日期時間': current_datetime,
                                '相機編號': f'cam{camID+1}',
                                '類別': class_name,
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2,
                                '面積': area,
                                '信心度': confidence,
                                'bbox面積': (x2 - x1) * (y2 - y1),
                                'mask面積': area,
                                'plate_row': well["row"],
                                'plate_col': well["col"]
                            }
                            all_detection_data.append(detection_data)
                    
                    # 獲取標註後的圖片
                    annotated_well = results[0].plot()
                    annotated_wells.append(annotated_well)

                # 合併標註後的圖片
                combined_annotated = combine_frames(annotated_wells, plate_rows, plate_cols)

                # 將數據轉換為 DataFrame
                df = pd.DataFrame(all_detection_data)
                
                # 儲存到 Excel 檔案
                excel_filename = f'cam{camID+1}_results.xlsx'
                
                try:
                    # 如果檔案存在，則附加數據；否則創建新檔案
                    if os.path.exists(folder + excel_filename):
                        existing_df = pd.read_excel(folder + excel_filename)
                        df = pd.concat([existing_df, df], ignore_index=True)
                    
                    df.to_excel(folder + excel_filename, index=False)
                except Exception as e:
                    print(f"Error saving Excel file for cam{camID+1}: {e}")

                # 儲存合併後的標註圖片
                filename = f'cam{camID+1}_anno_frame_{int(current_time)}.tiff'
                cv2.imwrite(folder+filename, combined_annotated)

                # 顯示合併後的圖片
                cv2.namedWindow('Webcam live, camera='+str(camID+1), cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Webcam live, camera='+str(camID+1), window_width, window_height)
                cv2.imshow('Webcam live, camera='+str(camID+1), combined_annotated)

                # 圖片檔案名稱，以時間戳記命名
                filename = f'cam{camID+1}_frame_{int(current_time)}.tiff'
                # 儲存圖片
                cv2.imwrite(folder+filename, frame)

                print(f"Saved {filename}")

                # 釋放資源
                cap.release()
                

        # 按下 'q' 鍵結束錄影
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break



if __name__ == "__main__":
    main()
