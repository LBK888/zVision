"""
v2025.05.22
To compare all yolo models in the folder.
Good for calculate recall

"""
import cv2
from ultralytics import YOLO
import os
import glob
import pandas as pd
from datetime import datetime

def predict_on_camera(model):
    # 開啟網路攝影機 (預設裝置編號0)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("按 'q' 可退出攝影機模式")
    while True:
        success, frame = cap.read()
        if not success:
            print("無法讀取攝影機影像")
            break
            
        # 執行 YOLO 預測
        results = model.predict(
            source=frame,
            conf=0.25,       # 信心度閾值
            device=0,   #  or 'cpu' 
            verbose=False   # 關閉詳細輸出以提升效能
        )
        
        # 在影像上繪製偵測結果
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv11/v12 即時偵測 - 相機模式", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def predict_on_videos(model, folder_path):
    # 定義支援的影片副檔名
    video_extensions = ['.mp4', '.avi', '.mov']
    video_files = []
    
    # 搜尋指定資料夾下所有影片檔
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(folder_path, file))
            
    if not video_files:
        print("指定的資料夾中沒有找到影片檔案。")
        return
    
    # 處理每個影片檔案
    for video_path in video_files:
        print(f"\n處理影片: {video_path}")
        cap = cv2.VideoCapture(video_path)
        window_name = f"YOLOv11/v12 預測 - {os.path.basename(video_path)}"
        cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
        print("播放中，按 'q' 可結束此影片的預測")
        
        # 新增：計數器變數
        frame_count = 0
        class_counts = {}
        
        while True:
            success, frame = cap.read()
            if not success:
                print(f"影片 {video_path} 播放完畢或讀取失敗")
                break
            
            # 更新幀計數
            frame_count += 1
            
            # 執行 YOLO 預測
            results = model.predict(
                source=frame,
                conf=0.20,
                device=0,
                verbose=False
            )
            
            # 新增：統計各類別數量
            for r in results:
                if len(r.boxes) > 0:
                    for box in r.boxes:
                        # 獲取類別 ID 和類別名稱
                        cls_id = int(box.cls[0].item())
                        cls_name = model.names[cls_id]
                        
                        # 更新計數
                        if cls_name in class_counts:
                            class_counts[cls_name] += 1
                        else:
                            class_counts[cls_name] = 1
            
            # 在畫面上顯示統計信息
            annotated_frame = results[0].plot()
            
            # 添加文字顯示統計信息
            cv2.putText(annotated_frame, f"Frames: {frame_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 顯示類別統計
            y_offset = 60
            cv2.putText(annotated_frame, "Total Counts:", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            
            for cls_name, count in class_counts.items():
                text = f"{cls_name}: {count}"
                cv2.putText(annotated_frame, text, (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
            
            annotated_frameS = ResizeWithAspectRatio(annotated_frame, width=1080)
            cv2.imshow(window_name, annotated_frameS)
            
            # 每100幀輸出一次統計到控制台
            if frame_count % 100 == 0:
                print(f"已處理 {frame_count} 幀")
                print("目前累計物件數量:")
                for cls_name, count in class_counts.items():
                    print(f"  {cls_name}: {count}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # 按 'q' 鍵離開當前影片播放
                break
        
        # 顯示最終統計結果
        print(f"\n影片處理完成。總共處理了 {frame_count} 幀")
        print("最終物件統計:")
        for cls_name, count in sorted(class_counts.items()):
            print(f"  {cls_name}: {count}")
        
        cap.release()
        cv2.destroyWindow(window_name)

def compare_models_on_videos(models_folder, output_folder=None):
    # 如果沒有指定輸出資料夾，則使用與模型資料夾相同的資料夾
    if output_folder is None:
        output_folder = models_folder
    
    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 獲取所有模型檔案
    model_files = glob.glob(os.path.join(models_folder, "*.pt"))
    if not model_files:
        print(f"在 {models_folder} 中沒有找到 .pt 模型檔")
        return
    
    # 獲取所有影片檔案
    video_extensions = ['.mp4', '.avi', '.mov']
    video_files = []
    for file in os.listdir(models_folder):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(models_folder, file))
    
    if not video_files:
        print(f"在 {models_folder} 中沒有找到影片檔案")
        return
    
    # 創建DataFrame來存儲結果
    columns = ['model_name', 'video_name', 'frame_count', 'class_name', 'count']
    all_results = []
    
    # 處理每個模型
    for model_path in model_files:
        model_name = os.path.basename(model_path)
        print(f"\n載入模型: {model_name}")
        
        try:
            model = YOLO(model_path)
            
            # 處理每個影片
            for video_path in video_files:
                video_name = os.path.basename(video_path)
                print(f"處理影片: {video_name}")
                
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                valid_frame_count = 0
                class_counts = {}
                
                # 跳到第50幀
                cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
                
                while True:
                    success, frame = cap.read()
                    if not success or valid_frame_count >= 1000:
                        break
                    
                    frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    valid_frame_count += 1
                    
                    # 執行YOLO預測
                    results = model.predict(
                        source=frame,
                        conf=0.20,
                        device=0,
                        verbose=False
                    )
                    
                    # 統計各類別數量
                    for r in results:
                        if len(r.boxes) > 0:
                            for box in r.boxes:
                                cls_id = int(box.cls[0].item())
                                cls_name = model.names[cls_id]
                                
                                if cls_name in class_counts:
                                    class_counts[cls_name] += 1
                                else:
                                    class_counts[cls_name] = 1
                    
                    # 每100幀保存一次數據
                    if valid_frame_count % 100 == 0:
                        print(f"已處理 {valid_frame_count} 幀")
                        
                        # 保存當前統計到結果列表
                        for cls_name, count in class_counts.items():
                            all_results.append({
                                'model_name': model_name,
                                'video_name': video_name,
                                'frame_count': frame_count,
                                'class_name': cls_name,
                                'count': count
                            })
                
                cap.release()
                
                # 保存最終統計
                for cls_name, count in class_counts.items():
                    all_results.append({
                        'model_name': model_name,
                        'video_name': video_name,
                        'frame_count': frame_count,
                        'class_name': cls_name,
                        'count': count
                    })
                
                print(f"影片 {video_name} 處理完成，共處理 {valid_frame_count} 幀")
        
        except Exception as e:
            print(f"處理模型 {model_name} 時發生錯誤: {str(e)}")
    
    # 將結果轉換為DataFrame並保存到Excel
    if all_results:
        df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = os.path.join(output_folder, f"model_comparison_{timestamp}.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"\n比較結果已保存至: {excel_path}")
    else:
        print("沒有獲取到任何比較結果")

def main():
    # 取得目前執行檔所在的資料夾路徑
    main_folder = os.path.dirname(os.path.realpath(__file__)) + os.sep
    # 載入模型，先找出所有的模型
    model_files = glob.glob(os.path.join(main_folder, "*.pt"))
    print(f"找到的模型: {model_files}")
    if len(model_files) == 0:
        print("沒有找到任何模型")
    elif len(model_files) == 1:
        model_path = model_files[0]
    else:
        model_path = input("請輸入模型路徑: ").strip()
        if not os.path.exists(model_path):
            print("輸入的模型路徑不存在，程式結束。")
            return
        
    # 或是自行修改下面這行
    # model_path = os.path.join(main_folder, 'M7best.pt')
    
    # 載入預訓練的 YOLO 模型
    model = YOLO(model_path)
    
    # 提供模式選擇介面
    print("請選擇模式：")
    print("1. 開啟相機模式")
    print("2. 讀取資料夾中的所有影片")
    print("3. 比較多個模型")
    choice = input("請輸入選項 (1/2/3): ").strip()
    
    if choice == '1':
        predict_on_camera(model)
    elif choice == '2':
        folder_path = input("請輸入影片所在資料夾路徑（若留空則使用目前資料夾）: ").strip()
        if folder_path == "":
            folder_path = main_folder
        elif not os.path.isdir(folder_path):
            print("輸入的資料夾路徑不存在，將使用目前資料夾")
            folder_path = main_folder
        predict_on_videos(model, folder_path)
    elif choice == '3':
        folder_path = input("請輸入模型和影片所在資料夾路徑（若留空則使用目前資料夾）: ").strip()
        if folder_path == "":
            folder_path = main_folder
        elif not os.path.isdir(folder_path):
            print("輸入的資料夾路徑不存在，將使用目前資料夾")
            folder_path = main_folder
        
        output_path = input("請輸入結果儲存資料夾路徑（若留空則使用與模型相同的資料夾）: ").strip()
        if output_path == "":
            output_path = None
        elif not os.path.isdir(output_path):
            try:
                os.makedirs(output_path)
            except:
                print("無法創建輸出資料夾，將使用與模型相同的資料夾")
                output_path = None
        
        compare_models_on_videos(folder_path, output_path)
    else:
        print("無效的選項，程式結束。")

if __name__ == "__main__":
    main()

