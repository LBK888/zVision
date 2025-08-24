def predict_single_video(model, video_path):
    """
    處理單個影片的預測功能 (從 predict_on_videos 提取出來)
    
    Args:
        model: YOLO 模型
        video_path (str): 影片檔案路徑
    """
    print(f"處理影片: {os.path.basename(video_path)}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法開啟影片: {video_path}")
        return
        
    # 獲取影片資訊
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"影片資訊: {total_frames} 幀, {fps:.1f} FPS, {duration:.1f} 秒")
    
    device = get_device()
    window_name = f"YOLO 預測 - {os.path.basename(video_path)}"
    print("播放中，按 'q' 可結束預測，按 's' 可跳過")
    
    frame_count = 0
    class_counts = {}
    start_time = time.time()
    
    while True:
        success, frame = cap.read()
        if not success:
            print(f"影片播放完畢或讀取失敗")
            break
        
        frame_count += 1
        
        try:
            # 執行 YOLO 預測
            results = model.predict(
                source=frame,
                conf=0.20,
                device=device,
                verbose=False
            )
            
            # 統計各類別數量
            for r in results:
                if len(r.boxes) > 0:
                    for box in r.boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = model.names[cls_id]
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            
            # 在畫面上顯示統計信息
            annotated_frame = results[0].plot()
            
            # 添加進度資訊
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames} ({progress:.1f}%)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 顯示類別統計
            y_offset = 60
            cv2.putText(annotated_frame, "Detections:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            
            for cls_name, count in sorted(class_counts.items()):
                text = f"{cls_name}: {count}"
                cv2.putText(annotated_frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
            
            cv2.imshow(window_name, annotated_frame)
            
            # 每100幀輸出一次統計到控制台
            if frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                print(f"已處理 {frame_count} 幀 (處理速度: {processing_fps:.1f} FPS)")
                if class_counts:
                    print("目前累計物件數量:")
                    for cls_name, count in sorted(class_counts.items()):
                        print(f"  {cls_name}: {count}")
            
        except Exception as e:
            print(f"處理第 {frame_count} 幀時發生錯誤: {str(e)}")
            continue
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("跳過此影片")
            break
    
    # 顯示最終統計結果
    elapsed_time = time.time() - start_time
    processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\n影片處理完成:")
    print(f"  總共處理: {frame_count} 幀")
    print(f"  處理時間: {elapsed_time:.1f} 秒")
    print(f"  處理速度: {processing_fps:.1f} FPS")
    
    if class_counts:
        print("最終物件統計:")
        for cls_name, count in sorted(class_counts.items()):
            print(f"  {cls_name}: {count}")
    else:
        print("  未檢測到任何物件")
    
    cap.release()
    cv2.destroyWindow(window_name)

"""
v2025.08.24 - 改進版
To compare all yolo models in the folder.
Good for calculate recall

改進項目:
1. 自動檢測和選擇模型檔案
2. 改善錯誤處理
3. 修復記憶體洩漏問題
4. 優化設備自動選擇
5. 改善使用者介面
6. 修復重複數據儲存問題
"""
import cv2
from ultralytics import YOLO
import os
import glob
import pandas as pd
from datetime import datetime
import torch
import time
import yt_dlp
import tempfile
import shutil

def get_device():
    """自動選擇最佳設備"""
    if torch.cuda.is_available():
        return 0  # GPU
    else:
        return 'cpu'

def get_available_models(folder_path):
    """獲取資料夾中所有可用的模型檔案"""
    model_extensions = ['*.pt', '*.pth', '*.onnx']
    model_files = []
    
    for ext in model_extensions:
        model_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    return [os.path.abspath(f) for f in model_files]

def select_model(folder_path):
    """智能選擇模型：單個自動選取，多個讓使用者選擇"""
    model_files = get_available_models(folder_path)
    
    if not model_files:
        print(f"錯誤: 在 {folder_path} 中沒有找到模型檔案 (.pt, .pth, .onnx)")
        return None
    
    if len(model_files) == 1:
        selected_model = model_files[0]
        print(f"自動選擇模型: {os.path.basename(selected_model)}")
        return selected_model
    
    # 多個模型讓使用者選擇
    print("\n發現多個模型檔案:")
    for i, model_file in enumerate(model_files, 1):
        print(f"{i}. {os.path.basename(model_file)}")
    print(f"{len(model_files) + 1}. 比較所有模型")
    
    while True:
        try:
            choice = input(f"請選擇模型 (1-{len(model_files) + 1}): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(model_files):
                selected_model = model_files[choice_num - 1]
                print(f"選擇模型: {os.path.basename(selected_model)}")
                return selected_model
            elif choice_num == len(model_files) + 1:
                return "compare_all"
            else:
                print("無效選項，請重新輸入")
        except ValueError:
            print("請輸入有效數字")

def load_model_safely(model_path):
    """安全載入模型，包含錯誤處理"""
    try:
        print(f"正在載入模型: {os.path.basename(model_path)}")
        model = YOLO(model_path)
        print("模型載入成功")
        return model
    except Exception as e:
        print(f"載入模型失敗: {str(e)}")
        return None

def check_camera_availability():
    """檢查攝影機是否可用"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("錯誤: 無法開啟攝影機，請檢查攝影機是否已連接")
        return False
    cap.release()
    return True

def predict_on_camera(model):
    """改進的攝影機預測功能"""
    if not check_camera_availability():
        return
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 檢查攝影機設定是否成功
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"攝影機解析度: {actual_width}x{actual_height}")
    
    device = get_device()
    print(f"使用設備: {'GPU' if device == 0 else 'CPU'}")
    print("按 'q' 可退出攝影機模式")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        success, frame = cap.read()
        if not success:
            print("無法讀取攝影機影像")
            break
        
        frame_count += 1
        
        # 執行 YOLO 預測
        try:
            results = model.predict(
                source=frame,
                conf=0.25,
                device=device,
                verbose=False
            )
            
            # 在影像上繪製偵測結果
            annotated_frame = results[0].plot()
            
            # 顯示 FPS
            if frame_count % 30 == 0:  # 每30幀計算一次FPS
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                print(f"FPS: {fps:.1f}")
            
            cv2.imshow("YOLO 即時偵測", annotated_frame)
            
        except Exception as e:
            print(f"預測過程中發生錯誤: {str(e)}")
            continue
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"攝影機模式結束，共處理 {frame_count} 幀")

def download_youtube_video(url, output_path=None, quality='720p'):
    """
    從 YouTube 下載影片並返回本地檔案路徑
    
    Args:
        url (str): YouTube 影片 URL
        output_path (str): 輸出路徑，若為 None 則使用臨時資料夾
        quality (str): 影片品質，預設 720p
    
    Returns:
        str: 下載的影片檔案路徑，失敗返回 None
    """
    try:
        if output_path is None:
            output_path = tempfile.mkdtemp()
        
        # yt-dlp 設定
        ydl_opts = {
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
            'format': f'best[height<={quality[:-1]}]/best',  # 移除 'p' 後綴
            'extract_flat': False,
            'writeinfojson': True,  # 儲存影片資訊
            'ignoreerrors': True,
        }
        
        print(f"正在下載 YouTube 影片...")
        print(f"URL: {url}")
        print(f"品質: {quality}")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # 獲取影片資訊
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'unknown_video')
            duration = info.get('duration', 0)
            
            print(f"影片標題: {title}")
            print(f"影片長度: {duration // 60}:{duration % 60:02d}")
            
            # 確認是否繼續下載
            user_confirm = input("是否繼續下載? (y/n): ").strip().lower()
            if user_confirm != 'y':
                print("取消下載")
                return None
            
            # 下載影片
            ydl.download([url])
            
            # 尋找下載的檔案
            video_files = []
            for ext in ['.mp4', '.webm', '.mkv', '.avi']:
                pattern = os.path.join(output_path, f"*{ext}")
                video_files.extend(glob.glob(pattern))
            
            if video_files:
                downloaded_file = video_files[0]  # 取第一個找到的影片檔案
                print(f"✓ 下載完成: {os.path.basename(downloaded_file)}")
                return downloaded_file
            else:
                print("❌ 下載失敗: 找不到下載的影片檔案")
                return None
                
    except Exception as e:
        print(f"❌ 下載 YouTube 影片時發生錯誤: {str(e)}")
        print("請確保已安裝 yt-dlp: pip install yt-dlp")
        return None

def predict_on_youtube(model, url, temp_folder=None):
    """
    從 YouTube URL 下載影片並進行預測
    
    Args:
        model: YOLO 模型
        url (str): YouTube 影片 URL
        temp_folder (str): 臨時資料夾路徑
    """
    if temp_folder is None:
        temp_folder = tempfile.mkdtemp()
    
    downloaded_file = None
    
    try:
        # 詢問影片品質
        print("\n請選擇下載品質:")
        print("1. 480p (較快下載)")
        print("2. 720p (推薦)")
        print("3. 1080p (較慢下載)")
        
        quality_choice = input("請選擇 (1/2/3，預設為 2): ").strip()
        quality_map = {'1': '480p', '2': '720p', '3': '1080p'}
        quality = quality_map.get(quality_choice, '720p')
        
        # 下載 YouTube 影片
        downloaded_file = download_youtube_video(url, temp_folder, quality)
        
        if downloaded_file is None:
            return
        
        # 檢查檔案是否存在且可讀取
        if not os.path.exists(downloaded_file):
            print("❌ 下載的檔案不存在")
            return
        
        # 使用現有的 predict_on_videos 功能處理單個影片
        print(f"\n開始分析影片...")
        predict_single_video(model, downloaded_file)
        
    except KeyboardInterrupt:
        print("\n下載被使用者中斷")
    except Exception as e:
        print(f"處理 YouTube 影片時發生錯誤: {str(e)}")
    finally:
        # 清理臨時檔案
        if downloaded_file and os.path.exists(downloaded_file):
            try:
                print(f"清理臨時檔案...")
                os.remove(downloaded_file)
                # 清理空的臨時資料夾
                if temp_folder and os.path.exists(temp_folder) and not os.listdir(temp_folder):
                    os.rmdir(temp_folder)
            except Exception as e:
                print(f"清理臨時檔案時發生錯誤: {str(e)}")

def predict_single_video(model, video_path):
    """
    處理單個影片的預測功能 (從 predict_on_videos 提取出來)
    
    Args:
        model: YOLO 模型
        video_path (str): 影片檔案路徑
    """
    print(f"處理影片: {os.path.basename(video_path)}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法開啟影片: {video_path}")
        return
        
    # 獲取影片資訊
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"影片資訊: {total_frames} 幀, {fps:.1f} FPS, {duration:.1f} 秒, {width}x{height}")
    
    device = get_device()
    window_name = os.path.basename(video_path)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 限制播放視窗大小到 720p
    max_w, max_h = 1280, 720
    if width > max_w or height > max_h:
        aspect = width / height
        if aspect > max_w / max_h:  # 寬比高更大 → 限制寬
            new_w, new_h = max_w, int(max_w / aspect)
        else:  # 限制高
            new_h, new_w = max_h, int(max_h * aspect)
        cv2.resizeWindow(window_name, new_w, new_h)
    else:
        cv2.resizeWindow(window_name, width, height)

    print("播放中，按 'q' 可結束全部，按 's' 可跳過，左右方向鍵可前後5秒")
    
    frame_count = 0
    class_counts = {}
    start_time = time.time()
    early_quit = False
    
    while True:
        success, frame = cap.read()
        if not success:
            print(f"影片播放完畢或讀取失敗")
            break
        
        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        try:
            # 執行 YOLO 預測
            results = model.predict(
                source=frame,
                conf=0.20,
                device=device,
                verbose=False
            )
            
            # 統計各類別數量
            for r in results:
                if len(r.boxes) > 0:
                    for box in r.boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = model.names[cls_id]
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            
            annotated_frame = results[0].plot()
            
            # 添加進度資訊
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames} ({progress:.1f}%)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(window_name, annotated_frame)
            
        except Exception as e:
            print(f"處理第 {frame_count} 幀時發生錯誤: {str(e)}")
            continue
        
        key = cv2.waitKeyEx(20) #& 0xFF
        
        if key == ord('q'):  # 全部結束
            early_quit = True
            print("結束所有影片分析")
            break
            
        elif key == ord('s'):  # 跳過影片
            print("跳過此影片")
            break
        elif key == 2424832:  # ← 左方向鍵，往前5秒
            new_frame = max(frame_count - int(fps * 5), 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            print(f"往前跳到 {new_frame}/{total_frames}")
        elif key == 2555904:  # → 右方向鍵，往後5秒
            new_frame = min(frame_count + int(fps * 5), total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            print(f"往後跳到 {new_frame}/{total_frames}")
    
    # 顯示最終統計結果
    elapsed_time = time.time() - start_time
    processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\n影片處理完成:")
    print(f"  總共處理: {frame_count} 幀")
    print(f"  處理時間: {elapsed_time:.1f} 秒")
    print(f"  處理速度: {processing_fps:.1f} FPS")
    
    if class_counts:
        print("最終物件統計:")
        for cls_name, count in sorted(class_counts.items()):
            print(f"  {cls_name}: {count}")
    else:
        print("  未檢測到任何物件")
    
    cap.release()
    cv2.destroyWindow(window_name)
    if early_quit:
        return "quit"


def get_video_files(folder_path):
    """獲取資料夾中的所有影片檔案"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_files = []
    
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            full_path = os.path.join(folder_path, file)
            # 檢查檔案是否可讀取
            if os.path.isfile(full_path) and os.access(full_path, os.R_OK):
                video_files.append(full_path)
    
    return video_files

def predict_on_videos(model, folder_path):
    """改進的影片預測功能"""
    video_files = get_video_files(folder_path)
            
    if not video_files:
        print("指定的資料夾中沒有找到可讀取的影片檔案。")
        return
    
    device = get_device()
    print(f"使用設備: {'GPU' if device == 0 else 'CPU'}")
    print(f"找到 {len(video_files)} 個影片檔案")
    
    # 使用單個影片處理功能
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] 處理影片: {os.path.basename(video_path)}")
        if predict_single_video(model, video_path)=='quit':
            break

def compare_models_on_videos(models_folder, output_folder=None):
    """改進的模型比較功能"""
    if output_folder is None:
        output_folder = models_folder
    
    os.makedirs(output_folder, exist_ok=True)
    
    # 獲取所有模型檔案
    model_files = get_available_models(models_folder)
    if not model_files:
        print(f"在 {models_folder} 中沒有找到模型檔")
        return
    
    # 獲取所有影片檔案
    video_files = get_video_files(models_folder)
    if not video_files:
        print(f"在 {models_folder} 中沒有找到影片檔案")
        return
    
    print(f"找到 {len(model_files)} 個模型檔案和 {len(video_files)} 個影片檔案")
    
    device = get_device()
    print(f"使用設備: {'GPU' if device == 0 else 'CPU'}")
    
    # 創建結果儲存列表
    all_results = []
    
    # 處理每個模型
    for model_idx, model_path in enumerate(model_files, 1):
        model_name = os.path.basename(model_path)
        print(f"\n[{model_idx}/{len(model_files)}] 載入模型: {model_name}")
        
        model = load_model_safely(model_path)
        if model is None:
            continue
        
        # 處理每個影片
        for video_idx, video_path in enumerate(video_files, 1):
            video_name = os.path.basename(video_path)
            print(f"  [{video_idx}/{len(video_files)}] 處理影片: {video_name}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"    警告: 無法開啟影片 {video_name}")
                continue
            
            valid_frame_count = 0
            class_counts = {}
            start_time = time.time()
            
            # 跳到第50幀開始處理
            cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
            
            try:
                while valid_frame_count < 1000:
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    valid_frame_count += 1
                    
                    # 執行YOLO預測
                    results = model.predict(
                        source=frame,
                        conf=0.20,
                        device=device,
                        verbose=False
                    )
                    
                    # 統計各類別數量
                    for r in results:
                        if len(r.boxes) > 0:
                            for box in r.boxes:
                                cls_id = int(box.cls[0].item())
                                cls_name = model.names[cls_id]
                                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    
                    # 每250幀輸出進度
                    if valid_frame_count % 250 == 0:
                        elapsed_time = time.time() - start_time
                        fps = valid_frame_count / elapsed_time if elapsed_time > 0 else 0
                        print(f"    已處理 {valid_frame_count} 幀 (速度: {fps:.1f} FPS)")
                
                # 儲存結果（只儲存最終結果，避免重複）
                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                for cls_name, count in class_counts.items():
                    all_results.append({
                        'model_name': model_name,
                        'video_name': video_name,
                        'total_frames_processed': valid_frame_count,
                        'final_frame_number': frame_number,
                        'class_name': cls_name,
                        'total_count': count,
                        'detection_rate': count / valid_frame_count if valid_frame_count > 0 else 0
                    })
                
                elapsed_time = time.time() - start_time
                fps = valid_frame_count / elapsed_time if elapsed_time > 0 else 0
                print(f"    完成: {valid_frame_count} 幀, {fps:.1f} FPS")
                if class_counts:
                    print(f"    檢測結果: {sum(class_counts.values())} 個物件")
                
            except Exception as e:
                print(f"    錯誤: 處理影片時發生問題 - {str(e)}")
            finally:
                cap.release()
    
    # 將結果轉換為DataFrame並保存到Excel
    if all_results:
        df = pd.DataFrame(all_results)
        
        # 創建摘要統計
        summary_df = df.groupby(['model_name', 'class_name']).agg({
            'total_count': 'sum',
            'detection_rate': 'mean'
        }).reset_index()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = os.path.join(output_folder, f"model_comparison_{timestamp}.xlsx")
        
        # 使用多個工作表儲存詳細和摘要結果
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='詳細結果', index=False)
            summary_df.to_excel(writer, sheet_name='摘要統計', index=False)
        
        print(f"\n✓ 比較結果已保存至: {excel_path}")
        print(f"  - 詳細結果: {len(df)} 筆記錄")
        print(f"  - 摘要統計: {len(summary_df)} 筆記錄")
    else:
        print("❌ 沒有獲取到任何比較結果")

def main():
    """主程式"""
    try:
        # 取得目前執行檔所在的資料夾路徑
        main_folder = os.path.dirname(os.path.abspath(__file__))
        print(f"當前工作目錄: {main_folder}")
        
        # 智能選擇模型
        selected_model = select_model(main_folder)
        
        if selected_model is None:
            return
        elif selected_model == "compare_all":
            # 直接進入比較模式
            folder_path = input("請輸入模型和影片所在資料夾路徑（若留空則使用目前資料夾）: ").strip()
            if folder_path == "" or not os.path.isdir(folder_path):
                folder_path = main_folder
            
            output_path = input("請輸入結果儲存資料夾路徑（若留空則使用與模型相同的資料夾）: ").strip()
            if output_path == "" or not os.path.isdir(output_path):
                output_path = None
            
            compare_models_on_videos(folder_path, output_path)
            return
        
        # 載入選定的模型
        model = load_model_safely(selected_model)
        if model is None:
            return
        
        # 提供模式選擇介面
        print("\n請選擇操作模式：")
        print("1. 開啟相機模式")
        print("2. 讀取資料夾中的所有影片")
        print("3. 比較多個模型")
        print("4. 分析 YouTube 影片")
        
        while True:
            choice = input("請輸入選項 (1/2/3/4): ").strip()
            
            if choice == '1':
                predict_on_camera(model)
                break
            elif choice == '2':
                folder_path = input("請輸入影片所在資料夾路徑（若留空則使用目前資料夾）: ").strip()
                if folder_path == "" or not os.path.isdir(folder_path):
                    folder_path = main_folder
                predict_on_videos(model, folder_path)
                break
            elif choice == '3':
                folder_path = input("請輸入模型和影片所在資料夾路徑（若留空則使用目前資料夾）: ").strip()
                if folder_path == "" or not os.path.isdir(folder_path):
                    folder_path = main_folder
                
                output_path = input("請輸入結果儲存資料夾路徑（若留空則使用與模型相同的資料夾）: ").strip()
                if output_path == "" or not os.path.isdir(output_path):
                    output_path = None
                
                compare_models_on_videos(folder_path, output_path)
                break
            elif choice == '4':
                youtube_url = input("請輸入 YouTube 影片 URL: ").strip()
                if youtube_url:
                    predict_on_youtube(model, youtube_url)
                else:
                    print("無效的 URL")
                break
            else:
                print("無效的選項，請輸入 1、2、3 或 4")
    
    except KeyboardInterrupt:
        print("\n程式被使用者中斷")
    except Exception as e:
        print(f"程式執行時發生錯誤: {str(e)}")

if __name__ == "__main__":
    main()
