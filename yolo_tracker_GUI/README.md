# Multi-Camera Timelapse Analyzer (多相機時間序列影像分析工具)

[English](#english) | [繁體中文](#繁體中文)

---

<a name="english"></a>
## 🇬🇧 English

### Introduction
**Multi-Camera Timelapse Analyzer** is a powerful GUI-based tool designed for analyzing time-lapse image sequences using YOLO object detection models. It integrates advanced object tracking algorithms to monitor and visualize object movements across frames. The tool supports multi-camera setups, allowing users to process image folders, crop regions of interest, and generate detailed analysis reports and visualized videos.

### Features
*   **YOLO Integration**: Supports YOLOv8/v11/v12 models for robust object detection.
*   **Advanced Tracking**: Includes UKF (Unscented Kalman Filter), SORT, and ByteTrack algorithms.
*   **GUI Control**: User-friendly interface for model selection, parameter tuning, and visualization settings.
*   **Region of Interest (ROI)**: Interactive cropping tool to focus analysis on specific image areas.
*   **Visualization**: Customizable overlays for bounding boxes, trails, masks, keypoints, and velocity vectors.
*   **Export**: Generates analyzed videos (MP4) and Excel reports (`.xlsx`) containing detailed tracking data.
*   **Class Similarity**: Configurable class similarity map to handle object classification jitter (e.g., confusing 'car' with 'truck').

### Installation

1.  **Prerequisites**:
    *   Python 3.8 or higher
    *   CUDA-capable GPU (recommended for faster YOLO inference)

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Key dependencies include: `PyQt5`, `ultralytics`, `opencv-python`, `pandas`, `numpy`, `scipy`.*

### GUI Usage

1.  **Launch the Application**:
    ```bash
    python multicam_timelapse_analyzer.py
    ```
   or for windows users, click: start.bat  
    
<img width="1525" height="1270" alt="image" src="https://github.com/user-attachments/assets/e0edcf44-843c-402d-bad5-3ae13c7cb245" />


2.  **Workflow**:
    *   **Select Model**: Choose a YOLO model (`.pt`) from the dropdown or load a custom one.
    *   **Load Images**: Select a folder containing your time-lapse image sequence.
    *   **Set ROI (Optional)**: Drag on the image preview to define a crop area.
    *   **Configure Tracker**:
        *   Select Algorithm: `UKF`, `SORT`, or `ByteTrack`.
        *   Adjust `IOU Threshold`, `Conf Threshold`, and `Max Age`.
    *   **Visualization Settings**: Toggle `BBox`, `Trails`, `Masks`, etc., in the "Overlay Settings" panel.
    *   **Run Analysis**: Click **"Start Analysis"** to begin processing. The tool will generate an output video and an Excel report in the `runs/` directory.

### Tracker Algorithms

This software implements a **YOLO Object Tracker Pro** system (`yolo_tracker_v2.py`) with the following core algorithms:

*   **UKF (Unscented Kalman Filter)**:
    *   **Description**: A non-linear Kalman Filter that uses sigma points to handle non-linear state transitions. It provides robust tracking for objects with complex motion patterns.
    *   **State Vector**: 8-dimensional `[x, y, w, h, vx, vy, vw, vh]`.
    *   **Best for**: Scenarios where objects may change speed or direction non-linearly.

*   **SORT (Simple Online and Realtime Tracking)**:
    *   **Description**: A standard approach using a Linear Kalman Filter and IOU matching. Fast and effective for simple tracking tasks.
    *   **State Vector**: 7-dimensional `[x, y, area, ratio, vx, vy, v_area]`.
    *   **Best for**: Real-time applications with predictable object motion.

*   **ByteTrack**:
    *   **Description**: An enhanced version of SORT that utilizes a two-stage matching process. It first matches high-confidence detections and then attempts to recover low-confidence detections using remaining tracks.
    *   **Best for**: Handling occlusion and maintaining tracks for objects with fluctuating detection confidence.

### License
This project is licensed under the MIT License. See the LICENSE file for details.

---

<a name="繁體中文"></a>
## 🇹🇼 繁體中文

### 簡介 (Introduction)
**多相機時間序列影像分析工具 (Multi-Camera Timelapse Analyzer)** 是一款基於 GUI 的強大工具，專為使用 YOLO 物件偵測模型分析縮時攝影影像而設計。它整合了先進的物件追蹤演算法，可監控並視覺化跨幀的物件移動軌跡。本工具支援多相機設置，允許使用者處理影像資料夾、裁切感興趣區域 (ROI)，並生成詳細的分析報表與視覺化影片。

### 功能特色 (Features)
*   **YOLO 整合**: 支援 YOLOv8/v11/v12 模型，提供強大的物件偵測能力。
*   **先進追蹤**: 內建 UKF (Unscented Kalman Filter)、SORT 與 ByteTrack 演算法。
*   **圖形介面**: 友善的使用者介面，可輕鬆進行模型選擇、參數調整與視覺化設定。
*   **感興趣區域 (ROI)**: 互動式裁切工具，可針對特定影像區域進行分析。
*   **視覺化**: 可自訂疊加層，包含邊界框 (BBox)、軌跡 (Trails)、遮罩 (Masks)、關鍵點 (Keypoints) 與速度向量。
*   **輸出**: 自動生成分析影片 (MP4) 與包含詳細追蹤數據的 Excel 報表 (`.xlsx`)。
*   **類別相似度**: 可設定類別相似度對照表，解決物件分類跳動的問題 (例如將「汽車」誤判為「卡車」的情況)。

### 安裝說明 (Installation)

1.  **環境需求**:
    *   Python 3.8 或更高版本
    *   支援 CUDA 的 GPU (建議使用，以加速 YOLO 推論)

2.  **安裝依賴套件**:
    ```bash
    pip install -r requirements.txt
    ```
    *主要依賴包含: `PyQt5`, `ultralytics`, `opencv-python`, `pandas`, `numpy`, `scipy`.*

### GUI 使用說明 (GUI Usage)

1.  **啟動程式**:
    ```bash
    python multicam_timelapse_analyzer.py
    ```
   或 windows 使用者, 點選: start.bat  
    
<img width="1525" height="1270" alt="image" src="https://github.com/user-attachments/assets/e0edcf44-843c-402d-bad5-3ae13c7cb245" />

2.  **操作流程**:
    *   **選擇模型**: 從下拉選單選擇 YOLO 模型 (`.pt`) 或載入自定義模型。
    *   **載入影像**: 選擇包含縮時攝影影像序列的資料夾。
    *   **設定 ROI (選用)**: 在影像預覽區拖曳滑鼠以定義裁切區域。
    *   **設定追蹤器**:
        *   選擇演算法: `UKF`、`SORT` 或 `ByteTrack`。
        *   調整 `IOU 閾值`、`信心度閾值` 與 `最大年齡 (Max Age)`。
    *   **視覺化設定**: 在「疊加設定 (Overlay Settings)」面板中切換 `BBox`、`Trails`、`Masks` 等顯示選項。
    *   **執行分析**: 點擊 **"開始分析 (Start Analysis)"** 按鈕開始處理。程式將在 `runs/` 目錄下生成輸出影片與 Excel 報表。

### 追蹤演算法 (Tracker Algorithms)

本軟體實作了 **YOLO Object Tracker Pro** 系統 (`yolo_tracker_v2.py`)，包含以下核心演算法：

*   **UKF (Unscented Kalman Filter)**:
    *   **說明**: 非線性卡爾曼濾波器，使用 Sigma 點來處理非線性狀態轉移。對於運動模式複雜的物件提供穩健的追蹤能力。
    *   **狀態向量**: 8 維 `[x, y, w, h, vx, vy, vw, vh]`。
    *   **適用於**: 物件速度或方向可能發生非線性變化的場景。

*   **SORT (Simple Online and Realtime Tracking)**:
    *   **說明**: 使用線性卡爾曼濾波器與 IOU 匹配的標準方法。對於簡單的追蹤任務快速且有效。
    *   **狀態向量**: 7 維 `[x, y, area, ratio, vx, vy, v_area]`。
    *   **適用於**: 物件運動可預測的即時應用。

*   **ByteTrack**:
    *   **說明**: SORT 的增強版本，採用兩階段匹配過程。首先匹配高信心度的偵測結果，然後嘗試利用剩餘的軌跡找回低信心度的偵測結果。
    *   **適用於**: 處理遮擋問題，以及維持偵測信心度波動較大的物件軌跡。

### 授權 (License)
本專案採用 MIT License 授權。詳細內容請參閱 LICENSE 文件。

