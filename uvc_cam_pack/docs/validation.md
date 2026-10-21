# 交付驗證紀錄

驗證日期：2026-10-21（Asia/Taipei）。這是可執行的整合版本 0.1.0；實體相機與馬達的驗收另列於下方。

## 自動化測試

```text
.venv\Scripts\python.exe -m pytest -q --basetemp .cache/pytest-release-final --junitxml=docs/test-results.xml
58 passed in 6.90s
```

機器可讀結果：[test-results.xml](test-results.xml)。環境記錄：[tested-environment.json](tested-environment.json)。

涵蓋：

- 設定 JSON 來回、遮蔽 token 的設定重新載入、錯字／非法參數拒絕。
- UKF 相關共變異數重建、SORT／ByteTrack 真實 dt、每個 stream ID 隔離、低 confidence 恢復舊軌跡、沒有觀測時不產生正例。
- 投票必須同類且同物體，一個視圖不會貢獻兩票；首尾不複製；鄰幀推定不產生本張面積。
- 非正方形影像 ±90° 框、mask 與 pose 逆轉；精確二元 mask 包含孔洞的編碼來回。
- 多模型屬性融合、校正／原圖座標轉換、矩形等分無餘數像素遺失、橢圓有效範圍、平移防抖與透視矩陣。
- 多相機實際模擬擷取與存檔、寫檔失敗、佇列滿載停止後排空已接受影格、手動停止與指定時長截止。
- HTTP bearer 驗證與到位要求、拒絕過期 request、重試不重設計時、完成確認；多目標等到檔案保存才允許移動。
- 完全沒有控制器回報時，指定拍攝時長仍能結束，不被硬體 timeout 無限拖長。
- 原始影像＋processed、原始影像單獨保存後還原拍攝 ROI、重新選擇新 ROI 時不重複分析同一張 raw。
- 原 TrackBox 同資料夾 cam1／cam2 序列隔離與自然排序、中文路徑讀寫。
- 真正呼叫 FFmpeg 輸出 MP4／AVI，檢查 FPS、影格數、重複／略過計數、片段收尾。
- Excel 可讀取、溫溼度單位與折線圖。
- 三種語言的六頁 UI、ROI 與模型參數實際進入 runtime、相機幾何覆寫跨語言保存。
- 真正從 Qt 背景工作執行擷取與完成結果，確認 Live 控制不被鎖住。
- 單檔發行版的模型資源定位、相機子程序 JSON IPC、失敗結果回報。

## 實際模型與影片測試

使用本機 Ultralytics **8.4.45**，下載並實際載入官方 **yolo26n-seg.pt** 與 **yolo26n-pose.pt**，以 CPU、imgsz=320 執行。使用 Ultralytics 套件內建 `assets/bus.jpg`，重複三張作為流程測試；這不是生物實驗資料，也不是運動追蹤準確度驗證。

| 模式 | 結果 |
|---|---|
| 單次多模型 | 3 張合計 18 個融合後物體記錄，其中 9 筆同時包含 segmentation + pose；MP4、Excel、SQLite、模型 hash 成功產生 |
| 前／本／後張 2/3 | 同樣成功得到 18 筆與 9 筆跨模型融合記錄；首尾各使用真實存在的另一張支持，不重複補票 |
| 原／CW／CCW 90° 2/3 | 工作完成，正例 0；模型原圖有偵測，但旋轉後無偵測，投票如實排除；原始候選仍保存供稽核 |

最後一項顯示旋轉投票可能嚴重增加漏判，不能把「三次取兩次」直接當作已證實的準確度改善。

可重跑：

```powershell
.\.venv\Scripts\python.exe tools/smoke_yolo.py --consensus none
.\.venv\Scripts\python.exe tools/smoke_yolo.py --consensus temporal
.\.venv\Scripts\python.exe tools/smoke_yolo.py --consensus rotation
```

另已執行無 UI CLI：

```powershell
.\.venv\Scripts\python.exe -m uvc_lab --config profiles/demo.json --capture
```

模擬相機拍攝 5 秒、每 0.5 秒一張，完成 10 次拍攝與 raw／processed／設定／manifest／SQLite／Excel 輸出。

## 視覺驗證

使用實際 PyQt 介面離屏渲染，不是 HTML 概念稿：

- [台灣繁體中文](ui-zh_TW.png)
- [English](ui-en.png)
- [日本語](ui-ja.png)

已檢查字型、六頁建立、模擬畫面的 ROI／格線、資料標示、主要操作與可捲動設定。圖片中的畫面明確標為模擬資料。

## Windows 單檔 EXE 發行驗證

`UVC-Lab-0.1.0-Windows-x64.exe` 已在 Windows 11 x64 直接執行，使用全新資料目錄，PATH 只保留 Windows 系統目錄。完整 `--self-test` 回傳退出碼 **0**，報告為 `ok: true`、`frozen: true`。版本、檔案大小、SHA-256 與機器可讀結果見 [windows-release.json](windows-release.json)。

- 內嵌 FFmpeg 正常找到，無須外部 FFmpeg 或 PATH 中的 Python。
- 相機輔助程序可由 EXE 再啟動，透過 JSON 檔案回報模擬模式清單；實際原生相機列舉正常回傳空清單。
- 實際建立繁體中文、英文、日文 Qt 視窗並輸出影像。
- 模擬拍攝 4 張，保存 raw／processed、設定、manifest、SQLite 與可讀 Excel。
- 實際 MP4 經 FFprobe 驗證為 H.264、160×120、20 FPS、4 frames；另成功寫入 AVI。
- 從 EXE 內嵌權重執行 YOLO26 segmentation＋pose，取得 6 個融合物體，並驗證有同時包含面積與 keypoints 的結果。
- EXE 包含預設兩個模型與第三方授權文件；原始碼 ZIP 不包含二進位依賴與權重。

建置時隔離 DLL 搜尋 PATH，避免其他工具的 `icuuc.dll` 覆蓋 Windows ICU，造成 Qt 的「找不到指定的程序」錯誤。PyTorch 2.11 啟動需要 `torch.testing._internal.logging_tensor`，因此不排除該內部套件。這兩項問題皆以實際 EXE 自測確認修正。

此 CPU 版未做程式碼簽章；需要 CUDA 時請使用原始碼與對應的 PyTorch 環境。這次是本機實際 EXE 測試，沒有宣稱已在另一台乾淨 Windows 電腦或實體相機上驗收。

## 仍需實體環境驗收

本機執行 `python -m uvc_lab.camera scan` 回傳 `[]`，沒有可供驗收的 UVC 相機。因此以下沒有冒充成已通過：

1. 每款 IMX179 板／韌體在目標電腦上的完整 mode 清單與真實 FPS。先檢查 MJPG／YUY2、選定解析度、不同 USB port／hub，分別做單相機與多相機完整輸出測試。
2. AE、AF、AWB、manual exposure／gain／WB 實際是否生效。驅動的 set／readback 測試需再以影像或外部量測確認；不支援的硬體不能由通用 API 補出光學能力。
3. 連續 24–72 小時的磁碟、USB 穩定性、相機溫升、儲存格式與完整 FPS／掉幀曲線。8 秒測試是篩選條件，不是長期負載認證。
4. ESP32／Pi／PC 的實際到位感測、網路斷線與馬達穩定時間。HTTP 協定已測試，未宣稱實體馬達已到位或可安全移動。
5. 實驗物種與自訓模型的偵測、segmentation、pose、tracking 與投票 precision／recall。官方 person pose 不能替代物種專用模型。
6. 透視後的 µm/pixel、mask 面積與相機背景防抖量測。靜態背景不足時，平移估計也可能把生物運動當作相機抖動；預設不開啟防抖。

Windows 為目前驗證平台；Linux／macOS 有後端與能力降級路徑，尚未在這次環境中實機驗證。相機驅動若完全卡在原生讀取，程式會回報需重啟；不以另一個執行緒強制關閉同一個 camera handle。

## 報表與資料限制

完整設定、候選、mask、transform 與軌跡以 JSON／SQLite 為準。Excel 超過單張工作表列數會分頁並保留標題；超過 Excel 單格長度的文字 payload 不能完整放入儲存格，完整內容仍保留在 SQLite。

無限長工作不代表硬碟永久足夠。估算會顯示可用時間，拍攝時持續檢查保留空間；影片依 stream 分段，以降低未完成片段的影響。
