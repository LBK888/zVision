"""
YOLO Object Tracker Pro
支援 YOLOv8/v11/v12, UKF, Gating, Vectorized Matching, ReID
"""

import numpy as np
from typing import Any, Iterable, List, Dict, Tuple, Optional, Sequence, Union
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment
import pandas as pd
from datetime import datetime

try:
    from ultralytics.engine.results import Results, Boxes
except ImportError:
    Results = None
    Boxes = None


@dataclass
class TrackedObject:
    """追蹤物件的資料結構"""
    id: int
    bbox: np.ndarray  # [x1, y1, x2, y2] (測量更新後的位置)
    class_id: int
    confidence: float
    frame_id: int
    
    # 狀態計數器
    age: int = 0            # 物件存在的總幀數
    hits: int = 0           # 成功匹配檢測的次數
    time_since_update: int = 0  # 距離上次檢測匹配經過的幀數
    
    # UKF 狀態
    # State vector: [x_center, y_center, width, height, v_xc, v_yc, v_w, v_h]
    state: np.ndarray = field(default_factory=lambda: np.zeros(8))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(8))
    
    # 預測位置 (用於記錄/除錯)
    pred_pos: Optional[Tuple[float, float]] = None  # (pred_cx, pred_cy)
    
    # 額外資訊
    mask: Optional[np.ndarray] = None
    keypoints: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None  # ReID
    trail: List[Tuple[float, float]] = field(default_factory=list)


class UnscentedKalmanFilter:
    """
    數值穩定的 Unscented Kalman Filter 實現 (支援動態 dt)
    """
    
    def __init__(self, dim_x=8, dim_z=4):
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        # UKF sigma point 參數 (Merwe Scaled)
        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0.0
        self.lambda_ = self.alpha**2 * (dim_x + self.kappa) - dim_x
        
        # 權重計算
        self.Wm = np.full(2 * dim_x + 1, 1 / (2 * (dim_x + self.lambda_)))
        self.Wc = np.full(2 * dim_x + 1, 1 / (2 * (dim_x + self.lambda_)))
        self.Wm[0] = self.lambda_ / (dim_x + self.lambda_)
        self.Wc[0] = self.lambda_ / (dim_x + self.lambda_) + (1 - self.alpha**2 + self.beta)
        
        # 過程噪聲 (Process Noise) Q
        self.Q = np.eye(dim_x)
        self.Q[0:4, 0:4] *= 0.01  # 位置變異小
        self.Q[4:8, 4:8] *= 0.1   # 速度變異大
        
        # 測量噪聲 (Measurement Noise) R
        self.R = np.eye(dim_z) * 0.1

    def generate_sigma_points(self, x, P):
        n = len(x)
        sigma = np.zeros((2 * n + 1, n))
        sigma[0] = x
        
        # 確保對稱正定
        P_stable = (P + P.T) / 2 + np.eye(n) * 1e-6
        
        try:
            U = np.linalg.cholesky((n + self.lambda_) * P_stable)
        except np.linalg.LinAlgError:
            u, s, _ = np.linalg.svd((n + self.lambda_) * P_stable)
            U = u @ np.diag(np.sqrt(s))
            
        for i in range(n):
            sigma[i + 1] = x + U[i]
            sigma[n + i + 1] = x - U[i]
        return sigma
    
    def state_transition(self, x, dt):
        """
        狀態轉移函數 (恆定速度模型)
        x' = F * x
        """
        # 這裡不顯式建立大矩陣 F 以節省資源，直接運算
        # x = [cx, cy, w, h, vx, vy, vw, vh]
        new_x = x.copy()
        new_x[0] += x[4] * dt  # cx += vx * dt
        new_x[1] += x[5] * dt  # cy += vy * dt
        new_x[2] += x[6] * dt  # w  += vw * dt
        new_x[3] += x[7] * dt  # h  += vh * dt
        return new_x
    
    def measurement_function(self, x):
        return x[:4]
    
    def predict(self, x, P, dt=1.0):
        """
        預測步驟 (加入 dt 參數)
        """
        sigma_points = self.generate_sigma_points(x, P)
        
        # 傳播 sigma points (代入 dt)
        sigma_points_f = np.array([self.state_transition(sp, dt) for sp in sigma_points])
        
        x_pred = np.dot(self.Wm, sigma_points_f)
        
        P_pred = self.Q.copy() # 若 dt 很大，Q 也應該隨之調整，這裡簡化假設 Q 為單位時間噪聲
        if dt != 1.0:
             # 簡單的時間縮放調整過程噪聲 ( heuristic )
             P_pred *= dt 

        diff = sigma_points_f - x_pred[None, :]
        P_pred += np.dot(diff.T, np.dot(np.diag(self.Wc), diff))
        
        return x_pred, P_pred
    
    def update(self, x_pred, P_pred, z):
        sigma_points = self.generate_sigma_points(x_pred, P_pred)
        sigma_points_h = np.array([self.measurement_function(sp) for sp in sigma_points])
        z_pred = np.dot(self.Wm, sigma_points_h)
        
        diff_z = sigma_points_h - z_pred[None, :]
        S = self.R.copy() + np.dot(diff_z.T, np.dot(np.diag(self.Wc), diff_z))
        
        diff_x = sigma_points - x_pred[None, :]
        T = np.dot(diff_x.T, np.dot(np.diag(self.Wc), diff_z))
        
        # 使用 pinv 避免奇異矩陣
        K = np.dot(T, np.linalg.pinv(S))
        
        residual = z - z_pred
        x_updated = x_pred + np.dot(K, residual)
        P_updated = P_pred - np.dot(K, np.dot(S, K.T))
        
        return x_updated, P_updated


class YOLOTracker:
    
    def __init__(
        self,
        max_age: int = 30,
        max_trail_len: int = 30,  # 新增: 獨立的軌跡長度參數
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        gating_threshold: float = 1000.0, # 新增: Gating 閾值 (像素距離)
        use_area_similarity: bool = True,
        area_weight: float = 0.2,
        appearance_weight: float = 0.0,
        # 新增: 類別相似度設定 (None=嚴格匹配, Dict={(0,1): 0.5} 表示 class 0 和 1 有 0.5 相似度)
        class_similarity_map: Optional[Dict[Tuple[int, int], float]] = None 
    ):
        self.max_age = max_age
        self.max_trail_len = max_trail_len
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.gating_threshold = gating_threshold
        
        self.use_area_similarity = use_area_similarity
        self.area_weight = area_weight
        self.appearance_weight = appearance_weight
        self.class_similarity_map = class_similarity_map
        
        self.tracked_objects: List[TrackedObject] = []
        self.next_id = 1
        self.frame_count = 0
        self.ukf = UnscentedKalmanFilter()
        
        self.tracking_history: List[Dict] = []

    @staticmethod
    def bbox_to_xywh(bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])
    
    @staticmethod
    def xywh_to_bbox(xywh: np.ndarray) -> np.ndarray:
        cx, cy, w, h = xywh
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2])
    
    @staticmethod
    def calculate_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """計算兩個邊界框的 IOU"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    @staticmethod
    def calculate_area_similarity(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """計算面積相似度"""
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        return min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0.0


    def _compute_cost_matrix(self, tracks: List[TrackedObject], detections: List[Dict]) -> np.ndarray:
        """
        向量化計算 Cost Matrix
        包含: IOU, Area, Embedding, Class Similarity, Gating
        """
        n_tracks = len(tracks)
        n_dets = len(detections)
        
        if n_tracks == 0 or n_dets == 0:
            return np.empty((n_tracks, n_dets))

        # 準備數據矩陣
        track_bboxes = np.array([t.bbox for t in tracks])   # [N, 4]
        det_bboxes = np.array([d['bbox'] for d in detections]) # [M, 4]
        
        track_centers = track_bboxes[:, :2] + (track_bboxes[:, 2:] - track_bboxes[:, :2]) / 2
        det_centers = det_bboxes[:, :2] + (det_bboxes[:, 2:] - det_bboxes[:, :2]) / 2
        
        # --- 1. 向量化 Gating (Distance Check) ---
        # 使用 Broadcasting 計算歐式距離矩陣 [N, M]
        # dist[i, j] = ||track_center[i] - det_center[j]||
        diff = track_centers[:, None, :] - det_centers[None, :, :] # [N, M, 2]
        dists = np.sqrt(np.sum(diff**2, axis=2)) # [N, M]
        
        # 標記超出 Gating 範圍的配對
        gating_mask = dists > self.gating_threshold
        
        # --- 2. 向量化 IOU 計算 ---
        # 擴展維度 [N, 1, 4] vs [1, M, 4]
        b1 = track_bboxes[:, None, :]
        b2 = det_bboxes[None, :, :]
        
        inter_x1 = np.maximum(b1[..., 0], b2[..., 0])
        inter_y1 = np.maximum(b1[..., 1], b2[..., 1])
        inter_x2 = np.minimum(b1[..., 2], b2[..., 2])
        inter_y2 = np.minimum(b1[..., 3], b2[..., 3])
        
        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
        area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
        union_area = area1 + area2 - inter_area
        
        iou_matrix = inter_area / (union_area + 1e-6)
        
        # 初始 Score 為 IOU
        score_matrix = iou_matrix.copy()
        
        # --- 3. Area Similarity (Optional) ---
        if self.use_area_similarity:
            min_area = np.minimum(area1, area2)
            max_area = np.maximum(area1, area2)
            area_sim = min_area / (max_area + 1e-6)
            # Broadcasting already produces [N, M], no squeeze needed
            score_matrix = (1 - self.area_weight) * score_matrix + self.area_weight * area_sim


        # --- 4. Appearance / ReID (Optional) ---
        if self.appearance_weight > 0:
            # 需確保所有物件都有 embedding，否則此步跳過或部分計算
            track_embs = [t.embedding for t in tracks]
            det_embs = [d.get('embedding') for d in detections]
            
            if all(e is not None for e in track_embs) and all(e is not None for e in det_embs):
                t_emb = np.array(track_embs)
                d_emb = np.array(det_embs)
                # Normalize
                t_emb /= np.linalg.norm(t_emb, axis=1, keepdims=True) + 1e-6
                d_emb /= np.linalg.norm(d_emb, axis=1, keepdims=True) + 1e-6
                
                # Cosine sim [N, M]
                emb_sim = np.dot(t_emb, d_emb.T)
                score_matrix = (1 - self.appearance_weight) * score_matrix + self.appearance_weight * emb_sim

        # --- 5. Class Similarity Logic ---
        track_classes = np.array([t.class_id for t in tracks]) # [N]
        det_classes = np.array([d['class_id'] for d in detections]) # [M]
        
        # 建立 [N, M] 的類別權重矩陣
        class_weights = np.ones((n_tracks, n_dets))
        
        # 如果沒有 map，執行嚴格匹配 (不同類別 weight=0)
        # 如果有 map，查找相似度
        for i in range(n_tracks):
            for j in range(n_dets):
                tid, did = track_classes[i], det_classes[j]
                if tid == did:
                    continue # weight = 1.0
                
                if self.class_similarity_map:
                    # 嘗試查找 (id1, id2) 或 (id2, id1)
                    sim = self.class_similarity_map.get((tid, did))
                    if sim is None:
                        sim = self.class_similarity_map.get((did, tid), 0.0) # 預設不同類別為 0
                    class_weights[i, j] = sim
                else:
                    class_weights[i, j] = 0.0 # 嚴格模式

        # 應用類別權重
        score_matrix *= class_weights
        
        # --- 轉換為 Cost ---
        cost_matrix = 1.0 - score_matrix
        
        # 應用 Gating (將距離過遠的 Cost 設為無限大)
        cost_matrix[gating_mask] = 10000.0
        
        # 將類別完全不匹配 (weight=0 -> score=0 -> cost=1) 或 gating 失敗的設為不可選
        # 注意: 原本 score=0 -> cost=1。如果 IOU threshold=0.3, 則 cost > 0.7 都不會選。
        # 我們可以保留 cost=1，後續 filter 會過濾掉。
        
        # 🔧 FIX: Ensure cost_matrix contains no NaN or Inf
        # This can happen if bbox dimensions are invalid (zero/negative width/height)
        cost_matrix = np.nan_to_num(cost_matrix, nan=10000.0, posinf=10000.0, neginf=10000.0)
        
        return cost_matrix

    def predict_tracks(self, dt: float):
        """預測所有軌跡，並記錄預測位置"""
        for obj in self.tracked_objects:
            obj.state, obj.covariance = self.ukf.predict(obj.state, obj.covariance, dt=dt)
            
            # 記錄預測位置 (Prediction)
            pred_xywh = obj.state[:4]
            obj.pred_pos = (float(pred_xywh[0]), float(pred_xywh[1]))
            
            # 更新 bbox 供後續 matching 使用
            obj.bbox = self.xywh_to_bbox(pred_xywh)
            
            obj.age += 1
            obj.time_since_update += 1

    def update(self, detections: Any, frame_id: Optional[int] = None, dt: float = 1.0) -> List[TrackedObject]:
        """
        Args:
            detections: 檢測結果
            frame_id: 當前幀號
            dt: 與上一幀的時間差 (秒或幀數單位)，預設 1.0
        """
        parsed_detections = self._normalize_detections(detections)
        
        if frame_id is None:
            frame_id = self.frame_count
        self.frame_count = frame_id + 1
        
        # 1. 預測 (Predict) - 傳入 dt
        self.predict_tracks(dt)
        
        # 邊界情況處理：若無軌跡且無檢測，直接返回
        if len(parsed_detections) == 0 and len(self.tracked_objects) == 0:
             return []
             
        # 邊界情況處理：若無檢測，只清理舊軌跡
        if len(parsed_detections) == 0:
            self._cleanup_tracks()
            return self.get_confirmed_tracks()

        # 邊界情況處理：若無現有軌跡，全部視為新物件 (避免 linear_sum_assignment 錯誤)
        if len(self.tracked_objects) == 0:
            self._create_new_tracks(parsed_detections, frame_id)
            return self.get_confirmed_tracks()

        # 2. 計算 Cost Matrix (向量化)
        cost_matrix = self._compute_cost_matrix(self.tracked_objects, parsed_detections)
        
        # 3. 匈牙利演算法
        # 雙重確認矩陣尺寸，雖然上面已有防護
        if cost_matrix.size == 0:
            row_indices, col_indices = [], []
        else:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matched_indices = set()
        unmatched_detections = set(range(len(parsed_detections)))
        
        # 4. 處理匹配
        for row, col in zip(row_indices, col_indices):
            # 檢查 Cost 閾值 (包含 Gating 的結果)
            # threshold: score < 0.3 (iou_thresh) => cost > 0.7
            if cost_matrix[row, col] > (1 - self.iou_threshold):
                continue
                
            obj = self.tracked_objects[row]
            det = parsed_detections[col]
            
            # UKF Update (Correction)
            measurement = self.bbox_to_xywh(np.array(det['bbox']))
            obj.state, obj.covariance = self.ukf.update(obj.state, obj.covariance, measurement)
            
            # 更新物件狀態
            obj.bbox = np.array(det['bbox'])
            obj.confidence = det['confidence']
            obj.class_id = det['class_id']
            obj.mask = det.get('mask')
            obj.keypoints = det.get('keypoints')
            obj.embedding = det.get('embedding')
            
            obj.hits += 1
            obj.time_since_update = 0
            obj.frame_id = frame_id
            
            # 更新 Trail (使用 max_trail_len)
            obj.trail = self._update_trail(obj.trail, obj.bbox)
            
            det['track_id'] = obj.id
            matched_indices.add(row)
            unmatched_detections.remove(col)
            
            self._record_tracking(obj)
            
        # 5. 創建新軌跡
        unmatched_dets_list = [parsed_detections[i] for i in unmatched_detections]
        self._create_new_tracks(unmatched_dets_list, frame_id)
        
        # 6. 清理過期軌跡
        self._cleanup_tracks()
        
        return self.get_confirmed_tracks()

    def _create_new_tracks(self, detections: List[Dict], frame_id: int):
        for det in detections:
            bbox = np.array(det['bbox'])
            xywh = self.bbox_to_xywh(bbox)
            
            new_obj = TrackedObject(
                id=self.next_id,
                bbox=bbox,
                class_id=det['class_id'],
                confidence=det['confidence'],
                frame_id=frame_id,
                mask=det.get('mask'),
                keypoints=det.get('keypoints'),
                embedding=det.get('embedding'),
                hits=1,
                age=1,
                time_since_update=0,
                pred_pos=(float(xywh[0]), float(xywh[1])) # 初始預測位置即為當前位置
            )
            
            # 初始化狀態與協方差
            new_obj.state = np.concatenate([xywh, np.zeros(4)])
            new_obj.covariance = np.eye(8) * 10.0 
            
            new_obj.trail = self._update_trail([], bbox)
            
            det['track_id'] = new_obj.id
            self.tracked_objects.append(new_obj)
            self.next_id += 1
            
            self._record_tracking(new_obj)

    def _cleanup_tracks(self):
        self.tracked_objects = [
            obj for obj in self.tracked_objects
            if obj.time_since_update <= self.max_age
        ]

    def get_confirmed_tracks(self) -> List[TrackedObject]:
        return [
            obj for obj in self.tracked_objects
            if obj.hits >= self.min_hits or self.min_hits == 0
        ]

    def _update_trail(self, trail: List[Tuple[float, float]], bbox: Sequence[float]) -> List[Tuple[float, float]]:
        center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
        new_trail = trail + [center]
        # 使用獨立的 max_trail_len
        if len(new_trail) > self.max_trail_len:
            new_trail = new_trail[-self.max_trail_len:]
        return new_trail

    def _record_tracking(self, obj: TrackedObject):
        """記錄包含預測位置的詳細資料"""
        pred_x, pred_y = obj.pred_pos if obj.pred_pos else (0.0, 0.0)
        
        record = {
            'frame_id': obj.frame_id,
            'object_id': obj.id,
            'class_id': obj.class_id,
            'x1': round(obj.bbox[0], 1),
            'y1': round(obj.bbox[1], 1),
            'x2': round(obj.bbox[2], 1),
            'y2': round(obj.bbox[3], 1),
            # 新增預測位置欄位
            'pred_x': round(pred_x, 1),
            'pred_y': round(pred_y, 1),

            'confidence': obj.confidence,
            'hits': obj.hits,
            'age': obj.age

        }
        self.tracking_history.append(record)

    def export_to_excel(self, filename: str = 'tracking_results.xlsx'):
        if not self.tracking_history:
            print("No data to export.")
            return
        df = pd.DataFrame(self.tracking_history)
        df.to_excel(filename, index=False)
        print(f"Exported to {filename}")

    # --- Helper Functions for Ultralytics format (同前) ---
    def _normalize_detections(self, detections: Any) -> List[Dict]:
        if detections is None: return []
        if isinstance(detections, dict): return [detections]
        if self._is_ultralytics_results(detections): return self._convert_results_sequence([detections])
        if self._is_ultralytics_boxes(detections): return self._convert_boxes(detections)
        if isinstance(detections, (list, tuple)):
            if not detections: return []
            first = detections[0]
            if isinstance(first, dict): return list(detections)
            if self._is_ultralytics_results(first): return self._convert_results_sequence(detections)
        return []

    def _is_ultralytics_results(self, obj: Any) -> bool:
        return Results is not None and isinstance(obj, Results)

    def _is_ultralytics_boxes(self, obj: Any) -> bool:
        return Boxes is not None and isinstance(obj, Boxes)

    def _convert_results_sequence(self, results_seq: Sequence[Any]) -> List[Dict]:
        detections = []
        for result in results_seq:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0: continue
            
            xyxy = self._tensor_to_numpy(boxes.xyxy)
            conf = self._tensor_to_numpy(boxes.conf)
            cls_ids = self._tensor_to_numpy(boxes.cls)
            
            masks_data = self._tensor_to_numpy(result.masks.data) if hasattr(result, "masks") and result.masks is not None else None
            kpts_data = self._tensor_to_numpy(result.keypoints.data) if hasattr(result, "keypoints") and result.keypoints is not None else None

            for i in range(len(xyxy)):
                det = {
                    "bbox": xyxy[i].tolist(),
                    "class_id": int(cls_ids[i]),
                    "confidence": float(conf[i])
                }
                if masks_data is not None and i < len(masks_data): det["mask"] = masks_data[i]
                if kpts_data is not None and i < len(kpts_data): det["keypoints"] = kpts_data[i]
                detections.append(det)
        return detections

    def _convert_boxes(self, boxes: Any) -> List[Dict]:
        xyxy = self._tensor_to_numpy(boxes.xyxy)
        conf = self._tensor_to_numpy(boxes.conf)
        cls_ids = self._tensor_to_numpy(boxes.cls)
        if xyxy is None: return []
        return [{"bbox": xyxy[i].tolist(), "class_id": int(cls_ids[i]), "confidence": float(conf[i])} for i in range(len(xyxy))]

    @staticmethod
    def _tensor_to_numpy(data: Any):
        if data is None: return None
        if isinstance(data, np.ndarray): return data
        if hasattr(data, "detach") and hasattr(data, "cpu"): return data.detach().cpu().numpy()
        if hasattr(data, "cpu"): return data.cpu().numpy()
        return np.array(data)

    def reset(self):
        """重置追蹤器"""
        self.tracked_objects = []
        self.next_id = 1
        self.frame_count = 0
        self.tracking_history = []


# --- New Trackers Implementation ---

class KalmanFilterLinear:
    """
    Standard Kalman Filter for SORT/ByteTrack (Constant Velocity Model)
    State: [x, y, s, r, vx, vy, vs] where s=area, r=aspect_ratio
    """
    def __init__(self):
        self.dim_x = 7
        self.dim_z = 4
        self.F = np.eye(7) # State transition
        self.H = np.eye(4, 7) # Measurement function
        self.R = np.eye(4) * 1.0 # Measurement noise
        self.P = np.eye(7) * 1000.0 # Error covariance
        self.Q = np.eye(7) * 0.01 # Process noise

        # Initialize F (dt=1)
        for i in range(4):
            if i < 3: # x, y, s have velocity
                self.F[i, i+4] = 1.0
                
    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.dim_x)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

    def initiate(self, measurement):
        self.x = np.zeros(7)
        self.x[:4] = measurement
        self.P = np.eye(7) * 10.0
        self.R = np.eye(4) * 1.0
        self.Q = np.eye(7) * 0.01

    @staticmethod
    def bbox_to_z(bbox):
        """[x1,y1,x2,y2] -> [x, y, s, r]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h) if h > 0 else 1.0
        return np.array([x, y, s, r])

    @staticmethod
    def x_to_bbox(x):
        """[x, y, s, r] -> [x1, y1, x2, y2]"""
        center_x, center_y, s, r = x[:4]
        
        # Ensure s (area) and r (aspect ratio) are positive to avoid NaN
        s = max(s, 1.0)  # Minimum area of 1 pixel
        r = max(r, 0.01)  # Minimum aspect ratio to avoid division issues
        
        w = np.sqrt(s * r)
        h = s / w if w > 0 else 1.0
        
        # Ensure width and height are positive
        w = max(w, 1.0)
        h = max(h, 1.0)
        
        return np.array([center_x-w/2, center_y-h/2, center_x+w/2, center_y+h/2])


class SORTTracker(YOLOTracker):
    """
    SORT: Simple Online and Realtime Tracking
    Uses Linear Kalman Filter and IOU matching.
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3, **kwargs):
        super().__init__(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold, **kwargs)
        # Override UKF with Linear KF for each object (managed manually in update)
        # We will store KF instance in TrackedObject.custom_kf
    
    def _create_new_tracks(self, detections: List[Dict], frame_id: int):
        for det in detections:
            bbox = np.array(det['bbox'])
            
            new_obj = TrackedObject(
                id=self.next_id,
                bbox=bbox,
                class_id=det['class_id'],
                confidence=det['confidence'],
                frame_id=frame_id,
                mask=det.get('mask'),
                keypoints=det.get('keypoints'),
                embedding=det.get('embedding'),
                hits=1,
                age=1,
                time_since_update=0
            )
            
            # Initialize Linear KF
            kf = KalmanFilterLinear()
            kf.initiate(KalmanFilterLinear.bbox_to_z(bbox))
            new_obj.state = kf.x # Use state to store KF state
            new_obj.covariance = kf.P
            new_obj.custom_kf = kf # Attach KF object
            
            # Predict pos
            new_obj.pred_pos = (float(bbox[0]+bbox[2])/2, float(bbox[1]+bbox[3])/2)
            
            new_obj.trail = self._update_trail([], bbox)
            
            det['track_id'] = new_obj.id
            self.tracked_objects.append(new_obj)
            self.next_id += 1
            self._record_tracking(new_obj)

    def predict_tracks(self, dt: float):
        # SORT assumes constant frame rate usually, but we can adapt if needed.
        # Standard SORT ignores dt in basic implementation, but we can scale process noise if we wanted.
        for obj in self.tracked_objects:
            if hasattr(obj, 'custom_kf'):
                obj.state = obj.custom_kf.predict()
                obj.bbox = KalmanFilterLinear.x_to_bbox(obj.state)
                
                # Update pred_pos
                cx = (obj.bbox[0] + obj.bbox[2]) / 2
                cy = (obj.bbox[1] + obj.bbox[3]) / 2
                obj.pred_pos = (float(cx), float(cy))
            
            obj.age += 1
            obj.time_since_update += 1

    def update(self, detections: Any, frame_id: Optional[int] = None, dt: float = 1.0) -> List[TrackedObject]:
        # Reuse YOLOTracker's update logic but ensure it uses our predict_tracks and _create_new_tracks
        # And we need to override the update step in the matching loop
        
        parsed_detections = self._normalize_detections(detections)
        if frame_id is None: frame_id = self.frame_count
        self.frame_count = frame_id + 1
        
        self.predict_tracks(dt)
        
        if len(parsed_detections) == 0:
            self._cleanup_tracks()
            return self.get_confirmed_tracks()

        # SORT uses IOU only
        # We can use _compute_cost_matrix from YOLOTracker but force weights
        # Or just implement simple IOU matching here.
        # For consistency with the requested feature "Class Similarity", we should use the parent's cost matrix
        # which supports class similarity.
        
        cost_matrix = self._compute_cost_matrix(self.tracked_objects, parsed_detections)
        
        if cost_matrix.size == 0:
            row_indices, col_indices = [], []
        else:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
        matched_indices = set()
        unmatched_detections = set(range(len(parsed_detections)))
        
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] > (1 - self.iou_threshold):
                continue
                
            obj = self.tracked_objects[row]
            det = parsed_detections[col]
            
            # KF Update
            if hasattr(obj, 'custom_kf'):
                obj.custom_kf.update(KalmanFilterLinear.bbox_to_z(np.array(det['bbox'])))
                obj.state = obj.custom_kf.x
                obj.covariance = obj.custom_kf.P
                obj.bbox = KalmanFilterLinear.x_to_bbox(obj.state)
            
            # Update info
            obj.confidence = det['confidence']
            obj.class_id = det['class_id']
            obj.mask = det.get('mask')
            obj.keypoints = det.get('keypoints')
            obj.embedding = det.get('embedding')
            obj.hits += 1
            obj.time_since_update = 0
            obj.frame_id = frame_id
            obj.trail = self._update_trail(obj.trail, obj.bbox)
            
            det['track_id'] = obj.id
            matched_indices.add(row)
            unmatched_detections.remove(col)
            self._record_tracking(obj)
            
        # Create new
        unmatched_dets_list = [parsed_detections[i] for i in unmatched_detections]
        self._create_new_tracks(unmatched_dets_list, frame_id)
        self._cleanup_tracks()
        
        return self.get_confirmed_tracks()


class ByteTracker(SORTTracker):
    """
    ByteTrack: Multi-stage matching (High conf -> IOU, Low conf -> IOU)
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3, high_thresh=0.5, low_thresh=0.1, **kwargs):
        super().__init__(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold, **kwargs)
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh

    def update(self, detections: Any, frame_id: Optional[int] = None, dt: float = 1.0) -> List[TrackedObject]:
        parsed_detections = self._normalize_detections(detections)
        if frame_id is None: frame_id = self.frame_count
        self.frame_count = frame_id + 1
        
        self.predict_tracks(dt)
        
        # Split detections
        high_dets = []
        low_dets = []
        for d in parsed_detections:
            if d['confidence'] >= self.high_thresh:
                high_dets.append(d)
            elif d['confidence'] >= self.low_thresh:
                low_dets.append(d)
        
        # First association: High confidence detections
        # Use IOU (and class similarity from parent)
        cost_matrix_1 = self._compute_cost_matrix(self.tracked_objects, high_dets)
        
        if cost_matrix_1.size > 0:
            row_ind_1, col_ind_1 = linear_sum_assignment(cost_matrix_1)
        else:
            row_ind_1, col_ind_1 = [], []
            
        matched_tracks = set()
        unmatched_high_dets = set(range(len(high_dets)))
        
        # Update matched
        for r, c in zip(row_ind_1, col_ind_1):
            if cost_matrix_1[r, c] <= (1 - self.iou_threshold):
                obj = self.tracked_objects[r]
                det = high_dets[c]
                self._update_object(obj, det, frame_id)
                matched_tracks.add(r)
                unmatched_high_dets.remove(c)
        
        # Second association: Low confidence detections with remaining tracks
        remaining_tracks_idx = [i for i in range(len(self.tracked_objects)) if i not in matched_tracks]
        remaining_tracks = [self.tracked_objects[i] for i in remaining_tracks_idx]
        
        if len(remaining_tracks) > 0 and len(low_dets) > 0:
            cost_matrix_2 = self._compute_cost_matrix(remaining_tracks, low_dets)
            if cost_matrix_2.size > 0:
                row_ind_2, col_ind_2 = linear_sum_assignment(cost_matrix_2)
            else:
                row_ind_2, col_ind_2 = [], []
                
            for r, c in zip(row_ind_2, col_ind_2):
                # Loose threshold for second stage usually? Or same? 
                # ByteTrack usually uses same or slightly looser IOU. Let's stick to iou_threshold.
                if cost_matrix_2[r, c] <= (1 - self.iou_threshold):
                    real_track_idx = remaining_tracks_idx[r]
                    obj = self.tracked_objects[real_track_idx]
                    det = low_dets[c]
                    self._update_object(obj, det, frame_id)
                    matched_tracks.add(real_track_idx)
        
        # Initialize new tracks from unmatched high confidence detections
        new_dets_list = [high_dets[i] for i in unmatched_high_dets]
        self._create_new_tracks(new_dets_list, frame_id)
        
        self._cleanup_tracks()
        return self.get_confirmed_tracks()

    def _update_object(self, obj, det, frame_id):
        if hasattr(obj, 'custom_kf'):
            obj.custom_kf.update(KalmanFilterLinear.bbox_to_z(np.array(det['bbox'])))
            obj.state = obj.custom_kf.x
            obj.covariance = obj.custom_kf.P
            obj.bbox = KalmanFilterLinear.x_to_bbox(obj.state)
        
        obj.confidence = det['confidence']
        obj.class_id = det['class_id']
        obj.mask = det.get('mask')
        obj.keypoints = det.get('keypoints')
        obj.embedding = det.get('embedding')
        obj.hits += 1
        obj.time_since_update = 0
        obj.frame_id = frame_id
        obj.trail = self._update_trail(obj.trail, obj.bbox)
        det['track_id'] = obj.id
        self._record_tracking(obj)


def create_tracker(name: str, config: Dict) -> YOLOTracker:
    """Factory method to create tracker"""
    if name.lower() == 'sort':
        return SORTTracker(**config)
    elif name.lower() == 'bytetrack':
        return ByteTracker(**config)
    else:
        # Default to original UKF based YOLOTracker
        return YOLOTracker(**config)