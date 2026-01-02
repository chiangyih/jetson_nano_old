import os  # 判斷顯示環境（目前不顯示預覽）
import platform  # 取得作業系統資訊（用於選擇攝影機後端）
import sys  # 程式結束時回傳 exit code 用
import time  # 計時用（用於計算 FPS）

import cv2  # OpenCV：讀取攝影機與顯示視窗
from ultralytics import YOLO  # Ultralytics：載入 YOLOv8 模型

MODEL_PATH = "yolov8n.pt"  # 指定要使用的 YOLOv8n 模型檔
CAMERA_INDEX = 0  # USB 攝影機編號（0 通常是第一支）
FRAME_WIDTH = 640  # 偵測/顯示用影像寬度
FRAME_HEIGHT = 480  # 偵測/顯示用影像高度
WINDOW_NAME = "YOLOv8m - person"  # 視窗標題文字
PERSON_CLASS_ID = 0  # COCO person（人類）的類別 id


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:  # 開啟指定編號的 USB 攝影機（指定解析度）
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else 0  # Windows 優先用 DirectShow，其他系統用預設後端
    cap = cv2.VideoCapture(index, backend)  # 建立攝影機擷取物件
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # 設定擷取寬度（不保證每台攝影機都會接受）
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # 設定擷取高度（不保證每台攝影機都會接受）
    return cap  # 回傳擷取物件


def ensure_frame_size(frame, width: int, height: int):  # 確保影像大小一致（必要時做 resize）
    if frame.shape[1] == width and frame.shape[0] == height:  # 若解析度已符合目標
        return frame  # 直接回傳原始影像
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)  # 不符合就 resize 成指定大小


def overlay_status(frame, model_path: str, fps: float) -> None:  # 在影像上疊加狀態文字
    cv2.putText(  # 在畫面左上角疊加文字資訊
        frame,  # 要畫字的影像
        f"Model: {model_path} | FPS: {fps:.1f}",  # 顯示模型名稱與 FPS
        (10, 30),  # 文字左上角座標
        cv2.FONT_HERSHEY_SIMPLEX,  # 字型
        0.8,  # 字體大小
        (0, 255, 0),  # 字體顏色（綠色）
        2,  # 線條粗細
        cv2.LINE_AA,  # 反鋸齒
    )  # 結束 putText 呼叫


def window_closed(window_name: str) -> bool:  # 檢查視窗是否被使用者關閉
    try:  # 某些環境可能對 getWindowProperty 支援不完整
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1  # 視窗不可見代表已關閉
    except cv2.error:  # OpenCV 例外（保守處理為未關閉）
        return False  # 回傳未關閉


def main() -> int:  # 主流程：回傳 0 代表正常結束，非 0 代表失敗
    try:  # 載入模型可能失敗（例如檔案不存在）
        model = YOLO(MODEL_PATH)  # 載入 YOLO 模型
    except Exception as exc:  # 捕捉任何載入模型的錯誤
        print(f"[error] 載入模型失敗：{exc}")  # 顯示錯誤原因
        return 1  # 以非 0 表示失敗

    cap = open_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)  # 開啟 USB 攝影機並嘗試設定解析度
    if not cap.isOpened():  # 檢查攝影機是否成功開啟
        print(f"[error] 無法開啟 USB 攝影機 (index={CAMERA_INDEX})")  # 顯示錯誤訊息
        return 2  # 以非 0 表示失敗

    print("[info] 已停用即時視窗預覽，僅在終端輸出狀態。") # 提示使用者目前不顯示視窗

    try:  # 主要執行區塊（確保最後會釋放資源）
        last_time = time.perf_counter()  # 記錄上一幀的時間點
        fps = 0.0  # 初始化 FPS 顯示值
        frame_count = 0  # 幀計數器，用於定期輸出 FPS

        while True:  # 持續讀取攝影機畫面並推論
            ret, frame = cap.read()  # 讀取一幀影像
            if not ret:  # 若讀取失敗（例如斷線/被占用）
                print("[warn] 無法讀取畫面")  # 顯示警告
                return 3  # 以非 0 表示失敗

            frame = ensure_frame_size(frame, FRAME_WIDTH, FRAME_HEIGHT)  # 確保輸入影像大小為 640x480

            results = model(frame, classes=[PERSON_CLASS_ID], verbose=False)  # 只偵測 person 類別並回傳結果
            annotated = results[0].plot()  # 將框與標籤畫在影像上

            now = time.perf_counter()  # 取得目前時間點
            dt = now - last_time  # 計算與上一幀的時間差
            fps = (1.0 / dt) if dt > 0 else fps  # 以時間差估算即時 FPS（避免除以 0）
            last_time = now  # 更新上一幀時間

            overlay_status(annotated, MODEL_PATH, fps)  # 疊加模型與 FPS 資訊

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"[info] FPS: {fps:.1f}")

            # 預覽已停用；若需避免忙等可微小 sleep
            time.sleep(0.01)
    except KeyboardInterrupt:  # 支援 Ctrl+C 中斷
        print("\n[info] 使用者中斷，正在釋放資源...")  # 顯示中斷提示
        return 0  # 將 Ctrl+C 視為正常結束（避免 Exit Code 1）
    finally:  # 一定會執行的清理區塊
        cap.release()  # 釋放攝影機資源
        cv2.destroyAllWindows()  # 關閉所有 OpenCV 視窗


if __name__ == "__main__":  # 直接執行此檔案時才會進入
    sys.exit(main())  # 以明確的 exit code 結束程式
