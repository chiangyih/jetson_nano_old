import os
import sys
import time

import cv2 # OpenCV 函式庫


def has_display() -> bool:
    return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))


display_available = has_display()
if display_available:
    cv2.namedWindow('YOLO', cv2.WINDOW_NORMAL)
else:
    print('[warn] 未偵測到顯示環境，將不開啟視窗。若需畫面預覽，請在具備 X/Wayland 的本機執行或設定遠端轉發。')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('[error] 無法開啟 USB 攝影機，請檢查裝置或裝置編號。')
    sys.exit(1)

frame_idx = 0
try:
    while True:
        ret, frame = cap.read()  #讀取畫面, ret 代表是否成功; frame 代表畫面
        if not ret:
            print('[warn] 無法讀取畫面，可能是裝置中斷或頻寬不足。')
            break

        if display_available:
            cv2.imshow('YOLO', frame)
            key = cv2.waitKey(1) & 0xFF  #等待鍵盤輸入, 1 毫秒, 並取得按鍵代碼 ;0xFF 為了相容性,確保只取低位元組
            if key in (27, ord('q')):
                break
        else:
            # Headless fallback:寫出採樣幀供檢查，避免 Qt xcb 失敗
            if frame_idx % 30 == 0:
                cv2.imwrite('headless_frame.jpg', frame)
                print('[info] 已寫入 headless_frame.jpg 供檢查 (每 30 幀)。按 Ctrl+C 結束。')
            frame_idx += 1
            time.sleep(0.03)
except KeyboardInterrupt:
    print('\n[info] 使用者中斷，正在釋放資源...')
finally:
    cap.release()
    if display_available:
        cv2.destroyAllWindows()