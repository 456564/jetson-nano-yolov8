import cv2
import os
import numpy as np

out_dir = "../models/calib_images"
os.makedirs(out_dir, exist_ok=True)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

target_size = 416
count = 0
print("开始抓取校准图，请稍微动一下摄像头...")

while count < 200:
    ret, frame = cap.read()
    if not ret: continue
    h, w = frame.shape[:2]
    scale = min(target_size / w, target_size / h)
    nw, nh = int(w * scale), int(h * scale)
    pad_w = (target_size - nw) // 2
    pad_h = (target_size - nh) // 2
    resized = cv2.resize(frame, (nw, nh))
    in_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    in_img[pad_h:pad_h+nh, pad_w:pad_w+nw] = resized
    cv2.imwrite(f"{out_dir}/{count:04d}.jpg", in_img)
    count += 1
    print(f"\r已抓取: {count}/200", end="")

cap.release()
print("\n校准图准备完成！")

