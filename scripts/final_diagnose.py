import ncnn
import numpy as np
import cv2

COCO_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush']

img = cv2.imread("../models/bus.jpg")
h, w = img.shape[:2]
mat_in = ncnn.Mat.from_pixels_resize(img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, w, h, 320, 320)
mat_in.substract_mean_normalize([0, 0, 0], [1.0/255.0, 1.0/255.0, 1.0/255.0])

net = ncnn.Net()
net.load_param("../models/yolov8n.ncnn.param")
net.load_model("../models/yolov8n.ncnn.bin")
ex = net.create_extractor()
ex.input("in0", mat_in)

ret, mat_out = ex.extract("out0")
output = np.array(mat_out).T # (2100, 84)

print("=== 前10个锚点的原始输出分析 ===")
for i in range(10):
    coords = output[i, :4]
    logits = output[i, 4:]
    scores = 1.0 / (1.0 + np.exp(-logits))
    max_idx = np.argmax(scores)
    
    # 假设是 cx, cy, w, h，尝试转换为 x1, y1, x2, y2
    x1 = coords[0] - coords[2] / 2
    y1 = coords[1] - coords[3] / 2
    x2 = coords[0] + coords[2] / 2
    y2 = coords[1] + coords[3] / 2
    
    print(f"锚点{i}: 原始坐标={coords}")
    print(f"       转换为xyxy=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    print(f"       最高置信度={scores[max_idx]:.4f} ({COCO_NAMES[max_idx]}), 对应logits={logits[max_idx]:.4f}")
