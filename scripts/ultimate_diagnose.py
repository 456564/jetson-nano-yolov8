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
img_h, img_w = img.shape[:2]
print(f"原图: {img_w}x{img_h}")

mat_in = ncnn.Mat.from_pixels_resize(img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, img_w, img_h, 320, 320)
mat_in.substract_mean_normalize([0,0,0], [1/255, 1/255, 1/255])

net = ncnn.Net()
net.load_param("../models/yolov8n.ncnn.param")
net.load_model("../models/yolov8n.ncnn.bin")
ex = net.create_extractor()
ex.input("in0", mat_in)
ret, mat_out = ex.extract("out0")
output = np.array(mat_out).T  # (2100, 84)

print(f"\n{'='*60}")
print("=== 两种布局对比 ===")
print(f"{'='*60}")

for layout_name, box_slice, cls_slice in [
    ("方案A: [前4列=box, 后80列=class]", slice(0,4), slice(4,84)),
    ("方案B: [前80列=class, 后4列=box]", slice(80,84), slice(0,80)),
]:
    boxes = output[:, box_slice]
    cls_logits = output[:, cls_slice]
    cls_sig = 1.0 / (1.0 + np.exp(-cls_logits))
    max_scores = np.max(cls_sig, axis=1)
    max_ids = np.argmax(cls_sig, axis=1)

    # 找出置信度 > 0.5 的锚点
    valid = max_scores > 0.5
    n_valid = np.sum(valid)

    # Top 15 by confidence
    top_idx = np.argsort(max_scores)[::-1][:15]

    print(f"\n{layout_name}")
    print(f"  Box值范围: [{boxes.min():.1f}, {boxes.max():.1f}]")
    print(f"  Class sigmoid范围: [{cls_sig.min():.4f}, {cls_sig.max():.4f}]")
    print(f"  置信度>0.5的锚点数: {n_valid}")
    print(f"  Top 15 检测结果:")
    for rank, i in enumerate(top_idx):
        b = boxes[i]
        print(f"    #{rank+1} {COCO_NAMES[max_ids[i]]:15s} conf={max_scores[i]:.3f} raw_box=[{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}]")

# 额外: 检查class分数的分布
print(f"\n{'='*60}")
print("=== 方案B 详细: class分数分布 ===")
print(f"{'='*60}")
cls_b = output[:, :80]
cls_sig_b = 1.0 / (1.0 + np.exp(-cls_b))
for i in range(80):
    max_val = np.max(cls_sig_b[:, i])
    mean_val = np.mean(cls_sig_b[:, i])
    if max_val > 0.6:
        print(f"  {COCO_NAMES[i]:15s}: max={max_val:.4f}  mean={mean_val:.4f}")
