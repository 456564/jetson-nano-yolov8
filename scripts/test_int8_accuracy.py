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

def run_results(param_path, bin_path, img_path, input_size=320, label=""):
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]

    mat_in = ncnn.Mat.from_pixels_resize(img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, img_w, img_h, input_size, input_size)
    mat_in.substract_mean_normalize([0, 0, 0], [1.0/255.0, 1.0/255.0, 1.0/255.0])

    net = ncnn.Net()
    net.load_param(param_path)
    net.load_model(bin_path)
    ex = net.create_extractor()
    ex.input("in0", mat_in)
    ret, mat_out = ex.extract("out0")

    output = np.array(mat_out)
    # ncnn (w=84, h=2100) → numpy (h, w) = (2100, 84) ← 直接就是 (2100, 84), 不需要 .T!
    print(f"\n=== {label} ===")
    print(f"numpy shape: {output.shape}")

    if output.shape == (84, 2100):
        print("  注意: 输出是 (84, 2100), 需要 .T")
        output = output.T  # → (2100, 84)
    print(f"实际 shape: {output.shape}")

    boxes = output[:, 0:4]   # (2100, 4) box值
    cls_logits = output[:, 4:84] # (2100, 80) class logits

    cls_sig = 1.0 / (1.0 + np.exp(-cls_logits))
    max_cls_val = np.max(cls_sig, axis=1)
    max_cls_id = np.argmax(cls_sig, axis=1)

    # 排序取 Top 20
    sorted_idx = np.argsort(max_cls_val)[::-1]

    results = []
    for i in sorted_idx[:20]:
        results.append({
            "class": COCO_NAMES[max_cls_id[i]],
            "conf": max_cls_val[i],
            "box": boxes[i],
        })
    return results

print("=" * 60)
print("FP32 model:")
fp32 = run_results("../models/yolov8n.ncnn.param", "../models/yolov8n.ncnn.bin", "../models/bus.jpg", 320, "FP32")
for i, r in enumerate(fp32):
    print(f"  #{i+1: {r['class']:15s} conf={r['conf']:.3f}  box=[{r['box'][0]:.1f}, {r['box'][1]:.1f}, {r['box'][2]:.1f}, {r['box'][3]:.1f}")

print("\n" + "=" * 60)
print("INT8 model:")
print("=" * 60)
int8 = run_results("../models/yolov8n_int8.param", "../models/yolovn_int8.bin", "../models/bus.jpg", 320, "INT8")
for i, r in enumerate(int8):
    print(f"  #{i+1: {r['class']:15s} conf={r['conf']:.3f}  box=[{r['box'][0]:.1f}, {r['box'][1]:.1f}, {r['box'][2]:.1f}, {r['box'][3]:.1f}")

print("\n" + "=" * 60)
if fp32 == int8:
    print("✅ 两个模型输出完全一致！INT8 量化无额外精度损失")
else:
    print("❌ 两个模型输出有差异，需要检查")
    print("FP32:", fp32)
    print("INT8:", int8)
