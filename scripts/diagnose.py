import ncnn
import numpy as np
import cv2

# 读取图片并预处理
img = cv2.imread("../models/bus.jpg")
h, w = img.shape[:2]
print(f"原图尺寸: {w}x{h}")

# 缩放到 320x320
img_resized = cv2.resize(img, (320, 320))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_float = img_rgb.astype(np.float32) / 255.0
# HWC -> CHW
img_chw = img_float.transpose(2, 0, 1)
print(f"输入 shape: {img_chw.shape}, min={img_chw.min():.3f}, max={img_chw.max():.3f}")

# 创建 ncnn.Mat
mat_in = ncnn.Mat(img_chw)
print(f"ncnn.Mat shape: w={mat_in.w}, h={mat_in.h}, c={mat_in.c}, dims={mat_in.dims}")

# FP32 推理
net = ncnn.Net()
net.load_param("../models/yolov8n.ncnn.param")
net.load_model("../models/yolov8n.ncnn.bin")
ex = net.create_extractor()
ex.input("in0", mat_in)

# 试试提取所有可能的输出名
for name in ["out0", "out1", "out2", "775", "776", "777", "778", "probs", "output"]:
    ret, mat_out = ex.extract(name)
    if ret == 0:
        arr = np.array(mat_out)
        print(f"\n输出 '{name}': shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
        print(f"  前20个值: {arr.flatten()[:20]}")
    else:
        pass  # 忽略不存在的输出

# 看看 param 文件里最后的输出节点
print("\n=== param 文件最后10行 ===")
with open("../models/yolov8n.ncnn.param") as f:
    lines = f.readlines()
    for line in lines[-10:]:
        print(line.rstrip())
