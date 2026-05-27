import numpy as np
import cv2

# ============ 第1步：用 ncnn 推理 ============
import ncnn

img = cv2.imread("../models/bus.jpg")
mat_in = ncnn.Mat.from_pixels_resize(img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 
                                       img.shape[1], img.shape[0], 320, 320)
mat_in.substract_mean_normalize([0,0,0], [1/255, 1/255, 1/255])

net = ncnn.Net()
net.load_param("../models/yolov8n.ncnn.param")
net.load_model("../models/yolov8n.ncnn.bin")
ex = net.create_extractor()
ex.input("in0", mat_in)
ret, mat_out = ex.extract("out0")
ncnn_out = np.array(mat_out)
if ncnn_out.shape == (84, 2100):
    ncnn_out = ncnn_out.T
print(f"ncnn output shape: {ncnn_out.shape}")
print(f"ncnn box  range: [{ncnn_out[:,:4].min():.2f}, {ncnn_out[:,:4].max():.2f}]")
print(f"ncnn cls  range: [{ncnn_out[:,4:].min():.4f}, {ncnn_out[:,4:].max():.4f}]")
cls_sig = 1.0 / (1.0 + np.exp(-ncnn_out[:, 4:]))
print(f"ncnn sigmoid range: [{cls_sig.min():.4f}, {cls_sig.max():.4f}]")
print(f"ncnn top5 class logits: {ncnn_out[:, 4:].max(axis=0).argsort()[::-1][:5]}")

# ============ 第2步：用 onnxruntime 推理（如果有的话） ============
try:
    import onnxruntime as ort
    # 尝试找 onnx 文件
    import glob
    onnx_files = glob.glob("*.onnx")
    if onnx_files:
        onnx_path = onnx_files[0]
        print(f"\n找到 ONNX 模型: {onnx_path}")
        sess = ort.InferenceSession(onnx_path)
        inp = sess.get_inputs()[0]
        print(f"  ONNX 输入名: {inp.name}, shape: {inp.shape}, dtype: {inp.type}")
        
        # 预处理
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (320, 320))
        img_float = img_resized.astype(np.float32) / 255.0
        if len(inp.shape) == 4:
            img_float = img_float.transpose(2, 0, 1)[np.newaxis]
        else:
            img_float = img_float.transpose(2, 0, 1)
        
        ort_out = sess.run(None, {inp.name: img_float})[0]
        print(f"  ONNX output shape: {ort_out.shape}")
        
        # 转成同样格式比较
        if ort_out.shape[0] == 1:
            ort_out = ort_out[0]
        if ort_out.shape[0] == 84 and ort_out.shape[1] == 2100:
            ort_out = ort_out.T
        print(f"  ONNX box  range: [{ort_out[:,:4].min():.2f}, {ort_out[:,:4].max():.2f}]")
        print(f"  ONNX cls  range: [{ort_out[:,4:].min():.4f}, {ort_out[:,4:].max():.4f}]")
        ort_cls_sig = 1.0 / (1.0 + np.exp(-ort_out[:, 4:]))
        print(f"  ONNX sigmoid range: [{ort_cls_sig.min():.4f}, {ort_cls_sig.max():.4f}]")
        
        # 比较
        if ncnn_out.shape == ort_out.shape:
            diff = np.abs(ncnn_out - ort_out)
            print(f"\n  ncnn vs ONNX 差异:")
            print(f"    box 最大差异: {diff[:,:4].max():.4f}")
            print(f"    cls 最大差异: {diff[:,4:].max():.4f}")
            print(f"    总体最大差异: {diff.max():.4f}")
            if diff.max() < 0.01:
                print("  ✅ ncnn 和 ONNX 输出几乎一致，导出正确！")
            elif diff.max() < 0.1:
                print("  ⚠️ ncnn 和 ONNX 有小差异，可能是精度损失")
            else:
                print("  ❌ ncnn 和 ONNX 差异很大，导出有问题！")
    else:
        print("\n未找到 .onnx 文件，跳过对比")
        print("请把原始 onnx 模型放到当前目录")
except ImportError:
    print("\nonnxruntime 未安装，跳过 ONNX 对比")
    print("安装: pip install onnxruntime")
