import numpy as np
import os

print("=" * 60)
print("第一步：查找所有模型相关文件")
print("=" * 60)
model_files = []
for ext in ['*.pt', '*.pth', '*.onnx', '*.param', '*.bin']:
    import glob
    found = glob.glob(f'../models/{ext}')
    for f in found:
        size = os.path.getsize(f) / 1024 / 1024
        print(f"  {f:40s} {size:.1f} MB")
        model_files.append(f)

print("\n" + "=" * 60)
print("第二步：尝试用ultralytics直接推理原始.pt模型")
print("=" * 60)
try:
    from ultralytics import YOLO
    
    # 找.pt文件
    pt_files = [f for f in model_files if f.endswith('.pt')]
    
    if pt_files:
        print(f"找到 .pt 文件: {pt_files}")
        model = YOLO(pt_files[0])
        
        # 用ultralytics直接推理bus.jpg
        results = model("../models/bus.jpg", verbose=False)
        
        print(f"\nultralytics推理结果（{pt_files[0]}）:")
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    print(f"  {cls_name:15s} conf={conf:.3f}  box=[{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]")
            else:
                print("  无检测结果！原始.pt模型也有问题！")
    else:
        print("未找到 .pt 文件")
        print("需要原始的 yolov8n.pt 来验证")
        
except ImportError:
    print("ultralytics 未安装，请安装: pip install ultralytics")
except Exception as e:
    print(f"出错: {e}")

print("\n" + "=" * 60)
print("第三步：尝试修复pnnx ONNX并推理")
print("=" * 60)
onnx_files = [f for f in model_files if f.endswith('.onnx') and 'pnnx' in f]
if onnx_files:
    print(f"找到 pnnx ONNX: {onnx_files[0]}")
    print("尝试添加opset后加载...")
    
    try:
        import onnx
        from onnx import helper, numpy_helper
        
        # 加载broken onnx
        model_proto = onnx.load(onnx_files[0])
        
        # 检查opset
        if len(model_proto.opset_import) == 0:
            print("  确认: 缺少opset，正在添加opset 17...")
            model_proto.opset_import.append(helper.make_operatorsetid(17))
            fixed_path = "../models/yolov8n_fixed.onnx"
            onnx.save(model_proto, fixed_path)
            print(f"  已保存修复后的模型: {fixed_path}")
            
            # 尝试加载
            import onnxruntime as ort
            sess = ort.InferenceSession(fixed_path)
            print("  ✅ onnxruntime 加载成功！")
            
            # 推理
            import cv2
            img = cv2.imread("../models/bus.jpg")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (320, 320))
            img_float = img_resized.astype(np.float32) / 255.0
            img_nhwc = img_float.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 320, 320)
            
            inp_name = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name
            print(f"  输入: {inp_name}, shape={sess.get_inputs()[0].shape}")
            print(f"  输出: {out_name}, shape={sess.get_outputs()[0].shape}")
            
            ort_out = sess.run(None, {inp_name: img_nhwc})[0]
            print(f"  ONNX输出shape: {ort_out.shape}")
            
            # 整理输出
            if ort_out.shape[0] == 1:
                ort_out = ort_out[0]
            if ort_out.shape[0] == 84 and ort_out.shape[1] == 2100:
                ort_out = ort_out.T  # -> (2100, 84)
            
            print(f"  整理后shape: {ort_out.shape}")
            print(f"  ONNX box  范围: [{ort_out[:,:4].min():.2f}, {ort_out[:,:4].max():.2f}]")
            print(f"  ONNX cls  范围: [{ort_out[:,4:].min():.4f}, {ort_out[:,4:].max():.4f}]")
            
            ort_cls_sig = 1.0 / (1.0 + np.exp(-ort_out[:, 4:]))
            print(f"  ONNX sigmoid范围: [{ort_cls_sig.min():.4f}, {ort_cls_sig.max():.4f}]")
            
            # Top 10
            max_val = ort_cls_sig.max(axis=1)
            max_id = ort_cls_sig.argmax(axis=1)
            COCO_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                    'scissors', 'teddy bear', 'hair drier', 'toothbrush']
            
            sorted_idx = np.argsort(max_val)[::-1]
            print("\n  ONNX Top 10 检测结果:")
            for i in sorted_idx[:10]:
                print(f"    #{i}: {COCO_NAMES[max_id[i]]:15s} conf={max_val[i]:.4f}  logit={ort_out[i,4+max_id[i]]:.4f}  box={ort_out[i,:4]}")
            
            # 与ncnn比较
            print("\n" + "=" * 60)
            print("第四步：对比ONNX vs ncnn")
            print("=" * 60)
            import ncnn
            ncnn_out = np.load("../models/ncnn_output.npy") if os.path.exists("../models/ncnn_output.npy") else None
            
            # 重新跑ncnn
            img2 = cv2.imread("../models/bus.jpg")
            mat_in = ncnn.Mat.from_pixels_resize(img2, ncnn.Mat.PixelType.PIXEL_BGR2RGB, img2.shape[1], img2.shape[0], 320, 320)
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
            
            if ncnn_out.shape == ort_out.shape:
                diff = np.abs(ncnn_out - ort_out)
                print(f"  shape一致: {ncnn_out.shape}")
                print(f"  box  最大差异: {diff[:,:4].max():.6f}")
                print(f"  cls  最大差异: {diff[:,4:].max():.6f}")
                print(f"  总体 最大差异: {diff.max():.6f}")
                
                if diff.max() < 0.01:
                    print("\n  ✅ ncnn 和 ONNX 输出几乎一致")
                    print("  → 问题出在原始模型（.pt/.onnx），不是ncnn导出！")
                elif diff.max() < 0.5:
                    print("\n  ⚠️ ncnn 和 ONNX 有差异但不大")
                    print("  → 可能是ncnn导出的微小精度损失，但两者都有问题")
                else:
                    print("\n  ❌ ncnn 和 ONNX 差异很大！")
                    print("  → ncnn导出过程有严重问题！")
            else:
                print(f"  shape不一致: ncnn={ncnn_out.shape} vs onnx={ort_out.shape}")
                
        else:
            print(f"  已有opset: {model_proto.opset_import}")
            print("  直接加载...")
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_files[0])
            print("  ✅ 加载成功")
    except ImportError as e:
        print(f"  缺少依赖: {e}")
        print("  安装: pip install onnx onnxruntime")
    except Exception as e:
        print(f"  处理失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print("未找到 pnnx ONNX 文件")

print("\n" + "=" * 60)
print("第五步：检查是否是自定义/裁剪模型")
print("=" * 60)
for pf in ['../models/yolov8n.ncnn.param']:
    if os.path.exists(pf):
        with open(pf, 'r') as f:
            content = f.read()
            lines = content.strip().split('\n')
            print(f"\n{pf} 文件头部信息:")
            for line in lines[:20]:
                print(f"  {line}")
            
            # 统计
            total_lines = len(lines)
            op_count = sum(1 for l in lines if l and not l.startswith('#') and not l.startswith('7767517'))
            print(f"\n  总行数: {total_lines}")
            print(f"  算子数量: {op_count}")
            
            # 检查特殊算子
            ops = set()
            for l in lines:
                parts = l.split()
                if len(parts) >= 3:
                    ops.add(parts[0])
            print(f"  使用的算子种类: {ops}")
