import os
import numpy as np
import onnxruntime as ort

def main():
    onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "models", "yolov8n.onnx")
    print("加载模型：", onnx_path)
    sess = ort.InferenceSession(onnx_path)

    # —— 输入信息 ——
    inp = sess.get_inputs()[0]
    print("\n[输入] 名称={}, 形状={}, 类型={}".format(inp.name, inp.shape, inp.type))

    # —— 输出信息（可能 1 个或多个） ——
    outs = sess.get_outputs()
    print("\n[输出] 张量个数：{}".format(len(outs)))
    for i, o in enumerate(outs):
        print("  输出{}：name={}, shape={}".format(i, o.name, o.shape))

    # —— 用假数据跑一次推理 ——
    dummy = np.random.rand(1, 3, 320, 320).astype(np.float32)
    print("\n正在执行一次推理（输入 shape={}）...".format(dummy.shape))
    res = sess.run(None, {inp.name: dummy})

    print("\n=== 推理结果维度 ===")
    for i, r in enumerate(res):
        print("  results[{}] shape = {}".format(i, r.shape))

    # —— 简单解读 ——
    r0 = res[0]
    if len(r0.shape) == 3:
        B, C, N = r0.shape
        print("\n解读第一个输出：")
        print("  batch={}, channels={}, num_predictions={}".format(B, C, N))
        print("  84 = 4(xywh) + 80(COCO 类)  -> 若 C==84，则这是“标准检测输出（含分类）”")
        print("  64 = 4(xywh) + 64(DFL 系数)  -> 若 C==64，则这是“回归分支（YOLOv8 DFL 模式）”")
        print("  通常 320×320 的预测点数 N = 40×40 + 20×20 + 10×10 = 2100")

if __name__ == "__main__":
    main()
