import os
from ultralytics import YOLO

def main():
    pt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "yolov8n.pt")
    model = YOLO(pt_path)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    os.makedirs(out_dir, exist_ok=True)

    print("开始导出 ONNX (640×640, 静态维度)...")
    model.export(
        format="onnx",
        imgsz=640,         # 与你当前 engine 的输入一致
        dynamic=False,
        simplify=True,
        opset=12,          # TRT 8.2.1 更稳
    )

    default_out = "yolov8n.onnx"
    target = os.path.join(out_dir, "yolov8n.onnx")
    if os.path.exists(default_out):
        os.rename(default_out, target)
    print("导出成功，文件位于：", target)

if __name__ == "__main__":
    main()
