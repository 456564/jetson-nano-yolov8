# YOLOv8n TensorRT Inference on Jetson Nano

Real-time object detection on **Jetson Nano** (Tegra X1, 128-core Maxwell GPU) using TensorRT FP16 engine.

| Component | Details |
|-----------|---------|
| Model | YOLOv8n 640×640 |
| Engine | TensorRT FP16 (SM 5.3) |
| Preproc | Custom CUDA kernel (resize + pad + BGR→RGB + norm + HWC→CHW) |
| Postproc | OpenMP parallel decode + greedy NMS |
| FPS | **~14** (GPU 51ms, total 70ms) |

## Quick Start

```bash
# Clone
git clone https://github.com/456564/jetson-nano-yolov8.git
cd jetson-nano-yolov8

# Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Run
./detect
```

> Requires: TensorRT 8.x, CUDA 10.2+, OpenCV 4.x, OpenMP.

## Model Setup

Place your YOLOv8n TensorRT engine at `models/yolov8n_fp16.engine`.

To generate from scratch:

```bash
# PT → ONNX (run on PC or Jetson)
pip install ultralytics
python scripts/export_onnx.py

# ONNX → TensorRT engine (on Jetson)
trtexec --onnx=models/yolov8n.onnx --fp16 \
  --saveEngine=models/yolov8n_fp16.engine \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:1x3x640x640
```

## Project Structure

```
├── src/
│   ├── main.cpp                 # Main entry: async GPU pipeline
│   ├── cuda_preprocess.cu       # Fused CUDA preprocess kernel
│   ├── camera/                  # GStreamer/V4L2 camera capture
│   ├── detector/                # TensorRT engine wrapper + postprocess
│   └── utils/                   # Timer, COCO class names
├── scripts/
│   ├── export_onnx.py           # PT → ONNX export
│   └── verify_onnx.py           # ONNX verification
├── models/
│   └── bus.jpg                  # Sample test image
├── CMakeLists.txt
└── 项目总体.md                  # Full documentation (Chinese)
```

## Architecture

```
Camera → V4L2 (3ms)
  → memcpy H2D async
  → CUDA preprocess kernel (resize+pad+color+norm+CHW)
  → TensorRT FP16 inference (51ms)
  → D2H async to pinned memory
  → OpenMP decode + NMS (12ms)
  → Draw & display
```

All GPU ops async, single `cudaStreamSynchronize` at the end.

## Performance

| Stage | Time |
|-------|------|
| Camera capture | 3.3ms |
| GPU preprocess | ~1ms |
| TensorRT inference | 51.3ms |
| Postprocess (4-core) | 12-15ms |
| Display | 0.2ms |
| **Total FPS** | **~14** |

Inference is GPU-bound — 51ms is the hardware limit for YOLOv8n 640×640 FP16 on Maxwell 128-core.

## License

MIT
