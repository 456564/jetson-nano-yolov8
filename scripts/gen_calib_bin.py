import os
import torch
from PIL import Image
from torchvision import transforms

calib_dir = "/home/user/nano_yolo_cpp/models/calib_images"
output_bin = "/home/user/nano_yolo_cpp/models/calib.bin"
img_size = 640

def preprocess(img_path):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    # YOLOv8 标准 Letterbox 逻辑
    scale = min(img_size / w, img_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # 用 114 灰度值填充黑边
    padded_img = Image.new('RGB', (img_size, img_size), (114, 114, 114))
    paste_x = (img_size - new_w) // 2
    paste_y = (img_size - new_h) // 2
    padded_img.paste(img, (paste_x, paste_y))
    
    # 转为 Tensor, 归一化到 0~1, 增加 Batch 维度
    img_tensor = transforms.ToTensor()(padded_img).unsqueeze(0) 
    return img_tensor

images = []
print("正在处理校准图片...")
for fname in sorted(os.listdir(calib_dir)):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            images.append(preprocess(os.path.join(calib_dir, fname)))
        except Exception as e:
            print(f"跳过错误图片 {fname}: {e}")

if images:
    # 拼接成 [N, 3, 640, 640] 的 Float32 张量并保存为连续二进制文件
    calib_data = torch.cat(images, dim=0).contiguous()
    calib_data.numpy().tofile(output_bin)
    print(f"成功生成: {output_bin}")
    print(f"包含 {calib_data.shape[0]} 张图片, 文件大小: {os.path.getsize(output_bin) / 1024 / 1024:.2f} MB")
else:
    print("未找到任何图片！")
