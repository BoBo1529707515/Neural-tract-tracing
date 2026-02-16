import torch
import cv2

print(f"OpenCV Version: {cv2.__version__}")
print(f"PyTorch Version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"✅ GPU 状态: 正常")
    print(f"🚀 显卡型号: {torch.cuda.get_device_name(0)}")
    print(f"💾 显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # 测试一下张量计算
    x = torch.rand(5, 3).cuda()
    print("   GPU 张量计算测试通过！")
else:
    print("❌ 警告: PyTorch 未检测到 GPU！我们将只能使用 CPU，速度会慢 50 倍以上。")
    print("   请检查 CUDA 驱动是否安装，或 pip install 命令是否正确。")
