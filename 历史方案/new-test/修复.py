import os

# 检查这两个可能的位置
paths_to_check = [
    r"F:\工作文件\RA\python\项目汇总\神经图像\yolo_dataset",
    r"F:\工作文件\RA\python\项目汇总\神经图像\dataset_yolo",
]

for path in paths_to_check:
    print(f"\n📂 检查: {path}")
    if os.path.exists(path):
        for item in os.listdir(path):
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                files = os.listdir(full_path)
                print(f"   └── {item}/ ({len(files)} 个文件)")
            else:
                print(f"   └── {item}")
    else:
        print("   ❌ 目录不存在")
