import json
import os
import shutil
from pathlib import Path

# ================= 配置 =================
# 你的标注目录（JSON 和 JPG 在一起）
LABELME_DIR = r"F:\工作文件\RA\python\项目汇总\神经图像\dataset_yolo\images"

# 输出目录（YOLO 格式）
OUTPUT_DIR = r"/历史方案/yolo_dataset"

# ⚠️ 类别名称 - 请改成你实际用的标签名！
CLASSES = ["tip"]  # 如果你用的中文，改成 ["尖端"] 或你用的名字


# ========================================

def convert_labelme_to_yolo():
    images_out = os.path.join(OUTPUT_DIR, "images")
    labels_out = os.path.join(OUTPUT_DIR, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    json_files = list(Path(LABELME_DIR).glob("*.json"))
    print(f"🔍 找到 {len(json_files)} 个标注文件")

    if len(json_files) == 0:
        print("❌ 没找到任何 .json 文件，请检查路径")
        return

    # 先看一个文件，检查标签名
    with open(json_files[0], 'r', encoding='utf-8') as f:
        sample = json.load(f)
    labels_found = set(s['label'] for s in sample['shapes'])
    print(f"📋 检测到的标签名: {labels_found}")

    converted = 0

    for json_path in json_files:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_w = data['imageWidth']
        img_h = data['imageHeight']

        # 图片名
        img_name = json_path.stem + '.jpg'
        img_src = os.path.join(LABELME_DIR, img_name)

        if not os.path.exists(img_src):
            # 尝试 png
            img_name = json_path.stem + '.png'
            img_src = os.path.join(LABELME_DIR, img_name)

        if not os.path.exists(img_src):
            print(f"⚠️ 图片不存在: {json_path.stem}.*")
            continue

        # 转换标注
        yolo_lines = []
        for shape in data['shapes']:
            label = shape['label']

            # 自动添加新类别
            if label not in CLASSES:
                CLASSES.append(label)
                print(f"➕ 自动添加类别: {label} (ID={len(CLASSES) - 1})")

            class_id = CLASSES.index(label)
            points = shape['points']

            if shape['shape_type'] == 'rectangle':
                x1, y1 = points[0]
                x2, y2 = points[1]
            else:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width = abs(x2 - x1) / img_w
            height = abs(y2 - y1) / img_h

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        if yolo_lines:
            shutil.copy(img_src, os.path.join(images_out, img_name))
            txt_name = json_path.stem + '.txt'
            with open(os.path.join(labels_out, txt_name), 'w') as f:
                f.write('\n'.join(yolo_lines))
            converted += 1

    # 保存类别文件
    with open(os.path.join(OUTPUT_DIR, "classes.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(CLASSES))

    print(f"\n✅ 转换完成！共 {converted} 张")
    print(f"📂 输出: {OUTPUT_DIR}")
    print(f"📋 类别: {CLASSES}")


if __name__ == "__main__":
    convert_labelme_to_yolo()
