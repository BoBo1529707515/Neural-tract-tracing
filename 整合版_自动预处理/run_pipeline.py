import argparse
import importlib.util
import os

import cv2
import numpy as np
import tifffile


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_video_stack(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 10
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            continue
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        frames.append(gray)
    cap.release()
    if len(frames) == 0:
        raise RuntimeError("未读取到任何帧")
    stack = np.stack(frames, axis=0)
    return stack, fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="输入视频或tif文件路径")
    parser.add_argument("--output", dest="output", default=None, help="输出根目录")
    parser.add_argument("--fps", dest="fps", type=float, default=10, help="tif输入的输出视频帧率")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_root = os.path.abspath(args.output) if args.output else os.path.join(base_dir, "输出")
    os.makedirs(output_root, exist_ok=True)

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        raise RuntimeError(f"输入不存在: {input_path}")

    ext = os.path.splitext(input_path)[1].lower()
    if ext in [".tif", ".tiff"]:
        input_tif = input_path
        fps = args.fps
    else:
        stack, fps = read_video_stack(input_path)
        input_tif = os.path.join(output_root, "input_stack.tif")
        tifffile.imwrite(input_tif, stack)

    modules_dir = os.path.abspath(os.path.join(base_dir, "..", "模块版"))
    step1_path = os.path.join(modules_dir, "步骤1：CLAHE二值化基础处理.py")
    step2_path = os.path.join(modules_dir, "步骤2：空间去噪（面积过滤）.py")
    step3_path = os.path.join(modules_dir, "步骤3：对齐裁剪.py")
    step4_path = os.path.join(modules_dir, "步骤4：时空去噪.py")
    step5_path = os.path.join(modules_dir, "步骤5：二次空间去噪.py")

    step1_output = os.path.join(output_root, "01_二值化基础")
    step2_output = os.path.join(output_root, "02_空间去噪")
    step3_output = os.path.join(output_root, "03_对齐裁剪")
    step4_output = os.path.join(output_root, "04_时空去噪")
    step5_output = os.path.join(output_root, "05_二次空间去噪")

    step1 = load_module(step1_path, "step1_module")
    step1.INPUT_FILE = input_tif
    step1.OUTPUT_DIR = step1_output
    step1.FPS = fps

    step2 = load_module(step2_path, "step2_module")
    step2.INPUT_DIR = os.path.join(step1_output, "frames_binary")
    step2.OUTPUT_DIR = step2_output
    step2.FPS = fps

    step3 = load_module(step3_path, "step3_module")
    step3.INPUT_DIR = os.path.join(step2_output, "frames_denoised")
    step3.OUTPUT_DIR = step3_output
    step3.FPS = fps

    step4 = load_module(step4_path, "step4_module")
    step4.INPUT_DIR = os.path.join(step3_output, "frames_aligned")
    step4.OUTPUT_DIR = step4_output
    step4.FPS = fps

    step5 = load_module(step5_path, "step5_module")
    step5.INPUT_DIR = os.path.join(step4_output, "frames_temporal_denoised")
    step5.OUTPUT_DIR = step5_output
    step5.FPS = fps

    print("✅ 开始步骤1")
    step1.main()
    print("✅ 开始步骤2")
    step2.main()
    print("✅ 开始步骤3")
    step3.main()
    print("✅ 开始步骤4")
    step4.main()
    print("✅ 开始步骤5")
    step5.main()

    print("✅ 全流程完成")
    print(f"最终输出: {os.path.join(step5_output, 'frames_final')}")


if __name__ == "__main__":
    main()
