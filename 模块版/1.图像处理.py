import cv2
import numpy as np
import tifffile
import os
import time
from skimage.filters import frangi
from skimage import exposure

# === 参数设置 ===
FILE_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\recover_0-72h_XY2.tif"

SIGMAS = [1, 2, 3]
REMOVE_STRIPES = True
STRIPE_WIDTH = 3


def remove_vertical_stripes_fft(img, stripe_width=3):
    """FFT 去除竖向条纹"""
    img_f = img.astype(np.float32)
    f_shift = np.fft.fftshift(np.fft.fft2(img_f))

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.ones((rows, cols), dtype=np.float32)
    mask[crow - stripe_width:crow + stripe_width, :] = 0
    mask[crow - stripe_width:crow + stripe_width, ccol - 50:ccol + 50] = 1

    f_filtered = f_shift * mask
    img_back = np.fft.ifft2(np.fft.ifftshift(f_filtered))
    result = np.abs(img_back)

    return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def process_frame(img_2d):
    img_8bit = cv2.normalize(img_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if REMOVE_STRIPES:
        img_clean = remove_vertical_stripes_fft(img_8bit, STRIPE_WIDTH)
    else:
        img_clean = img_8bit

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(img_clean)

    vesselness = frangi(gray_clahe, sigmas=SIGMAS, black_ridges=False)
    enhanced = exposure.rescale_intensity(vesselness, out_range=(0, 255)).astype(np.uint8)

    return img_8bit, img_clean, enhanced


def main():
    print("=" * 50)
    print("神经元生长追踪 · 全帧调试器 (CPU)")
    print("=" * 50)

    if not os.path.exists(FILE_PATH):
        print(f"❌ 文件不存在: {FILE_PATH}")
        return

    with open(FILE_PATH, 'rb') as f:
        stack = tifffile.imread(f)
    print(f"✅ 加载成功: {stack.shape}, {stack.dtype}")

    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]

    num_frames = stack.shape[0]
    print(f"🎬 共 {num_frames} 帧")

    win_name = "Neuro-Debugger"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1600, 500)

    cache = {'idx': -1, 'result': None}

    def update_view(frame_idx):
        if cache['idx'] == frame_idx:
            cv2.imshow(win_name, cache['result'])
            return

        print(f"处理帧 {frame_idx}...", end=" ", flush=True)
        t0 = time.perf_counter()

        original, destriped, enhanced = process_frame(stack[frame_idx])
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"完成 ({elapsed:.0f}ms)")

        scale = 0.4
        imgs = [cv2.resize(x, None, fx=scale, fy=scale) for x in [original, destriped, enhanced]]
        labels = [f"Original #{frame_idx}", "De-striped", f"Frangi ({elapsed:.0f}ms)"]

        for img, label in zip(imgs, labels):
            cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

        combined = np.hstack(imgs)
        cache['idx'], cache['result'] = frame_idx, combined
        cv2.imshow(win_name, combined)

    cv2.createTrackbar("Frame", win_name, 0, num_frames - 1, update_view)
    update_view(0)

    print("\n👉 [滑块]切换帧 | [A/D]逐帧 | [ESC]退出\n")

    current = 0
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 27:
            break
        elif key == ord('d'):
            current = min(current + 1, num_frames - 1)
        elif key == ord('a'):
            current = max(current - 1, 0)
        else:
            continue
        cv2.setTrackbarPos("Frame", win_name, current)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
