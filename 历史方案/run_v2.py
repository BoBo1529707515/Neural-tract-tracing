"""
程序入口
"""
import os
from neuron_trace_modular_v2.main_ui import NeuronTool

if __name__ == "__main__":
    # 默认视频路径
    VIDEO_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\preview_video.mp4"
    
    # 如果找不到默认视频，尝试使用preview_video
    if not os.path.exists(VIDEO_PATH):
        VIDEO_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\preview_video.mp4"

    tool = NeuronTool()
    tool.run(VIDEO_PATH)
