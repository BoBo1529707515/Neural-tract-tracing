"""
运行神经元标记与追踪工具
"""

from neuron_tool import NeuronTool

if __name__ == "__main__":
    # 视频路径
    VIDEO_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\neuron_growth_50.mp4"

    # 数据保存路径（可选，默认在视频同目录）
    DATA_PATH = r"F:\工作文件\RA\python\项目汇总\神经图像\neuron_data.json"

    # 创建工具并运行
    tool = NeuronTool()
    tool.run(VIDEO_PATH, DATA_PATH)
