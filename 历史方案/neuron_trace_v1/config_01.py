import os

标识_01_配置 = "01_配置"


class Config:
    def __init__(self, video_path, output_dir):
        self.video_path = video_path
        self.output_dir = output_dir
        self.output_video_path = os.path.join(output_dir, "01_轨迹结果.mp4")
        self.output_csv_path = os.path.join(output_dir, "02_轨迹数据.csv")
        self.window_name = "01_选择神经元"
        self.track_window_name = "02_跟踪显示"
        self.preview_scale = 0.5
        self.show_delay_ms = 1


def create_default_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = r"F:\工作文件\RA\python\项目汇总\神经图像\preview_video.mp4"
    output_dir = os.path.join(base_dir, "output")
    return Config(video_path, output_dir)
