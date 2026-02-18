from config_01 import create_default_config
from pipeline_04 import run_tracking

标识_05_入口 = "05_入口"


def main():
    config = create_default_config()
    run_tracking(config)


if __name__ == "__main__":
    main()
