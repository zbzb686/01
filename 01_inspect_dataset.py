import json  # 导入标准库 json，用于把 JSON 文本解析成 Python 字典。
from pathlib import Path  # 导入 Path，以便拼接路径和读取本地文件。

from pusht_project.data_io import download_dataset_file  # 导入统一的下载函数，获取数据仓库中的指定文件。
# 固定数据版本，让我们后续读取同一份数据。
def main():  # 定义主函数，作为运行入口。
    path = download_dataset_file("meta/info.json")  # 调用统一下载函数，获取元数据文件。
    info = json.loads(Path(path).read_text(encoding="utf-8"))  # 按 UTF-8 读取文件，再将 JSON 文本解析为字典。

    for key in ["codebase_version", "total_episodes", "total_frames", "fps"]:  # 依次选择格式版本、示范段数、总时间步数和每秒采样帧数。
        print(key, ":", info[key])  # 根据字段名称取出对应值，并打印字段名称和值。

    for key in ["observation.image", "observation.state", "action"]:  # 依次查看图像、状态和动作三个字段。
        feature = info["features"][key]  # 从 features 字典中取出当前字段的结构说明。
        print(key, ":", feature["dtype"], feature["shape"])  # 打印该字段的存储类型与数据形状。
if __name__ == "__main__":  # 如果当前脚本是直接运行而非被导入。
    main()  # 调用主函数，执行数据集检查.


