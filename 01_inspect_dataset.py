import json  # 导入标准库 json，用于把 JSON 文本解析成 Python 字典。
from pathlib import Path  # 导入 Path，以便拼接路径和读取本地文件。

from huggingface_hub import hf_hub_download  # 导入下载函数，用于获取 Hugging Face 仓库中的指定文件。

# 固定数据版本，让我们后续读取同一份数据。
revision = "7628202a2180972f291ba1bc6723834921e72c19"  # 保存数据仓库的提交编号，固定本实验使用的数据版本。

path = hf_hub_download(  # 调用下载函数，并将返回的本地文件路径保存到 path。
    repo_id="lerobot/pusht",  # 指定数据仓库：lerobot 组织下的 pusht 数据集。
    repo_type="dataset",  # 说明访问的是数据集仓库。
    revision=revision,  # 使用前面固定的提交编号，确保多次实验读取同一版本。
    filename="meta/info.json",  # 仅获取描述数据结构的元数据文件。
    local_dir=Path(__file__).parent / "data" / "pusht",  # 将文件保存到当前脚本所在文件夹下的 data/pusht 目录。
)  # 结束函数调用；path 指向下载完成后的 info.json 文件。

info = json.loads(Path(path).read_text(encoding="utf-8"))  # 按 UTF-8 读取文件，再将 JSON 文本解析为字典。

for key in ["codebase_version", "total_episodes", "total_frames", "fps"]:  # 依次选择格式版本、示范段数、总时间步数和每秒采样帧数。
    print(key, ":", info[key])  # 根据字段名称取出对应值，并打印字段名称和值。

for key in ["observation.image", "observation.state", "action"]:  # 依次查看图像、状态和动作三个字段。
    feature = info["features"][key]  # 从 features 字典中取出当前字段的结构说明。
    print(key, ":", feature["dtype"], feature["shape"])  # 打印该字段的存储类型与数据形状。



