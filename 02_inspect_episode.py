from pathlib import Path  # 导入路径工具，用来确定文件保存位置。

import pandas as pd  # 导入表格处理库 pandas，并使用简写 pd。
from huggingface_hub import hf_hub_download  # 导入下载函数，获取数据仓库中的指定文件。

path = hf_hub_download(  # 下载数值表，并将返回的本地文件路径保存到 path。
    repo_id="lerobot/pusht",  # 指定数据仓库：lerobot 组织下的 pusht 数据集。
    repo_type="dataset",  # 指定仓库类型为数据集。
    revision="7628202a2180972f291ba1bc6723834921e72c19",  # 固定数据版本，方便复现实验。
    filename="data/chunk-000/file-000.parquet",  # 指定要下载的数值表在仓库中的路径。
    local_dir=Path(__file__).parent / "data" / "pusht",  # 保存到当前脚本目录下的 data/pusht 文件夹。
)  # 结束下载函数调用；path 指向本地的 Parquet 文件。

df = pd.read_parquet(path)  # 读取 Parquet 文件，得到名为 df 的表格对象。

episode = df[df["episode_index"] == 0].sort_values("frame_index")  # 筛选第 0 段示范，并按时间步编号排序。

columns = ["frame_index", "timestamp", "observation.state", "action"]  # 指定要查看的时间步编号、时间戳、状态和动作四列。
print(episode[columns].head(5).to_string(index=False))  # 显示选定列的前 5 行，隐藏 pandas 自带的行索引。
print("第 0 段示范的时间步数量：", len(episode))  # len 返回这段示范的行数，即记录的时间步数量。