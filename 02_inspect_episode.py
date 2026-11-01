from pathlib import Path  # 导入路径工具，用来确定文件保存位置。

import pandas as pd  # 导入表格处理库 pandas，并使用简写 pd。
from pusht_project.data_io import download_dataset_file  # 导入统一的下载函数，获取数据仓库中的指定文件。
path = download_dataset_file("data/chunk-000/file-000.parquet")  # 调用统一下载函数，获取数值表文件。
df = pd.read_parquet(path)  # 读取 Parquet 文件，得到名为 df 的表格对象。

episode = df[df["episode_index"] == 0].sort_values("frame_index")  # 筛选第 0 段示范，并按时间步编号排序。

columns = ["frame_index", "timestamp", "observation.state", "action"]  # 指定要查看的时间步编号、时间戳、状态和动作四列。
print(episode[columns].head(5).to_string(index=False))  # 显示选定列的前 5 行，隐藏 pandas 自带的行索引。
print("第 0 段示范的时间步数量：", len(episode))  # len 返回这段示范的行数，即记录的时间步数量。