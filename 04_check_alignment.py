from pathlib import Path  # 导入路径工具，用于确定文件保存位置。

import pandas as pd  # 导入 pandas，用于读取和查看表格。
from huggingface_hub import hf_hub_download  # 导入下载函数，获取示范位置记录。

root = Path(__file__).parent  # 获取当前脚本所在的项目文件夹。

meta_path = hf_hub_download(  # 下载记录各段示范位置的元数据表。
    repo_id="lerobot/pusht",  # 指定 PushT 数据集仓库。
    repo_type="dataset",  # 指定仓库类型为数据集。
    revision="7628202a2180972f291ba1bc6723834921e72c19",  # 固定为前面使用的数据版本。
    filename="meta/episodes/chunk-000/file-000.parquet",  # 指定示范级别的元数据文件。
    local_dir=root / "data" / "pusht",  # 保存到项目内已有的数据目录。
)  # 结束下载调用，得到元数据表的本地路径。

episodes = pd.read_parquet(meta_path)  # 读取元数据表；这里每行描述一整段示范。

columns = [  # 定义本次需要查看的字段。
    "episode_index",  # 示范编号，用来与数值表中的示范编号对应。
    "length",  # 这段示范包含的采样时间步数量。
    "videos/observation.image/chunk_index",  # 视频所在的分块编号。
    "videos/observation.image/file_index",  # 视频在该分块中的文件编号。
    "videos/observation.image/from_timestamp",  # 这段示范在视频时间轴上的起始时间，单位为秒。
    "videos/observation.image/to_timestamp",  # 这段示范在视频时间轴上的结束边界，单位为秒。
]  # 结束字段列表。

selected = episodes[episodes["episode_index"].isin([0, 1])]  # 筛选编号为 0 和 1 的两段示范。
print(selected[columns].sort_values("episode_index").to_string(index=False))  # 按示范编号排序，显示选定字段并隐藏表格行索引。