from pathlib import Path  # 导入路径工具，定位本地元数据和配置目录。
import json  # 导入 JSON 工具，保存可重复使用的数据划分。
import pandas as pd  # 导入已经安装的 pandas，读取示范元数据表。

root = Path(__file__).resolve().parent  # 获取当前项目目录。
meta_dir = root / "data" / "pusht" / "meta" / "episodes"  # 定位记录各段示范信息的目录。
meta_files = sorted(meta_dir.glob("chunk-*/file-*.parquet"))  # 查找本地全部示范元数据分片。

if not meta_files:  # 检查元数据是否存在。
    raise FileNotFoundError(f"没有找到示范元数据：{meta_dir}")  # 路径不正确时停止，输出需要检查的位置。

episodes = pd.concat([pd.read_parquet(path) for path in meta_files], ignore_index=True)  # 合并元数据，每行描述一段示范。
train_episodes = list(range(180))  # 选择编号 0～179 的完整示范用于训练。
val_episodes = list(range(180,200))  # 选择编号 180～199 的完整示范用于验证。

if set(train_episodes) & set(val_episodes):  # 检查训练和验证是否包含同一段示范。
    raise RuntimeError("训练与验证示范发生重叠。")  # 出现重叠时停止，避免评估混入训练数据。

available = set(episodes["episode_index"].astype(int).tolist())  # 收集本地元数据中存在的示范编号。
missing = (set(train_episodes) | set(val_episodes)) - available  # 检查计划使用的编号是否全部存在。
if missing:  # 如果存在缺失编号，则不能继续生成有效划分。
    raise RuntimeError(f"元数据中缺少示范编号：{sorted(missing)}")  # 输出缺失编号，便于定位问题。

train_rows = episodes[episodes["episode_index"].isin(train_episodes)]  # 筛选训练示范对应的记录。
val_rows = episodes[episodes["episode_index"].isin(val_episodes)]  # 筛选验证示范对应的记录。
train_frames = int(train_rows["length"].sum())  # 汇总训练示范包含的时间步数量。
val_frames = int(val_rows["length"].sum())  # 汇总验证示范包含的时间步数量。

print("训练示范及长度：")  # 标明下面展示训练数据范围。
print(train_rows[["episode_index", "length"]].sort_values("episode_index").to_string(index=False))  # 打印每段训练示范的编号与长度。
print("验证示范及长度：")  # 标明下面展示验证数据范围。
print(val_rows[["episode_index", "length"]].sort_values("episode_index").to_string(index=False))  # 打印每段验证示范的编号与长度。
print("训练总时间步数：", train_frames)  # 查看训练数据规模。
print("验证总时间步数：", val_frames)  # 查看验证数据规模。
print("训练与验证是否重叠：", bool(set(train_episodes) & set(val_episodes)))  # 预期输出 False。

split = {  # 创建后续训练和评估共同读取的划分清单。
    "dataset_repo": "lerobot/pusht",  # 记录数据集名称。
    "dataset_revision": "7628202a2180972f291ba1bc6723834921e72c19",  # 固定数据版本。
    "train_episodes": train_episodes,  # 记录完整训练示范列表。
    "val_episodes": val_episodes,  # 记录完整验证示范列表。
    "train_frames": train_frames,  # 记录训练时间步总数。
    "val_frames": val_frames,  # 记录验证时间步总数。
}  # 完成划分清单。

config_path = root / "configs" / "pusht_split_v2.json"  # 将划分保存在项目配置目录，后续可以提交到 Git。
config_path.parent.mkdir(parents=True, exist_ok=True)  # 创建配置目录。
config_path.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")  # 保存格式清晰的 JSON 文件。
print("划分清单保存位置：", config_path)  # 输出保存位置，供下一步使用。