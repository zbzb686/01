from pathlib import Path  # 导入路径工具，定位配置与元数据。
import json  # 导入 JSON 工具，读取划分并保存统计量。
import numpy as np  # 导入 NumPy，处理统计量数组。
import pyarrow.parquet as pq  # 导入 Parquet 读取工具，正确还原嵌套的图像统计量。
from lerobot.datasets.compute_stats import aggregate_stats  # 导入官方合并工具，按样本数量合并均值和标准差。
root = Path(__file__).resolve().parent  # 获取项目目录。
split_path = root / "configs" / "pusht_split_v2.json"  # 定位上一轮保存的数据划分。
split = json.loads(split_path.read_text(encoding="utf-8"))  # 读取训练和验证示范编号。
train_ids = split["train_episodes"]  # 只选择训练示范参与统计。
meta_files = sorted((root / "data" / "pusht" / "meta" / "episodes").glob("chunk-*/file-*.parquet"))  # 查找本地示范元数据。
feature_shapes = {  # 指定模型需要的三个字段及其统计量形状。
    "observation.image": (3, 1, 1),  # 每个颜色通道各保存一个统计值。
    "observation.state": (2,),  # 推手位置的两个坐标各保存一个统计值。
    "action": (2,),  # 目标位置的两个坐标各保存一个统计值。
}  # 完成字段定义。
stat_names = ["min", "max", "mean", "std", "count"]  # 读取最小值、最大值、均值、标准差及统计计数。
columns = ["episode_index"] + [f"stats/{feature}/{name}" for feature in feature_shapes for name in stat_names]  # 只读取需要的元数据列。
records = []  # 收集训练示范的统计记录。
if not meta_files:  # 检查元数据文件是否存在。
    raise FileNotFoundError("没有找到本地示范元数据。")  # 文件缺失时停止，避免生成空统计量。

for path in meta_files:  # 依次读取元数据分片。
    missing = set(columns) - set(pq.read_schema(path).names)  # 检查这个文件是否包含所需统计列。
    if missing:  # 缺少字段时不能正确合并统计量。
        raise RuntimeError(f"元数据缺少统计列：{sorted(missing)}")  # 明确指出缺少哪些列。
    table = pq.read_table(path, columns=columns, filters=[("episode_index", "in", train_ids)])  # 只读取训练示范对应的统计记录。
    records.extend(table.to_pylist())  # 转为 Python 列表，保留图像统计量的嵌套结构。
record_ids = [int(record["episode_index"]) for record in records]  # 获取实际读到的示范编号。
if sorted(record_ids) != sorted(train_ids):  # 检查是否遗漏训练示范或重复读取记录。
    raise RuntimeError(f"统计记录与训练划分不一致：{record_ids}")  # 不一致时停止，避免保存错误范围的统计量。

episode_stats_list = []  # 收集每段训练示范的统计字典。
for record in records:  # 逐段整理训练统计量。
    episode_stats = {}  # 创建当前示范的统计字典。
    for feature, shape in feature_shapes.items():  # 依次处理图像、状态和动作。
        episode_stats[feature] = {}  # 创建当前字段的统计字典。
        for name in stat_names:  # 依次处理五种统计值。
            dtype = np.int64 if name == "count" else np.float64  # 计数使用整数，其他统计值使用浮点数。
            target_shape = (1,) if name == "count" else shape  # 计数只有一个值，其他统计量按字段形状保存。
            value = np.asarray(record[f"stats/{feature}/{name}"], dtype=dtype).reshape(target_shape)  # 将元数据转换为官方合并工具需要的数组。
            if not np.isfinite(value).all():  # 检查统计量是否出现 NaN 或无穷大。
                raise RuntimeError(f"示范 {record['episode_index']} 的 {feature}/{name} 数值异常。")  # 指明异常来源。
            episode_stats[feature][name] = value  # 保存当前统计值。
    episode_stats_list.append(episode_stats)  # 将整理好的当前示范加入合并列表。

train_stats = aggregate_stats(episode_stats_list)  # 使用官方算法合并训练示范，不能直接平均各段的标准差。
action_count = int(train_stats["action"]["count"][0])  # 读取动作统计包含的时间步数量。
if action_count != split["train_frames"]:  # 核对动作计数与划分清单中的训练帧数是否一致。
    raise RuntimeError(f"动作计数不一致：{action_count}，预期 {split['train_frames']}。")  # 数量不一致时停止保存。

payload = {  # 保存统计量及其来源，便于后续训练核对。
    "dataset_repo": split["dataset_repo"],  # 记录数据集名称。
    "dataset_revision": split["dataset_revision"],  # 记录固定数据版本。
    "train_episodes": train_ids,  # 记录参与统计的训练示范。
    "train_frames": action_count,  # 记录参与动作统计的时间步数量。
    "stats": {feature: {name: value.tolist() for name, value in stats.items()} for feature, stats in train_stats.items()},  # 将 NumPy 数组转换为合法 JSON 数据。
}  # 完成保存内容。
stats_path = root / "configs" / "pusht_train_stats_v2.json"  # 单独保存训练统计量，不覆盖原始数据集文件。
stats_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")  # 写入可供后续训练读取的 JSON。

print("参与统计的示范编号：", sorted(record_ids))  # 应当只有 0～9。
print("动作统计时间步数：", action_count)  # 应当为 1347。
print("验证示范是否参与统计：", bool(set(record_ids) & set(split["val_episodes"])))  # 应当为 False。
for feature in feature_shapes:  # 展示三个字段的核心统计量。
    print(feature, "均值：", train_stats[feature]["mean"].reshape(-1).tolist())  # 将统计量展平后打印，方便查看。
    print(feature, "标准差：", train_stats[feature]["std"].reshape(-1).tolist())  # 打印对应标准差。
print("训练统计量保存位置：", stats_path)  # 输出下一步训练需要读取的文件位置。