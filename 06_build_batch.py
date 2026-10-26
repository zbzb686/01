from pathlib import Path  # 导入路径工具，用于定位本地数据目录。

from torch.utils.data import DataLoader  # 导入批量读取工具，将多条样本组织成一个批次。
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # 导入 LeRobot 数据集类，负责读取和对齐数据。

root = Path(__file__).parent  # 获取当前脚本所在的项目目录。
fps = 10  # 使用已经核实的数据采样频率，每秒 10 个时间步。
chunk_size = 10  # 设置每条训练样本需要提供的目标动作数量。
action_times = [step / fps for step in range(chunk_size)]  # 生成相对时间偏移：0.0、0.1，直到 0.9 秒。

dataset = LeRobotDataset(  # 创建能够返回动作序列的数据集。
    repo_id="lerobot/pusht",  # 指定 PushT 数据集。
    root=root / "data" / "pusht",  # 使用项目内已经下载的数据。
    episodes=[0],  # 先使用第 0 段示范，方便与前面的结果对照。
    revision="7628202a2180972f291ba1bc6723834921e72c19",  # 固定数据版本。
    video_backend="pyav",  # 使用前一步已经验证的 PyAV 解码方式。
    delta_timestamps={"action": action_times},  # 为 action 字段读取指定相对时间上的动作序列。
)  # 完成数据集初始化。

loader = DataLoader(  # 创建批量读取器。
    dataset,  # 指定从哪个数据集读取样本。
    batch_size=4,  # 每批读取 4 条样本。
    shuffle=False,  # 按原顺序读取，方便核对最前面的真实数值。
    num_workers=0,  # 在当前进程读取数据，便于在 Windows 上观察和排错。
)  # 完成批量读取器的创建。

batch = next(iter(loader))  # 创建迭代器，并取出第一个批次。

print("图像形状：", tuple(batch["observation.image"].shape))  # 查看批量图像的维度。
print("状态形状：", tuple(batch["observation.state"].shape))  # 查看批量状态的维度。
print("动作形状：", tuple(batch["action"].shape))  # 查看批量动作序列的维度。
print("补齐标记形状：", tuple(batch["action_is_pad"].shape))  # 查看哪些目标动作属于越过示范边界后的补齐位置。
print("首个样本的前三个目标动作：", batch["action"][0, :3].tolist())  # 取第一个样本的前三个动作，与之前的表格核对。