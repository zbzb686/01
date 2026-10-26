from pathlib import Path  # 导入路径工具，用于定位本地数据目录。

from torch.utils.data import DataLoader  # 导入批量读取工具，将多条样本组织成一个批次。
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # 导入 LeRobot 数据集类，负责读取和对齐数据。
import torch  # 导入 PyTorch，用于设置随机种子和检查损失数值。
from lerobot.configs.types import FeatureType, PolicyFeature  # 导入特征类型与形状描述，告诉模型各字段代表什么。
from lerobot.policies.act.configuration_act import ACTConfig  # 导入 ACT 的配置类，用于设置输入、输出和动作序列长度。
from lerobot.policies.act.modeling_act import ACTPolicy  # 导入 ACT 模型类。
from lerobot.policies.act.processor_act import make_act_pre_post_processors  # 导入官方预处理工具，负责标准化数据和安排计算设备。

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
torch.manual_seed(0)  # 设置随机种子，方便后续在相同条件下比较实验结果。

config = ACTConfig(  # 创建 ACT 配置，描述模型应该接收和预测什么。
    input_features={  # 定义模型使用的观测字段；形状中不包含批量维度。
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),  # 图像属于视觉特征，按通道、高、宽排列。
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),  # 状态包含两个坐标值；逗号使 (2,) 成为元组。
    },  # 结束输入特征定义。
    output_features={  # 定义模型需要预测的动作字段。
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),  # 每个动作包含两个目标坐标值。
    },  # 结束输出特征定义。
    chunk_size=chunk_size,  # 使用前面设置的长度，一次预测 10 个动作。
    n_action_steps=chunk_size,  # 设置部署时每次预测后可执行的动作数量，本轮仅计算损失。
    pretrained_backbone_weights=None,  # 随机初始化视觉骨干，先检查计算流程，避免额外下载预训练权重。
    device="cuda",  # 指定在 NVIDIA 显卡上计算。
)  # 完成模型配置。

preprocessor, _ = make_act_pre_post_processors(config, dataset.meta.stats)  # 使用数据集统计量创建处理器；第二个返回值用于推理后恢复动作坐标，本轮暂不使用。
processed_batch = preprocessor(batch)  # 标准化图像、状态与目标动作，并将张量放到配置指定的设备上。

policy = ACTPolicy(config).to(config.device)  # 根据配置创建模型，并把模型参数放到显卡上。
policy.train()  # 切换到训练模式；此操作本身不会更新参数。

loss, metrics = policy(processed_batch)  # 执行前向计算，得到总损失及各项损失记录。

print("模型设备：", next(policy.parameters()).device)  # 查看模型参数所在的计算设备。
print("总损失：", loss.item())  # 将总损失这个标量张量转换为普通数值并打印。
print("各项损失：", metrics)  # 打印动作预测误差和潜变量约束等损失分项。
print("损失是否为有限数值：", torch.isfinite(loss).item())  # 检查损失是否出现 NaN 或无穷大。