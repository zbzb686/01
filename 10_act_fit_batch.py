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

preprocessor, postprocessor = make_act_pre_post_processors(config, dataset.meta.stats)  # 使用数据集统计量创建处理器；第二个返回值用于推理后恢复动作坐标，本轮暂不使用。
processed_batch = preprocessor(batch)  # 标准化图像、状态与目标动作，并将张量放到配置指定的设备上。

policy = ACTPolicy(config).to(config.device)  # 根据配置创建模型，并把模型参数放到显卡上。
policy.train()  # 切换到训练模式；此操作本身不会更新参数。

loss, metrics = policy(processed_batch)  # 执行前向计算，得到总损失及各项损失记录。

print("模型设备：", next(policy.parameters()).device)  # 查看模型参数所在的计算设备。
print("总损失：", loss.item())  # 将总损失这个标量张量转换为普通数值并打印。
print("各项损失：", metrics)  # 打印动作预测误差和潜变量约束等损失分项。
print("损失是否为有限数值：", torch.isfinite(loss).item())  # 检查损失是否出现 NaN 或无穷大。
if not torch.isfinite(loss).item():  # 如果前向损失不是有限数值，停止实验，避免用异常损失更新参数。
    raise RuntimeError("损失出现 NaN 或无穷大，已停止参数更新。")  # 输出明确错误，便于把问题定位在前向计算阶段。
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5, weight_decay=1e-4)  # 创建 AdamW 优化器，本次用较小学习率验证一次参数更新。
optimizer.zero_grad(set_to_none=True)  # 清空已有梯度，防止把之前计算的梯度意外累加进来。
loss.backward()  # 根据当前总损失计算模型中可训练参数的梯度；这一步尚未修改参数。
grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0, error_if_nonfinite=True)  # 限制梯度整体大小，若梯度出现非有限数值则停止更新。
tracked_name, tracked_parameter = next((name, parameter) for name, parameter in policy.named_parameters() if parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0)  # 选取一个确实收到非零梯度的参数，检查它是否发生变化。
parameter_before = tracked_parameter.detach().clone()  # 保存这个参数更新前的副本，不保留额外计算图。
optimizer.step()  # 使用本次梯度执行一次参数更新，这是本脚本实际改变模型权重的步骤。
max_change = (tracked_parameter.detach() - parameter_before).abs().max().item()  # 计算这个参数中最大的绝对变化，验证更新确实生效。
print("裁剪前梯度范数：", grad_norm.item())  # 打印梯度裁剪之前的整体大小，用来观察本次反向计算是否正常。
print("检查的参数名称：", tracked_name)  # 显示用于验证变化的模型参数名称。
print("参数最大绝对变化：", max_change)  # 大于零说明这个参数确实发生了变化。
print("参数是否已更新：", max_change > 0)  # 输出本轮实验最主要的检查结果，预期为 True。
print("已完成参数更新次数：", 1)  # 明确本轮只做了一次更新，尚未保存权重或完成正式训练。
total_steps=200;
for step in range(1,total_steps):  # 进行多轮训练，观察损失是否稳定下降。
    optimizer.zero_grad(set_to_none=True)  # 清空梯度。
    loss, metrics = policy(processed_batch)  # 前向计算损失。
    if not torch.isfinite(loss).item():  # 每一步都检查总损失是否为有限数值。
        raise RuntimeError(f"第 {step} 步损失异常，停止训练。")  # 出现 NaN 或无穷大时停止，并指出发生问题的步骤。
    
    loss.backward()  # 反向计算梯度。
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0, error_if_nonfinite=True)  # 裁剪梯度。
    optimizer.step()  # 更新参数。
    print(f"Step {step + 1}/{total_steps}: Loss={loss.item():.6f}, Grad Norm={grad_norm.item():.6f}")  # 打印每步的损失和梯度范数。
    print(f"Step{step+1}/{total_steps}: Loss1={metrics['l1_loss']:.6f}, Loss2={metrics['kld_loss']:.6f}\n")  # 打印每步的损失和梯度范数。
print("实验完成：同一批数据累计更新了", total_steps, "次。")  # 确认训练循环完整结束。
print("本轮尚未保存模型，下一步再添加保存功能。")  # 提醒当前更新后的模型仍只
#保存模型
from datetime import datetime  # 导入时间工具，为每次实验创建不同的保存目录。
import json  # 导入 JSON 工具，保存便于查看的实验记录。
run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 使用当前时间生成实验名称，避免覆盖之前保存的模型。
save_dir = root / "outputs" / "act_pusht" / run_name  # 将本次结果保存到项目 outputs 目录下。
save_dir.mkdir(parents=True, exist_ok=False)  # 创建输出目录；若同名目录已经存在则报错，避免混入旧文件。
policy.save_pretrained(save_dir)  # 使用 LeRobot 官方接口，将当前模型配置和训练后的权重保存到本地。
preprocessor.save_pretrained(save_dir, config_filename="policy_preprocessor.json")  # 保存输入处理流程及其标准化统计量。
postprocessor.save_pretrained(save_dir, config_filename="policy_postprocessor.json")  # 保存动作处理流程，供推理时恢复原始坐标。
run_info = {  # 创建实验记录，避免以后只看到权重却不知道它来自什么实验。
    "dataset_repo": "lerobot/pusht",  # 记录使用的数据集。
    "dataset_revision": "7628202a2180972f291ba1bc6723834921e72c19",  # 记录固定的数据版本。
    "episodes": [0],  # 记录本轮使用的示范编号。
    "batch_size": 4,  # 记录每批样本数量。
    "chunk_size": chunk_size,  # 记录每个样本预测的动作数量。
    "completed_updates": total_steps,  # 记录实际完成的参数更新次数。
    "last_loss_before_update": loss.item(),  # 记录最后一次更新前计算的训练损失。
    "experiment": "same_batch_smoke_test",  # 标明本轮是同一批数据上的流程验证实验。
}  # 完成实验记录。
with open(save_dir / "run_info.json", "w") as f:  # 打开文件，准备写入 JSON 格式的实验记录。
    json.dump(run_info, f, indent=4)  # 将实验记录写入文件，并使用缩进便于阅读。
f.close()  # 关闭文件，确保数据写入磁盘。
print("模型保存目录：", save_dir.resolve())  # 打印完整路径，方便下一步加载模型。
for saved_file in sorted(save_dir.iterdir()):  # 按名称遍历保存目录中的文件。
    print("已保存：", saved_file.name)  # 列出实际生成的文件，检查保存结果。
print("模型与处理器保存完成。")  # 只有前面的保存操作都成功后，才会打印这条提示。
