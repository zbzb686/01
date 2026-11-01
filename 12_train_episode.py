from pathlib import Path  # 导入路径工具，定位项目数据和输出目录。
from datetime import datetime  # 导入时间工具，为本次实验生成独立目录。
import csv  # 导入 CSV 工具，保存训练日志。
import json  # 导入 JSON 工具，保存实验配置。
import torch  # 导入 PyTorch，进行模型计算和参数更新。
from torch.utils.data import DataLoader  # 导入批次读取器，持续提供不同样本。
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # 导入已经验证可用的数据集读取工具。
from lerobot.configs.types import FeatureType, PolicyFeature  # 导入模型输入输出字段的描述工具。
from lerobot.policies.act.configuration_act import ACTConfig  # 导入 ACT 配置类。
from lerobot.policies.act.modeling_act import ACTPolicy  # 导入 ACT 模型类。
from lerobot.policies.act.processor_act import make_act_pre_post_processors  # 导入官方预处理和后处理工具。

root = Path(__file__).resolve().parent  # 获取脚本所在的项目目录。
torch.manual_seed(0)  # 固定随机种子，便于重复实验。
train_episodes = [0]  # 只训练第 0 段示范，保存记录时也使用同一个变量。
revision = "7628202a2180972f291ba1bc6723834921e72c19"  # 固定数据版本。
fps = 10  # 使用已经确认的数据采样频率。
chunk_size = 10  # 每条观测对应连续 10 个目标动作。
batch_size = 4  # 每批通常读取 4 个样本。
total_steps = 200  # 设置优化器总共更新 200 次。

dataset = LeRobotDataset(  # 创建提供动作序列的数据集。
    repo_id="lerobot/pusht",  # 指定 PushT 数据集。
    root=root / "data" / "pusht",  # 使用已经下载的数据。
    episodes=train_episodes,  # 允许读取选定示范中的全部时间步。
    revision=revision,  # 使用固定的数据版本。
    video_backend="pyav",  # 使用已经验证的视频解码后端。
    delta_timestamps={"action": [offset / fps for offset in range(chunk_size)]},  # 为每条输入读取当前及未来 9 个目标动作。
)  # 完成数据集创建。

loader = DataLoader(  # 创建不断提供不同批次的读取器。
    dataset,  # 从选定的整段示范读取样本。
    batch_size=batch_size,  # 每批最多包含 4 个输入。
    shuffle=True,  # 每轮遍历前打乱样本顺序，避免总按时间先后训练。
    num_workers=0,  # 在当前进程读取，沿用已验证的 Windows 设置。
    drop_last=False,  # 保留最后不足 4 条的批次，确保一整轮能覆盖全部样本。
)  # 完成读取器创建。

config = ACTConfig(  # 沿用之前实验的模型设置。
    input_features={  # 描述模型使用的两种输入。
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),  # 输入 RGB 图像。
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),  # 输入推手当前位置的两个坐标。
    },  # 结束输入配置。
    output_features={  # 描述模型需要预测的动作。
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),  # 每个动作包含目标位置的两个坐标。
    },  # 结束输出配置。
    chunk_size=chunk_size,  # 一次预测 10 个动作。
    n_action_steps=chunk_size,  # 保持与之前模型相同的动作执行配置。
    pretrained_backbone_weights=None,  # 仍然随机初始化骨干，不额外下载预训练权重。
    device="cuda",  # 使用已经验证可用的 GPU。
)  # 完成模型配置。

preprocessor, postprocessor = make_act_pre_post_processors(config, dataset.meta.stats)  # 创建与本次模型配套的处理器。
policy = ACTPolicy(config).to(config.device)  # 从头创建模型并放到显卡。
policy.train()  # 切换到训练模式。
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5, weight_decay=1e-4)  # 沿用之前的学习率和权重衰减。

step = 0  # 记录实际完成的更新次数，不再在循环外额外更新。
epoch = 0  # 记录当前进行到第几轮数据遍历。
seen_frames = set()  # 记录真正参与过更新的输入帧编号。
train_log = []  # 收集每一步的训练指标。

print("数据集样本数：", len(dataset))  # 当前应为 161。
print("每轮批次数：", len(loader))  # 当前应为 41，最后一批只有 1 个样本。

while step < total_steps:  # 数据遍历结束后重新开始，直到累计完成 200 次更新。
    epoch += 1  # 开始新一轮数据遍历。
    for batch in loader:  # 每次循环读取一个新批次，这是本次实验最主要的变化。
        frame_ids = batch["frame_index"].tolist()  # 获取本批输入的帧编号，用来核实数据确实在变化。
        processed_batch = preprocessor(batch)  # 对本批新数据进行标准化和设备安排。
        optimizer.zero_grad(set_to_none=True)  # 清空上一批梯度。
        loss, metrics = policy(processed_batch)  # 用本批数据重新计算总损失和分项损失。

        if not torch.isfinite(loss).item():  # 检查当前损失是否为有限数值。
            raise RuntimeError(f"第 {step + 1} 步损失异常，停止训练。")  # 异常时停止，防止继续更新。

        loss.backward()  # 计算本批损失对应的梯度。
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0, error_if_nonfinite=True)  # 裁剪梯度，遇到非有限梯度时停止。
        optimizer.step()  # 根据本批梯度更新模型一次。
        step += 1  # 只有更新成功后才增加计数。
        seen_frames.update(frame_ids)  # 将本批实际使用过的输入帧加入覆盖记录。
        train_log.append({  # 保存本步指标，数值均为普通 Python 数字。
            "step": step,  # 记录更新编号。
            "loss": loss.item(),  # 记录本次更新前计算的总损失。
            "l1_loss": metrics["l1_loss"],  # 记录动作预测误差。
            "kld_loss": metrics["kld_loss"],  # 记录 KL 损失。
            "grad_norm": grad_norm.item(),  # 记录裁剪前的梯度范数。
        })  # 完成本步日志记录。

        if step <= 3:  # 前三步显示帧编号，直接检查读取器是否在提供不同样本。
            print(f"第 {step} 步的输入帧编号：{frame_ids}")  # 打乱后通常不再是固定的 0、1、2、3。

        if step == 1 or step % 20 == 0:  # 第一步及之后每 20 步打印一次，避免输出过多。
            print(f"Step {step}/{total_steps} | Loss={loss.item():.4f} | L1={metrics['l1_loss']:.4f} | KL={metrics['kld_loss']:.4f} | 已见输入={len(seen_frames)}/{len(dataset)}")  # 同时观察损失与输入覆盖情况。

        if step >= total_steps:  # 达到设定次数后，结束当前数据遍历。
            break  # 返回外层循环，由外层条件结束训练。

run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 为本次训练生成独立实验名称。
save_dir = root / "outputs" / "act_pusht" / run_name  # 使用与之前一致的模型输出位置。
save_dir.mkdir(parents=True, exist_ok=False)  # 创建新目录，避免覆盖旧实验。

policy.save_pretrained(save_dir)  # 保存本次训练后的模型配置和权重。
preprocessor.save_pretrained(save_dir, config_filename="policy_preprocessor.json")  # 保存输入处理流程及统计量。
postprocessor.save_pretrained(save_dir, config_filename="policy_postprocessor.json")  # 保存动作恢复流程及统计量。

run_info = {  # 汇总本次实验设置和实际完成情况。
    "dataset_repo": "lerobot/pusht",  # 记录数据集名称。
    "dataset_revision": revision,  # 记录数据版本。
    "episodes": train_episodes,  # 从实际配置变量记录示范范围，避免手写不一致。
    "batch_size": batch_size,  # 记录通常使用的批次大小。
    "chunk_size": chunk_size,  # 记录动作块长度。
    "completed_updates": step,  # 记录实际更新次数。
    "unique_input_frames": len(seen_frames),  # 记录实际训练过的不同输入数量。
    "dataset_frames": len(dataset),  # 记录本轮允许使用的输入总数。
    "last_loss_before_update": loss.item(),  # 记录最后一次更新前的总损失。
    "experiment": "episode_multibatch",  # 标明这是整段示范的多批次训练实验。
}  # 完成实验记录。

with (save_dir / "run_info.json").open("w", encoding="utf-8") as file:  # 打开实验说明文件。
    json.dump(run_info, file, ensure_ascii=False, indent=2)  # 保存格式清晰的实验说明。

with (save_dir / "train_log.csv").open("w", newline="", encoding="utf-8-sig") as file:  # 打开训练日志文件。
    writer = csv.DictWriter(file, fieldnames=["step", "loss", "l1_loss", "kld_loss", "grad_norm"])  # 设置日志列名。
    writer.writeheader()  # 写入列名。
    writer.writerows(train_log)  # 保存全部更新步骤的指标。

print("实际更新次数：", step)  # 检查是否完成 200 次更新。
print("实际覆盖输入数量：", len(seen_frames), "/", len(dataset))  # 检查是否使用过全部 161 个输入。
print("模型保存目录：", save_dir.resolve())  # 输出新模型的路径，下一步评估时使用。
print("不同批次训练与保存完成。")  # 确认训练及保存过程都已结束。