from pathlib import Path  # 导入路径工具，定位配置、数据和输出目录。
from datetime import datetime  # 导入时间工具，区分不同训练实验。
import csv  # 导入 CSV 工具，保存训练与验证日志。
import json  # 导入 JSON 工具，读取配置并保存实验记录。
import torch  # 导入 PyTorch，进行模型计算与参数更新。
from torch.utils.data import DataLoader  # 导入批次读取工具。
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # 导入数据集读取类。
from lerobot.configs.types import FeatureType, PolicyFeature  # 导入输入输出字段描述工具。
from lerobot.policies.act.configuration_act import ACTConfig  # 导入 ACT 配置类。
from lerobot.policies.act.modeling_act import ACTPolicy  # 导入 ACT 模型类。
from lerobot.policies.act.processor_act import make_act_pre_post_processors  # 导入官方处理器创建函数。

root = Path(__file__).resolve().parent  # 获取当前项目目录。
split = json.loads((root / "configs" / "pusht_split_v1.json").read_text(encoding="utf-8"))  # 读取固定的数据划分。
stats_doc = json.loads((root / "configs" / "pusht_train_stats_v1.json").read_text(encoding="utf-8"))  # 读取训练集独立统计量。
torch.manual_seed(0)  # 设置随机种子，方便重复实验。
total_steps = 1000  # 设置本轮累计更新次数。
val_interval = 200  # 每完成 200 次更新进行一次完整验证。
batch_size = 4  # 沿用已验证的批次大小。
chunk_size = 10  # 每条观测预测 10 个目标动作。
fps = 10  # 沿用已核实的数据采样频率。
learning_rate = 1e-5  # 沿用之前的学习率。

if set(split["train_episodes"]) & set(split["val_episodes"]):  # 再次检查训练和验证示范是否重叠。
    raise RuntimeError("训练与验证示范重叠。")  # 发现重叠时停止。
if sorted(stats_doc["train_episodes"]) != sorted(split["train_episodes"]):  # 检查统计量是否来自当前训练划分。
    raise RuntimeError("统计量来源与训练划分不一致。")  # 防止误用其他实验的统计量。
if stats_doc["dataset_revision"] != split["dataset_revision"]:  # 检查数据版本是否匹配。
    raise RuntimeError("统计量与划分使用的数据版本不同。")  # 版本不匹配时停止。

train_stats = {feature: {name: torch.as_tensor(value) for name, value in stats.items()} for feature, stats in stats_doc["stats"].items()}  # 将 JSON 数组恢复为处理器需要的张量。
dataset_args = {  # 定义训练和验证共用的数据读取设置。
    "repo_id": split["dataset_repo"],  # 使用配置中的数据集名称。
    "root": root / "data" / "pusht",  # 使用项目中的数据目录。
    "revision": split["dataset_revision"],  # 使用固定数据版本。
    "video_backend": "pyav",  # 使用已经验证可用的视频后端。
    "delta_timestamps": {"action": [offset / fps for offset in range(chunk_size)]},  # 为每条观测提供连续 10 个目标动作。
}  # 完成共用数据配置。

train_dataset = LeRobotDataset(**dataset_args, episodes=split["train_episodes"])  # 创建训练集，** 将字典展开成函数参数。
val_dataset = LeRobotDataset(**dataset_args, episodes=split["val_episodes"])  # 创建验证集，使用不同的完整示范。
if len(train_dataset) != split["train_frames"] or len(val_dataset) != split["val_frames"]:  # 核对实际样本数量与清单。
    raise RuntimeError("实际数据集长度与划分清单不一致。")  # 数量异常时停止。

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)  # 打乱训练输入，并保留最后不足一批的样本。
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)  # 按顺序遍历完整验证集。

config = ACTConfig(  # 创建与之前相同的模型结构。
    input_features={  # 定义模型接收的观测。
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),  # 输入 RGB 图像。
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),  # 输入推手当前位置。
    },  # 结束输入定义。
    output_features={  # 定义模型输出的动作。
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),  # 每个动作包含两个目标坐标。
    },  # 结束输出定义。
    chunk_size=chunk_size,  # 每次预测 10 个动作。
    n_action_steps=chunk_size,  # 保持之前的动作执行配置。
    pretrained_backbone_weights=None,  # 从头初始化，不下载预训练骨干。
    device="cuda",  # 使用已经验证的 GPU。
)  # 完成模型配置。

preprocessor, postprocessor = make_act_pre_post_processors(config, train_stats)  # 训练和验证共同使用训练集统计量。
policy = ACTPolicy(config).to(config.device)  # 从头创建模型并放到 GPU。
optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=1e-4)  # 创建优化器。

run_dir = root / "outputs" / "act_train_val" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 创建本次实验的独立路径。
run_dir.mkdir(parents=True, exist_ok=False)  # 创建目录，保留之前所有实验。
for filename, content in [("split.json", split), ("train_stats.json", stats_doc)]:  # 保存本次实际使用的配置快照。
    (run_dir / filename).write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")  # 防止以后修改配置后丢失实验来源。

@torch.no_grad()  # 验证函数不记录梯度，也不进行反向传播。
def evaluate():  # 定义完整验证集的动作误差计算流程。
    was_training = policy.training  # 记录进入验证前的模型模式。
    policy.eval()  # 切换到推理模式。
    error_sum = 0.0  # 累积所有有效坐标的绝对误差。
    coordinate_count = 0  # 累积有效坐标数量。

    for batch_number, raw_batch in enumerate(val_loader, start=1):  # 遍历验证集的所有批次。
        obs = {key: raw_batch[key] for key in config.input_features}  # 只把图像和状态提供给模型，不提供示范动作答案。
        processed_obs = preprocessor(obs)  # 使用训练集统计量处理验证输入。
        normalized_prediction = policy.predict_action_chunk(processed_obs)  # 在推理模式下预测动作序列。
        prediction = postprocessor(normalized_prediction).detach().cpu()  # 将预测恢复为原始动作坐标并移到 CPU。
        target = raw_batch["action"].cpu()  # 取出未经标准化的示范目标坐标。
        valid_mask = (~raw_batch["action_is_pad"]).cpu().unsqueeze(-1)  # 有效位置为 True，排除越过示范末尾后的补齐动作。

        if prediction.shape != target.shape or not torch.isfinite(prediction).all().item():  # 检查预测形状与数值。
            raise RuntimeError("验证预测形状或数值异常。")  # 发现异常时停止，避免保存无效指标。

        error_sum += ((prediction - target).abs() * valid_mask).sum().item()  # 只累积有效位置的绝对坐标误差。
        coordinate_count += int(valid_mask.sum().item()) * target.shape[-1]  # 每个有效动作包含两个坐标。
        if batch_number % 25 == 0:  # 定期报告验证进度，避免长时间没有输出。
            print(f"  验证进度：{batch_number}/{len(val_loader)} 批")  # 显示验证读取到了哪里。

    policy.train(was_training)  # 恢复进入验证前的模式，避免后续误用推理模式训练。
    if coordinate_count == 0:  # 检查验证集是否包含有效目标。
        raise RuntimeError("验证集中没有有效动作坐标。")  # 无法计算指标时明确报错。
    return error_sum / coordinate_count  # 按全部有效坐标计算平均误差，不直接平均各批次均值。

def save_model(folder, current_step, val_mae):  # 定义模型导出函数，用于保存最佳模型或最后模型。
    policy.save_pretrained(folder)  # 保存模型结构配置和当前权重，并创建目标目录。
    preprocessor.save_pretrained(folder, config_filename="policy_preprocessor.json")  # 保存训练集输入统计量。
    postprocessor.save_pretrained(folder, config_filename="policy_postprocessor.json")  # 保存动作恢复流程与统计量。
    info = {"step": current_step, "val_coordinate_mae": val_mae}  # 记录这份权重对应的更新次数及验证误差。
    (folder / "checkpoint_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")  # 保存模型检查点说明。

print("训练样本数：", len(train_dataset))  # 应当输出 1347。
print("验证样本数：", len(val_dataset))  # 应当输出 337。
print("实验目录：", run_dir)  # 提前显示输出位置。
print("正在计算训练前的验证误差……")  # 先测量随机初始化模型，建立本轮基线。
best_mae = evaluate()  # 计算尚未训练的模型在验证集上的误差。
best_step = 0  # 将训练前模型记为第 0 步。
last_val_mae = best_mae  # 初始化最近一次验证结果。
save_model(run_dir / "best", 0, best_mae)  # 保存基线，后续只有验证误差更低时才覆盖。
print(f"Step 0 | 验证 MAE={best_mae:.4f}")  # 输出训练前基线。

step = 0  # 初始化实际更新次数。
seen_indices = set()  # 用数据集全局索引记录已使用过的输入，避免不同示范的帧编号重复。
train_log = []  # 保存每一步的训练指标。
val_log = [{"step": 0, "val_coordinate_mae": best_mae}]  # 保存每次完整验证的结果。
policy.train()  # 确保模型进入训练模式。

while step < total_steps:  # 遍历完训练集后重新读取，直到完成目标更新次数。
    for raw_batch in train_loader:  # 每一步取出新的训练批次。
        processed_batch = preprocessor(raw_batch)  # 使用训练统计量处理当前批次。
        optimizer.zero_grad(set_to_none=True)  # 清空上一批梯度。
        loss, metrics = policy(processed_batch)  # 计算训练总损失及 L1、KL 分项。
        if not torch.isfinite(loss).item():  # 检查训练损失是否正常。
            raise RuntimeError(f"第 {step + 1} 步损失异常。")  # 损失异常时停止更新。
        loss.backward()  # 反向计算梯度。
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0, error_if_nonfinite=True)  # 裁剪梯度并拒绝非有限梯度。
        optimizer.step()  # 执行一次参数更新。
        step += 1  # 记录实际完成的更新次数。
        seen_indices.update(raw_batch["index"].tolist())  # 记录实际参与过训练的输入索引。
        train_log.append({"step": step, "loss": loss.item(), "l1_loss": metrics["l1_loss"], "kld_loss": metrics["kld_loss"]})  # 保存本次训练指标。

        if step == 1 or step % 50 == 0:  # 第一步及之后每 50 步报告一次训练状态。
            print(f"Step {step}/{total_steps} | Loss={loss.item():.4f} | L1={metrics['l1_loss']:.4f} | KL={metrics['kld_loss']:.4f} | 已见输入={len(seen_indices)}/{len(train_dataset)}")  # 同时查看损失和数据覆盖。

        if step % val_interval == 0 or step == total_steps:  # 定期验证，并确保最后一步进行验证。
            last_val_mae = evaluate()  # 计算完整验证集的原始坐标误差。
            val_log.append({"step": step, "val_coordinate_mae": last_val_mae})  # 保存验证结果。
            print(f"Step {step} | 验证 MAE={last_val_mae:.4f}")  # 输出用于选择模型的指标。
            if last_val_mae < best_mae:  # 只有验证误差比历史最低值更小时才更新最佳模型。
                best_mae = last_val_mae  # 更新最低验证误差。
                best_step = step  # 记录最佳模型所在步骤。
                save_model(run_dir / "best", step, best_mae)  # 用更好的权重覆盖本次实验的 best。
                print(f"已更新最佳模型：Step {best_step}，MAE={best_mae:.4f}")  # 确认最佳模型已保存。

        if step >= total_steps:  # 达到目标更新次数后结束当前遍历。
            break  # 让外层循环依据步数结束训练。

save_model(run_dir / "last", step, last_val_mae)  # 保存最后一步模型，它不一定是验证表现最好的模型。
torch.save({"step": step, "optimizer_state_dict": optimizer.state_dict(), "torch_rng_state": torch.get_rng_state(), "cuda_rng_state_all": torch.cuda.get_rng_state_all()}, run_dir / "last" / "training_state.pt")  # 保存优化器及随机数状态，为后续继续训练准备。

for filename, rows in [("train_log.csv", train_log), ("val_log.csv", val_log)]:  # 分别保存训练日志和验证日志。
    with (run_dir / filename).open("w", newline="", encoding="utf-8-sig") as file:  # 创建方便 Windows 表格软件读取的文件。
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))  # 使用日志字典的字段作为列名。
        writer.writeheader()  # 写入列名。
        writer.writerows(rows)  # 写入全部记录。

run_info = {"seed": 0, "learning_rate": learning_rate, "completed_updates": step, "unique_train_inputs": len(seen_indices), "best_step": best_step, "best_val_mae": best_mae}  # 汇总本次实验的主要设置与结果。
(run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")  # 保存实验总结。
print("最佳模型步骤：", best_step)  # 查看最佳结果出现在训练的哪个阶段。
print("最低验证 MAE：", best_mae)  # 查看本轮验证误差最低值。
print("最佳模型目录：", run_dir / "best")  # 后续推理优先加载这里。
print("最后模型目录：", run_dir / "last")  # 后续继续训练可使用这里的模型和优化器状态。
print("训练与验证完成。")  # 确认训练、验证和保存全部结束。