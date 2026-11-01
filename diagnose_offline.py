import json  # 读取实验快照，并保存本次诊断结果。
from datetime import datetime  # 用时间戳区分不同诊断结果。
import torch  # 进行模型推理、张量运算和固定位置抽样。
from torch.utils.data import DataLoader, Subset  # 将选定样本组成固定顺序的批次。
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # 使用与训练相同的数据集读取方式。
from pusht_project.data_io import PROJECT_ROOT, DATA_ROOT  # 复用项目根目录和原始数据目录。
from pusht_project.policy_io import load_act_bundle  # 复用模型及配套预处理、后处理器的加载函数。
from PIL import Image, ImageDraw  # 使用已经安装的Pillow，保存观测画面与动作对比图。

@torch.inference_mode()  # 整个函数不记录梯度，也不会进行参数更新。
def coordinate_mae(loader, policy, preprocessor, postprocessor):  # 计算指定数据加载器上的原始动作坐标误差。
    policy.eval()  # 使用推理模式，不走训练时的示范动作辅助分支。
    error_sum = 0.0  # 累积所有有效坐标的绝对误差。
    coordinate_count = 0  # 累积有效坐标数量。
    episode_total={}#记录每个episode的总误差

    for batch_number, raw_batch in enumerate(loader, start=1):  # 按固定顺序遍历当前子集。
        obs = {key: raw_batch[key] for key in policy.config.input_features}  # 只提供图像和当前位置，不把示范动作交给模型。
        processed_obs = preprocessor(obs)  # 使用模型配套的训练统计量标准化输入。
        normalized_prediction = policy.predict_action_chunk(processed_obs)  # 预测完整动作块，不通过动作执行缓存。
        prediction = postprocessor(normalized_prediction).detach().cpu()  # 将预测恢复为原始目标坐标。
        target = raw_batch["action"].cpu()  # 读取同一起点的示范动作块作为答案。
        valid_mask = (~raw_batch["action_is_pad"]).cpu().unsqueeze(-1)  # 标记真实动作，排除示范末尾的补齐位置。

        if prediction.shape != target.shape or not torch.isfinite(prediction).all().item():  # 检查预测形状和数值。
            raise RuntimeError("预测动作形状或数值异常。")  # 发现问题时停止，避免生成无效结果。

        valid_errors = (prediction - target).abs() * valid_mask  # 得到每个有效坐标的绝对误差。
        sample_errors = valid_errors.sum(dim=(1, 2))  # 对每条样本内部求和，保留批次维度。
        sample_counts = valid_mask.sum(dim=(1, 2)) * target.shape[-1]  # 分别计算每条样本的有效坐标数量。
        error_sum += ((prediction - target).abs() * valid_mask).sum().item()  # 累积有效位置的绝对坐标误差，不裁剪预测动作。
        coordinate_count += int(valid_mask.sum().item()) * target.shape[-1]  # 每个有效动作包含横、纵两个坐标.
        for i in range(raw_batch["episode_index"].shape[0]):
            summary=episode_total.get(int(raw_batch["episode_index"][i].item()),{"error_sum":0.0,"coordinate_count":0,"sample_count":0})
            summary["error_sum"]+=sample_errors[i].item()
            summary["coordinate_count"]+=sample_counts[i].item()
            summary["sample_count"]+=1
            episode_total[int(raw_batch["episode_index"][i].item())]=summary
            if int(raw_batch["episode_index"][i].item()) == 189:  # 只展开误差最高的验证示范，不改变其他统计。
                frame_index = int(raw_batch["frame_index"][i].item())  # 获取输入在这段示范内部的帧编号，从0开始。
                sample_mae = sample_errors[i].item() / sample_counts[i].item()  # 用有效坐标数量求平均，排除末尾补齐动作。
                print(f"Episode 189 | 帧={frame_index:3d} | 动作块MAE={sample_mae:.4f}")  # 按读取顺序展示各输入起点的预测误差。
                if frame_index == 70:  # 只查看已经定位到的高误差输入，不改变原来的评估流程。
                    pixels = raw_batch["observation.image"][i].detach().cpu()  # 获取模型标准化之前的原始观测，形状为(3,96,96)。
                    pixels = (pixels.permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).numpy()  # 转成Pillow需要的HWC整数图像。
                    original = Image.fromarray(pixels).resize((512, 512))  # 放大供人查看，模型实际输入仍然是96×96。
                    overlay = original.copy()  # 在副本上画动作，保留左侧原始画面。
                    painter = ImageDraw.Draw(overlay)  # 创建叠加图的画笔。
                    valid_steps = valid_mask[i, :, 0].bool()  # 获取这条输入对应的真实动作位置，排除补齐部分。
                    expected = target[i][valid_steps]  # 取出有效示范目标坐标，形状为(有效步数,2)。
                    predicted = prediction[i][valid_steps]  # 取出同一位置的模型预测，已经恢复成原始坐标。
                    for actions, color in [(expected, "green"), (predicted, "red")]:  # 绿色表示示范，红色表示预测。
                        points = [(float(point[0]), float(point[1])) for point in actions]  # 使用512×512场地坐标，纵坐标向下增加。
                        if len(points) > 1:  # 至少有两个目标时才能连接成线。
                            painter.line(points, fill=color, width=3)  # 按动作顺序连接目标位置，并非T块的实际运动轨迹。
                        for offset, (x, y) in enumerate(points):  # 遍历每个目标，并标注它在动作块里的位置。
                            painter.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)  # 用圆点表示目标位置。
                            painter.text((x + 4, y + 4), str(offset), fill=color)  # 标出0～9，便于对比相同时刻的目标。
                    x, y = raw_batch["observation.state"][i].detach().cpu().tolist()  # 获取输入时推手的真实当前位置。
                    painter.ellipse((x - 5, y - 5, x + 5, y + 5), fill="blue")  # 蓝点表示当前位置，区别于未来目标。
                    comparison = Image.new("RGB", (1024, 552), "white")  # 创建左右对照画布，上方留出文字区域。
                    comparison.paste(original, (0, 40))  # 左侧展示没有叠加动作的观测画面。
                    comparison.paste(overlay, (512, 40))  # 右侧展示同一画面上的目标动作。
                    labels = ImageDraw.Draw(comparison)  # 为整张对照图添加说明。
                    labels.text((10, 10), "Episode 189 | Frame 70 | Original observation", fill="black")  # 标注输入来源。
                    labels.text((522, 10), "Green: demo | Red: prediction | Blue: pusher", fill="black")  # 说明颜色含义。
                    output_path = PROJECT_ROOT / "outputs" / "offline_mae" / "episode_189_frame_70.png"  # 用固定文件名保存这个诊断样本。
                    output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保图片保存目录存在。
                    comparison.save(output_path)  # 保存图片，不修改模型或原始数据。
                    print("当前推手位置：", [x, y])  # 帮助判断第一个目标相对于当前位置的方向。
                    for offset, (demo, pred) in enumerate(zip(expected.tolist(), predicted.tolist())):  # 逐个对齐示范与预测目标。
                        print(f"动作{offset} | 示范={demo} | 预测={pred}")  # 输出坐标，避免仅凭线条重叠判断误差。
                    print("帧70对比图：", output_path)  # 输出图片路径，方便在VS Code中打开。

        if batch_number % 32 == 0:  # 每完成32批报告一次进度。
            print(f"  推理进度：{batch_number}/{len(loader)}批")  # 显示当前数据组的评估进度。

    if coordinate_count == 0:  # 确认子集中存在有效动作坐标。
        raise RuntimeError("没有可用于计算误差的有效动作。")  # 防止空结果除零。
    episode_results=[]
    for episode_id, summary in episode_total.items():
        episode_mae=summary["error_sum"]/summary["coordinate_count"] if summary["coordinate_count"]>0 else None
        episode_results.append({"episode_index":episode_id,"sample_count":summary["sample_count"],"coordinate_mae":episode_mae})
    return error_sum / coordinate_count , episode_results  # 按全部有效坐标求平均，避免直接平均不同大小批次的误差.


def main():  # 组织加载模型、选择样本、评估和保存的完整流程。
    model_dir = PROJECT_ROOT / "outputs" / "act_train_val_v2" / "20261029_060540_204544" / "best"  # 固定使用当前实验的最佳模型。
    experiment_dir = model_dir.parent  # best的父目录保存了本次实验的数据与配置快照。
    samples_per_group = 512  # 每组最多选择512条观测，先做规模较小的诊断。
    batch_size = 4  # 沿用本机已经验证可运行的推理批次大小。

    split = json.loads((experiment_dir / "split.json").read_text(encoding="utf-8"))  # 读取实验实际采用的训练与验证划分。
    settings = json.loads((experiment_dir / "train_config.json").read_text(encoding="utf-8"))  # 读取实验保存的采样频率等设置。
    checkpoint_info = json.loads((model_dir / "checkpoint_info.json").read_text(encoding="utf-8"))  # 读取当前最佳模型对应的训练步数。

    if set(split["train_episodes"]) & set(split["val_episodes"]):  # 检查两组是否包含相同示范。
        raise RuntimeError("训练与验证示范发生重叠。")  # 避免使用错误划分做诊断。

    policy, preprocessor, postprocessor = load_act_bundle(model_dir)  # 从同一目录加载模型和配套处理器。
    print("模型目录：", model_dir)  # 确认实际使用的模型来源。
    print("模型步骤：", checkpoint_info["step"])  # 当前预计是25000步，以保存记录为准。

    dataset_args = {  # 定义训练和验证共同使用的数据读取方式。
        "repo_id": split["dataset_repo"],  # 使用实验记录的数据仓库。
        "root": DATA_ROOT,  # 使用现有本地数据目录。
        "revision": split["dataset_revision"],  # 固定为实验使用的数据版本。
        "video_backend": "pyav",  # 沿用已验证的视频解码后端。
        "delta_timestamps": {"action": [i / settings["fps"] for i in range(policy.config.chunk_size)]},  # 为当前观测读取对应的完整动作块。
    }  # 完成共同数据设置。

    report = {  # 建立本次诊断的来源记录。
        "model_dir": str(model_dir),  # 保存模型位置。
        "checkpoint_info": checkpoint_info,  # 保存模型步数及其原完整验证结果。
        "dataset_repo": split["dataset_repo"],  # 保存数据集名称。
        "dataset_revision": split["dataset_revision"],  # 保存数据集版本。
        "batch_size": batch_size,  # 保存本次推理批次大小。
        "results": {},  # 后面填入训练和验证两组的抽样位置与误差。
    }  # 完成报告初始化。

    for group in ["train", "val"]:  # 分别处理训练组和验证组，始终使用同一份模型。
        dataset = LeRobotDataset(**dataset_args, episodes=split[f"{group}_episodes"])  # 根据划分创建当前组的数据集。
        if len(dataset) != split[f"{group}_frames"]:  # 检查实际数据长度与实验快照是否一致。
            raise RuntimeError(f"{group}数据长度与划分快照不一致。")  # 数据不匹配时停止。

        sample_count = min(samples_per_group, len(dataset))  # 数据不足512条时使用全部，否则取512条。
        indices = torch.linspace(0, len(dataset) - 1, steps=sample_count, dtype=torch.float64).round().long().tolist()  # 在整个组内均匀选取固定样本位置。
        subset = Subset(dataset, indices)  # 创建只包含选定位置的数据视图，不复制原始视频。
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)  # 按固定顺序分批读取，便于复现。

        print(f"\n开始评估{group}固定子集：{sample_count}条输入")  # 提示当前处理的数据组与规模。
        mae, episode_results = coordinate_mae(loader, policy, preprocessor, postprocessor)  # 使用相同推理路径计算当前组MAE。
        report["results"][group] = {"sample_count": sample_count, "indices": indices, "coordinate_mae": mae, "episode_results": episode_results}  # 保存实际样本位置和误差。
        print(f"{group}固定子集MAE：{mae:.4f}")  # 显示原始动作坐标单位的误差。

        if group=="val":
            worst_episodes=sorted(episode_results,key=lambda x:x["coordinate_mae"] if x["coordinate_mae"] is not None else -1,reverse=True)[:5]
            print("\n验证组误差最大的五个示范：")
            for episode_result in worst_episodes:
                print(f"Episode {episode_result['episode_index']}: sample_count={episode_result['sample_count']}, coordinate_mae={episode_result['coordinate_mae']:.4f}")

    output_path = PROJECT_ROOT / "outputs" / "offline_mae" / f"offline_mae_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"  # 为本次诊断生成独立输出文件名。
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保结果目录存在。
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")  # 保存模型来源、固定索引和两组结果。
    print("\n诊断结果保存位置：", output_path)  # 输出可供后续核对的结果文件路径。

if __name__ == "__main__":  # 直接运行本文件时才加载模型和开始诊断。
    main()  # 启动独立的离线诊断流程.