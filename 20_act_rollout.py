from pathlib import Path  # 定位项目目录和模型文件。
import numpy as np  # 裁剪动作，使目标坐标处于环境允许的范围。
import torch  # 构造模型输入并执行 GPU 推理。
import gymnasium as gym  # 创建仿真环境并执行动作。
import gym_pusht  # 注册 PushT 环境。
from PIL import Image  # 保存执行模型动作后的仿真画面。
from lerobot.policies.act.modeling_act import ACTPolicy  # 加载已经训练好的 ACT。
from lerobot.processor import PolicyProcessorPipeline  # 加载训练时保存的处理器。
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action  # 在动作张量和处理器内部格式之间转换。
import csv  # 保存每一步的奖励、覆盖率和目标坐标，方便排查问题。
import json  # 保存本次仿真的模型来源和结果。
from datetime import datetime  # 使用时间戳区分不同实验，保留之前的结果。
root = Path(__file__).resolve().parent  # 获取当前项目目录。
model_dir = root / "outputs" / "act_train_val" / "20261028_070042_052971" / "best"  # 使用验证误差最低的第 200 步模型。
assert model_dir.is_dir(), f"模型目录不存在：{model_dir}"  # 路径错误时明确停止。
assert torch.cuda.is_available(), "当前解释器无法使用 CUDA"  # 确认本次仍使用已验证的 GPU 环境。

policy = ACTPolicy.from_pretrained(model_dir, local_files_only=True, strict=True).to("cuda")  # 从本地加载配置和权重，并放到 GPU。
policy.eval()  # 切换为推理模式。
policy.reset()  # 清空动作缓存，开始新的控制过程。

preprocessor = PolicyProcessorPipeline.from_pretrained(  # 加载与这份权重配套的输入处理器。
    model_dir,  # 使用同一个 best 目录中的统计量。
    config_filename="policy_preprocessor.json",  # 指定输入处理器配置。
    local_files_only=True,  # 仅从本地读取文件。
)  # 完成输入处理器加载。

postprocessor = PolicyProcessorPipeline.from_pretrained(  # 加载动作后处理器，把标准化输出恢复为目标坐标。
    model_dir,  # 必须与模型和输入处理器来自同一目录。
    config_filename="policy_postprocessor.json",  # 指定动作后处理器配置。
    local_files_only=True,  # 仅使用本地文件。
    to_transition=policy_action_to_transition,  # 将动作张量转为处理器内部格式。
    to_output=transition_to_policy_action,  # 将处理结果转回动作张量。
)  # 完成动作后处理器加载。

seed = 0  # 沿用之前的初始场景，本轮只测试这一个场景。
fps = 10  # 环境每秒推进 10 个控制步，动画也按每秒 10 帧播放。
run_dir = root / "outputs" / "act_rollouts" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 为本次运行设置独立目录。
run_dir.mkdir(parents=True, exist_ok=False)  # 创建目录，避免覆盖之前的实验。
env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array", visualization_width=256, visualization_height=256)  # 模型观测仍为96×96，仅将观看画面设为256×256。
max_steps = int(env.spec.max_episode_steps)  # 读取环境默认回合上限，当前为300步。
frames = []  # 保存每一步的观看画面，用于生成动画。
rows = []  # 保存每一步的数值记录。
success = False  # 记录本回合是否曾达到任务成功条件。
max_reward = 0.0  # 记录本回合执行过程中出现的最高奖励。
max_coverage = 0.0  # 记录本回合执行过程中出现的最高覆盖率。
clipped_steps = 0  # 记录有多少步预测目标超出范围，需要裁剪。

try:  # 无论运行成功还是出现异常，最后都释放环境资源。
    observation, info = env.reset(seed=seed)  # 生成固定种子的初始场景。
    policy.reset()  # 只在新回合开始时清空模型动作缓存。
    frames.append(Image.fromarray(env.render()))  # 记录尚未执行动作的初始画面。
    print("模型目录：", model_dir)  # 确认本次使用之前保存的best模型。
    print("随机种子：", seed, "最多执行步数：", max_steps)  # 明确本回合设置。
    print("每组执行动作数：", policy.config.n_action_steps)  # 当前应为10，表示一组动作的执行长度。

    for step in range(1, max_steps + 1):  # 持续执行动作，直到成功或达到回合上限。
        image = torch.from_numpy(observation["pixels"].copy()).permute(2, 0, 1).to(torch.float32) / 255.0  # 将当前图像转为CHW浮点数，范围为0～1。
        state = torch.as_tensor(observation["agent_pos"], dtype=torch.float32)  # 使用执行上一动作后得到的最新位置。
        model_input = {  # 将当前仿真观测映射为模型的输入字段。
            "observation.image": image.unsqueeze(0),  # 添加批次维度，形状为(1,3,96,96)。
            "observation.state": state.unsqueeze(0),  # 添加批次维度，形状为(1,2)。
        }  # 完成当前步输入构造。
        processed_input = preprocessor(model_input)  # 使用配套统计量标准化当前观测并移动到GPU。

        with torch.inference_mode():  # 仅推理，不计算梯度或更新模型参数。
            normalized_action = policy.select_action(processed_input)  # 从缓存取一个动作；缓存耗尽时才根据当前观测预测新的一组。
            predicted_action = postprocessor(normalized_action)  # 将这个动作恢复为原始目标坐标，形状为(1,2)。

        assert tuple(predicted_action.shape) == (1, 2), "动作形状异常"  # 确认返回的是一个二维目标位置。
        assert torch.isfinite(predicted_action).all().item(), "动作含异常数值"  # 禁止将NaN或无穷大交给环境。
        raw_action = predicted_action[0].detach().cpu().numpy()  # 去掉批次维度，转换为环境使用的NumPy数组。
        action = np.clip(raw_action, env.action_space.low, env.action_space.high).astype(np.float32)  # 将预测目标限制在合法范围内。
        clipped = not np.array_equal(raw_action, action)  # 判断本步目标是否经过范围裁剪。
        clipped_steps += int(clipped)  # 累计发生裁剪的步数。
        assert env.action_space.contains(action), "动作不符合环境要求"  # 执行前检查目标坐标是否合法。

        observation, reward, terminated, truncated, info = env.step(action)  # 执行动作，并用返回的新观测替换旧观测。
        reward = float(reward)  # 将奖励转换为普通Python数值，方便记录。
        coverage = float(info["coverage"])  # 读取真实覆盖率，避免把奖励直接当覆盖率。
        success = success or bool(info["is_success"])  # 使用环境给出的成功标记判断任务是否完成。
        max_reward = max(max_reward, reward)  # 更新最高奖励，它可能高于最后一步奖励。
        max_coverage = max(max_coverage, coverage)  # 更新最高覆盖率。
        rows.append({"step": step, "reward": reward, "coverage": coverage, "success": bool(info["is_success"]), "target_x": float(action[0]), "target_y": float(action[1]), "clipped": clipped})  # 保存本步结果与实际执行目标。
        frames.append(Image.fromarray(env.render()))  # 保存动作执行后的画面。

        if step == 1 or step % 20 == 0 or terminated or truncated:  # 第一步、每20步以及回合结束时打印进度。
            print(f"Step {step}/{max_steps} | 奖励={reward:.4f} | 覆盖率={coverage:.2%} | 成功={success}")  # 同时查看奖励与覆盖率。
        if terminated or truncated:  # 达到任务结束条件或时间上限后停止。
            break  # 不在已经结束的回合上继续调用step。

    log_path = run_dir / "step_log.csv"  # 指定逐步记录的保存位置。
    with log_path.open("w", newline="", encoding="utf-8-sig") as file:  # 创建Windows表格软件也能读取的CSV文件。
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))  # 使用记录字段作为表格列名。
        writer.writeheader()  # 写入表头。
        writer.writerows(rows)  # 写入本回合所有动作步的结果。

    summary = {  # 汇总本次实验的来源、设置和表现。
        "model_dir": str(model_dir), "seed": seed, "fps": fps,  # 记录模型来源、初始场景和控制频率。
        "n_action_steps": policy.config.n_action_steps, "steps": len(rows),  # 记录每组动作数和实际执行步数。
        "success": success, "max_reward": max_reward, "final_reward": reward,  # 记录是否成功、最高奖励和最终奖励。
        "max_coverage": max_coverage, "final_coverage": coverage,  # 记录最高与最终覆盖率。
        "clipped_steps": clipped_steps, "terminated": bool(terminated), "truncated": bool(truncated),  # 记录裁剪情况及结束原因。
    }  # 完成本次结果汇总。
    (run_dir / "run_info.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")  # 保存实验结果，方便后续比较。

    gif_path = run_dir / "rollout.gif"  # 指定动画保存位置。
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0, optimize=False)  # 按每帧100毫秒保存循环播放的GIF动画。
    print("实际执行步数：", len(rows))  # 显示本回合执行了多少个动作。
    print("是否成功：", success)  # 单独报告环境判定的成功结果。
    print("最高奖励：", max_reward, "最终奖励：", reward)  # 比较过程中最好表现与结束时表现。
    print("最高覆盖率：", f"{max_coverage:.2%}")  # 显示任务过程中最接近目标的程度。
    print("动作裁剪步数：", clipped_steps)  # 观察模型是否经常输出超范围目标。
    print("实验目录：", run_dir)  # 此目录包含动画、逐步日志和实验汇总。
    print("动画保存位置：", gif_path)  # 在VS Code中找到并打开动画。
finally:  # 无论正常结束还是报错，都执行清理。
    env.close()  # 释放仿真资源。