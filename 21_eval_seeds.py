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

seeds = [0, 1, 2, 3, 4]  # 固定五个初始场景，后续实验也使用同一组编号进行比较。
fps = 10  # 动画播放频率与环境的10Hz控制频率保持一致。
run_dir = root / "outputs" / "act_eval" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 为本次五场景评估创建独立路径。
run_dir.mkdir(parents=True, exist_ok=False)  # 创建输出目录并保留历史结果。
env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array", visualization_width=256, visualization_height=256)  # 保持原有观测和控制设置。
max_steps = int(env.spec.max_episode_steps)  # 使用环境默认的300步回合上限。
results = []  # 每个场景结束后，追加一条汇总结果。

try:  # 确保评估结束或发生异常时关闭环境。
    for seed in seeds:  # 依次测试五个固定场景。
        observation, info = env.reset(seed=seed)  # 根据当前编号重新生成初始场景。
        policy.reset()  # 清空上一回合剩余的动作，防止不同场景之间串用缓存。
        frames = [Image.fromarray(env.render())]  # 记录当前场景的初始画面。
        success = False  # 为当前场景重新初始化成功标记。
        max_reward = 0.0  # 为当前场景重新初始化最高奖励。
        max_coverage = 0.0  # 为当前场景重新初始化最高覆盖率。
        clipped_steps = 0  # 为当前场景重新初始化动作裁剪计数。
        print(f"\n开始评估 seed={seed}")  # 明确当前正在运行哪个场景。

        for step in range(1, max_steps + 1):  # 在当前场景中连续控制，最多执行300步。
            image = torch.from_numpy(observation["pixels"].copy()).permute(2, 0, 1).to(torch.float32) / 255.0  # 将当前RGB图像转为CHW浮点数，范围为0～1。
            state = torch.as_tensor(observation["agent_pos"], dtype=torch.float32)  # 将当前推手位置转换为浮点张量。
            model_input = {  # 构造与训练时一致的输入字段。
                "observation.image": image.unsqueeze(0),  # 图像批次形状为(1,3,96,96)。
                "observation.state": state.unsqueeze(0),  # 状态批次形状为(1,2)。
            }  # 完成当前步的模型输入构造。
            processed_input = preprocessor(model_input)  # 使用训练时保存的统计量处理观测。

            with torch.inference_mode():  # 只做推理，不更新模型参数。
                normalized_action = policy.select_action(processed_input)  # 按原有配置，每组10个动作依次执行。
                predicted_action = postprocessor(normalized_action)  # 将当前动作恢复为原始目标坐标。

            assert tuple(predicted_action.shape) == (1, 2), "动作形状异常"  # 确认当前返回一个二维动作。
            assert torch.isfinite(predicted_action).all().item(), "动作包含异常数值"  # 拒绝NaN或无穷大动作。
            raw_action = predicted_action[0].detach().cpu().numpy()  # 取出当前动作并转换为NumPy数组。
            action = np.clip(raw_action, env.action_space.low, env.action_space.high).astype(np.float32)  # 将目标位置限制在合法范围。
            clipped_steps += int(not np.array_equal(raw_action, action))  # 累计发生裁剪的动作数量。
            assert env.action_space.contains(action), "动作不符合环境要求"  # 执行前检查动作类型、形状和范围。

            observation, reward, terminated, truncated, info = env.step(action)  # 执行动作，并保存最新观测供下一步使用。
            reward = float(reward)  # 转为普通Python数值，便于输出和保存。
            coverage = float(info["coverage"])  # 获取当前T形块与目标区域的覆盖率。
            success = success or bool(info["is_success"])  # 记录当前场景是否达到过成功条件。
            max_reward = max(max_reward, reward)  # 更新当前场景的最高奖励。
            max_coverage = max(max_coverage, coverage)  # 更新当前场景的最高覆盖率。
            frames.append(Image.fromarray(env.render()))  # 保存当前画面，便于检查停滞或错误推动现象。

            if step == 1 or step % 100 == 0 or terminated or truncated:  # 定期打印进度，减少重复输出。
                print(f"seed={seed} | Step {step}/{max_steps} | 覆盖率={coverage:.2%} | 成功={success}")  # 显示当前场景的执行情况。
            if terminated or truncated:  # 检查是否成功结束或达到回合时间上限。
                break  # 结束当前场景，随后切换下一个seed。

        result = {  # 汇总刚刚完成的一个场景。
            "seed": seed, "steps": step, "success": success,  # 记录初始场景编号、实际步数和是否成功。
            "max_reward": max_reward, "final_reward": reward,  # 分别记录过程最高奖励和结束时奖励。
            "max_coverage": max_coverage, "final_coverage": coverage,  # 分别记录过程最高覆盖率和结束时覆盖率。
            "clipped_steps": clipped_steps, "terminated": bool(terminated), "truncated": bool(truncated),  # 记录裁剪次数和回合结束原因。
        }  # 完成当前场景的结果整理。
        results.append(result)  # 将当前结果加入五场景汇总表。

        with (run_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as file:  # 每完成一个场景就更新CSV，保留已完成的结果。
            writer = csv.DictWriter(file, fieldnames=list(result.keys()))  # 使用结果字典的字段作为表头。
            writer.writeheader()  # 写入列名。
            writer.writerows(results)  # 写入截至目前已经完成的所有场景。

        gif_path = run_dir / f"seed_{seed}.gif"  # 为每个场景分别保存动画。
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0, optimize=False)  # 按10fps保存当前场景的完整动画。
        print("当前场景动画：", gif_path)  # 方便找到并检查该场景的行为。

    success_count = sum(int(item["success"]) for item in results)  # 统计这五个场景中成功的数量。
    success_rate = success_count / len(results)  # 计算这组固定场景中的成功比例。
    summary = {  # 记录评估设置和汇总结果。
        "model_dir": str(model_dir), "seeds": seeds, "fps": fps,  # 保存模型来源和场景列表。
        "n_action_steps": policy.config.n_action_steps, "max_steps": max_steps,  # 保存动作执行配置和回合上限。
        "success_count": success_count, "success_rate": success_rate, "results": results,  # 保存成功数量、比例和每个场景的详细结果。
    }  # 完成汇总记录。
    (run_dir / "run_info.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")  # 保存可供后续比较的评估记录。

    print("\nseed | 成功 | 最高覆盖率 | 最终覆盖率")  # 输出最终结果表的标题。
    for item in results:  # 逐个展示五个场景的结果。
        print(f"{item['seed']:4d} | {str(item['success']):5s} | {item['max_coverage']:.2%} | {item['final_coverage']:.2%}")  # 方便比较不同场景的表现。
    print(f"本组场景成功数量：{success_count}/{len(results)}，成功比例：{success_rate:.0%}")  # 只报告本组五个场景中的结果。
    print("评估结果目录：", run_dir)  # 此目录包含CSV、实验记录和五段动画。
finally:  # 正常结束或报错后都执行清理。
    env.close()  # 释放仿真环境资源。