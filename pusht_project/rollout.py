import numpy as np  # 转换和裁剪动作目标坐标。
import torch  # 进行模型推理和数值检查。
from PIL import Image  # 将渲染数组转换为可保存的画面。
from .observations import observation_to_batch  # 从同一个包中复用观测转换函数。

def rollout_episode(env, policy, preprocessor, postprocessor, seed, record_frames=True):  # 使用传入的环境和模型完成一个仿真回合。
    observation, info = env.reset(seed=seed)  # 根据种子生成当前回合的初始场景。
    policy.reset()  # 清空上一回合剩余的动作缓存。
    max_steps = int(env.spec.max_episode_steps)  # 读取环境规定的回合步数上限。
    frames = []  # 保存需要返回给入口的画面。
    steps_records = []  # 保存本回合每一步的动作、推手位置和覆盖率，用于事后分析失败过程。
    if record_frames:  # 由调用者决定本回合是否收集画面。
        frames.append(Image.fromarray(env.render()))  # 保存执行动作前的初始画面。
    success = False  # 初始化本回合的成功标记。
    max_reward = 0.0  # 初始化本回合的最高奖励。
    max_coverage = 0.0  # 初始化本回合的最高覆盖率。
    clipped_steps = 0  # 初始化本回合的动作裁剪次数。
    for step in range(1, max_steps + 1):  # 在当前场景中连续控制，最多执行300步。
        model_input = observation_to_batch(observation)  # 使用公共函数转换本步最新观测。
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
        steps_records.append({  # 记录本步执行后的完整状态，供逐步分析。
        "step": step,  # 当前步数，从1开始。
        "action_x": float(action[0]), "action_y": float(action[1]),  # 本步实际执行的目标位置（裁剪后），即模型"命令"推手去哪。
        "pusher_x": float(observation["agent_pos"][0]), "pusher_y": float(observation["agent_pos"][1]),  # 本步执行后的推手实际位置。
        "coverage": coverage, "reward": reward,  # 本步执行后的覆盖率与奖励。
        })  # 完成本步记录，随后继续循环.

        if record_frames:  # 根据调用者的要求决定是否收集画面。
            frames.append(Image.fromarray(env.render()))  # 保存本步执行后的画面。

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
    # 在这里搬入原脚本的单回合循环，以及循环后的result字典构造。
    return result, frames, steps_records  # 返回数值结果和画面，由入口决定怎样保存。