from pathlib import Path  # 定位项目目录和模型文件。
import numpy as np  # 裁剪动作，使目标坐标处于环境允许的范围。
import torch  # 构造模型输入并执行 GPU 推理。
import gymnasium as gym  # 创建仿真环境并执行动作。
import gym_pusht  # 注册 PushT 环境。
from PIL import Image  # 保存执行模型动作后的仿真画面。
from pusht_project.policy_io import load_act_bundle  # 复用模型和配套处理器的加载流程。
from pusht_project.rollout import rollout_episode  # 复用一个完整回合的控制流程。
import csv  # 保存每一步的奖励、覆盖率和目标坐标，方便排查问题。
import json  # 保存本次仿真的模型来源和结果。
from datetime import datetime  # 使用时间戳区分不同实验，保留之前的结果。
def main():
    root = Path(__file__).resolve().parent  # 获取当前项目目录。
    model_dir = root / "outputs" / "act_train_val_v2" /"20261029_195213_819619"  / "best"  # 使用验证误差最低的第 200 步模型。
    assert model_dir.is_dir(), f"模型目录不存在：{model_dir}"  # 路径错误时明确停止。
    assert torch.cuda.is_available(), "当前解释器无法使用 CUDA"  # 确认本次仍使用已验证的 GPU 环境。

    policy, preprocessor, postprocessor = load_act_bundle(model_dir)  # 从指定目录加载三个配套对象。
    policy.config.n_action_steps = 6  # 每次预测只执行首个动作，下一步根据最新观测重新预测。
    policy.eval()  # 切换为推理模式。
    policy.reset()  # 清空动作缓存，开始新的控制过程。

    seeds =list( range(0,25))  # 固定五个初始场景，后续实验也使用同一组编号进行比较。
    fps = 10  # 动画播放频率与环境的10Hz控制频率保持一致。
    run_dir = root / "outputs" / "act_eval" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 为本次五场景评估创建独立路径。
    run_dir.mkdir(parents=True, exist_ok=False)  # 创建输出目录并保留历史结果。
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array", visualization_width=256, visualization_height=256)  # 保持原有观测和控制设置。
    max_steps = int(env.spec.max_episode_steps)  # 使用环境默认的300步回合上限。
    results = []  # 每个场景结束后，追加一条汇总结果。

    try:  # 确保评估结束或发生异常时关闭环境。
        for seed in seeds:  # 依次测试五个固定场景。
            result, frames,steps_records = rollout_episode(env, policy, preprocessor, postprocessor, seed=seed, record_frames=True)  # 委托公共函数执行一个回合。
            results.append(result)  # 将这个回合的结果加入多场景汇总列表。
            with (run_dir / f"steps_seed{seed}.csv").open("w", newline="", encoding="utf-8-sig") as file:  # 为当前场景单独保存逐步记录。
                writer = csv.DictWriter(file, fieldnames=["step", "action_x", "action_y", "pusher_x", "pusher_y", "coverage", "reward"])  # 按固定顺序写列名和数据。
                writer.writeheader()  # 写入表头行。
                writer.writerows(steps_records)  # 写入本场景全部300步记录。
   # 后面继续保留原来的CSV写入、GIF保存和当前场景结果提示。
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
if __name__ == "__main__":  # 直接运行本文件时才开始评估。
    main()  # 启动多场景评估主流程。    