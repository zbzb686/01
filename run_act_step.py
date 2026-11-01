from pathlib import Path  # 定位项目目录和模型文件。
import numpy as np  # 将预测动作转换并限制到环境允许的范围。
import torch  # 进行推理和数值检查。
import gymnasium as gym  # 创建仿真环境。
import gym_pusht  # 注册PushT环境。
from PIL import Image  # 保存执行动作后的画面。
from pusht_project.policy_io import load_act_bundle  # 复用模型与处理器加载函数。
from pusht_project.observations import observation_to_batch  # 复用观测转换函数。

def main():  # 将本次运行的组织流程放在一个函数中。
    root = Path(__file__).resolve().parent  # 获取入口脚本所在的项目目录。
    model_dir = root / "outputs" / "act_train_val" / "20261028_070042_052971" / "best"  # 指定要运行的模型。
    assert model_dir.is_dir(), f"模型目录不存在：{model_dir}"  # 路径错误时给出明确提示。
    assert torch.cuda.is_available(), "当前解释器无法使用CUDA"  # 保留原脚本的GPU要求。
    policy, preprocessor, postprocessor = load_act_bundle(model_dir)  # 加载模型和配套处理器。
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")  # 创建与上一轮相同的环境。

    try:  # 确保测试结束或发生异常时释放环境资源。
        observation, info = env.reset(seed=0)  # 沿用上一轮初始场景，便于比较。
        image = torch.from_numpy(observation["pixels"].copy()).permute(2, 0, 1).to(torch.float32) / 255.0  # 将 RGB 图像从 HWC 整数转换成 CHW 浮点数，范围为 0～1。
        state = torch.as_tensor(observation["agent_pos"], dtype=torch.float32)  # 将二维推手位置转换成模型使用的浮点张量。
        model_input = observation_to_batch(observation)  # 使用公共函数转换当前仿真观测。
        processed_input = preprocessor(model_input)  # 使用训练集统计量标准化输入，并移动到配置的设备。

        with torch.inference_mode():  # 关闭梯度记录，本次只推理。
            normalized_actions = policy.predict_action_chunk(processed_input)  # 预测标准化动作块，预期形状为 (1,10,2)。
            predicted_actions = postprocessor(normalized_actions)  # 恢复为原始目标坐标，并由处理器移到 CPU。

        assert tuple(predicted_actions.shape) == (1, 10, 2), "预测动作形状异常"  # 核对批次大小、动作数量和坐标维度。
        assert torch.isfinite(predicted_actions).all().item(), "预测动作含异常数值"  # 避免将 NaN 或无穷大送入环境。
        raw_action = predicted_actions[0, 0].detach().cpu().numpy()  # 取这个动作块的第一个二维目标位置。
        action = np.clip(raw_action, env.action_space.low, env.action_space.high).astype(np.float32)  # 将目标坐标限制在环境允许的范围内。
        assert env.action_space.contains(action), "动作不符合环境要求"  # 执行前检查形状、类型和范围。

        print("加载的模型目录：", model_dir)  # 确认实际加载的是 best。
        print("模型设备：", next(policy.parameters()).device)  # 预期显示 cuda:0。
        print("输入图像形状：", tuple(model_input["observation.image"].shape))  # 预期为 (1,3,96,96)。
        print("输入状态形状：", tuple(model_input["observation.state"].shape))  # 预期为 (1,2)。
        print("预测动作形状：", tuple(predicted_actions.shape))  # 预期为 (1,10,2)。
        print("当前推手位置：", observation["agent_pos"].tolist())  # 固定种子下应与上一轮初始位置一致。
        print("模型预测的首个目标：", raw_action.tolist())  # 查看模型给出的原始目标坐标。
        print("实际执行的目标：", action.tolist())  # 如果预测超出范围，这里会显示裁剪后的坐标。

        next_observation, reward, terminated, truncated, info = env.step(action)  # 将模型预测的目标交给物理仿真执行一次。
        print("执行后推手位置：", next_observation["agent_pos"].tolist())  # 观察模型动作如何改变环境。
        print("奖励：", float(reward))  # 记录这一步的任务奖励。
        print("任务终止：", terminated, "达到时间限制：", truncated)  # 查看这一步是否结束了回合。
        output_path = root / "outputs" / "act_env_step_reuse.png"  # 指定执行模型动作后的图片路径。
        output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
        Image.fromarray(env.render()).save(output_path)  # 保存动作执行后的 RGB 画面。
        print("图片保存位置：", output_path)  # 方便在 VS Code 中打开图片。
        print("ACT 与仿真的单步接口测试通过。")  # 表示加载、预测、动作执行和图片保存都已完成。
    finally:  # 成功或报错后都执行清理。
        env.close()  # 释放环境资源。    # 在这里搬入原脚本中的环境创建、单步预测、执行、保存和资源清理代码。

if __name__ == "__main__":  # 只有直接运行本文件时，才启动下面的主流程。
    main()  # 调用主函数，开始本次任务。