from pathlib import Path  # 定位项目目录和模型文件。
import numpy as np  # 裁剪动作，使目标坐标处于环境允许的范围。
import torch  # 构造模型输入并执行 GPU 推理。
import gymnasium as gym  # 创建仿真环境并执行动作。
import gym_pusht  # 注册 PushT 环境。
from PIL import Image  # 保存执行模型动作后的仿真画面。
from lerobot.policies.act.modeling_act import ACTPolicy  # 加载已经训练好的 ACT。
from lerobot.processor import PolicyProcessorPipeline  # 加载训练时保存的处理器。
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action  # 在动作张量和处理器内部格式之间转换。

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

env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")  # 创建与上一轮相同的环境。

try:  # 确保测试结束或发生异常时释放环境资源。
    observation, info = env.reset(seed=0)  # 沿用上一轮初始场景，便于比较。
    image = torch.from_numpy(observation["pixels"].copy()).permute(2, 0, 1).to(torch.float32) / 255.0  # 将 RGB 图像从 HWC 整数转换成 CHW 浮点数，范围为 0～1。
    state = torch.as_tensor(observation["agent_pos"], dtype=torch.float32)  # 将二维推手位置转换成模型使用的浮点张量。
    model_input = {  # 将仿真字段映射为训练时的字段名称。
        "observation.image": image.unsqueeze(0),  # 添加批次维度，得到 (1,3,96,96)。
        "observation.state": state.unsqueeze(0),  # 添加批次维度，得到 (1,2)。
    }  # 仅提供当前图像和位置。
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

    output_path = root / "outputs" / "act_env_step.png"  # 指定执行模型动作后的图片路径。
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    Image.fromarray(env.render()).save(output_path)  # 保存动作执行后的 RGB 画面。
    print("图片保存位置：", output_path)  # 方便在 VS Code 中打开图片。
    print("ACT 与仿真的单步接口测试通过。")  # 表示加载、预测、动作执行和图片保存都已完成。
finally:  # 成功或报错后都执行清理。
    env.close()  # 释放环境资源。