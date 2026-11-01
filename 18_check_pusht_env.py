from pathlib import Path  # 使用路径对象，把输出位置固定在当前项目中。
from importlib.metadata import version  # 读取已安装包的版本，方便以后复现和排错。
import numpy as np  # 构造二维目标位置，并检查数值是否正常。
import cv2  # 检查项目现有的 OpenCV 能否正常导入。
import gymnasium as gym  # 使用统一接口创建环境、重置环境和执行动作。
import gym_pusht  # 导入时注册 PushT 环境，让 gym.make 能找到它。
from PIL import Image  # 将仿真返回的 RGB 数组保存为 PNG 图片。

project_dir = Path(__file__).resolve().parent  # 获取本脚本所在的项目目录。
output_dir = project_dir / "outputs"  # 指定实验图片的保存目录。
output_dir.mkdir(parents=True, exist_ok=True)  # 目录不存在时创建，已经存在时保留原有内容。

print("gym-pusht 版本：", version("gym-pusht"))  # 确认本次实际导入的环境包版本。
print("OpenCV 版本：", cv2.__version__)  # 确认现有 cv2 可以导入。

env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")  # 创建同时返回图像和推手位置的环境；画面以数组返回。

try:  # 无论测试成功还是发生异常，最后都尝试释放环境资源。
    observation, info = env.reset(seed=0)  # 用固定随机种子生成初始场景，方便重复检查。
    print("观测字段：", list(observation.keys()))  # 预期包含 pixels 和 agent_pos。
    print("图像形状：", observation["pixels"].shape)  # 预期为高、宽、RGB 三通道，即 (96, 96, 3)。
    print("图像数据类型：", observation["pixels"].dtype)  # 预期为 uint8，颜色值范围为 0～255。
    print("初始推手位置：", observation["agent_pos"].tolist())  # 显示仿真坐标系中的当前位置。
    assert env.observation_space.contains(observation), "初始观测不符合环境声明"  # 检查观测的字段、形状、类型和范围。

    offset = np.array([20.0, 0.0], dtype=np.float32)  # 为测试设置目标：从当前位置向 x 增大方向偏移 20 个坐标单位。
    action = np.clip(observation["agent_pos"] + offset, env.action_space.low, env.action_space.high).astype(np.float32)  # 将目标限制在合法动作范围内，并转换成要求的数据类型。
    assert env.action_space.contains(action), "动作不符合环境声明"  # 检查二维目标位置是否合法。
    print("本次目标位置：", action.tolist())  # 动作表示目标坐标，推手不一定在一次执行后就到达目标。

    next_observation, reward, terminated, truncated, info = env.step(action)  # 执行动作，获得新观测、奖励、终止标记、截断标记和附加信息。
    assert env.observation_space.contains(next_observation), "执行后的观测不符合环境声明"  # 检查执行动作后环境是否仍返回合法观测。
    print("执行后推手位置：", next_observation["agent_pos"].tolist())  # 与初始位置比较，确认环境进行了更新。
    print("奖励：", float(reward))  # 奖励描述当前任务表现；一次测试动作不要求取得高奖励。
    print("任务终止：", terminated, "达到时间限制：", truncated)  # 分别显示任务结束和时间限制等截断情况。

    frame = env.render()  # 获取供人查看的 RGB 画面，默认分辨率比模型输入图像更大。
    assert frame is not None and np.isfinite(frame).all(), "渲染画面为空或含异常数值"  # 检查渲染是否产生有效数据。
    image_path = output_dir / "pusht_env_check.png"  # 指定本次测试图片的保存路径。
    Image.fromarray(frame).save(image_path)  # 直接按 RGB 保存，避免额外转换造成颜色颠倒。
    print("渲染画面形状：", frame.shape)  # 默认预期为 (680, 680, 3)，与模型输入的 96×96 图像用途不同。
    print("测试图片保存位置：", image_path)  # 输出绝对路径，方便在 VS Code 中找到图片。
    print("PushT 环境最小测试通过。")  # 只有前面的创建、执行和保存都成功，才会打印这一行。
finally:  # 即使中途报错，也执行下面的清理操作。
    env.close()  # 关闭环境并释放它占用的窗口等资源。