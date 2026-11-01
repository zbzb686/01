import numpy as np  # 构造用于检查的仿真格式数组。
import torch  # 检查输出张量的数据类型和数值。
from pusht_project.observations import observation_to_batch  # 导入你自己提取的观测转换函数。

observation = {  # 构造一条简单观测，使正确结果容易判断。
    "pixels": np.full((96, 96, 3), 255, dtype=np.uint8),  # 全白图像经过除以255后，应当全部等于1。
    "agent_pos": np.array([390.0, 304.0], dtype=np.float64),  # 故意使用float64，检查函数是否转换为float32。
}  # 完成测试输入构造。

batch = observation_to_batch(observation)  # 调用你写的函数，取得转换后的模型输入。
image = batch["observation.image"]  # 取出带批次维度的图像。
state = batch["observation.state"]  # 取出带批次维度的位置。

print("图像形状与类型：", tuple(image.shape), image.dtype)  # 预期为(1,3,96,96)和torch.float32。
print("图像最小值与最大值：", image.min().item(), image.max().item())  # 全白图像转换后，两个值都应为1。
print("位置形状与类型：", tuple(state.shape), state.dtype)  # 预期为(1,2)和torch.float32。
print("位置数值：", state.tolist())  # 预期仍为[[390.0,304.0]]。

assert tuple(image.shape) == (1, 3, 96, 96), "图像维度不正确"  # 检查批次、通道、高、宽的排列。
assert image.dtype == torch.float32 and torch.all(image == 1).item(), "图像类型或缩放不正确"  # 检查整数图像是否正确转为浮点数并缩放。
assert tuple(state.shape) == (1, 2) and state.dtype == torch.float32, "位置维度或类型不正确"  # 检查位置的批次维度和类型。
assert torch.equal(state, torch.tensor([[390.0, 304.0]])), "位置坐标被意外改变"  # 防止错误地对位置也除以255。
print("观测转换函数检查通过。")  # 只有所有检查通过后才打印这句话。