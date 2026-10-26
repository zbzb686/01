from pathlib import Path  # 导入路径工具，用于定位项目和保存图片。

from torchvision.transforms.functional import to_pil_image  # 导入转换函数，将图像张量转换成可保存的图片。
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # 导入 LeRobot 的数据集读取类。

root = Path(__file__).parent  # 获取当前脚本所在的项目目录。

dataset = LeRobotDataset(  # 创建数据集对象，负责组织数值表、元数据和视频。
    repo_id="lerobot/pusht",  # 指定要读取的 PushT 数据集。
    root=root / "data" / "pusht",  # 使用项目内的数据目录，缺失的必要文件会自动下载。
    episodes=[0],  # 只选择第 0 段示范作为本次实验的数据。
    revision="7628202a2180972f291ba1bc6723834921e72c19",  # 保持与前面实验相同的数据版本。
    video_backend="pyav",  # 明确使用 PyAV 解码视频。
)  # 完成数据集初始化。

sample = dataset[0]  # 读取当前所选数据中的第一条样本，包含匹配的图像、状态和动作。
image = sample["observation.image"]  # 从样本字典中取出图像张量。

print("样本数量：", len(dataset))  # 查看选中的这段示范包含多少个时间步。
print("图像形状：", tuple(image.shape))  # 查看图像张量各个维度的大小。
print("图像类型：", image.dtype)  # 查看像素采用哪种数值类型。
print("像素范围：", image.min().item(), image.max().item())  # 查看图像中的最小和最大像素值。
print("当前状态：", sample["observation.state"].tolist())  # 将当前状态张量转成列表并打印。
print("示范动作：", sample["action"].tolist())  # 将作为学习目标的示范动作转成列表并打印。

output_dir = root / "outputs"  # 定义保存实验结果的文件夹。
output_dir.mkdir(parents=True, exist_ok=True)  # 创建输出文件夹，已经存在时继续使用。
image_path = output_dir / "aligned_episode0_frame0.png"  # 定义经过数据集接口读取的图片保存位置。
to_pil_image(image).save(image_path)  # 将图像张量转换成图片并保存为 PNG。
print("图片保存位置：", image_path)  # 打印图片路径，方便在 VS Code 中打开。