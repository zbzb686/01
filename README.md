# LeRobot + ACT + SmolVLA 入门实验

以用户亲自动手为主的机器人学习项目。通过读取 PushT 示范、构造动作序列和理解 ACT，逐步过渡到 SmolVLA。

## 当前进度

已完成 Windows GPU 验证、PushT 数据探索、动作序列批次构造，以及 ACT 在 CUDA 上的一次前向计算。尚未开始参数更新、正式训练或仿真评估；SmolVLA 阶段尚未开始。

逐行中文注释和详细补课要求见 [AGENTS.md](AGENTS.md)，具体结果及待讲知识见 [PROJECT_MEMORY.md](PROJECT_MEMORY.md)。Claude Code 通过 [CLAUDE.md](CLAUDE.md) 导入两份共同文档。

## 文件说明

| 文件 | 内容 |
| --- | --- |
| `01_inspect_dataset.py` | 读取数据集说明 |
| `02_inspect_episode.py` | 查看示范的状态、动作和时间戳 |
| `03_view_first_frame.py` | 解码并保存视频第一帧 |
| `04_check_alignment.py` | 检查示范的视频起点和结束边界 |
| `05_load_sample.py` | 使用 LeRobot 读取完整样本 |
| `06_build_batch.py` | 构造 B=4、K=10 的小批次 |
| `07_act_forward.py` | 标准化数据并计算 ACT 的一次损失 |
| `GITHUB_GUIDE.md` | 首次上传、日常修改及切换助手的步骤 |
| `requirements.txt` | 项目的直接依赖 |
| `requirements-windows-lock.txt` | 从当前 Windows 环境导出的完整版本快照 |

## 环境与数据

当前验证环境：Windows、Python 3.13.7、RTX 5060 Laptop GPU 约 8 GB、PyTorch 2.11.0+cu130、torchvision 0.26.0+cu130、LeRobot 0.6.1。

数据集：`lerobot/pusht`，固定提交 `7628202a2180972f291ba1bc6723834921e72c19`，使用实际 v3.0 格式。视频后端明确选择 PyAV。

`.venv`、`data`、`outputs` 和权重不上传 GitHub；代码根据固定版本重新下载所需数据。不要将 Hugging Face 或 GitHub 的令牌写进脚本。

## 在另一台 Windows 电脑恢复

以下命令是恢复说明，不表示已经执行；每一步结束后检查输出，再继续。需要先准备 Git、Python 3.13 和支持 CUDA 13.0 的 NVIDIA 驱动。GPU、系统或 Python 版本不同，应先重新核对配置。

在克隆后的项目目录，用 VS Code 的 PowerShell 终端执行：

```powershell
py -3.13 -m venv .venv # 在当前项目目录创建独立 Python 环境。
& ".\.venv\Scripts\python.exe" -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130 # 从官方源安装匹配的 CUDA 版计算和图像库。
& ".\.venv\Scripts\python.exe" -m pip install -r requirements-windows-lock.txt # 按当前 Windows 环境的完整版本快照恢复其余依赖。
& ".\.venv\Scripts\python.exe" -m pip check # 检查安装包声明的依赖关系是否存在冲突。
```

快照主要记录包版本，不包含操作系统、驱动、下载源缓存或原始虚拟环境，因此安装完成后仍需验证实际导入与 GPU 运算。使用 `requirements.txt` 可以安装直接依赖，但间接依赖不一定与历史环境完全相同。

## 运行实验

按编号逐个阅读和运行；每个脚本的导入会执行其顶层代码，不要直接将这些入门脚本当作无副作用的公共模块导入。

```powershell
& ".\.venv\Scripts\python.exe" 01_inspect_dataset.py # 首先验证元数据下载与数据结构。
& ".\.venv\Scripts\python.exe" 06_build_batch.py # 检查图像、状态、动作序列和补齐标记的批次形状。
& ".\.venv\Scripts\python.exe" 07_act_forward.py # 检查一次 ACT 前向计算；此脚本不会更新参数。
```

部分脚本会联网下载公开数据集的缺失文件。尚未保证所有后续仿真环境可在 Windows 原生运行；LIBERO 的官方 LeRobot 集成要求 Linux。完整任务成功率需要仿真或真实机器人评估。

## 官方资料

- [LeRobot 安装](https://huggingface.co/docs/lerobot/installation)
- [ACT 说明](https://huggingface.co/docs/lerobot/act)
- [SmolVLA 说明](https://huggingface.co/docs/lerobot/smolvla)
- [PushT 数据集](https://huggingface.co/datasets/lerobot/pusht)

此仓库是学习实验，不代表已复现论文结果。引用上游代码、模型和数据时，按各自许可和引用要求处理。
