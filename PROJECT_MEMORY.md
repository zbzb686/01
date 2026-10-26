# 项目交接记忆

最后整理日期：2026-10-26。此文件用于不同助手、会话和电脑之间交接；它是项目记录，不是聊天历史的自动同步。后续请依据实际输出更新，保留已验证与待验证的区别。

## 1. 项目目标与当前优先事项

通过 LeRobot + ACT + SmolVLA 学习视觉动作模型和工程方法。顺序为：环境验证 → 数据探索 → 轻量 ACT 模仿学习 → SmolVLA 推理与微调。以用户亲自动手为主，每轮一个小步骤。

当前优先事项：用户要求暂时停下模型课程，整理可以交给 Claude Code 的项目记忆，并学习将项目上传到 GitHub、日常修改和同步代码。完成后再恢复 ACT 原理与参数更新实验。

模型进度：已经完成 ACT 单批次前向计算，但没有执行反向传播、优化器更新、正式训练或仿真评估。SmolVLA 尚未安装对应扩展、下载权重、运行推理或微调。

## 2. 学习者基础和教学要求

- 有线性代数、高等数学基本知识；学过 CNN 入门理论，Transformer 刚入门；曾用 VS Code 训练过 PyTorch 小模型。
- 不默认用户能独立实现完整训练流程，也不默认已掌握所有相关知识。
- 用户明确表示没有 ViT、VLM、LLM 基础，并指出此前不知道 ACT 是什么。必须在实验需要时从概念、动机和实例逐步补齐。
- 用户要求详细教学；先讲原理再给代码，必要时解释公式、张量形状、单位及代码对应关系。
- 所有教学代码的每个非空行配中文注释，包括终端命令。对不支持注释的格式，在代码块外逐行解释。
- 每轮只推进一个小操作或一个小概念，用户运行后贴回输出；不能为赶进度代做整个训练流程。
- 用小测确认理解；答错时解释原因。不能将程序运行成功等同于用户已经掌握原理。
- 安装前说明方案并确认；不要改坏用户原有环境。已授权的同一方案不重复确认。
- 完整共同规则见 `AGENTS.md`。

## 3. 已核实的本机环境

以下事实来自用户终端输出，并于本次交接时核对了当前项目环境中的主要安装包版本。换电脑、解释器或升级依赖后应重新确认。

| 项目 | 已验证情况 |
| --- | --- |
| 系统与工具 | Windows 原生、PowerShell、VS Code |
| 当前项目目录 | `E:\项目一` |
| 项目解释器 | `E:\项目一\.venv\Scripts\python.exe` |
| Python | 3.13.7 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，约 8 GB 显存 |
| NVIDIA 驱动 | 582.05；nvidia-smi 显示支持 CUDA 13.0 |
| PyTorch | 2.11.0+cu130 |
| torchvision | 0.26.0+cu130 |
| LeRobot | 0.6.1，已安装 training 扩展 |
| NumPy | 2.2.6 |
| 视频读取 | PyAV 15.1.0；LeRobot 明确设置 `video_backend="pyav"`，读取样本和批次成功 |
| TorchCodec | 已安装 0.11.1；其解码后端未独立验证，不能据此宣称可用 |
| Hugging Face Hub / datasets | 1.33.0 / 4.8.5 |
| pandas / pyarrow | 2.3.3 / 26.0.0 |
| 依赖关系检查 | 本次交接整理中实际执行 pip check，结果为 No broken requirements found. |
| WSL | 先前查询显示未安装；没有安装或改成 Linux 环境 |
| Git | 本次检查版本为 2.55.0.windows.2 |
| GitHub CLI | 本次终端未找到 gh 命令；不以它作为当前教学前提 |

旧环境 `D:\ai\pytorch_env` 中的 PyTorch 2.14.1+cu130 能使用 GPU，但超出当前 LeRobot 0.6.1 声明的 torch>=2.7,<2.12 范围，因此另建项目 `.venv`；不要覆盖旧环境。

新环境已实际完成 GPU 矩阵乘法，设备为 cuda:0，结果为 [[7.0, 10.0], [15.0, 22.0]]。nvidia-smi 显示的 CUDA 版本是驱动支持信息，不等同于已安装 Toolkit 的版本。

## 4. 数据集与读取约定

- 仓库：`lerobot/pusht`；固定提交：`7628202a2180972f291ba1bc6723834921e72c19`。
- 实际 `meta/info.json` 为 v3.0 格式；不要沿用旧数据卡的 v2.0 描述。
- 206 段示范，25,650 个时间步，10 fps，图像高×宽×通道为 [96,96,3]。
- `observation.state` 是推手的二维当前位置；`action` 是示范者给定的推手二维目标位置。不要当成速度、位移或下一帧必然到达的位置。
- 表格：`data/chunk-000/file-000.parquet`；视频：`videos/observation.image/chunk-000/file-000.mp4`；示范位置记录：`meta/episodes/chunk-000/file-000.parquet`。
- 本地数据统一保存到项目内 `data/pusht`；图像等生成结果保存到 `outputs`。
- 示范 0 长度 161，视频区间 [0.0,16.1) 秒，采样点 0.0～16.0 秒；示范 1 长度 118，视频区间 [16.1,27.9) 秒。
- 视频查询时间 = 示范在视频中的起始偏移 + 表格 timestamp；两段示范可以共享同一视频文件。
- 用户已从“10 fps 对应 10 秒”的误解，纠正到相邻采样间隔为 0.1 秒；对 20 fps 已答出 0.05 秒。

示范 0 前三个时间步的记录如下，便于继续检查对齐：

| frame_index | timestamp | observation.state | action |
| --- | --- | --- | --- |
| 0 | 0.0 | [222.0,97.0] | [233.0,71.0] |
| 1 | 0.1 | [225.2524,89.31253] | [229.0,83.0] |
| 2 | 0.2 | [227.59233,84.53438] | [229.0,86.0] |

## 5. 脚本与已验证的结果

这些脚本在交接时均存在；按需检查源码，不要为了整理记忆而自动执行训练。

| 脚本 | 用途与验证情况 |
| --- | --- |
| `01_inspect_dataset.py` | 下载并打印元数据；用户输出已验证格式、数量、fps、形状 |
| `02_inspect_episode.py` | 读取数值表并查看示范 0 的前五行；用户输出已验证 |
| `03_view_first_frame.py` | 用 PyAV 提取视频首帧并保存 PNG；没有单独收到完整输出或看图小测答案，不默认用户完成观察题 |
| `04_check_alignment.py` | 读取前两段示范的长度与视频时间边界；用户输出已验证 |
| `05_load_sample.py` | 用 LeRobotDataset 读取完整样本；用户回复可用，后续 06/07 又验证了同一数据读取流程 |
| `06_build_batch.py` | B=4、K=10，当前图像/状态与当前及未来 9 个动作；用户输出已验证 |
| `07_act_forward.py` | 在 06 的基础上创建 ACT、调用官方处理器、在 CUDA 上计算一次损失；用户输出已验证 |

当前批次参数：10 fps；chunk_size=10；action 相对时间偏移 0.0～0.9 秒；episodes=[0]；batch_size=4；shuffle=False；num_workers=0。

已验证形状：图像 (4,3,96,96)，状态 (4,2)，动作 (4,10,2)，action_is_pad (4,10)。首个样本前三个目标动作为 [[233,71],[229,83],[229,86]]。

`07_act_forward.py` 配置：ACT 默认主干和潜变量分支，chunk_size=n_action_steps=10，视觉骨干随机初始化（pretrained_backbone_weights=None），device=cuda。没有减掉 VAE 分支；没有加载论文训练好的模型。

使用 `make_act_pre_post_processors(config, dataset.meta.stats)` 进行标准化和设备安排，不能在后续实验中遗漏或重复预处理。数据集返回的 RGB 图像在 [0,1] 内，模型预处理进一步按统计量标准化。

最近一次用户报告：

| 指标 | 数值 |
| --- | --- |
| 模型设备 | cuda:0 |
| 总损失 | 83.39900970458984 |
| l1_loss | 0.7393173575401306 |
| kld_loss | 8.265969276428223 |
| 损失为有限值 | True |

默认 kl_weight=10，总损失约为 L1+10×KL。这是一次随机初始化模型的前向计算，不能当成训练成绩或任务成功率。

## 6. 原理讲解位置与下一步

已介绍，但不默认用户完全掌握：ACT 全称与模仿学习；CNN 提取图像特征、Transformer 融合信息、输出动作块；推手当前位置与目标位置的区别；B/K/动作维度；相邻采样时间；视频与表格对齐。

最近讲解：L1 采用平均绝对误差，当前批次有 4×10×2=80 个有效坐标；补齐位置应屏蔽。动作已标准化，L1 不能直接当像素误差。KL 只作了初步介绍，潜变量、概率分布、VAE/CVAE 还需要详细补课，不能默认已掌握。

尚未回答的小测：“如果 L1 下降，但 KL 上升，总损失一定下降吗？为什么？”

恢复模型教学时，先确认是否理解上述问题，再按需要补均值/标准差与标准化、概率分布和潜变量、KL 与 CVAE；分多轮讲，不一次堆叠术语。之后再进行一次参数更新、小批次拟合、训练验证划分与检查点保存、独立评估。

后续 SmolVLA 再补 ViT、语言 token、LLM 与 VLM 的必要基础。8 GB 显存下的配置需实测，不能保证论文默认微调配置可运行。Windows 可以先推进当前实验；LIBERO 官方集成要求 Linux，届时再讨论 WSL2/Linux。离线损失或动作输出不等于真实任务成功。

## 7. GitHub 与协作状态

2026-10-26 用户已亲手执行 `git init -b main`，终端报告在 `E:/项目一/.git/` 初始化成功。用户的 `git status --short` 输出包含脚本、文档和依赖清单，没有 `.venv`、`data`、`outputs`，忽略规则符合预期。助手随后只读核对 `git status --short --branch`，确认 `No commits yet on main`，所有项目文件仍未跟踪。尚未创建本地提交；尚未收到远程仓库创建、配置或推送的结果，不能视作已上传。GitHub 仓库名称、地址和可见性仍待用户确定。

本次已准备：`CLAUDE.md`、本交接文件、`README.md`、`.gitignore`、`requirements.txt`、`requirements-windows-lock.txt` 与 `GITHUB_GUIDE.md`。完整依赖快照记录了当前 Windows 环境中的 77 个包版本，不代表已在另一台电脑复现通过。

用户随后回复提交署名“已设置”。助手只读执行 `git var GIT_AUTHOR_IDENT` 并隐藏具体署名，确认当前仓库可取得有效作者信息；不在项目记忆中保存用户的姓名和邮箱。

当前下一小步：让用户执行 `git add .` 暂存项目文件，再用 `git diff --cached --stat` 查看第一次提交的文件清单与改动规模，确认没有数据、虚拟环境或凭据。尚未收到暂存结果，不能视作已经完成；清单确认后再创建首次本地提交，然后创建远程空仓库并推送。

日常流程：保存修改 → 查看差异 → 运行相关检查 → 更新本记忆中的进度 → commit → push。GitHub 不会自动同步每次按键；本地保存、提交、上传是不同步骤。切换助手前先结束旧助手的编辑任务，同一文件避免并发修改。

## 8. 官方参考

- LeRobot 安装：https://huggingface.co/docs/lerobot/installation
- LeRobot 0.6.1 依赖：https://github.com/huggingface/lerobot/blob/v0.6.1/pyproject.toml
- ACT：https://huggingface.co/docs/lerobot/act
- SmolVLA：https://huggingface.co/docs/lerobot/smolvla
- 固定数据说明：https://huggingface.co/datasets/lerobot/pusht/blob/7628202a2180972f291ba1bc6723834921e72c19/meta/info.json
- Claude Code 项目记忆：https://code.claude.com/docs/en/memory
- GitHub 上传本地项目：https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
