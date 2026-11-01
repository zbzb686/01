# LeRobot + ACT + SmolVLA 学习项目

从 PushT 模仿学习出发，拆解 ACT 原理并亲手实现其核心，最终进入 SmolVLA 的机器人学习项目。

**完整结论与数据见 [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**；逐日交接细节见 [PROJECT_MEMORY.md](PROJECT_MEMORY.md)；教学约定见 [AGENTS.md](AGENTS.md)。

---

## 项目有两条主线

| 主线 | 内容 | 状态 |
| --- | --- | --- |
| **A. 用现成 ACT 训 PushT** | LeRobot `ACTPolicy` + lerobot/pusht，完整走通数据 → 训练 → 验证 → 闭环评估 | ✅ 已跑完，成功率约 5–12% |
| **B. 手写 ACT 核心** | 不调用 `ACTPolicy`，从注意力、编码器、解码器一路写到 CVAE | 🟡 源码完成，尚未运行验证 |

---

## 当前进度

**主线 A 已完成，结果不理想但过程完整：**

- 数据划分：训练 0–179（22,207 帧）、验证 180–199（2,685 帧）、测试 200–205 预留
- 训练两轮各 32,000 步，唯一变量是学习率（1e-5 vs 3e-5）
- 最佳验证 MAE：22.2657 / 21.8819（只差 1.7%）
- 闭环评估：25 个固定场景，基线成功 3 个（12%），对照成功 2 个（8%）
- **结论：调学习率不能突破这个平台；离线 MAE 不能预测闭环成功率**

**主线 B 源码已完成，包含：**

```
视觉骨干 → 图像token → 状态token → 拼接 → 位置编码
   → 多头自注意力 → 编码器 → 动作查询 → 解码器 → 动作头
   → CVAE（VAE编码器 + 重参数化 + KL）
```

尚未实际运行，也尚未接入真实数据。

---

## 目录结构

```
.
├── act_core.py              # 手写的 ACT 全部模块（主线 B）
├── train_mini_act.py        # 手写 ACT 的训练脚本（主线 B）
├── configs/                 # 所有实验配置（JSON）
│   ├── pusht_split_v2.json         # 数据划分
│   ├── pusht_train_stats_v2.json   # 训练集统计量
│   ├── train_act_v2.json           # PushT 基线训练配置
│   ├── train_act_lr3e5.json        # 学习率对照配置
│   └── mini_act_debug.json         # 手写 ACT 配置
├── train.py                 # PushT 训练入口（主线 A）
├── evaluate.py              # PushT 闭环评估入口（主线 A）
├── run_act_step.py          # 单步推理调试入口
├── diagnose_offline.py      # 离线 MAE 诊断
├── pusht_project/           # 公共模块：观测转换、模型加载、单回合执行
├── 01–22_*.py               # 早期逐步教学脚本（数据探索 → 单步训练 → 评估）
├── PROJECT_SUMMARY.md       # 项目总结：数据、结果、结论
├── PROJECT_MEMORY.md        # 交接记忆：逐日进展与待办
├── AGENTS.md                # 教学约定
└── GITHUB_GUIDE.md / GIT_COMMANDS.md
```

`outputs/`（实验产物）、`data/`（数据集）、`.venv/`（虚拟环境）不纳入版本控制。

---

## 运行

主线 A（需要 GPU、数据集和 gym-pusht）：

```powershell
& ".\.venv\Scripts\python.exe" train.py       # 按 configs/train_act_v2.json 训练
& ".\.venv\Scripts\python.exe" evaluate.py    # 按脚本内 model_dir 做闭环评估
```

主线 B（仅验证结构，用模拟数据）：

```powershell
& ".\.venv\Scripts\python.exe" act_core.py         # 各模块单独演示
& ".\.venv\Scripts\python.exe" train_mini_act.py   # 跑 20 步模拟训练
```

> 注意：主线 B 的代码**尚未经过实际运行验证**，只通过了语法检查。首次运行时请留意形状是否与注释中的预期一致。

---

## 环境

| 项目 | 版本 |
| --- | --- |
| 系统 | Windows 11 原生 |
| Python | 3.13.7 |
| PyTorch | 2.11.0+cu130 |
| LeRobot | 0.6.1（含 training 扩展） |
| GPU | RTX 5060 Laptop，约 8 GB |
| 仿真 | gym-pusht 0.1.8 |

完整依赖快照见 `requirements-windows-lock.txt`。**注意**：`gym-pusht` 声明依赖 `opencv-python`，而 LeRobot 依赖 `opencv-python-headless`；本项目保留 headless 版本，用 `--no-deps` 单独安装 `gym-pusht`，因此 `pip check` 会报告缺少 `opencv-python`，属预期行为。

### 在另一台机器恢复

```powershell
py -3.13 -m venv .venv                                    # 创建独立环境
& ".\.venv\Scripts\python.exe" -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
& ".\.venv\Scripts\python.exe" -m pip install -r requirements-windows-lock.txt
& ".\.venv\Scripts\python.exe" -m pip check
```

---

## 三条值得记住的结论

1. **离线指标不能代替闭环评估。** 验证 MAE 只改善 1.7%，闭环成功率反而下降。
2. **小样本评估会严重误导。** 5 个场景得到的"40%"在 25 个场景上变成 8%。
3. **失败要分解到机制。** 同样是 0/5，"过冲撞歪"和"过早停止"是完全不同的两种病。

---

## 下一阶段

1. 跑通手写 ACT，确认损失下降、参数确实更新
2. 接入真实 PushT 数据（动作块、padding 掩蔽、统计量标准化）
3. 加验证集与检查点保存
4. 补二维位置编码、dropout、时间集成
5. 补 ViT / 语言 token / LLM / VLM 基础，再进 SmolVLA

---

## 官方资料

- [LeRobot 安装](https://huggingface.co/docs/lerobot/installation)
- [ACT 说明](https://huggingface.co/docs/lerobot/act)
- [SmolVLA 说明](https://huggingface.co/docs/lerobot/smolvla)
- [PushT 数据集](https://huggingface.co/datasets/lerobot/pusht)

本仓库是学习实验，不代表已复现论文结果。引用上游代码、模型和数据时，按各自许可与引用要求处理。
