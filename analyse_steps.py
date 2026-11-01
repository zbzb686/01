import pandas as pd
import pandas as pd  # 用pandas读取逐步记录表格，方便按列筛选。

run_dir = "outputs/act_eval/20261029_131137_138556"  # 本次评估输出目录，里面有每个场景的逐步CSV。

for seed in range(5):  # 先重点分析两个接近成功的失败场景：seed0（撞歪）和seed3（过早停止）。
    df = pd.read_csv(f"{run_dir}/steps_seed{seed}.csv")  # 读入该场景300步的逐步记录。
    peak_idx = df["coverage"].idxmax()  # 找到覆盖率最高那行在表格里的位置编号。
    peak_step = int(df.loc[peak_idx, "step"])  # 取出峰值对应的步数。
    peak_cov = float(df.loc[peak_idx, "coverage"])  # 取出峰值覆盖率数值。
    pos = (peak_step - 1) % 10  # 峰值是当前10步动作块里的第几个动作：0表示本块第一次预测，越大越接近块尾。
    print(f"seed{seed}: 峰值覆盖率 {peak_cov:.4f} 在第 {peak_step} 步，块内第 {pos} 个动作")  # 打印峰值位置和块内序号。
    print("=" * 70)  # 分隔两个场景，方便对照。
