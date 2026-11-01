from pathlib import Path  # 导入路径工具，定位数据集和已保存模型。

import torch  # 导入 PyTorch，用于推理和检查输出数值。
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # 导入之前使用的数据集读取类。
from lerobot.policies.act.modeling_act import ACTPolicy  # 导入 ACT 模型类，加载保存的配置与权重。
from lerobot.processor import PolicyProcessorPipeline  # 导入处理器流水线，加载保存的预处理与后处理流程。
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action  # 导入动作格式转换函数，供动作后处理器使用。
root=Path(__file__).resolve().parent  # 获取当前脚本所在的项目目录。
sample_index = 0  # 选择第一个检查索引对应的样本进行单条推理。
model_dir = root / "outputs" / "act_pusht" / "20261027_150704_767282"  # 明确指定刚才已经保存成功的模型目录。
policy = ACTPolicy.from_pretrained(model_dir, local_files_only=True,strict=True)  # 使用官方接口加载模型配置和权重。
policy.eval()  # 切换到推理模式；此操作本身不会更新参数.
policy.reset()  # 清空模型可能缓存的动作，确保从一个新的预测过程开始。
preprocessor = PolicyProcessorPipeline.from_pretrained(  # 加载训练时保存的输入预处理器。
    model_dir,  # 从同一个模型目录读取配置及统计量。
    config_filename="policy_preprocessor.json",  # 指定输入预处理器配置文件。
    local_files_only=True,  # 仅使用本地保存的文件。
)  # 完成输入预处理器加载。

postprocessor = PolicyProcessorPipeline.from_pretrained(  # 加载保存的动作后处理器，将标准化动作还原为原始坐标。
    model_dir,  # 从同一个模型目录读取动作处理配置及统计量。
    config_filename="policy_postprocessor.json",  # 指定动作后处理器配置文件。
    local_files_only=True,  # 仅使用本地保存的文件。
    to_transition=policy_action_to_transition,  # 把模型输出的动作张量转换为处理器内部格式。
    to_output=transition_to_policy_action,  # 把处理完成的结果转换回动作张量。
)  # 完成动作后处理器加载。
dataset = LeRobotDataset(  # 打开本地数据集，为推理提供一条观测。
    repo_id="lerobot/pusht",  # 使用与训练相同的 PushT 数据集。
    root=root / "data" / "pusht",  # 使用之前已经下载的数据。
    episodes=[0],  # 本轮仍从第 0段示范读取样本。
    revision="7628202a2180972f291ba1bc6723834921e72c19",  # 固定数据版本，与训练时一致。
    video_backend="pyav",  # 使用已经验证可用的视频读取方式。
)  # 完成数据集创建。
sample = dataset[sample_index]  # 取出第一个时间步的样本。
observation = {  # 构造推理输入，只提供当前观测。
    "observation.image": sample["observation.image"].unsqueeze(0),  # 将图像从 (3,96,96) 扩展为单样本批次 (1,3,96,96)。
    "observation.state": sample["observation.state"].unsqueeze(0),  # 将当前位置从 (2,) 扩展为 (1,2)。
}  # 完成输入构造，不把示范动作作为模型输入。
processed_observation = preprocessor(observation)  # 使用保存的统计量标准化输入，并安排计算设备。
with torch.inference_mode():  # 关闭梯度记录，本轮只预测，不更新参数。
    normalized_actions = policy.predict_action_chunk(processed_observation)  # 预测一整块标准化动作，预期形状为 (1,10,2)。
    predicted_actions = postprocessor(normalized_actions)  # 将预测动作恢复为数据集原始坐标，并移动到 CPU。

if not torch.isfinite(predicted_actions).all().item():  # 检查所有预测坐标是否为有限数值。
    raise RuntimeError("预测动作包含 NaN 或无穷大。")  # 出现异常数值时停止，避免误认为推理成功。

print("加载的模型目录：", model_dir)  # 确认实际使用的是哪一次保存的模型。
print("模型设备：", next(policy.parameters()).device)  # 查看模型当前所在设备。
print("预测动作形状：", tuple(predicted_actions.shape))  # 预期显示 (1,10,2)。
print("当前推手位置：", sample["observation.state"].tolist())  # 打印输入中的原始当前位置。
print("预测的前三个目标位置：", predicted_actions[0, :5].tolist())  # 查看模型预测的前三个动作，数值已经恢复到原始坐标。
print("示范的当前目标位置：", sample["action"].tolist())  # 打印示范答案供观察，这个答案没有提供给推理模型。
print("预测动作全部为有限数值：", torch.isfinite(predicted_actions).all().item())  # 预期显示 True。
print("模型加载与动作推理完成。")  # 确认本轮流程完整运行。
from PIL import Image, ImageDraw, ImageFont  # 使用环境里已有的 Pillow 生成对比图。

predicted = predicted_actions[0].detach().cpu()  # 去掉单样本批次维度，得到形状为 (10,2) 的原始坐标。
horizon = predicted.shape[0]  # 获取预测包含的动作数量，当前为 10。
future_samples = [dataset[sample_index + offset] for offset in range(horizon)]  # 从同一个起点读取连续 10 个时间步的示范。
if any(item["episode_index"].item() != sample["episode_index"].item() for item in future_samples):  # 检查目标序列有没有跨越示范边界。
    raise RuntimeError("目标序列跨越了示范边界，请选择更靠前的样本。")  # 阻止把不同示范的动作错误地拼接在一起。
expected = torch.stack([item["action"] for item in future_samples]).cpu()  # 将连续示范目标组成 (10,2) 张量，与预测逐个对应。
current = sample["observation.state"].cpu()  # 保存输入时推手的当前位置，形状为 (2,)。
coordinate_mae = (predicted - expected).abs().mean().item()  # 计算这 10 个动作的平均绝对坐标误差，使用原始坐标而非标准化数值。

all_points = torch.cat([predicted, expected, current.unsqueeze(0)], dim=0)  # 汇总所有点，让图的范围能容纳预测、示范和当前位置。
lower = all_points.min(dim=0).values - 10  # 在最小坐标外留出 10 个坐标单位的边距。
upper = all_points.max(dim=0).values + 10  # 在最大坐标外留出同样的边距。
left, top, right, bottom = 90, 100, 850, 590  # 设置绘图区的四个边界，单位是输出图片的像素。
image = Image.new("RGB", (900, 700), "white")  # 创建白色背景的图片。
draw = ImageDraw.Draw(image)  # 创建画笔，用于绘制网格、线条、文字和点。
font = ImageFont.load_default(size=16)  # 使用 Pillow 自带字体，避免依赖本机字体文件。

def to_canvas(point):  # 将原始动作坐标转换成输出图片中的绘图位置。
    x = left + (float(point[0]) - float(lower[0])) / float(upper[0] - lower[0]) * (right - left)  # 按横轴范围缩放 X 坐标。
    y = bottom - (float(point[1]) - float(lower[1])) / float(upper[1] - lower[1]) * (bottom - top)  # 按纵轴范围缩放 Y 坐标，使较大的 Y 位于图上方。
    return round(x), round(y)  # 返回画笔使用的整数像素位置。

draw.text((90, 15), f"ACT action comparison | sample {sample_index}", fill="black", font=font)  # 标明图的内容和样本编号。
draw.text((90, 45), "Green: demonstration", fill="green", font=font)  # 说明绿色代表示范目标序列。
draw.text((350, 45), "Red: prediction", fill="red", font=font)  # 说明红色代表模型预测序列。
draw.text((590, 45), "Blue: current position", fill="blue", font=font)  # 说明蓝色代表输入时的推手位置。

for tick in range(6):  # 将每个坐标轴划分成五段，绘制六条网格线。
    ratio = tick / 5  # 计算当前网格线在坐标轴上的相对位置。
    x = round(left + ratio * (right - left))  # 计算竖直网格线的图片位置。
    y = round(bottom - ratio * (bottom - top))  # 计算水平网格线的图片位置。
    draw.line((x, top, x, bottom), fill="#dddddd")  # 绘制竖直浅灰色网格线。
    draw.line((left, y, right, y), fill="#dddddd")  # 绘制水平浅灰色网格线。
    draw.text((x - 20, bottom + 10), f"{float(lower[0] + ratio * (upper[0] - lower[0])):.1f}", fill="black", font=font)  # 标注横轴的原始坐标值。
    draw.text((5, y - 8), f"{float(lower[1] + ratio * (upper[1] - lower[1])):.1f}", fill="black", font=font)  # 标注纵轴的原始坐标值。

for points, color in [(expected, "green"), (predicted, "red")]:  # 分别绘制示范和预测的目标位置序列。
    pixels = [to_canvas(point) for point in points]  # 将这一组原始坐标转换成图片位置。
    draw.line(pixels, fill=color, width=3)  # 按时间顺序连接目标位置。
    for index, (x, y) in enumerate(pixels):  # 遍历每一个目标位置，同时记录序列编号。
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)  # 用圆点标记目标位置。
        draw.text((x + 6, y - 15), str(index), fill=color, font=font)  # 标注动作编号 0～9，方便对比同一时间位置。

x, y = to_canvas(current)  # 将推手当前位置转换成图片位置。

draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="blue")  # 用较大的蓝点标记推手当前位置。
draw.text((90, 640), f"Mean absolute coordinate error: {coordinate_mae:.3f}", fill="black", font=font)  # 在图下方记录这条样本的平均绝对坐标误差。
output_path = root / "outputs" / f"action_comparison_{model_dir.name}_sample_{sample_index}.png"  # 将对比图保存到项目 outputs 目录。
output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
image.save(output_path)  # 保存 PNG 图片，方便直接在 VS Code 中打开。

print("示范动作形状：", tuple(expected.shape))  # 预期为 (10,2)，与单条预测动作块一致。
print("平均绝对坐标误差：", coordinate_mae)  # 打印当前样本在原始坐标中的误差。
print("对比图保存位置：", output_path)  # 打印图片路径，方便查看。
import csv  # 使用 Python 自带的 CSV 工具保存结果，无需安装新依赖.

check_indices = [0, 1, 2, 3, 20, 50, 80, 100]    # 前四个是训练输入，后四个是同一示范中未参与这批训练的输入。
action_count = policy.config.chunk_size  # 从模型配置读取动作块长度，当前为 10。
results = []  # 收集每个样本的误差，供分组汇总和保存使用。
print("\n样本编号 | 分组 | 平均绝对坐标误差")  # 打印表头，与前面的单样本输出分开。

for index in check_indices:  # 依次评估指定样本，每次只处理一条观测。
    if index + action_count > len(dataset):  # 检查后面是否还有足够的数据构成完整动作块。
        raise RuntimeError(f"样本 {index} 后面不足 {action_count} 个动作。")  # 数据不足时停止，避免越界或错误比较。

    sequence = [dataset[index + offset] for offset in range(action_count)]  # 从当前索引开始读取连续 10 个时间步。
    item = sequence[0]  # 直接使用序列起点作为模型输入，确保输入和答案从同一时间步开始。
    if any(part["episode_index"].item() != item["episode_index"].item() for part in sequence):  # 检查整个动作块是否都属于同一段示范。
        raise RuntimeError(f"样本 {index} 的动作块跨越了示范边界。")  # 发现跨段时停止，避免拼接不同任务过程。

    obs = {  # 构造当前样本的模型输入。
        "observation.image": item["observation.image"].unsqueeze(0),  # 增加批量维度，图像形状为 (1,3,96,96)。
        "observation.state": item["observation.state"].unsqueeze(0),  # 增加批量维度，状态形状为 (1,2)。
    }  # 输入只包含图像和状态，不提供示范答案。
    processed_obs = preprocessor(obs)  # 使用已保存的输入处理器完成标准化和设备安排。
    policy.reset()  # 在处理新的独立观测前清空模型缓存。

    with torch.inference_mode():  # 只进行推理，不记录梯度或更新模型参数。
        normalized_prediction = policy.predict_action_chunk(processed_obs)  # 根据当前观测预测一整块动作。
        prediction = postprocessor(normalized_prediction)[0].detach().cpu()  # 恢复原始动作坐标，并去掉批量维度，得到 (10,2)。

    target = torch.stack([part["action"] for part in sequence]).cpu()  # 将相同起点的示范动作组成 (10,2)。
    if not torch.isfinite(prediction).all().item():  # 检查预测坐标有没有 NaN 或无穷大。
        raise RuntimeError(f"样本 {index} 的预测包含异常数值。")  # 出现异常时停止，避免生成无效误差。

    error = (prediction - target).abs().mean().item()  # 对 10×2 个坐标误差取绝对值后求平均。
    group = "第一段示范检查点"  # 按本次固定批次实际使用过的输入编号进行分组。
    results.append({"sample_index": index, "group": group, "coordinate_mae": error})  # 保存这一条样本的结果。
    print(f"{index:>8} | {group} | {error:.4f}")  # 打印当前样本的原始坐标误差。

for group in ["第一段示范检查点"]:  # 分别汇总两组样本。
    errors = [row["coordinate_mae"] for row in results if row["group"] == group]  # 取出属于当前组的所有误差。
    print(f"{group}平均误差：{sum(errors) / len(errors):.4f}")  # 计算组内样本误差的平均值。

csv_path = root / "outputs" / f"sample_errors_{model_dir.name}.csv"  # 文件名包含模型目录名称，便于追踪结果属于哪个模型。
csv_path.parent.mkdir(parents=True, exist_ok=True)  # 确保结果目录存在。
with csv_path.open("w", newline="", encoding="utf-8-sig") as file:  # 使用适合 Windows 表格软件读取的编码保存 CSV。
    writer = csv.DictWriter(file, fieldnames=["sample_index", "group", "coordinate_mae"])  # 指定结果表格的三列。
    writer.writeheader()  # 写入列名。
    writer.writerows(results)  # 写入全部八条样本的结果。

print("误差表保存位置：", csv_path)  # 输出结果文件的完整路径。