import json 
import torch
from act_core import MiniACT, VisualBackbone, masked_l1_loss, kl_divergence  # 导入模型、骨干（推算特征图尺寸）、掩蔽L1与KL散度。
from pathlib import Path
root=Path(__file__).resolve().parent
def load_configs(config_path):
    with open(config_path, "r", encoding="utf-8") as file:   # 打开文件，离开这段代码块时自动关闭
        config=json.load(file)
    return config  # 返回解析好的字典；模块顶层不再自动加载，改由 main() 调用时读取。


def build_optimizer(model, config):  # 按配置里的优化器名称创建对应优化器，整个训练过程只调用一次。
    optimizer_cfg = config["optimizer"]
    name = optimizer_cfg["name"].lower()
    learning_rate = float(optimizer_cfg["learning_rate"])
    weight_decay = float(optimizer_cfg["weight_decay"])
    momentum = float(optimizer_cfg["momentum"])

    if name == "sgd":
        optimizer = torch.optim.SGD(
           model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay 
         )
    elif name == "adamw":   
        optimizer = torch.optim.AdamW(
          model.parameters(),
          lr=learning_rate,
          weight_decay=weight_decay  
        )
    else:
        raise ValueError(f"不支持的优化器名称：{name}，请填写 'sgd' 或 'adamw'")
    return optimizer


def build_model(config):  # 按配置里的 module 段创建 MiniACT，并放到指定设备。
    module_cfg = config["module"]  # 取出模型参数段落，避免后面反复写完整嵌套路径。
    feature_height = module_cfg["image_height"] // VisualBackbone.downsample_factor  # 图像经骨干4次减半后的高度：96→6。
    feature_width = module_cfg["image_width"] // VisualBackbone.downsample_factor  # 图像经骨干4次减半后的宽度：96→6。
    max_obs_token = 1 + feature_height * feature_width  # 1个状态token加特征图上全部位置，作为观测位置表容量。
    model = MiniACT(  # 用配置中的尺寸构造教学版ACT。
        image_channels=module_cfg["image_channels"],  # 原始彩色图像的通道数，通常为3。
        backbone_channels=module_cfg["backbone_channels"],  # 视觉骨干输出的特征通道数。
        state_dim=module_cfg["state_dim"],  # 推手当前位置的维度。
        action_dim=module_cfg["action_dim"],  # 每个动作包含的坐标数。
        token_dim=module_cfg["token_dim"],  # 每个token的特征长度。
        num_head=module_cfg["num_head"],  # 注意力头数，必须能整除token_dim。
        num_hidden=module_cfg["num_hidden"],  # 前馈网络中间层宽度。
        action_count=module_cfg["action_count"],  # 一次前向预测的动作数量。
        max_obs_token=max_obs_token,  # 观测位置表容量，由图像尺寸推导而不是手写，避免两处不一致。
        encoder_layer=module_cfg["encoder_layer"],  # 编码器堆叠层数。
        decoder_layer=module_cfg["decoder_layer"],  # 解码器堆叠层数。
        latent_dim=module_cfg["latent_dim"],  # 潜变量z的维度。
        vae_layers=module_cfg["vae_layers"],  # VAE编码器的堆叠层数。
    )  # 完成模型创建，此时参数为随机初始化。
    return model.to(config["device"])  # 把全部参数搬到配置指定设备后返回给调用者。


# 1) make_fake_batch(config)
def make_fake_batch(config):  # 造一批模拟数据，只用于验证训练流程能跑通，不代表真实任务数据。
    module_cfg = config["module"]  # 取出模型尺寸配置，用于确定各张量的形状。
    batch_size = config["training"]["batch_size"]  # 本批包含多少条样本。
    device = config["device"]  # 所有张量必须和模型放在同一设备上。

    torch.manual_seed(config["seed"])  # 固定随机种子，使每次运行造出的数据完全一致。

    images = torch.rand(  # 模拟原始RGB图像，取值0～1与数据集一致；特征图改由模型内部的骨干现场计算。
        batch_size,  # 批次大小 B。
        module_cfg["image_channels"],  # 彩色图像通道数，通常为3。
        module_cfg["image_height"],  # 图像高度，例如96。
        module_cfg["image_width"],  # 图像宽度，例如96。
        device=device,  # 与模型同设备，否则前向时会报设备不一致。
    )  # 得到 [B, 3, 96, 96]。

    state = torch.randn(batch_size, module_cfg["state_dim"], device=device)  # 模拟推手当前位置，得到 [B, state_dim]。
    target_actions = torch.rand(batch_size, module_cfg["action_count"], module_cfg["action_dim"], device=device) * 512.0  # 模拟示范动作：均匀取值再放大到 0～512，接近真实场地坐标量级。

    action_is_pad = torch.zeros(batch_size, module_cfg["action_count"], dtype=torch.bool, device=device)  # 先全部标为“有效”；dtype 必须是布尔，掩蔽函数要用按位取反。
    action_is_pad[0, -1] = True  # 把第 0 条样本的最后一个动作标成“补齐”，用来验证掩蔽是否真的生效。

    return {  # 打包成字典，训练步骤按名字取用。
        "images": images,  # 图像分支输入，交给模型内部的视觉骨干。
        "state": state,  # 状态分支输入。
        "target_actions": target_actions,  # 监督目标。
        "action_is_pad": action_is_pad,  # 哪些动作位置不参与损失。
    }  # 返回本批数据。

# 2) train_one_step(model, optimizer, batch)
def train_one_step(model, optimizer, batch, kl_weight):  # 完成一次完整参数更新，返回总损失、L1与KL三个数值。
    model.train()  # 切换到训练模式，模型内部的VAE编码器才会启用。
    optimizer.zero_grad()  # 清空上一步残留在各参数里的梯度；PyTorch 的梯度默认是累加而不是覆盖。
    predicted, mu, logvar = model(batch["images"], batch["state"], target_actions=batch["target_actions"])  # 训练时传入示范动作，走VAE采样分支。
    l1 = masked_l1_loss(predicted, batch["target_actions"], batch["action_is_pad"])  # 只在有效动作上计算平均绝对误差，负责“推得准”。
    kl = kl_divergence(mu, logvar)  # 让潜变量分布贴近标准正态，负责“别把示范死记进z”。
    loss = l1 + kl_weight * kl  # 两项加权求和，就是CVAE的训练目标。
    loss.backward()  # 反向传播：梯度同时流向主网络和VAE编码器。
    optimizer.step()  # 根据刚算出的梯度更新参数，这一步才真正改变模型数值。
    return loss.item(), l1.item(), kl.item()  # 三项分开返回，便于观察是哪一项在变化。


# 3) main() 与入口保护
def main():  # 串起整套流程：读配置→建模型→建优化器→造数据→循环更新→检查参数是否变化。
    config = load_configs(root / "configs" / "mini_act_debug.json")  # 每次运行都重新读取磁盘上的配置，改 JSON 不必改代码。
    model = build_model(config)  # 按配置创建 MiniACT，并搬到配置指定的设备上。
    optimizer = build_optimizer(model, config)  # 优化器必须在模型搬到设备之后创建，绑定的才是同一批参数张量。
    batch = make_fake_batch(config)  # 造一批模拟数据；整个循环复用同一批，用来观察损失能不能被拟合下去。
    total_steps = config["training"]["step"]  # 从配置读取本次要执行的更新次数。

    print("优化器：", config["optimizer"]["name"], "| 学习率：", config["optimizer"]["learning_rate"], "| 总步数：", total_steps)  # 打印本轮实验条件，方便和另一组优化器对照。
    print("模拟数据形状：", {name: tuple(value.shape) for name, value in batch.items()})  # 确认四个张量的形状符合预期。
    print("补齐标记：\n", batch["action_is_pad"])  # 确认第 0 条样本的最后一个动作确实被标为 True。

    kl_weight = float(config["training"]["kl_weight"])  # 从配置读取KL项权重，决定“分布规整”相对“推得准”的重要程度。
    print("KL权重：", kl_weight)  # 打印出来，便于和另一组设置对照。

    before = model.action_head.act_proj.weight.detach().clone()  # 更新前保存副本：detach 切断计算图，clone 复制独立内存。
    losses = []  # 保存每一步的总损失。
    l1_history = []  # 保存每一步的L1分量，用来观察“推得准”这一项。
    kl_history = []  # 保存每一步的KL分量，用来观察潜变量分布是否被压向标准正态。
    for step in range(1, total_steps + 1):  # 从 1 开始计数，循环执行 total_steps 次更新。
        total_loss, l1_value, kl_value = train_one_step(model, optimizer, batch, kl_weight)  # 执行一次完整更新。
        losses.append(total_loss)  # 记录总损失。
        l1_history.append(l1_value)  # 记录L1分量。
        kl_history.append(kl_value)  # 记录KL分量。
        if step == 1 or step % 5 == 0 or step == total_steps:  # 只打印首步、每 5 步和末步，避免输出刷屏。
            print(f"第 {step:3d} 步 | 总损失 {total_loss:8.4f} | L1 {l1_value:8.4f} | KL {kl_value:8.4f}")  # 三项一起看，才知道是哪一项在动。
    after = model.action_head.act_proj.weight.detach()  # 读取更新后的同一个参数张量。
    max_diff = (after - before).abs().max().item()  # 计算前后最大绝对变化，用于证明参数真的被改动过。

    print("首步损失：", round(losses[0], 4), "| 末步损失：", round(losses[-1], 4))  # 对比首末损失，判断模型是否在拟合这批数据。
    print("首步L1：", round(l1_history[0], 4), "| 末步L1：", round(l1_history[-1], 4))  # L1下降说明预测越来越接近示范。
    print("首步KL：", round(kl_history[0], 4), "| 末步KL：", round(kl_history[-1], 4))  # KL变化反映潜变量分布被拉向标准正态的程度。
    print("动作头权重最大变化量：", max_diff)  # 大于 0 才说明优化器确实更新到了参数。


if __name__ == "__main__":  # 只在直接运行本文件时执行训练；被其他文件导入时不执行。
    main()  # 启动一次更新的演示流程。


