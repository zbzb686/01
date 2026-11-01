import torch  # 使用PyTorch将仿真数组转换为模型输入张量。
def observation_to_batch(observation): # 接收一条仿真观测，返回包含批次维度的模型输入字典。
    
    image = torch.from_numpy(observation["pixels"].copy()).permute(2, 0, 1).to(torch.float32) / 255.0# 第一步：读取pixels，转成CHW排列的float32图像，并将颜色值除以255。
    state = torch.as_tensor(observation["agent_pos"], dtype=torch.float32)# 第二步：读取agent_pos，转成float32位置张量，保持原始坐标值。
    model_input = {  # 将仿真字段映射为训练时的字段名称。
            "observation.image": image.unsqueeze(0),  # 添加批次维度，得到 (1,3,96,96)。
            "observation.state": state.unsqueeze(0),  # 添加批次维度，得到 (1,2)。
        }  # 仅提供当前图像和位置。# 第三步：为图像和位置添加批次维度，使用训练时的字段名组成字典.
    
    return model_input  # 返回包含批次维度的模型输入字典。