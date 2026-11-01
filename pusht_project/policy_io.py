from pathlib import Path  # 将传入的目录字符串转换为路径对象。
from lerobot.policies.act.modeling_act import ACTPolicy  # 加载ACT模型。
from lerobot.processor import PolicyProcessorPipeline  # 加载保存的输入和动作处理器。
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action  # 为动作后处理提供格式转换函数。

def load_act_bundle(model_dir):  # 接收模型目录，加载同一实验的模型和配套处理器。
    model_dir = Path(model_dir)  # 允许调用者传入字符串或Path对象。
    policy = ACTPolicy.from_pretrained(model_dir, local_files_only=True, strict=True).to("cuda")  # 从本地加载配置和权重，并放到 GPU。
    policy.eval()  # 切换为推理模式。
    policy.reset()  # 清空动作缓存，开始新的控制过程。

    preprocessor = PolicyProcessorPipeline.from_pretrained(  # 加载与这份权重配套的输入处理器。
    model_dir,  # 使用同一个 best 目录中的统计量。
    config_filename="policy_preprocessor.json",  # 指定输入处理器配置。
    local_files_only=True,  # 仅从本地读取文件。
    )  # 完成输入处理器加载。

    postprocessor = PolicyProcessorPipeline.from_pretrained(  # 加载动作后处理器，把标准化输出恢复为目标坐标。
    model_dir,  # 必须与模型和输入处理器来自同一目录。
    config_filename="policy_postprocessor.json",  # 指定动作后处理器配置。
    local_files_only=True,  # 仅使用本地文件。
    to_transition=policy_action_to_transition,  # 将动作张量转为处理器内部格式。
    to_output=transition_to_policy_action,  # 将处理结果转回动作张量。
    )     # 完成动作后处理器加载。# 在这里放入原脚本中的模型和处理器加载代码，保持在函数内部缩进。
    return policy, preprocessor, postprocessor  # 将三个对象一起返回，供运行入口使用。