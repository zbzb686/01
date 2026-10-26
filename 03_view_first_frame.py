from pathlib import Path  # 导入路径工具，用于确定下载目录和图片保存位置。

import av  # 导入 PyAV，用于打开视频并解码其中的图像。
from huggingface_hub import hf_hub_download  # 导入函数，下载数据仓库中的指定文件。

root = Path(__file__).parent  # 获取当前脚本所在的项目文件夹。

video_path = hf_hub_download(  # 下载一个视频文件，并保存它在本地的路径。
    repo_id="lerobot/pusht",  # 指定 PushT 数据仓库。
    repo_type="dataset",  # 指定仓库类型为数据集。
    revision="7628202a2180972f291ba1bc6723834921e72c19",  # 使用与数值表相同的数据版本。
    filename="videos/observation.image/chunk-000/file-000.mp4",  # 指定图像字段对应的视频文件。
    local_dir=root / "data" / "pusht",  # 将视频保存到项目内的数据目录。
)  # 结束下载调用，video_path 指向本地视频文件。

output_dir = root / "outputs"  # 定义用于保存实验结果的文件夹。
output_dir.mkdir(parents=True, exist_ok=True)  # 创建输出文件夹；已经存在时继续使用。
image_path = output_dir / "video_first_frame.png"  # 定义第一帧图片的保存路径。

with av.open(video_path) as container:  # 打开视频，并在代码块结束后自动关闭文件。
    frame = next(container.decode(video=0))  # 从第一个视频流中解码并取出第一帧。
    image = frame.to_image()  # 将解码后的帧转换成 Pillow 图片对象。
    image.save(image_path)  # 将图片保存为 PNG 文件，方便在 VS Code 中查看。
    print("帧时间戳（秒）：", frame.time)  # 打印这一帧在视频时间轴上的时间。
    print("图片尺寸（宽、高）：", image.size)  # 打印图片的宽度和高度。
    print("颜色模式：", image.mode)  # 查看图片是否采用 RGB 三通道表示。

print("图片保存位置：", image_path)  # 输出图片的完整路径，方便查找。