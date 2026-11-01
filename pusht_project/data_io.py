from pathlib import Path  # 使用路径对象定位项目目录，并统一下载函数的返回类型。
from huggingface_hub import hf_hub_download  # 使用现有的Hugging Face单文件下载接口。

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 当前文件位于pusht_project内，向上两层得到项目根目录。
DATA_ROOT = PROJECT_ROOT / "data" / "pusht"  # 统一原始数据的本地保存位置。
REPO_ID = "lerobot/pusht"  # 统一使用的Hugging Face数据集仓库。
REVISION = "7628202a2180972f291ba1bc6723834921e72c19"  # 固定数据版本，保持各入口读取同一版数据。

def download_dataset_file(filename):  # 接收仓库内的文件路径，取得文件并返回本地Path对象。
    path = hf_hub_download(  # 沿用原下载接口，复用本地文件或获取所需文件。
        repo_id=REPO_ID,  # 使用公共仓库设置。
        repo_type="dataset",  # 指明访问数据集仓库。
        revision=REVISION,  # 使用固定提交版本。
        filename=filename,  # 由调用者指定本次需要的文件。
        local_dir=DATA_ROOT,  # 所有入口使用同一个本地数据目录。
    )  # 下载接口返回本地文件路径。
    return Path(path)  # 将路径统一为Path对象，方便读取和拼接。
