import getopt
import json
import os
import platform
import subprocess
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from Util import common


def gpu_error_message():
    """
    에러 메시지 출력
     - gpu 관련이라 nvidia-smi 출력 하는 함수
    Returns
    -------

    """
    print("available gpu is not exist. check your gpu env \n nvidia-smi")
    out = subprocess.check_output(["nvidia-smi"], encoding="utf-8")
    out.split("\n")
    print(out)


def get_usable_gpus():
    """
    우휴 gpu 의 번호를 가져오는 함수
    Returns
    -------

    """
    out = subprocess.check_output(["gpustat", "--json"], encoding="utf-8")
    out = out.replace("\n", "")
    gpu_config = json.loads(out).get("gpus", False)
    gpus = []

    if gpu_config == False or len(gpu_config) == 0:
        gpu_error_message()
        return None

    usable_gpu_number = -1
    for config in gpu_config:
        if (
                config["memory.used"] / config["memory.total"] < 0.01
                and config["utilization.gpu"] < 10
        ):
            usable_gpu_number = config["index"]
            gpus.append(str(usable_gpu_number))
        else:
            if config["memory.used"] / config["memory.total"] > 0.01:
                print("[get_usable_gpus()] : gpu memory usage is high .. ")
            if config["utilization.gpu"] > 10:
                print("[get_usable_gpus()] : gpu cpu usage is high .. ")
            gpu_error_message()

    if usable_gpu_number == -1:
        gpu_error_message()

    return gpus


# MODULE_HOME
home = os.environ.get("MODULE_HOME")
if home is None:
    print("plz export MODULE_HOME")
    home = os.path.dirname(os.path.abspath(__file__))

# MODULE_PATH
py_path = os.environ.get("MODULE_PATH")
if py_path is None:
    print("plz export MODULE_PATH")
    py_path = os.path.dirname(os.path.abspath(__file__))

# argv
if len(sys.argv) < 3:
    aicommon.Utils.usage()
    sys.exit(2)

try:
    opts, _ = getopt.getopt(
        sys.argv[1:],
        "g:d:m:l",
        ["gpu=", "datapath=", "model_dir=", "log_dir="],
    )
except getopt.GetoptError as err:
    print(err)
    aicommon.Utils.usage()
    sys.exit(2)

GPU = None
for opt, arg in opts:
    if opt == "-h":
        print("usage: command [OPTIONS] [MODULE]")
        print("    -m, --module   module name")
        print("    -d, --datapath   train data directory")
        print("    -g, --gpu   gpu number")
        sys.exit()
    elif opt in ("-m", "--module"):
        module_name = arg
    elif opt in ("-d", "--datapath"):
        train_data_path = arg
    elif opt in ("-g", "--gpu"):
        GPU = arg

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if GPU is None:
    usable_gpus = get_usable_gpus()
    if len(usable_gpus) == 0:
        print("[get_usable_gpus()] we dont have available gpus ..")
        sys.exit(2)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(usable_gpus)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU

if True if "windows" in platform.platform().lower() else False:
    train_info = json.loads((Path(train_data_path) / "train_info.json").read_text("UTF-8"))
else:
    train_info = json.loads((Path(train_data_path) / "train_info.json").read_text())

class_name = common.to_camel_case(module_name)

# model_dir
model_dir = Path(home) / "model" / module_name
model_dir.mkdir(exist_ok=True, parents=True)

# log_dir
log_dir = Path(home) / "logs" / module_name
Path(log_dir).mkdir(exist_ok=True, parents=True)

logger = logging.getLogger(module_name)
log_file = str(Path(log_dir) / "train.log")
handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, encoding='utf-8')
handler.suffix = "%Y%m%d"

fomatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)s] - %(message)s', "%Y-%m-%d %H:%M:%S")
handler.setFormatter(fomatter)

logger.setLevel(logging.INFO)
logger.addHandler(handler)

target_class = common.get_module_class(
    module_name, class_name, str(Path(py_path))
)
config = train_info
config["train_dir"] = train_data_path
config["model_dir"] = model_dir
config["log_dir"] = log_dir
config["module"] = module_name

instance = target_class(config, logger)

header = None
body = None
errno = 0
errmsg = None

logger.info("============== Start Training Process !! ==============")
train_result = dict()
try:
    train_result = instance.train()
    instance.end_train()

except MemoryError as error:
    logger.exception("Occur Memory error in training process.", error)
    aicommon.Utils.print_memory_usage(logger)

except Exception as e:

    logger.exception(f"Occur Error in training process : {e}")

logger.info("============== End Training Process !! ==============")
