import argparse
import json
import logging
import os
import sys
import time
import platform
from pathlib import Path
from Util import common
from logging.handlers import TimedRotatingFileHandler
import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import pandas as pd


def rmsle(y, pred):
    log_y = np.log1p(y)
    log_pred = np.log1p(pred)
    squared_error = (log_y - log_pred) ** 2
    rmsle = np.sqrt(np.mean(squared_error))
    return rmsle


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(
    prog="predict",
    description="predict",
    add_help=True
)
parser.add_argument("-m", "--module", help="module name", required=True)
parser.add_argument("-d", "--train_data_path", help="train data path", required=True)

args = parser.parse_args()
module_name = args.module
train_data_path = args.train_data_path

# MODULE_HOME
home = os.environ.get("MODULE_HOME")
# MODULE_PATH
py_path = os.environ.get("MODULE_PATH")

# log_dir = Path(home) / "logs"
log_dir = Path(home) / "logs" / module_name

if True if "windows" in platform.platform().lower() else False:
    train_info = json.loads((Path(train_data_path) / "train_info.json").read_text("UTF-8"))
else:
    train_info = json.loads((Path(train_data_path) / "train_info.json").read_text())

config = train_info
config["module"] = module_name
config["train_dir"] = train_data_path
config["model_dir"] = str(
    Path(home) / "model" / module_name
)
config["log_dir"] = log_dir

logger = logging.getLogger(module_name)
log_file = str(Path(log_dir) / "serving.log")
handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, encoding='utf-8')
handler.suffix = "%Y%m%d"

fomatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)s] - %(message)s', "%Y-%m-%d %H:%M:%S")
handler.setFormatter(fomatter)

logger.setLevel(logging.INFO)
logger.addHandler(handler)

# train model config
if config["model_dir"] is None:
    logger.error("model directory is not found: ", config["model_dir"])
    sys.exit(3)

model_path = config["model_dir"]

# class name
class_name = common.to_camel_case(module_name)

# load module class
target_class = common.get_module_class(
    module_name, class_name, str(Path(py_path))
)
logger.info("============== Start prediction Process !! ==============")

instance = target_class(config, logger)
# load model
instance.init_serve()

pred_result = instance.predict()

# prediction Visualization Using Plotly Express
predict_data_dir = train_data_path + "/" + "predict_data.csv"
predict_df = pd.read_csv(predict_data_dir, index_col=0)
pred_real_df = predict_df.loc[pred_result.index.values[0]:pred_result.index.values[-1]]

feat = 'Close'
RMSLE = rmsle(pred_real_df[f"{feat}"], pred_result[f"{feat}"])
fig = go.Figure()
fig.add_trace(go.Scatter(x=pred_real_df.index, y=pred_real_df[f"{feat}"], mode="lines+markers", name="real value"))
fig.add_trace(go.Scatter(x=pred_result.index, y=pred_result[f"{feat}"], mode="lines+markers", name="predict value",
                         marker=dict(color='violet')))
fig.add_trace(
    go.Scatter(x=pred_result.index, y=pred_result[f"{feat}_lower"], mode="lines+markers", name="predict lower",
               marker=dict(color='pink')))
fig.add_trace(
    go.Scatter(x=pred_result.index, y=pred_result[f"{feat}_upper"], mode="lines+markers", name="predict upper",
               marker=dict(color='pink')))
fig.update_layout(title=f'<b></b> BTC-USD {feat}, RMSLE: {RMSLE}')
fig.write_image("./BTC-USD_Close.png")
fig.show()

logger.info("============== End prediction Process !! ==============")