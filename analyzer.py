import os
import time
from pathlib import Path
from pprint import pformat
import pandas as pd
from SCINet import SCI


class Analyzer:
    def __init__(self, config, logger):

        self.config = config
        self.logger = logger

        self.logger.info(f"config :{pformat(config)}")

        self.sci = None

        self.model_dict = {}

        try:
            self.sci = SCI(config=config, logger=logger)
        except MemoryError as error:
            logger.exception(
                f"[Error] Unexpected memory error during serving : {error}"
            )
            aicommon.Utils.print_memory_usage(logger)
            raise error
        except Exception as exception:
            logger.exception(
                f"[Error] Unexpected exception during serving : {exception}"
            )
            raise exception

    def _save(self):
        model_dir = self.config["model_dir"]
        Path(model_dir).mkdir(exist_ok=True, parents=True)

        self.sci.save(model_dir)

    def _load(self):
        if not os.path.exists(self.config["model_dir"]):
            msg = f"model directory {self.config['model_dir']} not found"
            self.logger.critical(msg)
            raise Exception(msg)

        model_dir = self.config["model_dir"]

        res = self.sci.load(model_dir)

        return True

    # API for training
    def train(self):
        self.logger.info(f"module start training")

        process_start_time = time.time()

        train_directory = self.config["train_dir"] + "/" + "BTC-USD.csv"
        df = pd.read_csv(train_directory, index_col=0)

        sci_result = self.sci.fit(df)

        self._save()

        # train_result
        self.logger.info(f"module training result: {sci_result}")

        return sci_result

    def end_train(self):
        pass

    # API for serving
    def init_serve(self):
        res = self._load()
        return res

    def predict(self):

        model = self.sci
        pred_result = model.predict()

        return pred_result

    def end_serve(self):
        pass
