import tensorflow as tf
import pandas as pd
import numpy as np
import os
import pickle
import time
from sklearn.preprocessing import RobustScaler
from gtda.time_series import SlidingWindow
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MSE, MAE
from pathlib import Path
from timelogger import TimeLogger
from algorithms.common.data_preprocessing import DataPreprocessing


class InnerConv1DBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, h: float, kernel_size: int, neg_slope: float = .01, dropout: float = .5,
                 **kwargs):
        if filters <= 0 or h <= 0:
            raise ValueError('filters and h must be positive')

        super().__init__(**kwargs)
        self.conv1d = tf.keras.layers.Conv1D(max(round(h * filters), 1), kernel_size, padding='same')
        self.leakyrelu = tf.keras.layers.LeakyReLU(neg_slope)

        self.dropout = tf.keras.layers.Dropout(dropout)

        self.conv1d2 = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')
        self.tanh = tf.keras.activations.tanh

    def call(self, input_tensor, training=None):
        x = self.conv1d(input_tensor)
        x = self.leakyrelu(x)

        ## only training
        if training:
            x = self.dropout(x)

        x = self.conv1d2(x)
        x = self.tanh(x)
        return x


class SCIBlock(tf.keras.layers.Layer):
    def __init__(self, features: int, kernel_size: int, h: int, name='sciblock', **kwargs):
        """
        :param features: number of features in the output
        :param kernel_size: kernel size of the convolutional layers
        :param h: scaling factor for convolutional module
        """
        super().__init__(name=name, **kwargs)
        self.features = features
        self.kernel_size = kernel_size
        self.h = h

        self.conv1ds = {k: InnerConv1DBlock(filters=self.features, h=self.h, kernel_size=self.kernel_size, name=k)
                        for k in ['psi', 'phi', 'eta', 'rho']}  # regularize?

    def call(self, inputs):
        F_odd, F_even = inputs[:, ::2], inputs[:, 1::2]

        # Interactive learning as described in the paper
        F_s_odd = F_odd * tf.math.exp(self.conv1ds['phi'](F_even))
        F_s_even = F_even * tf.math.exp(self.conv1ds['psi'](F_odd))

        F_prime_odd = F_s_odd + self.conv1ds['rho'](F_s_even)
        F_prime_even = F_s_even - self.conv1ds['eta'](F_s_odd)

        return F_prime_odd, F_prime_even

    def get_config(self):
        config = super().get_config()
        config.update({'features': self.features, 'kernel_size': self.kernel_size, 'h': self.h})
        return config


class Interleave(tf.keras.layers.Layer):
    """A layer used to reverse the even-odd split operation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _interleave(self, slices):
        if not slices:
            return slices
        elif len(slices) == 1:
            return slices[0]

        mid = len(slices) // 2
        even = self._interleave(slices[:mid])
        odd = self._interleave(slices[mid:])

        shape = tf.shape(even)
        return tf.reshape(tf.stack([even, odd], axis=3), (shape[0], shape[1] * 2, shape[2]))

    def call(self, inputs):
        return self._interleave(inputs)


class SCINet(tf.keras.layers.Layer):
    def __init__(self, horizon=2, features=1, levels=3, h=4, kernel_size=5,
                 kernel_regularizer=None, activity_regularizer=None, train_mode=False, name='scinet', **kwargs):
        """
        :param horizon: number of time stamps in output
        :param levels: height of the binary tree + 1
        :param h: scaling factor for convolutional module in each SCIBlock
        :param kernel_size: kernel size of convolutional module in each SCIBlock
        :param kernel_regularizer: kernel regularizer for the fully connected layer at the end
        :param activity_regularizer: activity regularizer for the fully connected layer at the end
        """
        if levels < 1:
            raise ValueError('Must have at least 1 level')

        super().__init__(name=name, **kwargs)
        self.horizon = horizon
        self.features = features
        self.levels = levels
        self.h = h
        self.kernel_size = kernel_size

        self.interleave = Interleave()
        self.flatten = tf.keras.layers.Flatten()

        self.dropout = tf.keras.layers.Dropout(0.01)
        self.train_mode = train_mode

        # tree of sciblocks
        self.sciblocks = [SCIBlock(features=features, kernel_size=self.kernel_size, h=self.h)
                          for _ in range(2 ** self.levels - 1)]
        self.dense = tf.keras.layers.Dense(
            self.horizon * features,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer
        )

    def build(self, input_shape):
        if input_shape[1] / 2 ** self.levels % 1 != 0:
            raise ValueError(f'timestamps {input_shape[1]} must be evenly divisible by a tree with '
                             f'{self.levels} levels')
        super().build(input_shape)

    def call(self, inputs, training=None):
        # cascade input down a binary tree of sci-blocks
        lvl_inputs = [inputs]  # inputs for current level of the tree
        for i in range(self.levels):
            i_end = 2 ** (i + 1) - 1
            i_start = i_end - 2 ** i
            lvl_outputs = [output for j, tensor in zip(range(i_start, i_end), lvl_inputs)
                           for output in self.sciblocks[j](tensor)]
            lvl_inputs = lvl_outputs

        x = self.interleave(lvl_outputs)
        x += inputs
        x = tf.ensure_shape(x, [None, inputs.shape[1], inputs.shape[2]])  # 추가..

        # not sure if this is the correct way of doing it. The paper merely said to use a fully connected layer to
        # produce an output. Can't use TimeDistributed wrapper. It would force the layer's timestamps to match that of
        # the input -- something SCINet is supposed to solve
        x = self.flatten(x)
        ## using only serving
        if training and not self.train_mode:
            x = self.dropout(x)

        x = self.dense(x)
        x = tf.reshape(x, (-1, self.horizon, self.features))

        return x

    def get_config(self):
        config = super().get_config()
        config.update({'horizon': self.horizon, 'levels': self.levels})
        return config


class StackedSCINet(tf.keras.layers.Layer):
    """Layer that implements StackedSCINet as described in the paper.
    When called, outputs a tensor of shape (K, -1, n_steps, n_features) containing the outputs of all K internal
    SCINets (e.g., output[k-1] is the output of the kth SCINet, where k is in [1, ..., K]).
    To use intermediate supervision, pass the layer's output to StackedSCINetLoss as a separate model output.
    """

    '''
    def __init__(self, horizon: int, features: int, stacks: int, levels: int, h: int, kernel_size: int,
                 kernel_regularizer=None, activity_regularizer=None, name='stacked_scinet', **kwargs):
        """
        :param horizon: number of time stamps in output
        :param stacks: number of stacked SCINets
        :param levels: number of levels for each SCINet
        :param h: scaling factor for convolutional module in each SCIBlock
        :param kernel_size: kernel size of convolutional module in each SCIBlock
        :param kernel_regularizer: kernel regularizer for each SCINet
        :param activity_regularizer: activity regularizer for each SCINet
        """
        if stacks < 2:
            raise ValueError('Must have at least 2 stacks')

        super().__init__(name=name, **kwargs)
        self.stacks = stacks
        self.scinets = [SCINet(horizon=horizon, features=features, levels=levels, h=h,
                               kernel_size=kernel_size, kernel_regularizer=kernel_regularizer,
                               activity_regularizer=activity_regularizer) for _ in range(stacks)]
    '''

    def __init__(self, horizon=2, features=1, stacks=3, levels=3, h=4, kernel_size=5,
                 kernel_regularizer=tf.keras.regularizers.L1L2(0.1, 0.1), activity_regularizer=None, train_mode=False,
                 **kwargs):
        self.horizon = horizon
        self.features = features
        self.stacks = stacks
        self.levels = levels
        self.h = h
        self.kernel_size = kernel_size
        self.kernel_regularizer = kernel_regularizer
        self.activity_regularizer = activity_regularizer

        super(StackedSCINet, self).__init__(**kwargs)
        self.scinets = [SCINet(horizon=horizon, features=features, levels=levels, h=h,
                               kernel_size=kernel_size, kernel_regularizer=kernel_regularizer,
                               activity_regularizer=activity_regularizer, train_mode=train_mode) for _ in range(stacks)]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'horizon': self.horizon,
            'features': self.features,
            'stacks': self.stacks,
            'levels': self.levels,
            'h': self.h,
            'kernel_size': self.kernel_size,
            'kernel_regularizer': self.kernel_regularizer,
            'activity_regularizer': self.activity_regularizer,
        })
        return config

    def call(self, inputs):
        outputs = []
        for scinet in self.scinets:
            x = scinet(inputs)
            outputs.append(x)  # keep each stack's output for intermediate supervision
            inputs = tf.concat([x, inputs[:, x.shape[1]:, :]], axis=1)  # X_hat_k concat X_(t-(T-tilda)+1:t)
        return tf.stack(outputs)


class Identity(tf.keras.layers.Layer):
    """Identity layer used solely for the purpose of naming model outputs and properly displaying outputs when plotting
    some multi-output models.
    Returns input without changing them.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.identity(inputs)


class StackedSCINetLoss(tf.keras.losses.Loss):
    """Compute loss for a Stacked SCINet via intermediate supervision.
    `loss = sum of mean normalised difference between each stack's output and ground truth`
    `y_pred` should be the output of a StackedSCINet layer.
    """

    def __init__(self, name='stacked_scienet_loss', **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        stacked_outputs = y_pred
        horizon = stacked_outputs.shape[2]
        errors = stacked_outputs - y_true
        loss = tf.linalg.normalize(errors, axis=3)[1]
        loss = tf.reduce_sum(loss, 2)
        loss /= horizon
        loss = tf.reduce_sum(tf.clip_by_value(loss, 1e-10, 1.0))

        return loss


class SCI:
    def __init__(self, id, config, logger):
        self.config = config
        self.logger = logger

        self.model_id = f"SCI_{id}"
        self.model_desc = 'sci'
        self.scaler_dict = dict()
        self.models = dict()

        self.features = None
        self.init_param(config)

        # SCINet model structure parameter
        self.level = 2
        self.Stack = 3

    def make_simple_stacked_scinet(self, input_shape, horizon: int, K: int, L: int, h: int, kernel_size: int,
                                   learning_rate: float, kernel_regularizer=None, activity_regularizer=None,
                                   diagram_path=None, train_mode=False):
        """Compiles a simple StackedSCINet and saves model diagram if given a path.
        Intended to be a demonstration of simple model construction. See paper for details on the hyperparameters.
        """
        inputs = tf.keras.Input(shape=(input_shape[1], input_shape[2]), name='lookback_window')
        x = StackedSCINet(horizon=horizon, features=input_shape[-1], stacks=K, levels=L, h=h,
                          kernel_size=kernel_size, kernel_regularizer=kernel_regularizer,
                          activity_regularizer=activity_regularizer, train_mode=train_mode)(inputs)
        outputs = Identity(name='outputs')(x[-1])
        intermediates = Identity(name='intermediates')(x)
        model = tf.keras.Model(inputs=inputs, outputs=[outputs, intermediates])

        model.summary()
        if diagram_path:
            tf.keras.utils.plot_model(model, to_file=diagram_path, show_shapes=True)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss={
                          'intermediates': StackedSCINetLoss()
                      },
                      metrics={'outputs': ['mse', 'mae']}
                      )

        return model

    def make_simple_scinet(self, input_shape, horizon: int, L: int, h: int, kernel_size: int, learning_rate: float,
                           kernel_regularizer=None, activity_regularizer=None, diagram_path=None):
        """Compiles a simple SCINet and saves model diagram if given a path.
        Intended to be a demonstration of simple model construction. See paper for details on the hyperparameters.
        """
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(input_shape[1], input_shape[2]), name='inputs'),
            SCINet(horizon, features=input_shape[-1], levels=L, h=h, kernel_size=kernel_size,
                   kernel_regularizer=kernel_regularizer, activity_regularizer=activity_regularizer)
        ])

        model.summary()
        if diagram_path:
            tf.keras.utils.plot_model(model, to_file=diagram_path, show_shapes=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mse',  # mse
                      metrics=['mse', 'mae']
                      )

        return model

    def train_data_preprocessing(self, train_df):
        train_df.index = pd.to_datetime(train_df.index, format="%Y-%m-%d %H:%M:%S")
        train_df["minute"] = train_df.index.map(lambda x: int((x.hour * 60 + x.minute) / 1))
        train_df["weekday"] = train_df.index.map(lambda x: x.weekday())
        train_df["yymmdd"] = train_df.index.map(lambda x: x.strftime("%Y-%m-%d"))

        return train_df

    def fit(self, df, biz_dict, train_progress=None):
        fit_start = time.time()
        df = df[~df.index.duplicated()]
        df = self.train_data_preprocessing(df)

        train_df = df.copy()
        train_cutoff = int(len(train_df) * 0.7)
        train_data, val_data = train_df[:train_cutoff], train_df[
                                                        train_cutoff:]  # df[train_cutoff:val_cutoff], df[val_cutoff:]

        for feat in self.features:
            self.logger.info(f"SCI fit start feat : {feat}")
            feat_train_data = train_data[[feat]].copy()
            feat_val_data = val_data[[feat]].copy()

            scaler = RobustScaler()
            scaled_train_data = scaler.fit_transform(feat_train_data)
            scaled_val_data = scaler.transform(feat_val_data)

            lag_length, horizon = 32, 1  # lag_length is multiple of eight
            learning_rate = 0.009
            h, kernel_size, L, K = 4, 5, self.level, self.Stack

            # select to prevent overfitting
            kernel_regularizer = tf.keras.regularizers.L1L2(0.1, 0.1)

            windows = SlidingWindow(size=lag_length + horizon, stride=1)
            window_train_data = windows.fit_transform(scaled_train_data)
            windows = SlidingWindow(size=lag_length + horizon, stride=lag_length + horizon)
            window_val_data = windows.fit_transform(scaled_val_data)

            # Split all time series segments into x and y
            X_train, Y_train = window_train_data[:, :-horizon, :], window_train_data[:, -horizon:, :]
            X_val, Y_val = window_val_data[:, :-horizon, :], window_val_data[:, -horizon:, :]

            model = self.make_simple_stacked_scinet(X_train.shape, horizon=horizon, K=K, L=L, h=h,
                                                    kernel_size=kernel_size,
                                                    learning_rate=learning_rate, kernel_regularizer=None,
                                                    diagram_path=None, train_mode=True)

            # select to prevent overfitting
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0, verbose=1,
                                           restore_best_weights=True)  # patience=5

            history = model.fit(X_train,
                                Y_train,
                                validation_data=(X_val, Y_val),
                                batch_size=16,
                                epochs=30,
                                )
            # callbacks=[early_stopping]) # callbacks=[early_stopping, tensorboard_callback]

            self.models[feat] = model
            self.scaler_dict[feat] = scaler

        train_time = int(time.time() - fit_start)

        train_result = {
            "from_date": self.config["date"][0],
            "to_date": self.config["date"][-1],
            "except_failure_date_list": None,
            "except_business_list": None,
            "business_list": None,
            "train_business_status": None,
            "train_mode": None,
            "outlier_mode": None,
            "results": {
                "mse": -1,
                "rmse": -1,
                "duration_time": train_time,
                "hyper_params": None,
            },
            "train_metrics": None,
        }

        self.logger.info(f"[SCI] model training finish !!")

        return train_result, None, 0, None

    def make_serving_data(self, input_df):
        input_df["tstmp"] = pd.to_datetime(input_df.time, format="%Y-%m-%d %H:%M:%S")
        input_df["dmin"] = input_df.tstmp.map(
            lambda x: int((x.hour * 60 + x.minute) / 1)
        )
        input_df["wday"] = input_df.tstmp.map(lambda x: x.weekday())

        return input_df

    def predict(self, input_df, sbiz_df):
        with TimeLogger("[SCI] Serving elapsed time :", self.logger):

            result_df = pd.DataFrame([])
            serv_df = pd.read_csv("./20220928.csv")
            serv_window = 60
            for i in range(len(serv_df) - serv_window):
                input_df = serv_df.iloc[i:serv_window + i].copy()
                input_df = self.make_serving_data(input_df)
                input_df = input_df.reset_index()
                pred_dict = dict()
                window_size = 32
                for feat in self.features:
                    feat_input_df = input_df.loc[len(input_df) - window_size:len(input_df), feat].copy()
                    scaler = self.scaler_dict[feat]
                    serv_scaled_value = scaler.transform(feat_input_df.values.reshape(-1, 1))
                    X_serv = []
                    X_serv.append(serv_scaled_value)
                    X_serv = np.array(X_serv)
                    ## Using MC Dropout serving
                    X_serv_rep = np.tile(X_serv, 30).T
                    pred_rep, _ = self.models[feat](X_serv_rep, training=True)
                    pred_rep = scaler.inverse_transform(np.squeeze(pred_rep).reshape(-1, 1))
                    y_mean = np.mean(pred_rep)
                    y_std = np.std(pred_rep)
                    pred_dict[feat] = y_mean
                    pred_dict[f"{feat}_std"] = y_std

                pred_df = pd.DataFrame(pred_dict, index=[input_df.iloc[-1]['time']])
                # make prediction intervals
                for feat in self.features:
                    pred_df[f"{feat}_lower"] = pred_df[feat] - 1.5 * pred_df[f"{feat}_std"]
                    pred_df[f"{feat}_upper"] = pred_df[feat] + 1.5 * pred_df[f"{feat}_std"]
                pred_df[pred_df < 0] = 0
                pred_df = pred_df.drop([f"{feat}_std"], axis=1)
                result_df = result_df.append(pred_df)

        result_df[result_df < 0] = 0
        result_df.to_csv(f"./sci_result_df_{self.level}_{self.Stack}.csv")

        return result_df

    def _get_model_file_path(self, model_dir):
        return os.path.join(model_dir, f"{self.model_id}.pkl")

    def save_model_files(self, model_dir, feat):
        model_file_path = os.path.join(model_dir, f"{self.model_desc}_{feat}.h5")
        self.models[feat].save(model_file_path)
        self.logger.info(f"{self.model_desc} {feat} model file saving success")

    def save(self, model_dir):
        current_dir = Path(model_dir) / self.model_desc

        try:
            if not os.path.exists(current_dir):
                os.makedirs(current_dir)

            model_pickle_file_path = self._get_model_file_path(current_dir)
            with open(model_pickle_file_path, "wb") as f:
                pickle.dump(self.features, f)
                pickle.dump(self.scaler_dict, f)

            for feat in self.features:
                self.save_model_files(current_dir, feat)

        except Exception as e:
            self.logger.exception(f"[SCI] model saving failed : {e}")
            return False

        return True

    def load(self, model_dir):

        current_dir = os.path.join(model_dir, self.model_desc)

        if not os.path.exists(current_dir):
            self.logger.info(
                f"[SCI] warning: model directory {current_dir} not exist, plz training and again try!!"
            )
            return False
        try:
            with TimeLogger("[SCI] model loading time :", self.logger):
                model_pickle_file_path = self._get_model_file_path(current_dir)

                with open(model_pickle_file_path, "rb") as f:
                    self.features = pickle.load(f)
                    self.scaler_dict = pickle.load(f)
                    self.logger.info(
                        f"{self.model_desc} model pickle file loaded successfully"
                    )
                    f.close()
                    f = None

                for feat in self.features:
                    model_file_path = os.path.join(current_dir, f"{self.model_desc}_{feat}.h5")
                    if os.path.exists(model_file_path):
                        self.logger.info(
                            f"{self.model_desc} model {feat} is loading start"
                        )
                        self.models[feat] = tf.keras.models.load_model(model_file_path, custom_objects={
                            'StackedSCINetLoss': StackedSCINetLoss(),
                            'StackedSCINet': StackedSCINet,
                            'L1L2': tf.keras.regularizers.L1L2(0.1, 0.1),
                            'Identity': Identity()})
                    else:
                        self.logger.info(f"{self.model_desc} {feat} model does not exist")

                self.logger.info(
                    f"{self.model_desc} model h5 file for all features loaded successfully"
                )
        except Exception as e:
            self.logger.exception(f"[SCI] model loading failed : {e}")
            return False

        return True