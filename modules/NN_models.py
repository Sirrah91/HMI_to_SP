import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from tensorflow.keras.layers import Input, Conv2D, Dropout, Activation, add, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
import tensorflow.keras.optimizers as opt
from tensorflow.keras.constraints import MaxNorm
from keras_tuner import HyperModel
import numpy as np
import keras_tuner as kt

from modules.utilities import to_list
from modules.NN_losses_metrics_activations import gimme_metrics, gimme_loss, my_output_activation
from modules.NN_layers import ReflectionPadding2D

from modules._constants import _wp, _num_eps

# defaults only
from modules.NN_config import conf_model_setup, conf_output_setup

K.set_epsilon(_num_eps)
K.set_floatx(str(_wp).split(".")[-1].split("'")[0])


class MyHyperModel(HyperModel):
    # hp = keras_tuner.HyperParameters()

    def __init__(self, input_shape: tuple[int, ...],
                 params: dict[str, str | int | float | bool | list[int]] | None = None,
                 metrics: list[str] | None = None,
                 used_quantities: np.ndarray | None = None
                 ):
        self._input_shape = input_shape
        self._params = params
        self._metrics = metrics
        self._used_quantities = used_quantities

        self._for_tuning = "tuning_method" in params.keys()

        if self._params is None: self._params = conf_model_setup["params"]
        if self._metrics is None: self._metrics = conf_model_setup["params"]["metrics"]
        if self._used_quantities is None: self._used_quantities = conf_output_setup["used_quantities"]

    def _common_hp(self, hp):
        self._activation_input = hp.Choice("input_activation", values=to_list(self._params["input_activation"]))
        # self._activation_output = hp.Choice("output_activation", values=to_list(self._params["output_activation"]))

        self._kernel_size = hp.Int("kernel_size", min_value=np.min(self._params["kern_size"]),
                                   max_value=np.max(self._params["kern_size"]), step=2)
        self._kernel_padding = hp.Choice("kernel_padding", values=to_list(self._params["kern_pad"]))

        self._opt_choice = hp.Choice("optimizer", values=to_list(self._params["optimizer"]))

        # add learning rate to conditional_scope for optimizer?
        self._lr = hp.Float("learning_rate", min_value=np.min(self._params["learning_rate"]),
                            max_value=np.max(self._params["learning_rate"]), sampling="log")

        self._drop_in_hid = hp.Float("dropout_input_hidden", min_value=np.min(self._params["dropout_input_hidden"]),
                                     max_value=np.max(self._params["dropout_input_hidden"]), step=0.05)
        self._drop_hid_out = hp.Float("dropout_hidden_output", min_value=np.min(self._params["dropout_hidden_output"]),
                                      max_value=np.max(self._params["dropout_hidden_output"]), step=0.1)

        if self._for_tuning:
            L1_trade_off = np.clip(self._params["L1_trade_off"], a_min=K.epsilon(), a_max=None)
            L2_trade_off = np.clip(self._params["L2_trade_off"], a_min=K.epsilon(), a_max=None)
            sampling = "log"
        else:
            L1_trade_off = self._params["L1_trade_off"]
            L2_trade_off = self._params["L2_trade_off"]
            sampling = "linear"  # Allow for zero min_value

        self._l1 = hp.Float("L1_trade_off", min_value=np.min(L1_trade_off), max_value=np.max(L1_trade_off),
                            sampling=sampling)
        self._l2 = hp.Float("L2_trade_off", min_value=np.min(L2_trade_off), max_value=np.max(L2_trade_off),
                            sampling=sampling)

        self._max_norm = hp.Float("max_norm", min_value=np.min(self._params["max_norm"]),
                                  max_value=np.max(self._params["max_norm"]))

        self._batch_size = hp.Int("batch_size", min_value=np.min(self._params["batch_size"]),
                                  max_value=np.max(self._params["batch_size"]), step=4)
        self._bs_norm_before_activation = hp.Choice("batch_norm_before_activation",
                                                    values=to_list(self._params["bs_norm_before_activation"]))

    def _cnn_residual_hp(self, hp):
        self._num_residuals = hp.Int("num_residuals", min_value=np.min(self._params["num_residuals"]),
                                     max_value=np.max(self._params["num_residuals"]), step=1)

        if self._num_residuals > 1:
            with hp.conditional_scope("num_residuals", list(range(2, np.max(self._params["num_residuals"]) + 1))):
                self._drop_res_res = hp.Float("dropout_residual_residual",
                                              min_value=np.min(self._params["dropout_residual_residual"]),
                                              max_value=np.max(self._params["dropout_residual_residual"]), step=0.1)

        default = None if self._for_tuning else self._params["num_nodes"]
        self._filters = hp.Int("num_filters", default=default,
                               min_value=np.min(self._params["num_nodes"]),
                               max_value=np.max(self._params["num_nodes"]), step=4)

    def _cnn_standard_hp(self, hp):
        self._num_layers = hp.Int("num_layers", min_value=np.min(self._params["num_layers"]),
                                  max_value=np.max(self._params["num_layers"]), step=1)

        if self._num_layers > 1:
            with hp.conditional_scope("num_layers", list(range(2, np.max(self._params["num_layers"]) + 1))):
                self._drop_hid_hid = hp.Float("dropout_hidden_hidden",
                                              min_value=np.min(self._params["dropout_hidden_hidden"]),
                                              max_value=np.max(self._params["dropout_hidden_hidden"]), step=0.1)

        self._filters = {}
        for i in range(self._num_layers):
            if i < self._num_layers:
                with hp.conditional_scope("num_layers", list(range(i + 1, np.max(self._params["num_layers"]) + 1))):
                    default = None if self._for_tuning else self._params["num_nodes"][i]
                    self._filters[f"num_filters_{i}"] = hp.Int(f"num_filters_{i}",
                                                               default=default,
                                                               min_value=np.min(self._params["num_nodes"]),
                                                               max_value=np.max(self._params["num_nodes"]), step=4)

    def _return_optimizer(self, hp):
        # Is the conditional_scope needed here?
        if self._opt_choice == "Adam":
            with hp.conditional_scope("optimizer", ["Adam"]):
                optimizer_tuner = opt.Adam(learning_rate=self._lr)
        elif self._opt_choice == "SGD":
            with hp.conditional_scope("optimizer", ["SGD"]):
                optimizer_tuner = opt.SGD(learning_rate=self._lr, nesterov=True)
        else:
            raise ValueError(f'Unknown optimizer. Must be one of "Adam" or "SGD" but is "{self._opt_choice}". '
                             f'Add it to MyHyperModel._return_optimizer in NN_models.py.')

        return optimizer_tuner

    def _return_loss_and_metrics(self, hp):
        metrics = gimme_metrics(metrics=self._metrics, all_to_one=True)

        if self._for_tuning:
            c = np.clip(self._params["c"], a_min=K.epsilon(), a_max=None)
            sampling_c = "log"
        else:
            c = self._params["c"]
            sampling_c = "linear"  # Allow for zero min_value

        # no need of alpha if no continuum or only continuum
        if not self._used_quantities[0] or not np.any(self._used_quantities[1:]):
            self._alpha = hp.Choice("alpha", values=[1.])

        else:
            if self._for_tuning:
                alpha = np.clip(self._params["alpha"], a_min=K.epsilon(), a_max=None)
                sampling_a = "log"
            else:
                alpha = self._params["alpha"]
                sampling_a = "linear"  # Allow for zero min_value

            self._alpha = hp.Float("alpha", min_value=np.min(alpha), max_value=np.max(alpha), sampling=sampling_a)
        self._c = hp.Float("c", min_value=np.min(c), max_value=np.max(c), sampling=sampling_c)

        self._loss_type = self._params["loss_type"]
        loss = gimme_loss(used_quantities=self._used_quantities, alpha=self._alpha, c=self._c, loss_type=self._loss_type)

        return loss, metrics

    def _return_output_activation(self):
        return my_output_activation(used_quantities=self._used_quantities)

    def _residual(self, inputs):
        x = ReflectionPadding2D(self._kernel_size // 2)(inputs) if self._kernel_padding == "valid" else inputs
        x = Conv2D(filters=self._filters,
                   kernel_size=self._kernel_size,
                   padding=self._kernel_padding,
                   kernel_initializer="he_normal",
                   kernel_regularizer=regularizers.l1_l2(l1=self._l1, l2=self._l2)
                   )(x)

        if self._bs_norm_before_activation:
            x = BatchNormalization()(x)
            x = Activation(self._activation_input)(x)
        else:
            x = Activation(self._activation_input)(x)
            x = BatchNormalization()(x)

        x = ReflectionPadding2D(self._kernel_size // 2)(x) if self._kernel_padding == "valid" else x
        x = Conv2D(filters=self._filters,
                   kernel_size=self._kernel_size,
                   padding=self._kernel_padding,
                   kernel_initializer="he_normal",
                   kernel_regularizer=regularizers.l1_l2(l1=self._l1, l2=self._l2)
                   )(x)

        x = BatchNormalization()(x)
        x = add([x, inputs])

        return x

    def _hidden_layers(self, inputs):  # ResNet architecture
        x = Dropout(rate=self._drop_in_hid)(inputs)

        # initial convolution layer
        x = ReflectionPadding2D(self._kernel_size // 2)(x) if self._kernel_padding == "valid" else x
        x = Conv2D(filters=self._filters,
                   kernel_size=self._kernel_size,
                   padding=self._kernel_padding,
                   kernel_initializer="he_normal",
                   kernel_regularizer=regularizers.l1_l2(l1=self._l1, l2=self._l2)
                   )(x)

        x = Activation(self._activation_input)(x)

        # building residual blocks
        for i in range(self._num_residuals):
            x = self._residual(x)

        # final convolution layer
        x = ReflectionPadding2D(self._kernel_size // 2)(x) if self._kernel_padding == "valid" else x
        x = Conv2D(filters=np.shape(inputs)[-1],
                   kernel_size=self._kernel_size,
                   padding=self._kernel_padding,
                   kernel_initializer="he_normal",
                   kernel_regularizer=regularizers.l1_l2(l1=self._l1, l2=self._l2)
                   )(x)

        # adding the input to the output (skip connection)
        outputs = add([x, inputs])

        return outputs

    def _build_CNN(self):
        # Input layer
        inputs = Input(shape=self._input_shape)
        x = self._hidden_layers(inputs)
        outputs = Activation(self._return_output_activation())(x)

        return Model(inputs=inputs, outputs=outputs)

    def _build_CNN_separate(self):
        # Input layer
        inputs = Input(shape=self._input_shape)
        # i:i+1 to keep the last dimension ([i] does not work in tensorflow)
        x = Concatenate()([self._hidden_layers(inputs[..., i:i+1]) for i in range(self._input_shape[-1])])
        outputs = Activation(self._return_output_activation())(x)

        return Model(inputs=inputs, outputs=outputs)

    def _build_CNN_classic(self):
        # Input layer
        inputs = Input(shape=self._input_shape)

        # Dropout between input and the first hidden layer
        x = Dropout(rate=self._drop_in_hid)(inputs)

        # Adding hidden layers
        for i in range(self._num_layers):
            x = ReflectionPadding2D(self._kernel_size // 2)(x) if self._kernel_padding == "valid" else x
            x = Conv2D(filters=self._filters[f"num_filters_{i}"],
                       kernel_size=self._kernel_size,
                       padding=self._kernel_padding,
                       kernel_constraint=MaxNorm(self._max_norm),
                       bias_constraint=MaxNorm(self._max_norm),
                       kernel_regularizer=regularizers.l1_l2(l1=self._l1, l2=self._l2),
                       bias_regularizer=regularizers.l1_l2(l1=self._l1, l2=self._l2)
                       )(x)

            if self._bs_norm_before_activation:
                x = BatchNormalization()(x)
                x = Activation(self._activation_input)(x)
            else:
                x = Activation(self._activation_input)(x)
                x = BatchNormalization()(x)

            # Dropout layer for stabilisation of the network
            if i < self._num_layers - 1:  # Last layer has different dropout
                x = Dropout(rate=self._drop_hid_hid)(x)

        x = Dropout(rate=self._drop_hid_out)(x)

        x = Conv2D(filters=np.shape(inputs)[-1],
                   kernel_size=(1, 1),
                   padding="same",
                   kernel_constraint=MaxNorm(self._max_norm),
                   bias_constraint=MaxNorm(self._max_norm),
                   kernel_regularizer=regularizers.l1_l2(l1=self._l1, l2=self._l2),
                   bias_regularizer=regularizers.l1_l2(l1=self._l1, l2=self._l2)
                   )(x)
        outputs = Activation(self._return_output_activation())(x)

        return Model(inputs=inputs, outputs=outputs)

    def build(self, hp):
        self._model_type = hp.Choice("model_type", values=to_list(self._params["model_type"]))

        if self._model_type == "CNN":
            with hp.conditional_scope("model_type", ["CNN"]):
                self._common_hp(hp)
                self._cnn_residual_hp(hp)
                model = self._build_CNN()
                optimizer_tuner = self._return_optimizer(hp)
        elif self._model_type == "CNN_sep":
            with hp.conditional_scope("model_type", ["CNN_sep"]):
                self._common_hp(hp)
                self._cnn_residual_hp(hp)
                model = self._build_CNN_separate()
                optimizer_tuner = self._return_optimizer(hp)
        elif self._model_type == "CNN_classic":
            with hp.conditional_scope("model_type", ["CNN_classic"]):
                self._common_hp(hp)
                self._cnn_standard_hp(hp)
                model = self._build_CNN_standard()
                optimizer_tuner = self._return_optimizer(hp)
        elif self._model_type == "test":
            with hp.conditional_scope("model_type", ["test"]):
                self._common_hp(hp)
                self._cnn_standard_hp(hp)
                model = self._test()
                optimizer_tuner = self._return_optimizer(hp)
        else:
            raise ValueError(f'Unknown model type. Must be one of "CNN", "CNN_sep", or "CNN_classic"'
                             f' but is "{self._model_type}".\n'
                             f'Add it to MyHyperModel.build in NN_models.py.')

        # Compiling the model to train. Declare the loss function and the optimizer (SGD, Adam, etc.)
        loss, metrics = self._return_loss_and_metrics(hp)
        model.compile(loss=loss, metrics=metrics, optimizer=optimizer_tuner)

        return model

    def fit(self, hp, model: Model, *args, **kwargs):

        if "batch_size" in kwargs:
            del kwargs["batch_size"]
        if "verbose" in kwargs:
            del kwargs["verbose"]

        return model.fit(
            *args,
            # Tune batch size
            batch_size=self._batch_size,
            verbose=0,
            **kwargs,
        )


def build_model(input_shape: tuple[int, ...],
                params: dict[str, str | int | float | bool | list[int]] | None = None,
                metrics: list[str] | None = None,
                used_quantities: np.ndarray | None = None) -> Model:
    if params is None: params = conf_model_setup["params"]
    if metrics is None: metrics = conf_model_setup["params"]["metrics"]
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    if params["model_type"] in ["CNN", "CNN_sep", "CNN_classic", "test"]:
        hypermodel = MyHyperModel(input_shape=input_shape,
                                  params=params,
                                  metrics=metrics,
                                  used_quantities=used_quantities)
        model = hypermodel.build(kt.HyperParameters())
    else:
        raise NameError('unknown p["model_type"]')

    return model
