import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.activations import softplus, linear
import tensorflow.experimental.numpy as tnp
from tensorflow.python.framework.ops import EagerTensor
from typing import Callable, Literal
from collections import Counter

from modules._constants import _wp, _num_eps

# defaults only
from modules.NN_config import conf_output_setup

K.set_epsilon(_num_eps)
K.set_floatx(str(_wp).split(".")[-1].split("'")[0])


def my_mse_loss(used_quantities: np.ndarray | None = None,
                alpha: float = 1.
                ) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    if alpha < 0.:
        raise ValueError(f'"alpha" must be a non-negative number but is {alpha}.')

    # no need of alpha if no continuum or only continuum
    if not used_quantities[0] or not np.any(used_quantities[1:]):
        # @tf.function
        @tf.autograph.experimental.do_not_convert
        def mse_loss(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
            return K.mean(K.square(y_true - y_pred))
    else:
        alpha_full = K.arange(len(used_quantities), dtype=_wp)
        alpha_full = K.cast(tf.where(alpha_full == 0, float(alpha), 1.)[used_quantities], dtype=_wp)

        # @tf.function
        @tf.autograph.experimental.do_not_convert
        def mse_loss(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
            return K.mean(alpha_full * K.square(y_true - y_pred))

    return mse_loss


def my_cauchy_loss(used_quantities: np.ndarray | None = None, c: float = 0.1, alpha: float = 1.
                   ) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]
    # low c -> high sensitivity to outliers

    if alpha < 0.:
        raise ValueError(f'"alpha" must be a non-negative number but is {alpha}.')

    if c <= 0.:
        raise ValueError(f'"c" must be a positive number but is {c}.')

    # no need of alpha if no continuum or only continuum
    if not used_quantities[0] or not np.any(used_quantities[1:]):
        # @tf.function
        @tf.autograph.experimental.do_not_convert
        def cauchy_loss(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
            return K.mean(tf.math.log1p(tf.square((y_true - y_pred) / c)))
    else:
        alpha_full = K.arange(len(used_quantities), dtype=_wp)
        alpha_full = K.cast(tf.where(alpha_full == 0, float(alpha), 1.)[used_quantities], dtype=_wp)

        # @tf.function
        @tf.autograph.experimental.do_not_convert
        def cauchy_loss(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
            return K.mean(alpha_full * tf.math.log1p(tf.square((y_true - y_pred) / c)))

    return cauchy_loss


def my_ssim_loss() -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    @tf.autograph.experimental.do_not_convert
    def ssim_loss(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        # Compute the rank of the tensors
        rank_true = tf.rank(y_true)
        rank_pred = tf.rank(y_pred)

        # Generate the axes for reduction (all axes except the last one)
        axes_true = tf.range(rank_true - 1)
        axes_pred = tf.range(rank_pred - 1)

        # Compute max and min values for each component
        max_vals_true = tf.reduce_max(y_true, axis=axes_true)
        min_vals_true = tf.reduce_min(y_true, axis=axes_true)
        max_vals_pred = tf.reduce_max(y_pred, axis=axes_pred)
        min_vals_pred = tf.reduce_min(y_pred, axis=axes_pred)

        # Calculate SSIM for each component
        ssim_components = []
        for i in range(np.shape(y_true)[-1]):
            y_pred_rescaled = (y_pred[..., i] - min_vals_pred[i]) / (max_vals_pred[i] - min_vals_pred[i])
            y_true_rescaled = (y_true[..., i] - min_vals_true[i]) / (max_vals_true[i] - min_vals_true[i])
            ssim_comp = tf.image.ssim(y_true_rescaled, y_pred_rescaled, max_val=1.)
            ssim_components.append(ssim_comp)

        # Calculate the mean SSIM across components
        ssim_mean = tf.reduce_mean(ssim_components)

        # Return 1 - mean SSIM as the loss (to minimize)
        return 1. - ssim_mean

    return ssim_loss


def my_ae() -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    @tf.autograph.experimental.do_not_convert
    def ae(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        return K.abs(y_true - y_pred)

    return ae


def my_mae(all_to_one: bool = False) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    @tf.autograph.experimental.do_not_convert
    def mae(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        axis = tuple(range(K.ndim(K.cast(y_true, dtype=_wp)) - (not all_to_one)))

        return K.reshape(tnp.nanmean(my_ae()(y_true, y_pred), axis=axis), (-1,))

    return mae


def my_mse(all_to_one: bool = False) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    @tf.autograph.experimental.do_not_convert
    def mse(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        axis = tuple(range(K.ndim(K.cast(y_true, dtype=_wp)) - (not all_to_one)))

        return K.reshape(tnp.nanmean(K.square(my_ae()(y_true, y_pred)), axis=axis), (-1,))

    return mse


def my_rmse(all_to_one: bool = False) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    @tf.autograph.experimental.do_not_convert
    def rmse(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:

        return K.reshape(K.sqrt(my_mse(all_to_one=all_to_one)(y_true, y_pred)), (-1,))

    return rmse


def my_Lp_norm(p_coef: float, all_to_one: bool = False) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    if p_coef < 1.:
        raise ValueError("p_coef >= 1 in Lp_norm.")

    @tf.autograph.experimental.do_not_convert
    def Lp_norm(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        abs_error = my_ae()(y_true, y_pred)
        axis = tuple(range(K.ndim(K.cast(y_true, dtype=_wp)) - (not all_to_one)))

        return K.reshape(K.pow(tnp.nansum(K.pow(abs_error, p_coef), axis=axis), 1. / p_coef), (-1,))

    return Lp_norm


def my_quantile(percentile: np.ndarray | float,
                all_to_one: bool = False) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    if not np.all(np.logical_and(percentile >= 0., percentile <= 100.)):
        raise ValueError("Percentile must be in the range [0, 100].")

    @tf.autograph.experimental.do_not_convert
    def quantile(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        abs_error = my_ae()(y_true, y_pred)
        axis = tuple(range(K.ndim(K.cast(y_true, dtype=_wp)) - (not all_to_one)))

        res = tf.numpy_function(lambda error, perc:
                                K.cast(np.nanpercentile(error, perc, method="median_unbiased", axis=axis), dtype=_wp),
                                inp=[abs_error, percentile], Tout=_wp)
        if axis is None:
            return K.reshape(res, (-1, 1))
        return res

    return quantile


def my_r2(all_to_one: bool = False) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    @tf.autograph.experimental.do_not_convert
    def r2(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        axis = tuple(range(K.ndim(K.cast(y_true, dtype=_wp)) - (not all_to_one)))

        SS_res = tnp.nansum(K.square(y_true - y_pred), axis=axis)
        SS_tot = tnp.nansum(K.square(y_true - tnp.nanmean(y_true, axis=axis)), axis=axis)

        SS_tot = K.clip(SS_tot, K.epsilon(), None)

        return K.reshape(1.0 - SS_res / SS_tot, (-1,))

    return r2


def my_sam(all_to_one: bool = False) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    @tf.autograph.experimental.do_not_convert
    def sam(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        axis = tuple(range(K.ndim(K.cast(y_true, dtype=_wp)) - (not all_to_one)))

        s1_s2_norm = K.sqrt(tnp.nansum(K.square(y_true), axis=axis)) * K.sqrt(tnp.nansum(K.square(y_pred), axis=axis))
        sum_s1_s2 = tnp.nansum(y_true * y_pred, axis=axis)

        s1_s2_norm = K.clip(s1_s2_norm, K.epsilon(), None)

        tf.math.acos(sum_s1_s2 / s1_s2_norm)

        return K.reshape(tf.math.acos(sum_s1_s2 / s1_s2_norm), (-1,))

    return sam


def my_output_activation(used_quantities: np.ndarray | None = None) -> Callable[[EagerTensor], EagerTensor]:
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    @tf.function
    def output_activation(x: EagerTensor) -> EagerTensor:
        x_new = K.zeros_like(x[..., 0:0])
        used_indices = np.where(used_quantities)[0]

        for i, q in enumerate(used_indices):
            if q == 0:  # intensity -> softplus
                tmp = softplus(x[..., i:i+1])
            else:  # B -> linear
                tmp = linear(x[..., i:i+1])

            x_new = K.concatenate([x_new, tmp], axis=-1)

        return x_new

    return output_activation


def gimme_loss(used_quantities: np.ndarray | None = None,
               alpha: float = 1., c: float = 0.1, loss_type: Literal["mse", "Cauchy", "SSIM"] = "mse"):
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    if loss_type.lower() == "mse":
        return my_mse_loss(used_quantities=used_quantities, alpha=alpha)
    elif loss_type.lower() == "Cauchy":
        return my_cauchy_loss(used_quantities=used_quantities, c=c, alpha=alpha)
    else:
        return my_ssim_loss()


def gimme_metrics(metrics: list | tuple, all_to_one: bool = True) -> list[str]:
    custom_objects = create_custom_objects(all_to_one=all_to_one)

    # Metrics used in model training
    metrics = [custom_objects[met] for met in metrics if met in custom_objects]

    # to make the list of metrics unique (just in case...)
    metrics = list(Counter(metrics).keys())

    return metrics


def create_custom_objects(used_quantities: np.ndarray | None = None,
                          alpha: float = 1., c: float = 0.1, loss_type: Literal["mse", "Cauchy", "SSIM"] = "mse",
                          p_coef: float = 1.5, percentile: float = 50., all_to_one: bool = True) -> dict:
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    # losses
    loss_composition = gimme_loss(used_quantities=used_quantities, alpha=alpha, c=c, loss_type=loss_type)
    loss_composition_name = loss_composition.__name__

    # metrics
    mae = my_mae(all_to_one=all_to_one)
    mse = my_mse(all_to_one=all_to_one)
    rmse = my_rmse(all_to_one=all_to_one)
    Lp_norm = my_Lp_norm(p_coef=p_coef, all_to_one=all_to_one)
    quantile = my_quantile(percentile=percentile, all_to_one=all_to_one)
    r2 = my_r2(all_to_one=all_to_one)
    sam = my_sam(all_to_one=all_to_one)

    mae_name, mse_name, rmse_name = mae.__name__, mse.__name__, rmse.__name__
    Lp_norm_name, quantile_name, r2_name, sam_name = Lp_norm.__name__, quantile.__name__, r2.__name__, sam.__name__

    # activation functions
    output_activation = my_output_activation(used_quantities=used_quantities)
    activation_name = output_activation.__name__,

    custom_objects = {loss_composition_name: loss_composition,
                      #
                      mse_name: mse,
                      rmse_name: rmse,
                      quantile_name: quantile,
                      mae_name: mae,
                      Lp_norm_name: Lp_norm,
                      r2_name: r2,
                      sam_name: sam,
                      #
                      activation_name: output_activation,
                      }

    return custom_objects
