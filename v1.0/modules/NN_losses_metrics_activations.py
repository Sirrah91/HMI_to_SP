from modules._constants import _wp, _num_eps
from modules.NN_layers import ReflectionPadding2D
# defaults only
from modules.NN_config import conf_output_setup

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.activations import softplus, linear
import tensorflow.experimental.numpy as tnp
from tensorflow.python.framework.ops import EagerTensor
from typing import Callable, Literal
from collections import Counter

K.set_epsilon(_num_eps)
K.set_floatx(str(_wp).split(".")[-1].split("'")[0])


def create_alpha_full(used_quantities: np.ndarray, alpha: float) -> EagerTensor:
    alpha_full = K.arange(len(used_quantities), dtype=_wp)
    alpha_full = K.cast(tf.where(alpha_full == 0, float(alpha), 1.)[used_quantities], dtype=_wp)

    return alpha_full


def input_check(alpha: float | None = None,
                c: float | None = None,
                percentile: np.ndarray | float | None = None,
                p_coef: float | None = None) -> None:
    if alpha is not None and alpha < 0.:
        raise ValueError(f'Trade off parameter "alpha" must be a non-negative number but is {alpha}.')

    if c is not None and c <= 0.:
        raise ValueError(f'Cauchy loss trade off parameter "c" must be a positive number but is {c}.')

    if percentile is not None and not np.all(np.logical_and(percentile >= 0., percentile <= 100.)):
        raise ValueError("percentile must be in the range [0, 100].")

    if p_coef is not None and p_coef < 1.:
        raise ValueError(f'Lp norm "p_coef" parameter must be larger or equal to 1 but is {p_coef}.')


def gimme_axis(array: EagerTensor | np.ndarray, all_to_one) -> tuple[int] | None:
    axis = tuple(range(K.ndim(K.cast(array, dtype=_wp)) - (not all_to_one)))

    if not axis:
        return None
    return axis


def calc_train_weights(y_true: np.ndarray | EagerTensor, bins: tuple | list,
                       mu: float = 0.15) -> list | tuple:
    final_weights = []

    for i in range(len(bins)):
        # Select the i-th label
        y_true_label = y_true[..., i]
        bin = bins[i]

        # Determine which bin each value in y_true belongs to
        bin_indices = np.digitize(y_true_label, bins=bin)
        """
        bin_indices = tf.numpy_function(lambda _arr, _bin: K.cast(np.digitize(_arr, bins=_bin), dtype=tf.int32),
                                        inp=[y_true_label, bin], Tout=tf.int32)
        """
        # Compute counts in each bin
        # +1 in num_bins because digitize in principle creates output in range [0, len(bins)]
        # that has length = num_bins + 1 (0 and len(bins) are for edges)
        counts = np.array([np.sum(bin_indices == i) for i in range(len(bin) + 1)])
        total = np.sum(counts)
        """"
        counts = tf.numpy_function(lambda indices: K.cast([np.sum(indices == i) for i in range(len(bin) + 1)], dtype=_wp),
                                   inp=[bin_indices], Tout=_wp)
        total = K.sum(counts)  # Total count for normalization
        """
        # Avoid division by zero
        counts[counts < 1.] = 1.
        # counts = tf.where(counts > 1., counts, 1.)

        # Compute the weights for each bin (lower populated bins get higher weights)
        weights = np.log(mu * total / counts)
        # weights = K.log(mu * total / counts)

        # Ensure weights are >= 1
        weights[weights < 1.] = 1.
        # weights = tf.where(weights > 1., weights, 1.)

        final_weights.append(np.array(weights, dtype=_wp))

    return final_weights


def create_regression_weights(y_true: EagerTensor,
                              weights: tuple | list | None,
                              bins: tuple | list | None) -> EagerTensor | float:
    if weights is None or bins is None:
        return 1.

    # Create an empty tensor to hold the final weights
    final_weights = []

    for i in range(len(bins)):
        # Select the i-th label
        y_true_label = y_true[..., i]
        bin = bins[i]

        # Determine which bin each value in y_true belongs to
        bin_indices = tf.numpy_function(lambda _arr, _bin: K.cast(np.digitize(_arr, bins=_bin), dtype=tf.int32),
                                        inp=[y_true_label, bin], Tout=tf.int32)

        # Gather the corresponding weights for each bin index
        label_weights = tf.gather(weights[i], bin_indices, axis=0)

        # Add the weights for the current label to the final list
        final_weights.append(label_weights)

    # Stack the weights for each label to create the final weights tensor
    final_weights_tensor = K.stack(final_weights, axis=-1)

    return final_weights_tensor


def my_mse_loss(used_quantities: np.ndarray | None = None,
                alpha: float = 1.,
                weights: EagerTensor | None = None, bins: tuple | list | None = None
                ) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    input_check(alpha=alpha)

    # no need of alpha if no continuum or only continuum
    if not used_quantities[0] or not np.any(used_quantities[1:]):
        # @tf.function
        @tf.autograph.experimental.do_not_convert
        def mse_loss(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
            weights_tensor = create_regression_weights(y_true=y_true, weights=weights, bins=bins)
            return K.mean(K.square(weights_tensor * (y_true - y_pred)))
    else:
        alpha_full = create_alpha_full(alpha=alpha, used_quantities=used_quantities)

        # @tf.function
        @tf.autograph.experimental.do_not_convert
        def mse_loss(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
            weights_tensor = create_regression_weights(y_true=y_true, weights=weights, bins=bins)
            return K.mean(alpha_full * K.square(weights_tensor * (y_true - y_pred)))

    return mse_loss


def my_cauchy_loss(used_quantities: np.ndarray | None = None,
                   c: float = 0.1, alpha: float = 1.,
                   weights: EagerTensor | None = None, bins: tuple | list | None = None,
                   ) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]
    # low c -> high sensitivity to outliers

    input_check(alpha=alpha, c=c)

    # no need of alpha if no continuum or only continuum
    if not used_quantities[0] or not np.any(used_quantities[1:]):
        # @tf.function
        @tf.autograph.experimental.do_not_convert
        def cauchy_loss(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
            weights_tensor = create_regression_weights(y_true=y_true, weights=weights, bins=bins)
            return K.mean(tf.math.log1p(tf.square(weights_tensor * (y_true - y_pred) / c)))
    else:
        alpha_full = create_alpha_full(alpha=alpha, used_quantities=used_quantities)

        # @tf.function
        @tf.autograph.experimental.do_not_convert
        def cauchy_loss(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
            weights_tensor = create_regression_weights(y_true=y_true, weights=weights, bins=bins)
            return K.mean(alpha_full * tf.math.log1p(tf.square(weights_tensor * (y_true - y_pred) / c)))

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
    def ae(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        return K.abs(y_true - y_pred)

    return ae


def my_mae(all_to_one: bool = False) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    def _mae(array1: EagerTensor, array2: EagerTensor, axis: tuple[int, ...] | None = None) -> EagerTensor:
        return tnp.nanmean(a=my_ae()(array1, array2), axis=axis)

    def mae(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        if all_to_one:
            return K.reshape(_mae(y_true, y_pred), shape=(-1,))
        else:
            return K.reshape([_mae(y_true[..., i], y_pred[..., i]) for i in range(np.shape(y_true)[-1])], shape=(-1,))

        """
        # THIS IS NUMERICALLY UNSTABLE FOR LARGE ARRAYS
        axis = gimme_axis(array=y_true, all_to_one=all_to_one)
        return K.reshape(_mae(y_true, y_pred, axis=axis), shape=(-1,))
        """

    return mae


def my_mse(all_to_one: bool = False) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    def _mse(array1: EagerTensor, array2: EagerTensor, axis: tuple[int, ...] | None = None) -> EagerTensor:
        return tnp.nanmean(a=K.square(my_ae()(array1, array2)), axis=axis)

    @tf.autograph.experimental.do_not_convert
    def mse(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        if all_to_one:
            return K.reshape(_mse(y_true, y_pred), shape=(-1,))
        else:
            return K.reshape([_mse(y_true[..., i], y_pred[..., i]) for i in range(np.shape(y_true)[-1])], shape=(-1,))

        """
        # THIS IS NUMERICALLY UNSTABLE FOR LARGE ARRAYS
        axis = gimme_axis(array=y_true, all_to_one=all_to_one)
        return K.reshape(_mse(y_true, y_pred, axis=axis), shape=(-1,))
        """

    return mse


def my_rmse(all_to_one: bool = False) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    @tf.autograph.experimental.do_not_convert
    def rmse(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        return K.reshape(K.sqrt(my_mse(all_to_one=all_to_one)(y_true, y_pred)), shape=(-1,))

    return rmse


def my_Lp_norm(p_coef: float, all_to_one: bool = False) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    input_check(p_coef=p_coef)

    def _Lp_norm(array1: EagerTensor, array2: EagerTensor, axis: tuple[int, ...] | None = None) -> EagerTensor:
        return K.pow(tnp.nansum(K.pow(my_ae()(array1, array2), p_coef), axis=axis), 1. / p_coef)

    def Lp_norm(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        if all_to_one:
            return K.reshape(_Lp_norm(y_true, y_pred), shape=(-1))
        else:
            return K.reshape([_Lp_norm(y_true[..., i], y_pred[..., i]) for i in range(np.shape(y_true)[-1])], shape=(-1,))

        """
        # THIS IS NUMERICALLY UNSTABLE FOR LARGE ARRAYS
        axis = gimme_axis(array=y_true, all_to_one=all_to_one)
        return K.reshape(_Lp_norm(y_true, y_pred, axis=axis), shape=(-1,))
        """

    return Lp_norm


def my_quantile(percentile: np.ndarray | float,
                all_to_one: bool = False) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    input_check(percentile=percentile)

    def _quantile_error(array1: EagerTensor, array2: EagerTensor, axis: tuple[int, ...] | None = None) -> EagerTensor:
        return tf.numpy_function(lambda error, perc:
                                 K.cast(np.nanpercentile(error, perc, method="median_unbiased", axis=axis), dtype=_wp),
                                 inp=[my_ae()(array1, array2), percentile], Tout=_wp)

    def quantile(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        if all_to_one:
            return K.transpose(K.reshape(_quantile_error(y_true, y_pred), shape=(-1, np.size(percentile))))
        else:
            return K.transpose(K.reshape([_quantile_error(y_true[..., i], y_pred[..., i]) for i in range(np.shape(y_true)[-1])], shape=(-1, np.size(percentile))))

        """
        # THIS IS NUMERICALLY UNSTABLE FOR LARGE ARRAYS
        axis = gimme_axis(array=y_true, all_to_one=all_to_one)
        return K.transpose(K.reshape(_quantile_error(y_true, y_pred, axis=axis), shape=(-1, np.size(percentile))))
        """

    return quantile


def my_r2(all_to_one: bool = False) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    def _ss_res(array1: EagerTensor, array2: EagerTensor, axis: tuple[int, ...] | None = None) -> EagerTensor:
        return tnp.nansum(a=K.square(array1 - array2), axis=axis)

    def _ss_tot(array: EagerTensor, axis: tuple[int, ...] | None = None) -> EagerTensor:
        return tnp.nansum(a=K.square(array - tnp.nanmean(a=array, axis=axis)), axis=axis)

    def r2(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:

        if all_to_one:
            SS_res = _ss_res(y_true, y_pred)
            SS_tot = _ss_tot(y_true)
        else:
            SS_res = K.reshape([_ss_res(y_true[..., i], y_pred[..., i]) for i in range(np.shape(y_true)[-1])], shape=(-1))
            SS_tot = K.reshape([_ss_tot(y_true[..., i]) for i in range(np.shape(y_true)[-1])], shape=(-1))

        """
        # THIS IS NUMERICALLY UNSTABLE FOR LARGE ARRAYS
        axis = gimme_axis(array=y_true, all_to_one=all_to_one)
        SS_res = _ss_res(y_true, y_pred, axis=axis)
        SS_tot = _ss_tot(y_true, axis=axis)
        """

        SS_tot = K.clip(SS_tot, K.epsilon(), None)

        return K.reshape(1.0 - SS_res / SS_tot, shape=(-1,))

    return r2


def my_sam(all_to_one: bool = False) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    def _s1_s2_norm(array1: EagerTensor, array2: EagerTensor, axis: tuple[int, ...] | None = None) -> EagerTensor:
        return K.sqrt(tnp.nansum(a=K.square(array1), axis=axis)) * K.sqrt(tnp.nansum(a=K.square(array2), axis=axis))

    def _sum_s1_s2(array1: EagerTensor, array2: EagerTensor, axis: tuple[int, ...] | None = None) -> EagerTensor:
        return tnp.nansum(a=array1 * array2, axis=axis)

    def sam(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        if all_to_one:
            s1_s2_norm = _s1_s2_norm(y_true, y_pred)
            sum_s1_s2 = _sum_s1_s2(y_true, y_pred)
        else:
            s1_s2_norm = K.reshape([_s1_s2_norm(y_true[..., i], y_pred[..., i]) for i in range(np.shape(y_true)[-1])], shape=(-1))
            sum_s1_s2 = K.reshape([_sum_s1_s2(y_true[..., i], y_pred[..., i]) for i in range(np.shape(y_true)[-1])], shape=(-1))

        """
        # THIS IS NUMERICALLY UNSTABLE FOR LARGE ARRAYS
        axis = gimme_axis(array=y_true, all_to_one=all_to_one)
        s1_s2_norm = _s1_s2_norm(y_true, y_pred, axis=axis)
        sum_s1_s2 = _sum_s1_s2(y_true, y_pred, axis=axis)
        """

        s1_s2_norm = K.clip(s1_s2_norm, K.epsilon(), None)

        return K.reshape(tf.math.acos(sum_s1_s2 / s1_s2_norm), shape=(-1,))

    return sam


def my_output_activation(used_quantities: np.ndarray | None = None,
                         name: str = "softplus_linear") -> Callable[[EagerTensor], EagerTensor]:
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    if name == "softplus_linear":
        @tf.autograph.experimental.do_not_convert
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
               alpha: float = 1., c: float = 0.1, loss_type: Literal["mse", "Cauchy", "SSIM"] = "mse",
               weights: EagerTensor | None = None, bins: tuple | list | None = None):
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    if loss_type.lower() == "ssim":
        return my_ssim_loss()
    elif loss_type.lower() == "cauchy":
        return my_cauchy_loss(used_quantities=used_quantities, c=c, alpha=alpha, weights=weights, bins=bins)
    else:
        return my_mse_loss(used_quantities=used_quantities, alpha=alpha, weights=weights, bins=bins)


def gimme_metrics(metrics: list | tuple, all_to_one: bool = True) -> list[str]:
    custom_objects = create_custom_objects(all_to_one=all_to_one)

    # Metrics used in model training
    metrics = [custom_objects[met] for met in metrics if met in custom_objects]

    # to make the list of metrics unique (just in case...)
    metrics = list(Counter(metrics).keys())

    return metrics


def create_custom_objects(used_quantities: np.ndarray | None = None,
                          alpha: float = 1., c: float = 0.1, loss_type: Literal["mse", "Cauchy", "SSIM"] = "mse",
                          weights: EagerTensor | None = None,
                          p_coef: float = 1.5, percentile: float = 50., all_to_one: bool = True) -> dict:
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    # losses
    loss_composition = gimme_loss(used_quantities=used_quantities, alpha=alpha, c=c, loss_type=loss_type,
                                  weights=weights)
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
    activation_name = output_activation.__name__

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
                      #
                      ReflectionPadding2D.__name__: ReflectionPadding2D
                      }

    return custom_objects
