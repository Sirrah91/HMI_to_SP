from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import argparse
from tensorflow.keras import backend as K
from tensorflow.keras import Model

from modules.NN_models import build_model
from modules.NN_HP import gimme_hyperparameters
from modules.utilities_data import extract_params_from_weights
from modules.NN_config import conf_model_setup, conf_output_setup, conf_grid_setup
from modules._constants import _wp, _num_eps

K.set_epsilon(_num_eps)
K.set_floatx(str(_wp).split(".")[-1].split("'")[0])


def get_model_memory_usage(model: Model, batch_size: int, additional_memory: float = 0.):
    """
    Return the estimated memory usage of a given Keras model in gigabytes.
    This includes the model weights and layers, but excludes the dataset.

    The model shapes are multiplied by the batch size, but the weights are not.

    Args:
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
        additional_memory: Additional memory usage, for example data.
    Returns:
        An estimate of the Keras model's memory usage in gigabytes.

    """

    shapes_mem_count = 0.
    internal_model_mem_count = 0.

    for layer in model.layers:

        if isinstance(layer, Model):
            internal_model_mem_count += get_model_memory_usage(layer, batch_size, additional_memory=0.)

        single_layer_mem = 1.
        out_shape = layer.output_shape

        if isinstance(out_shape, list):
            out_shape = out_shape[0]

        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s

        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    if K.floatx() == 'float16':
        number_size = 2.0
    elif K.floatx() == 'float64':
        number_size = 8.0
    else:
        number_size = 4.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = total_memory / (1024.0 ** 3) + internal_model_mem_count + additional_memory

    return gbytes


if __name__ == "__main__":
    p = conf_model_setup["params"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--tune_hp", type=bool, default=False, const=True, nargs="?")
    parser.add_argument("--add_mem", type=float, default=0.)

    parser.add_argument("--model_type", type=str, default=p["model_type"])
    parser.add_argument("--num_residuals", type=int, default=p["num_residuals"])
    parser.add_argument("--num_nodes", type=int, default=p["num_nodes"])
    parser.add_argument("--kern_size", type=int, default=p["kern_size"])
    parser.add_argument("--batch_size", type=float, default=p["batch_size"])

    parser.add_argument("--model_to_retrain", type=str, default="")

    args, _ = parser.parse_known_args()

    used_quantities = conf_output_setup["used_quantities"]
    metrics = conf_model_setup["params"]["metrics"]
    patch_size = conf_grid_setup["patch_size"]

    if args.tune_hp:
        params = gimme_hyperparameters(for_tuning=True)()
        # take the model with the maximum memory usage
        params["model_type"] = "CNN_sep" if "CNN_sep" in params["model_type"] else "CNN"
        params["num_residuals"] = np.max(params["num_residuals"])
        params["num_nodes"] = np.max(params["num_nodes"])
        params["kern_size"] = np.max(params["kern_size"])
        params["batch_size"] = np.max(params["batch_size"])
        N = 50.  # 50 models in memory

    else:
        kwargs = {"model_type": args.model_type,
                  "num_residuals": args.num_residuals,
                  "num_nodes": args.num_nodes,
                  "kern_size": args.kern_size,
                  "batch_size": args.batch_size,
                  }

        if args.model_to_retrain:
            kwargs |= extract_params_from_weights(args.model_to_retrain, "HMI_to_SOT")

        params = p | kwargs
        N = 1.  # 1 model in memory

    model = build_model(input_shape=(patch_size, patch_size, np.sum(used_quantities)), params=params, metrics=metrics,
                        used_quantities=used_quantities)

    mem_gb = get_model_memory_usage(model, params["batch_size"]) * 1.5  # scaling to real usage

    # print usage for qsub
    print(f"{np.ceil(mem_gb * N + args.add_mem):.0f}")
