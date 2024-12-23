from modules.NN_models import build_model, get_model_memory_usage
from modules.NN_HP import gimme_hyperparameters
from modules.utilities_data import extract_params_from_weights
from modules.NN_config_parse import bin_to_used
from modules.NN_config import conf_model_setup, conf_output_setup, conf_grid_setup

import numpy as np
import argparse


if __name__ == "__main__":
    p = conf_model_setup["params"]

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--tune_hp", action="store_true")
    parser.add_argument("--add_mem", type=float, default=0.)

    parser.add_argument("--model_type", type=str, default=p["model_type"])
    parser.add_argument("--num_residuals", type=int, default=p["num_residuals"])
    parser.add_argument("--num_nodes", type=int, default=p["num_nodes"])
    parser.add_argument("--kern_size", type=int, default=p["kern_size"])
    parser.add_argument("--batch_size", type=float, default=p["batch_size"])

    parser.add_argument("--model_to_retrain", type=str, default="")

    parser.add_argument("--used_quantities", type=str, default=conf_output_setup["bin_code"])

    parser.add_argument("--patch_size", type=int, default=conf_grid_setup["patch_size"])

    args, _ = parser.parse_known_args()

    used_quantities = bin_to_used(bin_code=args.used_quantities)
    metrics = conf_model_setup["params"]["metrics"]  # this does not change the estimated job size
    patch_size = args.patch_size

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
            kwargs |= extract_params_from_weights(args.model_to_retrain, subfolder="HMI_to_SOT")

        params = p | kwargs
        N = 1.  # 1 model in memory

    model = build_model(input_shape=(patch_size, patch_size, np.sum(used_quantities)), params=params, metrics=metrics,
                        used_quantities=used_quantities)

    mem_gb = get_model_memory_usage(model, params["batch_size"])

    # print usage for qsub
    print(f"{np.clip(np.ceil(1.5 * mem_gb * N + args.add_mem), a_min=1., a_max=None):.0f}")  # scaling to real usage
