# --------------------------------------------------------- #
# Neural Network to enhance SDO/HMI data to Hinode/SOT data #
# by David Korda (david.korda@asu.cas.cz)                   #
# --------------------------------------------------------- #
# Run with Python 3.10                                      #
# Install: requirements.txt                                 #
# pip install -r requirements.txt                           #
# --------------------------------------------------------- #

"""
This code is provided under the MIT licence (https://opensource.org/license/mit/).

Copyright 2024 David Korda (Astronomical Institute of the Czech Academy of Sciences).

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from modules.NN_config_parse import bin_to_used
from modules._constants import _sep_out, _sep_in
from modules.NN_config import (conf_output_setup, conf_grid_setup, conf_model_setup, conf_filtering_setup,
                               conf_data_split_setup)
from main import pipeline
import argparse
import socket


if __name__ == "__main__":
    hostname = socket.gethostname()
    print(f"Running on: {hostname}\n")

    p = conf_model_setup["params"]

    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--data_filename", type=str, default="SP_HMI-like.npz")
    #
    # model hyperparameters
    #
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--tune_hp", action="store_true")
    parser.add_argument("--model_to_retrain", type=str, default="")

    parser.add_argument("--model_type", type=str, default=p["model_type"])
    parser.add_argument("--num_residuals", type=int, default=p["num_residuals"])
    parser.add_argument("--num_nodes", type=int, default=p["num_nodes"])
    parser.add_argument("--kern_size", type=int, default=p["kern_size"])
    parser.add_argument("--kern_pad", type=str, default=p["kern_pad"])
    parser.add_argument("--input_activation", type=str, default=p["input_activation"])
    parser.add_argument("--output_activation", type=str, default=p["output_activation"])

    parser.add_argument("--dropout_input_hidden", type=float, default=p["dropout_input_hidden"])
    parser.add_argument("--dropout_residual_residual", type=float, default=p["dropout_residual_residual"])
    parser.add_argument("--dropout_hidden_output", type=float, default=p["dropout_hidden_output"])
    parser.add_argument("--L1_trade_off", type=float, default=p["L1_trade_off"])
    parser.add_argument("--L2_trade_off", type=float, default=p["L2_trade_off"])
    parser.add_argument("--optimizer", type=str, default=p["optimizer"])
    parser.add_argument("--learning_rate", type=float, default=p["learning_rate"])
    parser.add_argument("--batch_size", type=int, default=p["batch_size"])
    parser.add_argument("--apply_bs_norm", action="store_true")
    parser.add_argument("--bs_norm_before_activation", action="store_true")

    parser.add_argument("--loss_type", type=str, default=p["loss_type"])
    parser.add_argument("--use_weights", action="store_true")
    parser.add_argument("--weight_scale", type=float, default=p["weight_scale"])
    parser.add_argument("--metrics", type=str, default=p["metrics"], nargs="*")
    parser.add_argument("--alpha", type=float, default=p["alpha"])
    parser.add_argument("--c", type=float, default=p["c"])
    parser.add_argument("--num_epochs", type=int, default=p["num_epochs"])

    #
    # other model options
    #
    parser.add_argument("--objective", type=str, default=conf_model_setup["monitoring"]["objective"])
    parser.add_argument("--objective_direct", type=str, default=conf_model_setup["monitoring"]["direction"])
    parser.add_argument("--trim_mean_cut", type=float, default=conf_model_setup["trim_mean_cut"])
    parser.add_argument("--model_subdir", type=str, default=conf_model_setup["model_subdir"])

    #
    # model output space
    #
    parser.add_argument("--used_quantities", type=str, default=conf_output_setup["bin_code"])

    #
    # model training patch size
    #
    parser.add_argument("--patch_size", type=int, default=conf_grid_setup["patch_size"])

    #
    # model data filtering options
    #
    parser.add_argument("--filter_base_data", type=str, default=conf_filtering_setup["base_data"])
    parser.add_argument("--filter_I", type=float, default=conf_filtering_setup["I"])
    parser.add_argument("--filter_Bp_kG", type=float, default=conf_filtering_setup["Bp"])
    parser.add_argument("--filter_Bt_kG", type=float, default=conf_filtering_setup["Bt"])
    parser.add_argument("--filter_Br_kG", type=float, default=conf_filtering_setup["Br"])

    #
    # model data split options
    #
    parser.add_argument("--val_portion", type=float, default=conf_data_split_setup["val_portion"])
    parser.add_argument("--test_portion", type=float, default=conf_data_split_setup["test_portion"])
    parser.add_argument("--use_random", action="store_true")

    args, _ = parser.parse_known_args()

    # update the instructions
    model_setup = {"params": {"model_type": args.model_type,
                              "num_residuals": args.num_residuals,
                              "num_nodes": args.num_nodes,
                              "kern_size": args.kern_size,
                              "kern_pad": args.kern_pad,
                              "input_activation": args.input_activation,
                              "output_activation": args.output_activation,

                              "dropout_input_hidden": args.dropout_input_hidden,
                              "dropout_residual_residual": args.dropout_residual_residual,
                              "dropout_hidden_output": args.dropout_hidden_output,
                              "L1_trade_off": args.L1_trade_off,
                              "L2_trade_off": args.L2_trade_off,
                              "optimizer": args.optimizer,
                              "learning_rate": args.learning_rate,
                              "batch_size": args.batch_size,
                              "apply_bs_norm": args.apply_bs_norm,
                              "bs_norm_before_activation": args.bs_norm_before_activation,

                              "loss_type": args.loss_type,
                              "use_weights": args.use_weights,
                              "weight_scale": args.weight_scale,
                              "metrics": args.metrics,
                              "alpha": args.alpha,
                              "c": args.c,
                              "num_epochs": args.num_epochs,
                              },
                   "model_name": f"HMI{_sep_in}to{_sep_in}SOT{_sep_out}{args.used_quantities}",
                   "monitoring": {"objective": args.objective,
                                  "direction": args.objective_direct
                                  },
                   "trim_mean_cut": args.trim_mean_cut,
                   "model_subdir": args.model_subdir
                   }

    output_setup = {"used_quantities": bin_to_used(bin_code=args.used_quantities),
                    "bin_code": args.used_quantities}

    grid_setup = {"patch_size": args.patch_size}

    filtering_setup = {"I": args.filter_I,
                       "Bp": args.filter_Bp_kG,
                       "Bt": args.filter_Bt_kG,
                       "Br": args.filter_Br_kG
                       }

    data_split_setup = {"val_portion": args.val_portion,
                        "test_portion": args.test_portion,
                        "use_random": args.use_random
                        }

    if not args.model_to_retrain:
        args.model_to_retrain = None

    # layer adjustment for a specific model_to_retrain is done automatically in the pipeline
    pipeline(data_filename=args.data_filename,
             num_models=args.num_repeats, train_new_model=args.train, tune_hyperparameters=args.tune_hp,
             model_to_retrain=args.model_to_retrain,
             model_setup=model_setup, output_setup=output_setup, grid_setup=grid_setup,
             filtering_setup=filtering_setup, data_split_setup=data_split_setup)
