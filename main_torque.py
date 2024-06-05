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

from main import pipeline
import argparse
from tqdm import tqdm
import socket
from modules.NN_HP import gimme_hyperparameters
from modules.utilities_data import extract_params_from_weights


if __name__ == "__main__":
    hostname = socket.gethostname()
    print(f"Running on: {hostname}\n")

    p = gimme_hyperparameters(for_tuning=False)()

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--train", type=bool, default=True, const=True, nargs="?")
    parser.add_argument("--tune_hp", type=bool, default=False, const=True, nargs="?")
    parser.add_argument("--model_to_retrain", type=str, default="")

    parser.add_argument("--model_type", type=str, default=p["model_type"])
    parser.add_argument("--num_residuals", type=int, default=p["num_residuals"])
    parser.add_argument("--num_nodes", type=int, default=p["num_nodes"])
    parser.add_argument("--kern_size", type=int, default=p["kern_size"])
    parser.add_argument("--kern_pad", type=str, default=p["kern_pad"])
    parser.add_argument("--input_activation", type=str, default=p["input_activation"])

    parser.add_argument("--dropout_input_hidden", type=float, default=p["dropout_input_hidden"])
    parser.add_argument("--dropout_residual_residual", type=float, default=p["dropout_residual_residual"])
    parser.add_argument("--dropout_hidden_output", type=float, default=p["dropout_hidden_output"])
    parser.add_argument("--L1_trade_off", type=float, default=p["L1_trade_off"])
    parser.add_argument("--L2_trade_off", type=float, default=p["L2_trade_off"])
    parser.add_argument("--optimizer", type=str, default=p["optimizer"])
    parser.add_argument("--learning_rate", type=float, default=p["learning_rate"])
    parser.add_argument("--batch_size", type=int, default=p["batch_size"])
    parser.add_argument("--bs_norm_before_activation", type=bool, default=p["bs_norm_before_activation"],
                        const=True, nargs="?")

    parser.add_argument("--loss_type", type=str, default=p["loss_type"])
    parser.add_argument("--metrics", type=str, default=p["metrics"], nargs="+")
    parser.add_argument("--alpha", type=float, default=p["alpha"])
    parser.add_argument("--c", type=float, default=p["c"])
    parser.add_argument("--num_epochs", type=int, default=p["num_epochs"])

    args, _ = parser.parse_known_args()

    kwargs = {"model_type": args.model_type,
              "num_residuals": args.num_residuals,
              "num_nodes": args.num_nodes,
              "kern_size": args.kern_size,
              "kern_pad": args.kern_pad,
              "input_activation": args.input_activation,

              "dropout_input_hidden": args.dropout_input_hidden,
              "dropout_residual_residual": args.dropout_residual_residual,
              "dropout_hidden_output": args.dropout_hidden_output,
              "L1_trade_off": args.L1_trade_off,
              "L2_trade_off": args.L2_trade_off,
              "optimizer": args.optimizer,
              "learning_rate": args.learning_rate,
              "batch_size": args.batch_size,
              "bs_norm_before_activation": args.bs_norm_before_activation,

              "loss_type": args.loss_type,
              "metrics": args.metrics,
              "alpha": args.alpha,
              "c": args.c,
              "num_epochs": args.num_epochs,
              }

    if args.model_to_retrain:
        kwargs |= extract_params_from_weights(args.model_to_retrain, subfolder="HMI_to_SOT")
    else:
        args.model_to_retrain = None

    if args.tune_hp:
        pipeline(num_models=1, train_new_model=args.train, tune_hyperparameters=args.tune_hp)
    elif args.train:
        if args.num_repeats > 1:
            for _ in tqdm(range(args.num_repeats)):
                pipeline(train_new_model=args.train, tune_hyperparameters=args.tune_hp,
                         model_to_retrain=args.model_to_retrain, **kwargs)
        else:
            pipeline(train_new_model=args.train, tune_hyperparameters=args.tune_hp,
                     model_to_retrain=args.model_to_retrain, **kwargs)
    else:
        pipeline(num_models=1, train_new_model=args.train, tune_hyperparameters=args.tune_hp, **kwargs)
