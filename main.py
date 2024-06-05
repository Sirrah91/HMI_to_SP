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

from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from modules.NN_data import load_prepared_data, split_data, clean_data
from modules.NN_train import train, hp_tuner
from modules.NN_evaluate import evaluate_test_data, evaluate
import numpy as np
from tqdm import tqdm

from modules.utilities_data import (collect_all_models, save_results, model_name_to_result_name,
                                    adjust_params_from_weights)
from modules.utilities import stack
from modules.NN_config import (conf_output_setup, conf_model_setup, conf_filtering_setup, conf_data_split_setup,
                               conf_grid_setup)


def pipeline(num_models: int = 1,
             train_new_model: bool = True,
             tune_hyperparameters: bool = False,
             model_to_retrain: str | None = None,
             **kwargs) -> np.ndarray:
    # num_models: computes a trimmed mean of num_models and returns predictions or print best hyperparameters
    # train_new_model: train a new model or evaluate existing ones?
    # tune_hyperparameters: tune hyperparameters? If so, it ignores the previous parameters

    conf_model_setup["params"] |= kwargs

    if model_to_retrain is not None:
        conf_model_setup["params"] = adjust_params_from_weights(filename=model_to_retrain,
                                                                params=conf_model_setup["params"],
                                                                subfolder=conf_model_setup["model_subdir"])

    if train_new_model or tune_hyperparameters:
        # Name of the train data in _path_data
        filename_train_data = "SP_HMI_aligned.npz"
        print(f"Data file: {filename_train_data}")

        # Load the data
        x_train, y_train = load_prepared_data(filename_data=filename_train_data,
                                              used_quantities=conf_output_setup["used_quantities"],
                                              use_simulated_hmi=True)

        # Split the data
        x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_data=x_train, y_data=y_train,
                                                                    val_portion=conf_data_split_setup["val_portion"],
                                                                    test_portion=conf_data_split_setup["test_portion"])

        # Filter out the quiet Sun patches
        x_train, y_train, x_del, y_del = clean_data(x_train, y_train,
                                                    filtering_setup=conf_filtering_setup,
                                                    used_quantities=conf_output_setup["used_quantities"],
                                                    return_deleted=True)
        x_test, y_test = stack((x_test, x_del), axis=0), stack((y_test, y_del), axis=0)

        x_val, y_val, x_del, y_del = clean_data(x_val, y_val,
                                                filtering_setup=conf_filtering_setup,
                                                used_quantities=conf_output_setup["used_quantities"],
                                                return_deleted=True)
        x_test, y_test = stack((x_test, x_del), axis=0), stack((y_test, y_del), axis=0)

        x_del, y_del = [], []  # deallocate

        if tune_hyperparameters:
            # Tuning of the hyperparameters defined in the p_for_tuning dictionary
            model_names = hp_tuner(x_train=x_train, y_train=y_train,
                                   x_val=x_val, y_val=y_val,
                                   monitoring=conf_model_setup["monitoring"],
                                   model_subdir=conf_model_setup["model_subdir"],
                                   model_name=conf_model_setup["model_name"],
                                   metrics=conf_model_setup["params"]["metrics"],
                                   used_quantities=conf_output_setup["used_quantities"])
        else:
            # Create, train, and save the neural network
            model_names = [train(x_train=x_train, y_train=y_train,
                                 x_val=x_val, y_val=y_val,
                                 params=conf_model_setup["params"],
                                 monitoring=conf_model_setup["monitoring"],
                                 model_subdir=conf_model_setup["model_subdir"],
                                 model_to_retrain=model_to_retrain,
                                 model_name=conf_model_setup["model_name"],
                                 metrics=conf_model_setup["params"]["metrics"])
                           for _ in range(num_models)]

        # Evaluate it on the test data
        predictions, accuracy = evaluate_test_data(model_names=model_names,
                                                   x_test=x_test, y_test=y_test,
                                                   x_val=x_val, y_val=y_val,
                                                   x_train=x_train, y_train=y_train,
                                                   proportiontocut=conf_model_setup["trim_mean_cut"],
                                                   subfolder_model=conf_model_setup["model_subdir"])

        results_name = model_name_to_result_name(model_names[0])
        save_results(results_name, predictions, x_true=x_test, y_true=y_test,
                     output_setup=conf_output_setup, grid_setup=conf_grid_setup,
                     filtering_setup=conf_filtering_setup, data_split_setup=conf_data_split_setup,
                     model_setup=conf_model_setup, model_names=model_names, subfolder=conf_model_setup["model_subdir"])

    else:
        # List of the models in ./_path_model/subfolder_model/
        model_names = ["weights_1111_20240513105130.h5"]

        filename_data = "2011-03-27_2011-03-28.npz"
        print(f"Data file: {filename_data}")

        # Evaluate the models; second argument can be a path to the data or the data itself
        predictions = evaluate(model_names=model_names,
                               filename_or_data=filename_data,
                               proportiontocut=conf_model_setup["trim_mean_cut"],
                               subfolder_model=conf_model_setup["model_subdir"])

        results_name = model_name_to_result_name(model_names[0])
        save_results(results_name, predictions, output_setup=conf_output_setup, grid_setup=conf_grid_setup,
                     filtering_setup=conf_filtering_setup, data_split_setup=conf_data_split_setup,
                     model_setup=conf_model_setup, model_names=model_names, subfolder=conf_model_setup["model_subdir"])

    return predictions


if __name__ == "__main__":
    num_models, train_model, tune_hp = 1, True, False

    if tune_hp:
        pipeline(num_models=1, train_new_model=train_model, tune_hyperparameters=tune_hp)
    elif train_model:
        if num_models > 1:
            for _ in tqdm(range(num_models)):
                y_pred = pipeline(train_new_model=train_model, tune_hyperparameters=tune_hp)
        else:
            y_pred = pipeline(train_new_model=train_model, tune_hyperparameters=tune_hp)
    else:
        y_pred = pipeline(num_models=1, train_new_model=train_model, tune_hyperparameters=tune_hp)
