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

from modules.NN_data import load_prepared_data, split_data, clean_data
from modules.NN_train import train, hp_tuner
from modules.NN_evaluate import evaluate_test_data, evaluate_by_parts, set_observation_name
from modules.utilities_data import (save_results, model_name_to_result_name, adjust_params_from_weights,
                                    gimme_used_from_name)
from modules.utilities import stack, update_dict
from modules.NN_config_parse import config_check
from modules.NN_config import (conf_output_setup, conf_model_setup, conf_filtering_setup, conf_data_split_setup,
                               conf_grid_setup)
from modules._base_models import load_base_models
from modules._constants import _model_config_file, _wp

import numpy as np


def pipeline(data_filename: str = "SP_HMI-like.npz",
             num_models: int = 1,
             train_new_model: bool = True,
             tune_hyperparameters: bool = False,
             model_to_retrain: str | None = None,
             #
             model_setup: dict | None = None,
             output_setup: dict | None = None,
             filtering_setup: dict | None = None,
             data_split_setup: dict | None = None,
             grid_setup: dict | None = None
             ) -> np.ndarray:
    # num_models: computes a trimmed mean of num_models and returns predictions or print best hyperparameters
    # train_new_model: train a new model or evaluate existing ones?
    # tune_hyperparameters: tune hyperparameters? If so, it ignores the previous parameters

    model_setup = update_dict(new_dict=model_setup, base_dict=conf_model_setup)
    output_setup = update_dict(new_dict=output_setup, base_dict=conf_output_setup)
    filtering_setup = update_dict(new_dict=filtering_setup, base_dict=conf_filtering_setup)
    data_split_setup = update_dict(new_dict=data_split_setup, base_dict=conf_data_split_setup)
    grid_setup = update_dict(new_dict=grid_setup, base_dict=conf_grid_setup)

    # basic check of config
    config_check(output_setup=output_setup, data_split_setup=data_split_setup, model_setup=model_setup)

    # Name of the train data in _path_data
    print(f"Data file: {data_filename}")

    if model_to_retrain is not None:  # if there is a model to retrain, it means training...
        train_new_model, tune_hyperparameters = True, False

    if train_new_model or tune_hyperparameters:
        if model_to_retrain is not None:
            model_setup["params"] = adjust_params_from_weights(filename=model_to_retrain, params=model_setup["params"],
                                                               subfolder=model_setup["model_subdir"])
            output_setup["used_quantities"] = gimme_used_from_name(model_name=model_to_retrain)

        # Load the data
        x_train, y_train, metadata = load_prepared_data(filename_data=data_filename,
                                                        used_quantities=output_setup["used_quantities"],
                                                        use_simulated_hmi=True,
                                                        out_type=_wp,
                                                        convert_b=True)
        
        # Filter out the quiet Sun patches and evaluate them separately
        x_train, y_train, metadata_train, x_qs, y_qs, metadata_qs = clean_data(x_train, y_train,
                                                                               metadata=metadata,
                                                                               filtering_setup=filtering_setup,
                                                                               used_quantities=output_setup["used_quantities"],
                                                                               return_deleted=True)

        # Split the data
        x_train, y_train, meta_train, x_val, y_val, meta_val, x_test, y_test, meta_test = split_data(x_data=x_train, y_data=y_train,
                                                                                                     metadata=metadata_train,
                                                                                                     val_portion=data_split_setup["val_portion"],
                                                                                                     test_portion=data_split_setup["test_portion"],
                                                                                                     use_random=data_split_setup["use_random"])

        if tune_hyperparameters:
            # Tuning of the hyperparameters defined in the p_for_tuning dictionary
            model_names = hp_tuner(x_train=x_train, y_train=y_train,
                                   x_val=x_val, y_val=y_val,
                                   monitoring=model_setup["monitoring"],
                                   model_subdir=model_setup["model_subdir"],
                                   model_name=model_setup["model_name"],
                                   metrics=model_setup["params"]["metrics"],
                                   used_quantities=output_setup["used_quantities"])
        else:
            # Create, train, and save the neural network
            model_names = [train(x_train=x_train, y_train=y_train,
                                 x_val=x_val, y_val=y_val,
                                 params=model_setup["params"],
                                 monitoring=model_setup["monitoring"],
                                 model_subdir=model_setup["model_subdir"],
                                 model_to_retrain=model_to_retrain,
                                 model_name=model_setup["model_name"],
                                 metrics=model_setup["params"]["metrics"])
                           for _ in range(num_models)]

        # Evaluate it on the test data
        predictions = evaluate_test_data(model_names=model_names, x_test=x_test, y_test=y_test, x_val=x_val,
                                         y_val=y_val, x_train=x_train, y_train=y_train, x_others=x_qs, y_others=y_qs,
                                         proportiontocut=model_setup["trim_mean_cut"],
                                         subfolder_model=model_setup["model_subdir"])

        predictions = predictions["test"]["predictions"]

    else:
        model_names = load_base_models(_model_config_file)
        x_test, y_test = None, None

        # Evaluate the models; second argument can be a path to the data or the data itself
        predictions = evaluate_by_parts(model_names=model_names,
                                        filename_or_data=data_filename,
                                        observation_name=set_observation_name(data_filename),
                                        proportiontocut=model_setup["trim_mean_cut"],
                                        subfolder_model=model_setup["model_subdir"])
    """
    results_name = model_name_to_result_name(model_names[0])
    save_results(results_name, predictions, x_true=x_test, y_true=y_test,
                 output_setup=output_setup, grid_setup=grid_setup,
                 filtering_setup=filtering_setup, data_split_setup=data_split_setup,
                 model_setup=model_setup, model_names=model_names, subfolder=model_setup["model_subdir"])
    """
    return predictions


if __name__ == "__main__":
    data_filename = "SP_HMI-like.npz"
    num_models = 1
    train_new_model = True
    tune_hyperparameters = False
    model_to_retrain = "HMI-to-SOT_1000_20241121145011.weights.h5"
    #
    model_setup = None  # {"params": {"metrics": []}}  # don't need to compute MSE/RMSE metrics twice
    output_setup = None
    grid_setup = None
    filtering_setup = None
    data_split_setup = None

    predictions = pipeline(data_filename=data_filename,
                           num_models=num_models, train_new_model=train_new_model, tune_hyperparameters=tune_hyperparameters,
                           model_to_retrain=model_to_retrain,
                           model_setup=model_setup, output_setup=output_setup, grid_setup=grid_setup,
                           filtering_setup=filtering_setup, data_split_setup=data_split_setup)
