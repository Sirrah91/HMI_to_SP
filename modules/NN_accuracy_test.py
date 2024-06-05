from os import path
import pandas as pd
from datetime import datetime
from sklearn.model_selection import KFold, LeaveOneOut
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path

from modules.utilities import check_dir, stack
from modules.utilities_data import print_header, print_info, unit_to_gauss

from modules.control_plots import result_plots

from modules.NN_data import load_data
from modules.NN_train import train
from modules.NN_evaluate import evaluate_test_data

from modules._constants import _path_accuracy_tests, _metadata_name, _sep_out
from modules._constants import _metadata_key_name, _label_true_name, _label_pred_name, _config_name, _wp

# defaults only
from modules.NN_config import conf_model_setup, conf_filtering_setup, conf_output_setup, conf_grid_setup
from modules._constants import _rnd_seed


def split_data_for_testing(x_data: np.ndarray, y_data: np.ndarray,
                           options: tuple[str, int, int]) -> tuple[np.ndarray, ...]:
    method, index, K = options

    if method == "LOO":  # Leave-one-out
        # This can be written using KFold(n_splits=len(x_train)); that s definition of K for this case
        train_indices, test_indices = list(LeaveOneOut().split(x_data))[index]
    elif method == "K-fold":  # K-fold method
        train_indices, test_indices = list(KFold(n_splits=K).split(x_data))[index]
    else:
        raise ValueError('Method must be one of "LOO" and "K-fold".')

    x_data, x_test = deepcopy(x_data[train_indices]), deepcopy(x_data[test_indices])
    y_data, y_test = deepcopy(y_data[train_indices]), deepcopy(y_data[test_indices])

    return x_data, y_data, x_test, y_test


def gimme_info(model_option: tuple[str, int, int],
               output_setup: dict | None = None,
               grid_setup:  dict | None = None,
               filtering_setup: dict | None = None,
               model_setup: dict | None = None) -> dict[str, any]:

    if output_setup is None: output_setup = conf_output_setup
    if grid_setup is None: grid_setup = conf_grid_setup
    if filtering_setup is None: filtering_setup = conf_filtering_setup
    if model_setup is None: model_setup = conf_model_setup

    output = {}

    output["output_setup"] = deepcopy(output_setup)

    output["grid_setup"] = deepcopy(grid_setup)

    output["model_setup"] = deepcopy(model_setup)
    output["model_setup"]["num_models"] = model_option[2]

    output["data_split_setup"] = deepcopy({})
    output["data_split_setup"]["method"] = model_option[0]
    output["data_split_setup"]["num_splits"] = model_option[1]

    output["filtering_setup"] = deepcopy(filtering_setup)

    return output


def save_results(final_name: str, spectra: np.ndarray, wavelengths: np.ndarray, y_true: np.ndarray,
                 y_pred: np.ndarray, metadata: np.ndarray, config_setup: dict[str, any],
                 labels_key: np.ndarray | None = None, metadata_key: np.ndarray | None = None,
                 subfolder: str = "") -> None:
    filename = path.join(_path_accuracy_tests, subfolder, final_name)
    check_dir(filename)

    tmp = Path(filename)
    if tmp.suffix == "":
        filename += ".npz"

    # collect data and metadata
    data_and_metadata = {_spectra_name: np.array(spectra, dtype=_wp),  # save spectra
                         _wavelengths_name: np.array(wavelengths, dtype=_wp),  # save wavelengths
                         _label_true_name: np.array(y_true, dtype=_wp),  # save labels
                         _label_pred_name: np.array(y_pred, dtype=_wp),  # save results
                         _metadata_name: np.array(metadata, dtype=object),  # save metadata
                         _config_name: config_setup}  # save config file

    if metadata_key is not None:
        data_and_metadata[_metadata_key_name] = np.array(metadata_key, dtype=str)

    if labels_key is not None:
        data_and_metadata[_label_key_name] = np.array(labels_key, dtype=str)

    with open(filename, "wb") as f:
        np.savez_compressed(f, **data_and_metadata)


def gimme_method(maximum_splits: int, len_data: int) -> tuple[str, int]:
    if maximum_splits >= len_data:
        # If LOO then maximum training size
        method = "LOO"
        K = len_data
    else:
        # K-fold otherwise
        method = "K-fold"
        K = maximum_splits

    return method, K


def shuffle_data(x_data: np.ndarray, y_data: np.ndarray, metadata: pd.DataFrame,
                 rnd_seed: int | None = _rnd_seed) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed=rnd_seed)  # to always get the same permutation
    idx = rng.permutation(len(x_data))
    x_data = x_data[idx]
    y_data = y_data[idx]
    metadata = metadata.iloc[idx]

    return x_data, y_data, metadata


if __name__ == "__main__":

    taxonomy = False

    max_splits = 100
    num_models = 1

    filename_train_data = "HMI_7376.npz"

    # data = load_npz(filename_train_data, list_keys=[_metadata_key_name])
    # metadata_key = data[_metadata_key_name]
    # labels_key = data[_label_key_name]

    conf_model_setup["model_subdir"] = path.join("accuracy_tests", conf_model_setup["model_subdir"])

    dt_string = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    results_name = f"weights{_sep_out}{conf_output_setup['bin_code']}{_sep_out}{dt_string}.npz"

    # Load the data
    x_train, y_train = load_data(filename_train_data,
                                 return_meta=False,
                                 output_setup=conf_output_setup,
                                 grid_setup=conf_grid_setup,
                                 filtering_setup=conf_filtering_setup)

    method, K = gimme_method(maximum_splits=max_splits, len_data=len(x_train))
    if method == "K-fold":  # Shuffle the data (it is better to do it for K-fold)
        x_train, y_train, meta = shuffle_data(x_train, y_train, meta)

    info = gimme_info(model_option=(method, K, num_models),
                      output_setup=conf_output_setup,
                      grid_setup=conf_grid_setup,
                      filtering_setup=conf_filtering_setup,
                      model_setup=conf_model_setup)

    y_pred = np.zeros(np.shape(y_train))

    # Splitting test indices
    _, _, indices, _ = zip(*[split_data_for_testing(np.arange(len(x_train)), y_train, (method, i, K))
                             for i in range(K)])
    indices = stack(indices)

    start, stop = 0, 0
    for i in tqdm(range(K)):
        print("")  # To ensure the tqdm prints on a single line

        # Split them to train and test parts
        x_train_part, y_train_part, x_test_part, y_test_part = split_data_for_testing(x_train, y_train, (method, i, K))

        # Create and train the neural network and save the model
        model_names = [train(x_train, y_train, np.array([]), np.array([]),
                             params=conf_model_setup["params"],
                             monitoring=conf_model_setup["monitoring"],
                             model_subdir=conf_model_setup["model_subdir"],
                             model_name=conf_model_setup["model_name"],
                             metrics=conf_model_setup["metrics"])
                       for _ in range(num_models)]

        y_pred_part, accuracy_part = evaluate_test_data(model_names, x_test_part, y_test_part,
                                                        proportiontocut=conf_model_setup["trim_mean_cut"],
                                                        subfolder_model=conf_model_setup["model_subdir"])
        start, stop = stop, stop + len(y_test_part)
        y_pred[start:stop] = y_pred_part

    save_results(results_name, spectra=x_train[indices], wavelengths=wavelengths,
                 y_true=y_train[indices], y_pred=y_pred, labels_key=labels_key,
                 metadata=meta.iloc[indices], metadata_key=metadata_key,
                 config_setup=info)

    # One can get config with
    # data = load_npz(path.join(_path_accuracy_tests, results_name))
    # config = data[_config_name][()]

    suf = "_accuracy_test"
    x_train = unit_to_gauss(x_train, conf_output_setup["used_quantities"])
    y_train = unit_to_gauss(y_train, conf_output_setup["used_quantities"])
    y_pred = unit_to_gauss(y_pred, conf_output_setup["used_quantities"])
    result_plots(x_train[indices], y_train[indices], y_pred, density_plot=True, suf=suf, quiet=False)

    print("\n-----------------------------------------------------")
    print_header()
    print_info(y_train[indices], y_pred, which=method)
    print("-----------------------------------------------------\n")
