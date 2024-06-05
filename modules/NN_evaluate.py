from os import path
import numpy as np
from scipy.stats import trim_mean

from modules.utilities_data import (load_npz, load_txt, print_header, print_info, gimme_bin_code_from_name,
                                    unit_to_gauss, gimme_used_from_name, load_keras_model, adjust_params_from_weights)

from modules.control_plots import result_plots

from modules.utilities import is_empty, to_list

from modules.NN_models import build_model

from modules._constants import _wp, _quiet, _show_control_plot, _show_result_plot, _path_model, _observations_name

# defaults only
from modules.NN_config import conf_model_setup


def average_predictions(predictions: np.ndarray, proportiontocut: float) -> np.ndarray:
    # Trimmed mean
    predictions = trim_mean(predictions, proportiontocut, axis=-1)

    return np.array(predictions, dtype=_wp)


def check_models(model_names: list[str]) -> str:
    bin_codes = [gimme_bin_code_from_name(model_name) for model_name in to_list(model_names)]

    # must all be the same
    if not np.all([x == bin_codes[0] for x in bin_codes]):
        raise ValueError("Not all models have the same specification (grid and output labels).")

    return bin_codes[0]


def filename_data_to_data(filename_or_data: str | np.ndarray, sep: str = "\t", quiet: bool = False) -> np.ndarray:
    if isinstance(filename_or_data, str):
        # Import the test dataset
        if not quiet:
            print("Loading dataset")

        if filename_or_data[-4:] == ".npz":
            filename_or_data = load_npz(filename_or_data, subfolder="")
            filename_or_data = np.array(filename_or_data[_observations_name], dtype=_wp)

        else:
            filename_or_data = np.array(load_txt(filename_or_data, subfolder="", sep=sep, header=None), dtype=_wp)

    elif isinstance(filename_or_data, np.lib.npyio.NpzFile):
        filename_or_data = np.array(filename_or_data[_observations_name], dtype=_wp)

    else:
        filename_or_data = np.array(filename_or_data, dtype=_wp)

    # convert data to working precision
    return np.array(filename_or_data, dtype=_wp)


def make_predictions(model_names: list[str],
                     x_data: np.ndarray,
                     params: dict[str, str | int | float | bool | list[int]] | None = None,
                     which: str | None = None,
                     subfolder_model: str = "") -> np.ndarray:

    if not _quiet and which is not None:
        print(f"Evaluating the neural network on the {which} data")

    used_quantities = gimme_used_from_name(model_names[0])

    if params is not None:  # I expect that all models in model_names has the same architecture
        # initialize the model
        params = adjust_params_from_weights(filename=model_names[0], params=params, subfolder=subfolder_model)
        model = build_model(input_shape=np.shape(x_data)[1:], params=params, used_quantities=used_quantities)

    # Calc average prediction over the models
    for idx, model_name in enumerate(model_names):
        model_full_path = path.join(_path_model, subfolder_model, model_name)

        if params is None:
            # unknown parameters -> recreate the model in every iteration and load the weights into it
            model = load_keras_model(filename=model_name, input_shape=np.shape(x_data)[1:], subfolder=subfolder_model)
        else:
            # load weights into the initialized model
            model.load_weights(model_full_path)

        if idx == 0:
            predictions = np.zeros((len(x_data), *model.output_shape[1:], len(model_names)), dtype=_wp)

        # Evaluate the model on test data
        predictions[..., idx] = model.predict(x_data, verbose=0)

    return predictions


def evaluate(model_names: list[str], filename_or_data: str | np.ndarray,
             proportiontocut: float | None = None,
             params: dict[str, str | int | float | bool | list[int]] | None = None,
             subfolder_model: str = "",
             b_unit: str | None = None) -> np.ndarray:
    # This function evaluates the mean model on new a dataset

    if not model_names:
        raise ValueError('"model_names" is empty')

    check_models(model_names=model_names)
    used_quantities = gimme_used_from_name(model_names[0])

    if proportiontocut is None: proportiontocut = conf_model_setup["trim_mean_cut"]

    filename_or_data = filename_data_to_data(filename_or_data, quiet=_quiet)
    data = filename_or_data

    # only convolution layers -> can be used for any input dimensions
    predictions = make_predictions(model_names=model_names, x_data=data, params=params,
                                   subfolder_model=subfolder_model, which="new")

    print("-----------------------------------------------------")

    # Trimmed mean
    if b_unit is None or b_unit == "kG":
        return average_predictions(predictions, proportiontocut=proportiontocut)
    else:  # conversion to Gauss
        return unit_to_gauss(average_predictions(predictions, proportiontocut=proportiontocut), used_quantities)


def evaluate_test_data(model_names: list[str], x_test: np.ndarray, y_test: np.ndarray,
                       x_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
                       x_train: np.ndarray | None = None, y_train: np.ndarray | None = None,
                       proportiontocut: float | None = None,
                       params: dict[str, str | int | float | bool | list[int]] | None = None,
                       subfolder_model: str = "",
                       b_unit: str | None = None) -> tuple[np.ndarray, ...]:
    if not model_names:
        raise ValueError('"model_names" is empty')

    bin_code = check_models(model_names=model_names)

    if not _quiet:
        print(f"Evaluated models:\n\t{model_names}")

    used_quantities = gimme_used_from_name(model_names[0])

    if proportiontocut is None: proportiontocut = conf_model_setup["trim_mean_cut"]

    # loading needed values
    if "accuracy_test" in subfolder_model:
        show_result_plot, show_control_plot = False, False
    else:
        show_result_plot, show_control_plot = _show_result_plot, _show_control_plot

    # convert data to working precision
    x_test, y_test = np.array(x_test, dtype=_wp), np.array(y_test, dtype=_wp)

    do_train = not is_empty(x_train) and not is_empty(y_train)
    do_val = not is_empty(x_val) and not is_empty(y_val)

    if not do_train: y_train = None
    if not do_val: y_val = None

    if do_train:
        # convert data to working precision
        x_train, y_train = np.array(x_train, dtype=_wp), np.array(y_train, dtype=_wp)

    if do_val:
        # convert data to working precision
        x_val, y_val = np.array(x_val, dtype=_wp), np.array(y_val, dtype=_wp)

    # only convolution layers -> can be used for any input dimensions
    predictions = make_predictions(model_names=model_names, x_data=x_test, params=params,
                                   subfolder_model=subfolder_model, which="test")

    if do_train:
        predictions_train = make_predictions(model_names=model_names, x_data=x_train, params=params,
                                             subfolder_model=subfolder_model, which="train")

    if do_val:
        predictions_val = make_predictions(model_names=model_names, x_data=x_val, params=params,
                                           subfolder_model=subfolder_model, which="validation")

    # Trimmed means and conversion to Gauss (final predictions are in kG to keep it consistent with inputs)
    predictions = average_predictions(predictions, proportiontocut=proportiontocut)
    x_test, y_test = unit_to_gauss(x_test, used_quantities), unit_to_gauss(y_test, used_quantities)

    if do_train:
        predictions_train = unit_to_gauss(average_predictions(predictions_train, proportiontocut=proportiontocut),
                                          used_quantities)
        x_train, y_train = unit_to_gauss(x_train, used_quantities), unit_to_gauss(y_train, used_quantities)
    if do_val:
        predictions_val = unit_to_gauss(average_predictions(predictions_val, proportiontocut=proportiontocut),
                                        used_quantities)
        x_val, y_val = unit_to_gauss(x_val, used_quantities), unit_to_gauss(y_val, used_quantities)

    # Evaluate the accuracy (this is always printed)
    print("\n-----------------------------------------------------")
    print_header(used_quantities=used_quantities)

    if do_train:
        print_info(y_train, predictions_train, which="train")
    if do_val:
        print_info(y_val, predictions_val, which="validation")
    acc = print_info(y_test, unit_to_gauss(predictions, used_quantities), which="test")
    print("-----------------------------------------------------\n")

    # These are result plots
    if show_result_plot:
        print("Result plots")
        result_plots(x_test, y_test, unit_to_gauss(predictions, used_quantities), used_quantities=used_quantities,
                     suf=f"_{bin_code}")

    if show_control_plot:
        print("Control plots")
        if do_val:
            result_plots(x_val, y_val, predictions_val, used_quantities=used_quantities, suf=f"_{bin_code}_val")
        if do_train:
            result_plots(x_train, y_train, predictions_train, used_quantities=used_quantities, suf=f"_{bin_code}_train")

    # Trimmed mean
    if b_unit is None or b_unit == "kG":
        return predictions, acc
    else:  # conversion to Gauss
        return unit_to_gauss(predictions, used_quantities), acc


