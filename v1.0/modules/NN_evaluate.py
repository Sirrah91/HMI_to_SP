from modules.utilities_data import (load_npz, load_txt, print_header, print_info, gimme_bin_code_from_name,
                                    convert_unit, gimme_used_from_name, load_keras_model, adjust_params_from_weights,
                                    gimme_combined_used_quantities)

from modules.control_plots import result_plots

from modules.utilities import is_empty, to_list, robust_load_weights

from modules.NN_models import build_model, batch_size_estimate

from modules._constants import (_wp, _quiet, _show_control_plot, _show_result_plot, _path_model, _observations_name,
                                _b_unit)

# defaults only
from modules.NN_config import conf_model_setup

from os import path
import numpy as np
from typing import Literal
from scipy.stats import trim_mean
import h5py


def average_predictions(predictions: np.ndarray, proportiontocut: float) -> np.ndarray:
    # Trimmed mean
    predictions = trim_mean(predictions, proportiontocut, axis=-1)

    return np.array(predictions, dtype=_wp)


def check_models(model_names: list[str]) -> str:
    bin_codes = [gimme_bin_code_from_name(model_name) for model_name in to_list(model_names)]

    # must all be the same
    if not np.all([x == bin_codes[0] for x in bin_codes]):
        raise ValueError("Not all models have the same quantity specification.")

    return bin_codes[0]


def filename_data_to_data(filename_or_data: str | np.ndarray,
                          observation_name: str = _observations_name,
                          sep: str = "\t", quiet: bool = False) -> np.ndarray:
    if isinstance(filename_or_data, str):
        # Import the test dataset
        if not quiet:
            print("Loading dataset")

        if filename_or_data[-4:] == ".npz":
            filename_or_data = load_npz(filename_or_data, subfolder="")
            filename_or_data = np.array(filename_or_data[observation_name], dtype=_wp)

        else:
            filename_or_data = np.array(load_txt(filename_or_data, subfolder="", sep=sep, header=None), dtype=_wp)

    elif isinstance(filename_or_data, np.lib.npyio.NpzFile):
        filename_or_data = np.array(filename_or_data[observation_name], dtype=_wp)

    else:
        filename_or_data = np.array(filename_or_data, dtype=_wp)

    # convert data to working precision
    return np.array(filename_or_data, dtype=_wp)


def set_observation_name(filename_or_data) -> str:
    if isinstance(filename_or_data, str) and filename_or_data == "SP_HMI-like.npz":
        observation_name = f"{_observations_name}_simulated"
    else:
        observation_name = _observations_name

    return observation_name


def filter_data_to_used(array: np.ndarray, used_quantities: np.ndarray) -> np.ndarray:
    no_quantities = np.shape(array)[-1]
    if no_quantities != np.sum(used_quantities):
        if no_quantities == len(used_quantities):
            # data are not filtered to real used quantities (e.g. filename_or_data is a path to data)
            array = array[..., used_quantities]
        else:
            raise ValueError("Invalid input: Data does not have correct number of quantities.")

    return array


def make_predictions(model_names: list[str],
                     x_data: np.ndarray,
                     params: dict[str, str | int | float | bool | list[int]] | None = None,
                     proportiontocut: float | None = None,
                     which: str | None = None,
                     subfolder_model: str = "") -> np.ndarray:

    if not _quiet and which is not None:
        print(f"Evaluating the neural network on the {which} data")

    if proportiontocut is None: proportiontocut = conf_model_setup["trim_mean_cut"]

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
            model = robust_load_weights(model, model_full_path)

        if idx == 0:
            predictions = np.zeros((len(x_data), *model.output_shape[1:], len(model_names)), dtype=_wp)

        # batch_size to aim for 1 GB
        batch_size = batch_size_estimate(model=model, gb_memory_target=1.)

        # Evaluate the model on test data
        predictions[..., idx] = model.predict(x_data, verbose=0, batch_size=batch_size)

    # Trimmed mean
    return average_predictions(predictions, proportiontocut=proportiontocut)


def evaluate(model_names: list[str], filename_or_data: str | np.ndarray,
             observation_name: str = "auto",
             proportiontocut: float | None = None,
             params: dict[str, str | int | float | bool | list[int]] | None = None,
             subfolder_model: str = "",
             initial_b_unit: Literal["kG", "G", "T", "mT"] = "kG",
             final_b_unit: Literal["kG", "G", "T", "mT"] | None = None) -> np.ndarray:
    if final_b_unit is None:
        final_b_unit = initial_b_unit

    # This function evaluates the mean model on new a dataset
    if observation_name == "auto":
        observation_name = set_observation_name(filename_or_data)

    if not model_names:
        raise ValueError('"model_names" is empty')

    check_models(model_names=model_names)
    used_quantities = gimme_used_from_name(model_names[0])

    if proportiontocut is None: proportiontocut = conf_model_setup["trim_mean_cut"]

    filename_or_data = filename_data_to_data(filename_or_data, observation_name=observation_name, quiet=_quiet)
    data = filename_or_data
    data = filter_data_to_used(array=data, used_quantities=used_quantities)

    data = convert_unit(data, initial_unit=initial_b_unit, final_unit=_b_unit, used_quantities=used_quantities)

    # only convolution layers -> can be used for any input dimensions
    predictions = make_predictions(model_names=model_names, x_data=data, params=params,
                                   proportiontocut=proportiontocut,
                                   subfolder_model=subfolder_model, which="new")

    print("-----------------------------------------------------")

    # Convert predictions to desired unit
    return convert_unit(predictions, initial_unit=_b_unit, final_unit=final_b_unit, used_quantities=used_quantities)


def evaluate_test_data(model_names: list[str], x_test: np.ndarray, y_test: np.ndarray,
                       x_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
                       x_train: np.ndarray | None = None, y_train: np.ndarray | None = None,
                       x_others: np.ndarray | None = None, y_others: np.ndarray | None = None,
                       proportiontocut: float | None = None,
                       params: dict[str, str | int | float | bool | list[int]] | None = None,
                       subfolder_model: str = "",
                       initial_b_unit: Literal["kG", "G", "T", "mT"] = "kG",
                       final_b_unit: Literal["kG", "G", "T", "mT"] | None = None) -> dict:
    def _evaluate_data(x_data: np.ndarray | None, y_data: np.ndarray | None,
                       b_unit: Literal["kG", "G", "T", "mT"] = "G",
                       which: str = ""
                       ) -> tuple[np.ndarray, ...]:
        do_it = not is_empty(x_data) and not is_empty(y_data)

        if not do_it:
            return np.array([], dtype=_wp), np.array([], dtype=_wp), np.array([], dtype=_wp)

        # convert data to working precision
        x_data = convert_unit(np.array(x_data, dtype=_wp), initial_unit=initial_b_unit, final_unit=_b_unit,
                              used_quantities=used_quantities)

        # only convolution layers -> can be used for any input dimensions
        predictions = make_predictions(model_names=model_names, x_data=x_data, params=params,
                                       proportiontocut=proportiontocut,
                                       subfolder_model=subfolder_model, which=which)

        # Conversion to Gauss for plots and statistics
        predictions = convert_unit(predictions, initial_unit=_b_unit,
                                   final_unit=b_unit, used_quantities=used_quantities)
        x_data = convert_unit(x_data, initial_unit=_b_unit, final_unit=b_unit, used_quantities=used_quantities)
        y_data = convert_unit(np.array(y_data, dtype=_wp), initial_unit=initial_b_unit, final_unit=b_unit,
                              used_quantities=used_quantities)

        return x_data, y_data, predictions

    if final_b_unit is None:
        final_b_unit = initial_b_unit

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

    x_train, y_train, predictions_train = _evaluate_data(x_data=x_train, y_data=y_train, which="train", b_unit="G")
    x_val, y_val, predictions_val = _evaluate_data(x_data=x_val, y_data=y_val, which="validation", b_unit="G")
    x_test, y_test, predictions_test = _evaluate_data(x_data=x_test, y_data=y_test, which="test", b_unit="G")
    x_others, y_others, predictions_others = _evaluate_data(x_data=x_others, y_data=y_others, which="other", b_unit="G")

    did_train = not is_empty(x_train) and not is_empty(y_train)
    did_val = not is_empty(x_val) and not is_empty(y_val)
    did_test = not is_empty(x_test) and not is_empty(y_test)  # this should be always True
    did_other = not is_empty(x_others) and not is_empty(y_others)

    # Evaluate the accuracy (this is always printed)
    print("\n-----------------------------------------------------")
    print_header(used_quantities=used_quantities)

    if did_train:
        acc_train = print_info(y_train, predictions_train, which="train")
    else:
        acc_train = np.array([], dtype=_wp)
    if did_val:
        acc_val = print_info(y_val, predictions_val, which="validation")
    else:
        acc_val = np.array([], dtype=_wp)
    if did_test:
        acc_test = print_info(y_test, predictions_test, which="test")
    else:
        acc_test = np.array([], dtype=_wp)
    if did_other:
        acc_others = print_info(y_others, predictions_others, which="other")
    else:
        acc_others = np.array([], dtype=_wp)
    print("-----------------------------------------------------\n")

    # These are result plots
    if show_result_plot and did_test:
        print("Result plots")
        result_plots(x_test, y_test, predictions_test, used_quantities=used_quantities, suf=f"_{bin_code}_test")

    if show_control_plot:
        print("Control plots")
        if did_val:
            result_plots(x_val, y_val, predictions_val, used_quantities=used_quantities, suf=f"_{bin_code}_val")
        if did_train:
            result_plots(x_train, y_train, predictions_train, used_quantities=used_quantities, suf=f"_{bin_code}_train")
        if did_other:
            result_plots(x_others, y_others, predictions_others, used_quantities=used_quantities, suf=f"_{bin_code}_others")

    # Convert predictions from G to target unit
    if did_train:
        predictions_train = convert_unit(predictions_train, initial_unit="G", final_unit=final_b_unit, used_quantities=used_quantities)
    if did_val:
        predictions_val = convert_unit(predictions_val, initial_unit="G", final_unit=final_b_unit, used_quantities=used_quantities)
    if did_test:
        predictions_test = convert_unit(predictions_test, initial_unit="G", final_unit=final_b_unit, used_quantities=used_quantities)
    if did_other:
        predictions_others = convert_unit(predictions_others, initial_unit="G", final_unit=final_b_unit, used_quantities=used_quantities)

    return {"train": {"predictions": predictions_train, "accuracy": acc_train},
            "val": {"predictions": predictions_val, "accuracy": acc_val},
            "test": {"predictions": predictions_test, "accuracy": acc_test},
            "others": {"predictions": predictions_others, "accuracy": acc_others}}


def evaluate_by_parts(model_names: list[str], filename_or_data: str | np.ndarray,
                      observation_name: str = "auto",
                      proportiontocut: float | None = None,
                      params: dict[str, str | int | float | bool | list[int]] | None = None,
                      subfolder_model: str = "",
                      initial_b_unit: Literal["kG", "G", "T", "mT"] = "kG",
                      final_b_unit: Literal["kG", "G", "T", "mT"] | None = None) -> np.ndarray:
    # This function evaluates the single-quantity models on new a dataset and concatenates the outputs
    if final_b_unit is None:
        final_b_unit = initial_b_unit

    if observation_name == "auto":
        observation_name = set_observation_name(filename_or_data)

    if not model_names:
        raise ValueError('"model_names" is empty')

    used_quantities_total = gimme_combined_used_quantities(model_names)
    filtered_indices = np.cumsum(used_quantities_total) - 1

    if proportiontocut is None: proportiontocut = conf_model_setup["trim_mean_cut"]

    filename_or_data = filename_data_to_data(filename_or_data, observation_name=observation_name, quiet=_quiet)
    data = filename_or_data
    data = filter_data_to_used(array=data, used_quantities=used_quantities_total)

    data = convert_unit(data, initial_unit=initial_b_unit, final_unit=_b_unit, used_quantities=used_quantities_total)

    # only convolution layers -> can be used for any input dimensions
    predictions = np.zeros_like(data)
    bin_codes = np.array([gimme_bin_code_from_name(model_name) for model_name in model_names])

    # Collect indices of models that contain the specific quantity
    list_indices = [[bin_code[i] == "1" for bin_code in bin_codes] for i in range(len(used_quantities_total))]

    for i_quantity, indices in enumerate(list_indices):  # over unique bin_codes
        model_names_to_evaluate = np.array(model_names)[indices]
        # single quantity, all models that contain the quantity
        predictions_part = np.zeros((*np.shape(data)[:-1], 1, len(model_names_to_evaluate)))

        for i_model, model_name in enumerate(model_names_to_evaluate):
            # filtered used_quantities
            filtered_used_quantities = gimme_used_from_name(model_name)[used_quantities_total]
            filtered_quantity_index = np.cumsum(gimme_used_from_name(model_name)) - 1
            filtered_quantity_index = filtered_quantity_index[i_quantity]

            predictions_part[..., i_model] = make_predictions(model_names=[model_name],
                                                              x_data=data[..., filtered_used_quantities],
                                                              params=params,
                                                              proportiontocut=0.0,
                                                              subfolder_model=subfolder_model,
                                                              which="new")[..., [filtered_quantity_index]]

        if not is_empty(model_names_to_evaluate):  # if no models, it averages empty array and returns NaN
            predictions[..., [filtered_indices[i_quantity]]] = average_predictions(predictions=predictions_part,
                                                                                   proportiontocut=proportiontocut)

    print("-----------------------------------------------------")

    # Convert predictions to desired unit
    return convert_unit(predictions, initial_unit=_b_unit, final_unit=final_b_unit, used_quantities=used_quantities_total)


def process_patches(model_names: list[str], image_4d: np.ndarray, kernel_size: int | str = "auto",
                    proportiontocut: float | None = None,
                    initial_b_unit: Literal["kG", "G", "T", "mT"] = "kG",
                    final_b_unit: Literal["kG", "G", "T", "mT"] | None = None,
                    max_valid_size: int = 256,
                    subfolder_model: str = "") -> np.ndarray:
    """Processes the image by dividing it into patches and performing convolution."""

    if proportiontocut is None: proportiontocut = conf_model_setup["trim_mean_cut"]

    if not isinstance(kernel_size, int):
        kernel_size = np.zeros(len(model_names))
        for im, model_name in enumerate(model_names):
            with h5py.File(path.join(_path_model, subfolder_model, model_name), "r") as f:
                if "layer_names" in f.attrs:
                    layers = f.attrs["layer_names"]
                elif "layer_names" in f.keys():
                    layers = list(f["layer_names"].keys())
                elif "layers" in f.keys():
                    layers = list(f["layers"].keys())
                elif "_layer_checkpoint_dependencies" in f.keys():
                    layers = list(f["_layer_checkpoint_dependencies"].keys())
                else:
                    layers = None
                n_conv_layers = np.sum([key.startswith("conv2d") for key in layers]) if layers is not None else 10

                if "conv2d" in f.keys():
                    kern_size, _, _, _ = f["conv2d"]["conv2d"]["kernel:0"].shape
                elif "layers" in f.keys():
                    kern_size, _, _, _ = f["layers"]["conv2d"]["vars"]["0"].shape
                elif "_layer_checkpoint_dependencies" in f.keys():
                    kern_size, _, _, _ = f["_layer_checkpoint_dependencies"]["conv2d"]["vars"]["0"].shape
                else:
                    kern_size = 3

            kernel_size[im] = kern_size + (n_conv_layers - 1) * (kern_size - 1)

        kernel_size = int(np.max(kernel_size))

    _, nrows, ncols, _ = np.shape(image_4d)

    # Calculate padding size
    pad_size = kernel_size // 2

    # Initialize the result array (same size as padded image)
    result = np.zeros_like(image_4d)

    # Iterate over patches
    n_patches = len(list(range(0, nrows, max_valid_size))) * len(list(range(0, ncols, max_valid_size)))
    i_patch = 0
    for i in range(0, nrows, max_valid_size):
        for j in range(0, ncols, max_valid_size):
            i_patch += 1
            print(f"--------------- Progress: {i_patch}/{n_patches} ---------------")
            row_start = i
            row_end = min(row_start + max_valid_size, nrows)
            col_start = j
            col_end = min(col_start + max_valid_size, ncols)

            # Extract the patch from the padded image
            patch = image_4d[:, max(row_start - pad_size, 0):min(row_end + pad_size, nrows),
                    max(col_start - pad_size, 0):min(col_end + pad_size, ncols), :]

            # Perform same-like convolution
            conv_patch = evaluate_by_parts(model_names=model_names,
                                           filename_or_data=patch,
                                           proportiontocut=proportiontocut,
                                           params=None,
                                           subfolder_model=subfolder_model,
                                           initial_b_unit=initial_b_unit,
                                           final_b_unit=final_b_unit)

            # Determine valid region to cut from conv_patch
            patch_shape = np.shape(conv_patch)
            valid_row_start = 0 if row_start - pad_size <= 0 else pad_size
            valid_row_end = patch_shape[1] if row_end + pad_size >= nrows else patch_shape[1] - pad_size
            valid_col_start = 0 if col_start - pad_size <= 0 else pad_size
            valid_col_end = patch_shape[2] if col_end + pad_size >= ncols else patch_shape[2] - pad_size

            # Cut conv_patch to valid region
            conv_patch = conv_patch[:, valid_row_start:valid_row_end, valid_col_start:valid_col_end, :]

            # Place the valid part of the convolved patch into the padded result
            result[:, row_start:row_end, col_start:col_end, :] = conv_patch

    return result
