from os import path
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Model
from glob import glob
import h5py
import ast
from astropy.io import fits

from modules.NN_config_parse import bin_to_used
from modules.utilities import check_dir, safe_arange, is_empty, stack, check_file, split_path

from modules._constants import (_path_data, _path_model, _model_suffix, _wp, _observations_key_name, _label_true_name,
                                _label_name, _sep_out, _path_hmi, _observations_name, _config_name,
                                _path_accuracy_tests, _label_pred_name, _lat_name, _lon_name, _data_dir, _quiet)

from modules.NN_config import quantity_names_short

# defaults only
from modules.NN_config import (conf_output_setup, conf_filtering_setup, conf_grid_setup, conf_model_setup,
                               conf_data_split_setup)


def save_data(final_name: str, observations: np.ndarray,
              lat: np.ndarray | None = None, lon: np.ndarray | None = None,
              labels: np.ndarray | None = None,
              order: str | None = None,
              other_info: dict | None = None, subfolder: str = "") -> str:
    """
    if len(observations) != len(metadata):
        raise ValueError("Each observation must have its metadata. Length of observations != length of metadata.")
    """

    final_name = filename_adjustment(final_name)
    final_name = path.join(_data_dir, subfolder, final_name)

    # collect data and metadata
    data_and_metadata = {_observations_name: np.array(observations, dtype=_wp),  # save observations
                         _observations_key_name: add_unit_to_names(quantity_names_short, order=order),  # quantity order
                         # _metadata_name: np.array(metadata, dtype=object)
                         }  # save metadata

    if lat is not None:
        if len(lat) not in np.shape(observations):
            raise ValueError("Each image must have its latitude.")
        data_and_metadata[_lat_name] = np.array(lat, dtype=_wp)  # save lat

    if lon is not None:
        if len(lon) not in np.shape(observations):
            raise ValueError("Each image must have its longitude.")
        data_and_metadata[_lon_name] = np.array(lon, dtype=_wp)  # save lon

    if labels is not None:
        if len(observations) != len(labels):
            raise ValueError("Each image must have its label. Length of observations != length of labels.")

        data_and_metadata[_label_name] = np.array(labels, dtype=_wp)  # save labels

    """
    if metadata_key is not None:
        data_and_metadata[_metadata_key_name] = np.array(metadata_key, dtype=str)
    """

    if other_info is not None:  # existing keys are not updated
        data_and_metadata = other_info | data_and_metadata

    check_dir(final_name)
    with open(final_name, "wb") as f:
        np.savez_compressed(f, **data_and_metadata)

    return final_name


def save_results(final_name: str, y_pred: np.ndarray,
                 x_true: np.ndarray | None = None,
                 y_true: np.ndarray | None = None,
                 output_setup: dict | None = None,
                 grid_setup: dict | None = None,
                 filtering_setup: dict | None = None,
                 data_split_setup: dict | None = None,
                 model_setup: dict | None = None,
                 model_names: list[str] | None = None,
                 subfolder: str = "") -> None:
    if output_setup is None: output_setup = conf_output_setup
    if grid_setup is None: grid_setup = conf_grid_setup
    if filtering_setup is None: filtering_setup = conf_filtering_setup
    if data_split_setup is None: data_split_setup = conf_data_split_setup
    if model_setup is None: model_setup = conf_model_setup
    if model_names is None: model_names = []

    final_name = filename_adjustment(final_name)
    filename = path.join(_path_accuracy_tests, subfolder, final_name)
    check_dir(filename)

    # collect data and metadata
    model_setup["model_names"] = model_names

    data_and_metadata = {_label_pred_name: np.array(y_pred, dtype=_wp),  # save results
                         _config_name: {"output_setup": output_setup,  # save config file
                                        "grid_setup": grid_setup,
                                        "filtering_setup": filtering_setup,
                                        "data_split_setup": data_split_setup,
                                        "model_setup": model_setup}}

    if x_true is not None:
        data_and_metadata[_observations_name] = np.array(x_true, dtype=_wp)

    if y_true is not None:
        data_and_metadata[_label_true_name] = np.array(y_true, dtype=_wp)

    with open(filename, "wb") as f:
        np.savez_compressed(f, **data_and_metadata)


def rescale_intensity(image: np.ndarray, thresh: float = 0.9) -> np.ndarray:
    new_image = np.copy(image)  # deepcopy here

    new_image /= np.nanmax(new_image)
    new_image /= np.nanmedian(new_image[new_image >= thresh])

    return new_image


def load_npz(filename: str, subfolder: str = "", list_keys: list[str] | None = None,
             allow_pickle: bool = True, **kwargs):
    filename = check_file(filename, _path_data, subfolder)

    data = np.load(filename, allow_pickle=allow_pickle, **kwargs)

    if list_keys is None:
        return data

    return {key: data[key][()] for key in list_keys if key in data.files}


def load_fits(filename: str, subfolder: str = "", **kwargs):
    filename = check_file(filename, _path_hmi, subfolder)

    data = fits.open(filename, **kwargs)[1]

    return data


def load_h5(filename: str, subfolder: str = "", list_keys: list[str] | None = None) -> h5py.File | dict:
    filename = check_file(filename, _path_data, subfolder)

    if list_keys is None:
        print("Do not forget to close the file.")
        return h5py.File(filename, "r")

    with h5py.File(filename, "r") as f:
        return {key: np.array(f[key]) for key in list_keys if key in f.keys()}


def load_keras_model_old(filename: str, subfolder: str = "", custom_objects: dict | None = None,
                         compile: bool = True, custom_objects_dict: dict | None = None, **kwargs) -> Model:
    if custom_objects is None:
        if custom_objects_dict is None: custom_objects_dict = {}
        custom_objects = gimme_custom_objects(**custom_objects_dict)

    filename = check_file(filename, _path_model, subfolder)

    # compile=True is needed to get metrics names for composition vs. taxonomy check
    model = load_model(filename, custom_objects=custom_objects, compile=compile, **kwargs)

    return model


def extract_params_from_weights(filename: str, subfolder: str = "") -> dict:
    filename = check_file(filename, _path_model, subfolder)
    used_quantities = gimme_used_from_name(filename)

    n_conv_layer_outside_residuals = 2
    n_conv_layer_per_residual = 2

    with h5py.File(filename, "r") as f:
        kern_size, _, _, num_nodes = f["conv2d"]["conv2d"]["kernel:0"].shape
        model_type = "CNN_sep" if "concatenate" in f.keys() else "CNN"
        n_conv_layers = np.sum(["conv2d" in key for key in f.keys()])

    if model_type == "CNN_sep":
        n_conv_layers_per_quantity = n_conv_layers / np.sum(used_quantities)
    else:
        n_conv_layers_per_quantity = n_conv_layers

    # n_conv_layers_per_quantity = num_residuals * n_conv_layer_per_residual + n_conv_layer_outside_residuals
    num_residuals = int(np.round((n_conv_layers_per_quantity - n_conv_layer_outside_residuals) / n_conv_layer_per_residual))

    return {"model_type": model_type, "num_residuals": num_residuals, "num_nodes": num_nodes, "kern_size": kern_size}


def adjust_params_from_weights(filename: str, params: dict, subfolder: str = "") -> dict:
    return params | extract_params_from_weights(filename, subfolder=subfolder)


def load_keras_model(filename: str, input_shape: tuple[int, ...], subfolder: str = "",
                     params: dict | None = None, quiet: bool = _quiet) -> Model:
    from modules.NN_models import build_model

    filename = check_file(filename, _path_model, subfolder)
    used_quantities = gimme_used_from_name(filename)

    default_params = conf_model_setup["params"]  # default (if some keywords are missing in params)

    if params is None:
        params = default_params

        try:  # if params are stored in weights file, update the params
            with h5py.File(filename, "r") as f:
                params |= ast.literal_eval(f["params"][()].decode())
            if not quiet:
                print("Loading params from the weight file.")

        except KeyError:  # use default
            if not quiet:
                print("Original params were not stored together with the weights.")
                print("Restoring architecture from the weights. Using default for others.")
    else:
        params = default_params | params  # update the defaults

    # adjust params based on the architecture
    params = adjust_params_from_weights(filename=filename, params=params, subfolder=subfolder)
    model = build_model(input_shape=input_shape, params=params, used_quantities=used_quantities)
    model.load_weights(filename)

    return model


def load_xlsx(filename: str, subfolder: str = "", **kwargs) -> pd.DataFrame:
    filename = check_file(filename, _path_data, subfolder)
    excel = pd.read_excel(filename, **kwargs)

    return excel


def load_txt(filename: str, subfolder: str = "", **kwargs) -> pd.DataFrame:
    filename = check_file(filename, _path_data, subfolder)
    data = pd.read_csv(filename, **kwargs)

    return data


def combine_files(filenames: tuple[str, ...], final_name: str, subfolder: str = "") -> str:
    final_name = filename_adjustment(final_name)
    final_name = path.join(_path_data, subfolder, final_name)

    combined_file = dict(load_npz(filenames[0]))
    for filename in filenames[1:]:
        file_to_merge = dict(load_npz(filename))

        combined_file[_observations_name] = stack((combined_file[_observations_name], file_to_merge[_observations_name]), axis=0)
        combined_file[_lat_name] = stack((combined_file[_lat_name], file_to_merge[_lat_name]), axis=0)
        combined_file[_lon_name] = stack((combined_file[_lon_name], file_to_merge[_lon_name]), axis=0)

    check_dir(final_name)
    with open(final_name, "wb") as f:
        np.savez_compressed(f, **combined_file)

    return final_name


def filename_adjustment(filename: str) -> str:
    return f"{split_path(filename)[1]}.npz"


def center_crop_to_patch_size(image: np.ndarray, patch_size: int | None = None):
    if patch_size is None: patch_size = conf_grid_setup["patch_size"]

    _, nx, ny, _ = np.shape(image)

    # Calculate the target dimensions based on the patch size
    target_height = (nx // patch_size) * patch_size
    target_width = (ny // patch_size) * patch_size

    # Calculate how much to crop from each side
    crop_height = nx - target_height
    crop_width = ny - target_width

    # Calculate cropping offsets
    crop_top = crop_height // 2
    crop_bottom = crop_height - crop_top
    crop_left = crop_width // 2
    crop_right = crop_width - crop_left

    # Crop the image
    cropped_image = image[:, crop_top:nx - crop_bottom, crop_left:ny - crop_right, :]

    return cropped_image


def model_name_to_result_name(model_name: str) -> str:
    return f"{split_path(model_name)[1].replace('weights', 'results', 1)}.npz"


def if_no_test_data(x_train: np.ndarray | None, y_train: np.ndarray,
                    x_val: np.ndarray | None, y_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not is_empty(y_val):  # If the test portion is zero then use validation data
        x_test, y_test = x_val, y_val
    else:  # If even the val portion is zero, use train data (just for visualisation purposes)
        x_test, y_test = x_train, y_train

    return x_test, y_test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    all_to_one: bool = False,
                    return_r2: bool = False, return_sam: bool = False, return_mae: bool = False,
                    return_dict: bool = False) -> tuple[np.ndarray, ...] | dict[str, np.ndarray]:
    from modules.NN_losses_metrics_activations import my_mae, my_rmse, my_r2, my_sam

    results = {"RMSE": my_rmse(all_to_one=all_to_one)(y_true, y_pred).numpy()}

    if return_r2:
        results["R2"] = my_r2(all_to_one=all_to_one)(y_true, y_pred).numpy()

    if return_sam:
        results["SAM (deg)"] = np.rad2deg(my_sam(all_to_one=all_to_one)(y_true, y_pred).numpy())

    if return_mae:
        results["MAE"] = my_mae(all_to_one=all_to_one)(y_true, y_pred).numpy()

    if return_dict:
        return results

    return tuple([*results.values()])


def compute_within(y_true: np.ndarray, y_pred: np.ndarray, error_limit: tuple[float, ...],
                   all_to_one: bool = False,
                   step_percentile: float | None = 0.1,
                   return_dict: bool = False) -> tuple[np.ndarray, ...] | dict[str, np.ndarray]:
    from modules.NN_losses_metrics_activations import my_quantile, my_ae

    if step_percentile is None:
        ae = my_ae()(y_true, y_pred)
        axis = None if all_to_one else (0, 1, 2)
        within = np.array([100. * np.sum(ae <= x, axis=axis) / np.sum(np.isfinite(ae), axis=axis) for x in error_limit])

    else:
        percentile = safe_arange(0., 100., step_percentile, endpoint=True)
        quantile = my_quantile(percentile=percentile, all_to_one=all_to_one)(y_true, y_pred).numpy()

        # quantiles are always sorted
        within = np.transpose([np.interp(error_limit, quantile[:, i], percentile, left=0., right=100.)
                               for i in range(np.shape(quantile)[1])])

    within = np.array(within, dtype=_wp)

    if return_dict:
        return {f"within {limit} AE (%)": within_limit for limit, within_limit in zip(error_limit, within)}

    return tuple(within)


def compute_one_sigma(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, ...]:
    from modules.NN_losses_metrics_activations import my_quantile

    one_sigma = my_quantile(percentile=68.27)(y_true, y_pred).numpy()

    return one_sigma


def gimme_custom_objects(used_quantities: np.ndarray | None = None,
                         p_coef: float = 1.5, percentile: float = 50.) -> dict:
    from modules.NN_losses_metrics_activations import create_custom_objects
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    return create_custom_objects(used_quantities=used_quantities, p_coef=p_coef, percentile=percentile)


def cut_error_bars(y_true: np.ndarray, y_true_error: np.ndarray | float, y_pred: np.ndarray, y_pred_error: np.ndarray,
                   lim_min: float | None = None, lim_max: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    # equivalently
    # lower_error, upper_error = y_true - y_true_error, y_true_error + y_true
    # lower_error[lower_error < lim_min], upper_error[upper_error > lim_max] = lim_min, lim_max
    # lower_error, upper_error = y_true - lower_error, upper_error - y_true

    lower_error = y_true - np.clip(y_true - y_true_error, lim_min, lim_max)
    upper_error = np.clip(y_true_error + y_true, lim_min, lim_max) - y_true
    axis = np.ndim(lower_error)  # to create a new axes using stack
    actual_errorbar_reduced = np.moveaxis(stack((lower_error, upper_error), axis=axis),
                                          source=0, destination=-1)  # to shift the last axes to the beginning

    lower_error = y_pred - np.clip(y_pred - y_pred_error, lim_min, lim_max)
    upper_error = np.clip(y_pred_error + y_pred, lim_min, lim_max) - y_pred
    predicted_errorbar_reduced = np.moveaxis(stack((lower_error, upper_error), axis=axis),
                                             source=0, destination=-1)  # to shift the last axes to the beginning

    predicted_errorbar_reduced[predicted_errorbar_reduced < 0.] = 0.
    actual_errorbar_reduced[actual_errorbar_reduced < 0.] = 0.

    return predicted_errorbar_reduced, actual_errorbar_reduced


def error_estimation_overall(y_true: np.ndarray, y_pred: np.ndarray, actual_error: np.ndarray | float = 3.
                             ) -> tuple[np.ndarray, np.ndarray]:
    from modules.NN_losses_metrics_activations import my_rmse

    errors = my_rmse(all_to_one=False)(y_true, y_pred).numpy()

    # return cut_error_bars(y_true, actual_error, y_pred, errors)
    return errors, actual_error


def error_estimation_bin_like(y_true: np.ndarray, y_pred: np.ndarray, actual_error: np.ndarray | float = 3.
                              ) -> tuple[np.ndarray, np.ndarray]:
    from modules.NN_losses_metrics_activations import my_rmse

    # N bins
    N = 10

    # split array to N bins
    ibin = np.digitize(y_pred, bins=np.linspace(np.min(y_pred), np.max(y_pred), N + 1))
    ibin[ibin == N + 1] = N  # last one to the previous bin (otherwise it is there alone)
    ibin -= 1  # count from 0

    errors = np.zeros(np.shape(y_pred))  # final errors for each point

    for i in range(N):
        mask = ibin == i

        # mask only the values in the bin
        y_t, y_p = np.where(mask, y_true, np.nan), np.where(mask, y_pred, np.nan)
        errors = np.where(mask, my_rmse(all_to_one=False)(y_t, y_p).numpy(), errors)

    # return cut_error_bars(y_true, actual_error, y_pred, errors)
    return errors, actual_error


def used_indices(used_quantities: np.ndarray | None = None, return_digits: bool = False) -> np.ndarray:
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    if return_digits:
        return np.where(used_quantities)[0]
    return used_quantities


def unit_to_gauss(array: np.ndarray, used_quantities: np.ndarray | None = None) -> np.ndarray:
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    convert_units = np.array([1., 1000., 1000., 1000.])[used_indices(used_quantities)]  # kG -> G

    return array * convert_units


def print_header(used_quantities: np.ndarray | None = None) -> None:
    # Function to print header for an accuracy metric
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    used_names = add_unit_to_names(quantity_names_short)[used_indices(used_quantities)]
    print(f"{'':23} {'   '.join(f'{cls:8}' for cls in used_names)}")


def print_info(y_true: np.ndarray, y_pred: np.ndarray, which: str = "test") -> np.ndarray:
    from modules.NN_losses_metrics_activations import my_rmse

    accuracy = my_rmse(all_to_one=False)(y_true, y_pred).numpy()
    pref = f"Mean {which.lower()} RMSE score:"

    print(f"{pref:27} {'     '.join(f'{acc:6.3f}' for acc in np.round(accuracy, 3))}")

    return accuracy


def collect_all_models(subfolder_model: str, prefix: str | None = None, suffix: str | None = None,
                       regex: str | None = None, file_suffix: str = _model_suffix, full_path: bool = True) -> list[str]:

    final_suffix = "" if file_suffix == "SavedModel" else f".{file_suffix}"

    if prefix is not None:
        model_str = path.join(_path_model, subfolder_model, f"{prefix}*{final_suffix}")
    elif suffix is not None:
        model_str = path.join(_path_model, subfolder_model, f"*{suffix}{final_suffix}")
    elif regex is not None:
        model_str = path.join(_path_model, subfolder_model, f"{regex}{final_suffix}")
    else:
        model_str = path.join(_path_model, subfolder_model, f"*{final_suffix}")

    if full_path:
        return glob(model_str)
    else:
        return [path.basename(x) for x in glob(model_str)]


def add_unit_to_names(all_quantity_names: np.ndarray, order: str | None = None,
                      return_units: bool = False, latex_output: bool = False) -> np.ndarray | tuple[np.ndarray, ...]:
    names = np.array(all_quantity_names)
    if order is None:
        if latex_output:
            units = np.array([r"$\left(\text{cont. norm.}\right)$", r"$\left(\text{G}\right)$",
                              r"$\left(\text{G}\right)$", r"$\left(\text{G}\right)$"])
        else:
            units = np.array(["(cont. norm.)", "(G)", "(G)", "(G)"])
    else:
        if latex_output:
            units = np.array([r"$\left(\text{cont. norm.}\right)$", f"$\\left(\\text{{{order}G}}\\right)$",
                              f"$\\left(\\text{{{order}G}}\\right)$", f"$\\left(\\text{{{order}G}}\\right)$"])
        else:
            units = np.array(["(cont. norm.)", f"({order}G)", f"({order}G)", f"({order}G)"])

    names = np.array([f"{name} {unit}" for name, unit in zip(names, units)])

    if return_units:
        return names, units
    return names


def gimme_bin_code_from_name(model_name: str) -> str:
    bare_name = split_path(model_name)[1]

    name_parts = np.array(bare_name.split(_sep_out))

    # dt_string is made of 14 decimals
    dt_string_index = np.where([part.isdecimal() and len(part) == 14 for part in name_parts])[0]

    if np.size(dt_string_index) > 0:
        return _sep_out.join(name_parts[1:dt_string_index[0]])  # cut model_type, dt_string and following parts
    return _sep_out.join(name_parts[1:])  # cut model_type


def gimme_used_from_name(model_name: str) -> np.ndarray:
    return bin_to_used(bin_code=gimme_bin_code_from_name(model_name=model_name))


def gimme_params(model_name: str, subfolder: str = "") -> dict:
    # first check, if there is the result file with stored parameters
    result_name = path.join(_path_accuracy_tests, subfolder, model_name_to_result_name(model_name))

    if path.isfile(result_name):
        data = load_npz(result_name)
        params = data["config"][()]["model_setup"]["params"]

    else:  # maybe the unknown models are in model_names in one of the known model
        results = collect_all_models(subfolder_model=subfolder,
                                     prefix=f"weights{_sep_out}{gimme_bin_code_from_name(model_name)}")

        for result in results:
            result_name = path.join(_path_accuracy_tests, subfolder, model_name_to_result_name(result))
            if path.isfile(result_name):
                data = load_npz(result_name)
                if model_name in data["config"][()]["model_setup"]["model_names"]:
                    params = data["config"][()]["model_setup"]["params"]

    if "params" in locals():
        return params

    raise ValueError(f"Unknown parameters for {model_name}.")
