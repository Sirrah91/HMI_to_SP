from modules.NN_config_parse import bin_to_used
from modules.utilities import (check_dir, safe_arange, is_empty, stack, check_file, split_path, make_video_from_arrays,
                               make_video_from_images, concatenate_videos, remove_if_exists, standardize, update_dict,
                               robust_load_weights, remove_outliers_2d, pad_zeros_or_crop, return_mean_std)

from modules._constants import (_path_data, _path_model, _model_suffix, _wp, _observations_key_name, _label_true_name,
                                _label_name, _sep_out, _path_hmi, _observations_name, _config_name, _result_dir, _quiet,
                                _path_accuracy_tests, _label_pred_name, _lat_name, _lon_name, _data_dir, _path_figures,
                                _num_eps)

from modules.NN_config import quantity_names_short

# defaults only
from modules.NN_config import (conf_output_setup, conf_filtering_setup, conf_grid_setup, conf_model_setup,
                               conf_data_split_setup)

from os import path, listdir
import os
import gc
from typing import Literal
import numpy as np
import pandas as pd
from scipy.sparse.linalg import lsmr
from scipy.fft import fft2, ifft2, fftshift
import scipy.io as sio
from tensorflow.keras.models import load_model, Model
from tensorflow.python.framework.ops import EagerTensor
from glob import glob
import h5py
import ast
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from sunpy.coordinates import Helioprojective, HeliographicStonyhurst
from scipy.special import j1
from skimage.transform import resize
from skimage import io, filters, feature, color
from skimage.restoration import estimate_sigma
import drms
import time
from datetime import datetime, timedelta
import warnings


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
        if not is_empty(observations) and len(lat) not in np.shape(observations):
            raise ValueError("Each image must have its latitude.")
        data_and_metadata[_lat_name] = np.array(lat, dtype=_wp)  # save lat

    if lon is not None:
        if not is_empty(observations) and len(lon) not in np.shape(observations):
            raise ValueError("Each image must have its longitude.")
        data_and_metadata[_lon_name] = np.array(lon, dtype=_wp)  # save lon

    if labels is not None:
        if not is_empty(observations) and len(observations) != len(labels):
            raise ValueError("Each image must have its label. Length of observations != length of labels.")

        data_and_metadata[_label_name] = np.array(labels, dtype=_wp)  # save labels

    """
    if metadata_key is not None:
        data_and_metadata[_metadata_key_name] = np.array(metadata_key, dtype=str)
    """

    if other_info is not None:  # existing keys are not updated
        data_and_metadata = update_dict(new_dict=data_and_metadata, base_dict=other_info)

    check_dir(final_name, is_file=True)
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
    check_dir(filename, is_file=True)

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


def quiet_sun_mask(bvec: np.ndarray, threshold: float = 500.) -> np.ndarray:
    if np.ndim(bvec) == 3:
        axis = 0
    else:
        axis = (0, 3)

    return np.all(np.abs(bvec) < threshold, axis=axis)


def normalise_intensity(image: np.ndarray,
                        threshold: float | tuple[float, float] = (0.6, 0.95),
                        mask: np.ndarray | None = None) -> np.ndarray:
    masked_image = 1. * image  # deepcopy

    if mask is not None:
        masked_image[~mask] = np.nan
    if isinstance(threshold, float | int):
        threshold = (float(threshold), 1.)
    nanmax = np.nanmax(masked_image)

    qs_mask = np.logical_and(threshold[0] * nanmax <= masked_image, masked_image <= threshold[1] * nanmax)
    return image / np.nanmedian(masked_image[qs_mask]).astype(np.result_type(image))


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


def extract_params_from_weights(filename: str,
                                n_conv_layer_outside_residuals: int = 2,
                                n_conv_layer_per_residual: int = 2,
                                subfolder: str = "") -> dict:
    filename = check_file(filename, _path_model, subfolder)
    used_quantities = gimme_used_from_name(filename)

    with h5py.File(filename, "r") as f:
        exact_names = "layer_names" in f.attrs or "layer_names" in f.keys()
        if "layer_names" in f.attrs:
            layers = f.attrs["layer_names"]
        elif "layer_names" in f.keys():
            layers = list(f["layer_names"].keys())
        elif "layers" in f.keys():
            layers = list(f["layers"].keys())
        elif "_layer_checkpoint_dependencies" in f.keys():
            layers = list(f["_layer_checkpoint_dependencies"].keys())
        else:
            raise ValueError("Cannot get layer names from the saved weights.")

        if "conv2d" in f.keys():
            kern_size, _, _, num_nodes = f["conv2d"]["conv2d"]["kernel:0"].shape
        elif "layers" in f.keys():
            kern_size, _, _, num_nodes = f["layers"]["conv2d"]["vars"]["0"].shape
        elif "_layer_checkpoint_dependencies" in f.keys():
            kern_size, _, _, num_nodes = f["_layer_checkpoint_dependencies"]["conv2d"]["vars"]["0"].shape
        else:
            raise ValueError("Cannot get conv2d size from the saved weights.")

    model_type = "CNN_sep" if "concatenate" in layers else "CNN"
    apply_bs_norm = True if "batch_normalization" in layers else False
    n_conv_layers = np.sum([key.startswith("conv2d") for key in layers])
    kern_pad = "valid" if "reflection_padding2d" in layers else "same"

    if exact_names:  # "layer_names" should be present in f.attrs
        activation_position = np.where([key.startswith("activation_") for key in layers])[0]
        input_activation = "_".join(layers[activation_position[0]].split("_")[1:-1])  # contain a number
        output_activation = "_".join(layers[activation_position[-1]].split("_")[1:])  # does not contain number
        bs_norm_before_activation = True if "batch_norm_before_activation_0" in layers else False
    else:
        # Unfortunately, here must be some defaults
        bs_norm_before_activation = True
        input_activation = "relu"
        output_activation = "softplus_linear"

    if model_type == "CNN_sep":
        n_conv_layers_per_quantity = n_conv_layers / np.sum(used_quantities)
    else:
        n_conv_layers_per_quantity = n_conv_layers

    # n_conv_layers_per_quantity = num_residuals * n_conv_layer_per_residual + n_conv_layer_outside_residuals
    num_residuals = int(np.round((n_conv_layers_per_quantity - n_conv_layer_outside_residuals) / n_conv_layer_per_residual))

    return {"model_type": model_type, "num_residuals": num_residuals, "num_nodes": num_nodes,
            "kern_size": kern_size, "kern_pad": kern_pad,
            "input_activation": input_activation, "output_activation": output_activation,
            "apply_bs_norm": apply_bs_norm, "bs_norm_before_activation": bs_norm_before_activation}


def adjust_params_from_weights(filename: str, params: dict,
                               subfolder: str = "") -> dict:
    return update_dict(new_dict=extract_params_from_weights(filename, subfolder=subfolder), base_dict=params)


def load_keras_model(filename: str, input_shape: tuple[int, ...], subfolder: str = "",
                     params: dict | None = None, weights: np.ndarray | EagerTensor | None = None,
                     bins: tuple | list | None = None, quiet: bool = _quiet) -> Model:
    from modules.NN_models import build_model

    filename = check_file(filename, _path_model, subfolder)
    used_quantities = gimme_used_from_name(filename)

    default_params = conf_model_setup["params"]  # default (if some keywords are missing in params)
    adjust_params = True  # if not stored in weights

    if params is None:
        try:  # if params are stored in weights file, update the params
            with h5py.File(filename, "r") as f:
                if "params" in f.attrs:
                    params = update_dict(new_dict=ast.literal_eval(f.attrs["params"]), base_dict=default_params)
                elif "params" in f.keys():
                    params = update_dict(new_dict=ast.literal_eval(f["params"][()].decode()), base_dict=default_params)
                else:
                    raise KeyError("Unknown params.")
            adjust_params = False
            if not quiet:
                print("Loading params from the weight file.")

        except KeyError:  # use default
            params = default_params
            if not quiet:
                print("Original params were not stored together with the weights.")
                print("Restoring architecture from the weights. Using default for others.")
    else:
        params = update_dict(new_dict=params, base_dict=default_params)  # update the defaults

    if adjust_params:  # if not stored in weights
        # adjust params based on the architecture
        params = adjust_params_from_weights(filename=filename, params=params, subfolder=subfolder)

    metrics = params["metrics"] if "metrics" in params else None

    model = build_model(input_shape=input_shape, params=params, used_quantities=used_quantities,
                        metrics=metrics, weights=weights, bins=bins)
    model = robust_load_weights(model, filename=filename)

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

    check_dir(final_name, is_file=True)
    with open(final_name, "wb") as f:
        np.savez_compressed(f, **combined_file)

    return final_name


def filename_adjustment(filename: str) -> str:
    return f"{split_path(filename)[1]}.npz"


def center_crop_to_patch_size(image: np.ndarray, patch_size: int | None = None) -> np.ndarray:
    if patch_size is None: patch_size = conf_grid_setup["patch_size"]

    if np.ndim(image) == 2:
        nx, ny = np.shape(image)
        image = np.reshape(image, (1, nx, ny, 1))

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


def check_hmi_header(list_of_headers: list) -> bool:
    rotated = np.array(["history" in index and "rotated" in str(index["history"]) for index in list_of_headers])
    return all(rotated) or all(~rotated)


def prepare_hmi_data(fits_ic: str | None = None,
                     fits_b: str | None = None,
                     fits_inc: str | None = None,
                     fits_azi: str | None = None,
                     fits_disamb: str | None = None,
                     from_row: int | None = None,
                     to_row: int | None = None,
                     from_col: int | None = None,
                     to_col: int | None = None,
                     remove_limb_dark: bool = True,
                     disambiguate: bool = True,
                     interpolate_outliers: bool = False,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, any]:
    from modules.align_data import calc_hmi_to_sp_resolution

    hmi_to_sp_rows, hmi_to_sp_cols = calc_hmi_to_sp_resolution(fast=True)

    def read_and_process_fits(file: str, dtype: type, process_fn=None) -> np.ndarray:
        """Helper function to read and process FITS data in chunks."""
        with fits.open(file, memmap=True) as hdu:
            data = np.array(hdu[1].data)
            header = hdu[1].header
        return process_fn(data, header).astype(dtype) if callable(process_fn) else data.astype(dtype)

    def resize_data(data: np.ndarray) -> np.ndarray:
        """Resize data efficiently."""
        output_shape = np.round([hmi_to_sp_rows * data.shape[0], hmi_to_sp_cols * data.shape[1]]).astype(int)
        return resize(data, output_shape, anti_aliasing=True)

    def crop_data(data: np.ndarray) -> np.ndarray:
        """Crop data to the specified row and column range."""
        if from_row is not None or from_col is not None:
            return data[from_row:to_row, from_col:to_col]
        return data

    def process_intensity(fits_file: str, b_mask: np.ndarray | None = None) -> np.ndarray:
        print("Processing Ic...")
        if remove_limb_dark:
            process_fn = lambda image, header: crop_data(remove_limb_darkening_exact(image, header))
        else:
            process_fn = lambda image, header: crop_data(image)

        ic = read_and_process_fits(fits_file, dtype=_wp, process_fn=process_fn)
        if interpolate_outliers:
            ic = remove_outliers_2d(ic, kernel_size=7, n_std=3., interp_nans=True, maximum_value=None)
        ic = resize_data(ic)
        return normalise_intensity(ic, threshold=(0.6, 0.95), mask=b_mask)[np.newaxis, ..., np.newaxis]

    def process_vector(fits_b: str, fits_inc: str, fits_azi: str, fits_disamb: str | None) -> np.ndarray:
        print("Processing B vector...")

        index = []
        # Read data one by one to save memory
        with fits.open(fits_b, memmap=True) as hdu:
            index.append(hdu[1].header)
        with fits.open(fits_inc, memmap=True) as hdu:
            index.append(hdu[1].header)
        with fits.open(fits_azi, memmap=True) as hdu:
            index.append(hdu[1].header)

        if disambiguate:
            with fits.open(fits_disamb, memmap=True) as hdu:
                index.append(hdu[1].header)

        if not check_hmi_header(index):
            raise ValueError("(At least) one of fits file is not compatible with others.")
        index = index[0]

        if disambiguate:
            bvec = np.stack([
                read_and_process_fits(fits_b, dtype=_wp),
                read_and_process_fits(fits_inc, dtype=_wp),
                disambigue_azimuth(read_and_process_fits(fits_azi, dtype=_wp),
                                   read_and_process_fits(fits_disamb, dtype=int),
                                   method=1,
                                   rotated_image="history" in index and "rotated" in str(index["history"]))
            ], axis=0)
        else:
            bvec = np.stack([
                read_and_process_fits(fits_b, dtype=_wp),
                read_and_process_fits(fits_inc, dtype=_wp),
                read_and_process_fits(fits_azi, dtype=_wp)
            ], axis=0)

        # Convert to local reference frame
        bvec = list(data_b2ptr(index=index, bvec=bvec))

        # Interpolate and resize each component individually
        for i, data_part in enumerate(bvec):
            data_part = crop_data(data_part)  # Crop after `data_b2ptr`
            if interpolate_outliers:
                data_part = remove_outliers_2d(data_part, kernel_size=7, n_std=3., interp_nans=True, maximum_value=10000.)
            bvec[i] = resize_data(data_part)
        return np.transpose(bvec, axes=(1, 2, 0)).astype(_wp)[np.newaxis, ...]

    # Process data (overwrite variables to minimise memory use)
    data, b_mask = None, None
    if None not in [fits_b, fits_inc, fits_azi]:
        if fits_disamb is None and disambiguate:
            raise ValueError("Invalid input: Disambig fits not available.")

        data = process_vector(fits_b, fits_inc, fits_azi, fits_disamb)
        b_mask = quiet_sun_mask(data, threshold=500.)
    if fits_ic:
        ic = process_intensity(fits_ic, b_mask=b_mask)
        data = ic if data is None else np.concatenate([ic, data], axis=-1)

    if data is None:
        raise ValueError("Invalid input: No fits in the function input.")

    print("Correction for disk orientation")
    # Determine longitude and latitude from header
    header_source = fits_ic or fits_b
    with fits.open(header_source, memmap=True) as hdu:
        header = hdu[1].header
    lon, lat = return_lonlat(header)

    # Rotate coordinates to ensure North is up and East is left
    data = rot_coordinates_to_NW(longitude=lon, latitude=lat, array_to_flip=data)
    lon, lat = rot_coordinates_to_NW(longitude=lon, latitude=lat, array_to_flip=[lon, lat])
    lon, lat = resize_data(lon), resize_data(lat)
    lon, lat = np.clip(lon, a_min=-180., a_max=180.), np.clip(lat, a_min=-90., a_max=90.)

    # If B is included, Bp: +W, Bt: +S, Br: -grav
    return data, lon, lat, header


def arcsec_to_Mm(arcsec: float, distanceAU: float = 1., center_distance: bool = True) -> float:
    fun = np.tan if center_distance else np.sin
    return 2. * u.AU.to(u.Mm, distanceAU) * fun(u.arcsec.to(u.rad, arcsec) / 2.)


def jsoc_query(obs_date: str,  # "%Y-%m-%d" or "%Y-%m-%dT%H:%M:%S.%f", e.g. "2020-01-01" or "2020-01-01T12:15:10.012"
               quantity: str,
               data_type: Literal["", "dcon", "dconS"] = "",
               locunits: Literal["arcsec", "pixels", "stony", "carrlong"] = "stony",
               locref: tuple[float, float] | list[float] | np.ndarray | str = (0., 0.),
               integration_time_sec: int = 720,
               time_step_sec: int | None = None,
               total_obs_time_hours: float = 0.,
               t_ref: str | None = None,
               boxunits: Literal["pixels", "arcsec", "degrees"] = "pixels",
               boxsize: tuple[int, int] | list[int] | np.ndarray | int = (512, 512),
               outdir_prefix: str = "") -> str:

    warnings.filterwarnings("ignore")

    if time_step_sec is None:
        time_step_sec = integration_time_sec

    # E-mail registered in JSOC
    email = "d.korda@seznam.cz"

    if isinstance(boxsize, int):
        boxsize = (boxsize, boxsize)

    if isinstance(locref, str):
        if locunits == "stony":
            locref = str_coordinate_to_tuple(locref)
        else:
            locref = ast.literal_eval(locref)

    locref_str = np.array(np.round(locref), dtype=int)
    if locunits == "stony":
        tmp = f"N{locref_str[0]:02d}" if locref_str[0] >= 0 else f"S{-locref_str[0]:02d}"
        locref_str = f"{tmp}W{locref_str[1]:02d}" if locref_str[1] >= 0 else f"{tmp}E{-locref_str[1]:02d}"
    else:
        unit = "px" if locunits == "pixels" else "deg"
        locref_str = f"({';'.join([str(round(loc)) for loc in locref])}){unit}"

    # margin of timespan in minutes
    margin_frames = 0
    margin_time = margin_frames * integration_time_sec / 60.

    if "T" not in obs_date:
        start_obs = f"{obs_date}T00:00:00.000"
    else:
        start_obs = obs_date
        obs_date = obs_date.split("T")[0]
    start_obs = parse_datetime(start_obs)

    if start_obs is None:
        raise ValueError('Invalid date format. Check "parse_datetime" for options.')

    start_obs -= timedelta(minutes=margin_time)

    end_obs = start_obs + timedelta(hours=total_obs_time_hours, minutes=2*margin_time)

    obs_length = end_obs - start_obs
    # cannot be (start_obs + end_obs) / 2 because datetime does not support "+" but timedelta does
    obs_centre = start_obs + obs_length / 2.

    date_str_start = start_obs.strftime("%Y.%m.%d")
    time_str_start = start_obs.strftime("%H:%M:%S")

    if t_ref is None:
        date_str_centre = obs_centre.strftime("%Y-%m-%d")  # different format from the start
        time_str_centre = obs_centre.strftime("%H:%M:%S")
        t_ref = f"{date_str_centre}T{time_str_centre}"
    elif "T" not in t_ref:
        t_ref = f"{t_ref}T00:00:00.000"

    obs_length = int(np.ceil(obs_length.total_seconds() / 3600.))  # in hours

    if data_type:
        data_type = f"_{data_type}"

    if quantity in ["I", "continuum", "intensity"]:  # Ic data duration@lagImages
        query_str = f"hmi.Ic_{integration_time_sec}s{data_type}[{date_str_start}_{time_str_start}_TAI/{obs_length}h@{time_step_sec}s]{{continuum}}"
    elif quantity in ["B"]:  # magnetic field vector data
        query_str = f"hmi.B_{integration_time_sec}s{data_type}[{date_str_start}_{time_str_start}_TAI/{obs_length}h@{time_step_sec}s]{{field,inclination,azimuth,disambig}}"
    else:  # magnetic field component
        query_str = f"hmi.B_{integration_time_sec}s{data_type}[{date_str_start}_{time_str_start}_TAI/{obs_length}h@{time_step_sec}s]{{{quantity}}}"
    print(f"Data export query:\n\t{query_str}\n")

    process = {"im_patch": {
        "t_ref": t_ref,
        # there must be a non-missing image within +- 2 hours of t_ref
        "t": 0,  # tracking?
        "r": 0,  # register the images to the first frame?
        "c": 0,  # cropping?
        "locunits": locunits,  # units for x and y
        "x": locref[1],  # center_x in locunits
        "y": locref[0],  # center_y in locunits
        "boxunits": boxunits,  # units for width and height
        "width": boxsize[1],
        "height": boxsize[0],
    }}

    print("Submitting export request...")
    client = drms.Client()
    result = client.export(
        query_str,
        method="url",
        protocol="fits",
        email=email,
        process=process,
    )

    # Print request URL.
    print(f"\nRequest URL: {result.request_url}")
    print(f"{int(len(result.urls))} file(s) available for download.")

    out_dir = path.join(_path_hmi, f"{outdir_prefix}{obs_date.replace('-', '')}{_sep_out}{locref_str}")
    check_dir(out_dir)

    # Skip existing files.
    stored_files = listdir(out_dir)
    new_file_indices = np.where([file not in stored_files for file in result.data["filename"]])[0]
    print(f"{len(new_file_indices)} file(s) haven't been downloaded yet.\n")

    # Download selected files.
    result.wait()
    result.download(out_dir, index=new_file_indices)
    print("Download finished.")
    print(f'Download directory:\n\t"{path.abspath(out_dir)}"\n')

    print("Pausing the code for 10 seconds to avoid errors caused by pending requests.\n")
    time.sleep(10)

    return out_dir


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


def gimme_combined_used_quantities(model_names: list[str]):
    return np.any([gimme_used_from_name(model_name=model_name) for model_name in model_names], axis=0)


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


def error_estimation_bin_like(array_master: np.ndarray,
                              array_slave: np.ndarray,
                              bins: list | None = None,
                              used_quantities: np.ndarray | None = None) -> tuple[list, ...]:
    # returns mean RMSE in a specific bin

    from modules.NN_losses_metrics_activations import my_rmse

    if bins is None:
        # _max = np.nanmax(array_master, axis=(0, 1, 2))
        # _min = np.nanmin(array_master, axis=(0, 1, 2))
        bins = [[0., *np.arange(0.3, 1.4, 0.1), 5.],
                [*np.arange(-10000., -2000., 500.), *np.arange(-2000., 2001., 100.), *np.arange(2000., 10000., 500.)],
                [*np.arange(-10000., -2000., 500.), *np.arange(-2000., 2001., 100.), *np.arange(2000., 10000., 500.)],
                [*np.arange(-10000., -2000., 500.), *np.arange(-2000., 2001., 100.), *np.arange(2000., 10000., 500.)]]

        if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]
        bins = [_bin for i, _bin in enumerate(bins) if used_quantities[i]]

    errors = []
    counts = []
    for i, _bin in enumerate(bins):
        master, slave = array_master[..., i], array_slave[..., i]
        indices = np.digitize(master, bins=_bin) -1  # -1 to count from 0

        # Calculate counts for all bins
        bin_counts = [np.sum(indices == j) for j in range(len(_bin) - 1)]
        counts.append(bin_counts)

        # Calculate RMSE for each bin, using the most recent bin_counts to check for empty bin
        bin_errors = np.array([my_rmse(all_to_one=True)(master[indices == j], slave[indices == j]).numpy()
                               if bin_counts[j] > 0 else np.nan for j in range(len(_bin) - 1)])
        errors.append(bin_errors)

    return bins, errors, counts


def used_indices(used_quantities: np.ndarray | None = None, return_digits: bool = False) -> np.ndarray:
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    if return_digits:
        return np.where(used_quantities)[0]
    return used_quantities


def weight_bins(minimum: list, maximum: list, step: float = 0.1) -> list:
    # +- 5 kG and +- 500% intensity is enough...
    master_bins = safe_arange(-5., 5. + step, step, endpoint=True) - step / 2.
    return [master_bins[np.logical_and(minimum[i] - step <= master_bins, maximum[i] + step >= master_bins)] for i in range(len(minimum))]


def invert_bt_sign(array_4d: np.ndarray | None, used_quantities: np.ndarray | None = None,
                   inverted_axis: int = -1) -> np.ndarray | None:
    if array_4d is None:
        return None

    def _invert_sign(array: np.ndarray) -> np.ndarray:
        return -array

    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    new_array_4d = np.copy(array_4d)  # deepcopy here

    if used_quantities[2]:  # Bt in set
        indices = np.cumsum(used_quantities) - 1  # -1 to count from 0
        new_array_4d[..., indices[2]] = np.apply_along_axis(_invert_sign, axis=inverted_axis, arr=new_array_4d[..., indices[2]])

    return new_array_4d


def filter_empty_data(list_of_arrays: list[np.ndarray], used_quantities: np.ndarray | None = None
                      ) -> tuple[list[np.ndarray], np.ndarray]:
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]
    indices_to_keep = [i for i, array in enumerate(list_of_arrays) if used_quantities[i] and not is_empty(array)]
    filtered_array = [list_of_arrays[i] for i in indices_to_keep]
    used_quantities = np.array(len(used_quantities) * [False])
    used_quantities[indices_to_keep] = True

    return filtered_array, used_quantities


def convert_unit(array: np.ndarray | None,
                 initial_unit: Literal["kG", "G", "T", "mT"] = "kG",
                 final_unit: Literal["kG", "G", "T", "mT"] = "G",
                 used_quantities: np.ndarray | None = None) -> np.ndarray | None:
    if array is None or initial_unit == final_unit:
        return array

    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    # relative orders computed from Gauss
    orders = {"G": 0.,   # 1 G  = 10 ** 0 G
              "kG": 3.,  # 1 kG = 10 ** 3 G
              "T": 4.,   # 1 T  = 10 ** 5 G
              "mT": 1.}  # 1 mT = 10 ** 1 G

    factor = orders[initial_unit] - orders[final_unit]
    converson_factor = np.array([1., 10.**factor, 10.**factor, 10.**factor],
                                dtype=np.result_type(array))[used_indices(used_quantities)]

    if np.shape(array)[-1] != len(converson_factor):
        raise ValueError(f"Cannot convert the array. Incorrect dimensions. \n"
                         f"Check used_quantities: {used_quantities}")
    return array * converson_factor


def remove_constant_patches(x_data: np.ndarray) -> np.ndarray:
    if np.size(x_data) == 0:
        return x_data

    var = np.array([np.var(patch, axis=(0, 1)) for patch in x_data])

    return x_data[np.all(var > 0., axis=1)]


def split_data_to_patches(x_data: np.ndarray, patch_size: int | None = None,
                          crop_edges_to_patch: bool = True,
                          used_quantities: np.ndarray | None = None) -> np.ndarray:
    if patch_size is None: patch_size = conf_grid_setup["patch_size"]
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    if is_empty(x_data):
        return np.zeros((0, patch_size, patch_size, np.sum(used_quantities)))

    if crop_edges_to_patch:
        x_data = center_crop_to_patch_size(x_data, patch_size)

    nz, ny, nx, nq = np.shape(x_data)

    nx_full, ny_full = nx // patch_size, ny // patch_size
    if nz == 0 or nx_full == 0 or ny_full == 0:  # e.g. no validation data
        return np.zeros((0, patch_size, patch_size, nq), dtype=_wp)

    patches = np.zeros((nz, nx_full * ny_full, patch_size, patch_size, nq), dtype=_wp)

    i = 0
    for iy in range(ny_full):
        for ix in range(nx_full):
            patches[:, i, ...] = x_data[:, iy * patch_size: (iy + 1) * patch_size, ix * patch_size: (ix + 1) * patch_size, :]
            i += 1

    return remove_constant_patches(stack(patches, axis=0))


def print_header(used_quantities: np.ndarray | None = None) -> None:
    # Function to print header for an accuracy metric
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    used_names = add_unit_to_names(quantity_names_short)[used_indices(used_quantities)]
    print(f"{'':18} {'   '.join(f'{cls:8}' for cls in used_names)}")


def print_info(y_true: np.ndarray | None, y_pred: np.ndarray | None, which: str = "") -> np.ndarray | None:
    from modules.NN_losses_metrics_activations import my_rmse

    accuracy = my_rmse(all_to_one=False)(y_true, y_pred).numpy()
    if which:
        pref = f"{which.capitalize()} RMSE score:"
    else:
        pref = f"RMSE score:"
    print(f"{pref:22} {'     '.join(f'{acc:6.3f}' for acc in np.round(accuracy, 3))}")

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
    if order is None: order = ""

    names = np.array(all_quantity_names)

    if latex_output:
        units = np.array([r"$\left(\text{quiet-Sun norm.}\right)$", f"$\\left(\\text{{{order}G}}\\right)$",
                          f"$\\left(\\text{{{order}G}}\\right)$", f"$\\left(\\text{{{order}G}}\\right)$"])
    else:
        units = np.array(["(quiet-Sun norm.)", f"({order}G)", f"({order}G)", f"({order}G)"])

    names = np.array([f"{name} {unit}" for name, unit in zip(names, units)])

    if return_units:
        return names, units
    return names


def rot_coordinates_to_NW(longitude: np.ndarray, latitude: np.ndarray, array_to_flip: np.ndarray,
                          lon_index: int | None = None, lat_index: int | None = None) -> np.ndarray:

    if lon_index is None:
        lon_index = 1 if np.ndim(array_to_flip) > 2 else 0
    if lat_index is None:
        lat_index = 2 if np.ndim(array_to_flip) > 2 else 1

    nr, _ = np.shape(longitude)
    linear_lon = longitude[nr // 2, :]
    linear_lon = linear_lon[np.isfinite(linear_lon)]

    _, nc = np.shape(latitude)
    linear_lat = latitude[:, nc // 2]
    linear_lat = linear_lat[np.isfinite(linear_lat)]

    # axes along which to flip the map
    axes = []
    if linear_lon[0] > linear_lon[-1]:
        axes.append(lon_index)
    if linear_lat[0] > linear_lat[-1]:
        axes.append(lat_index)

    return np.flip(m=array_to_flip, axis=axes)


def gimme_bin_code_from_name(model_name: str) -> str:
    bare_name = split_path(model_name)[1]

    name_parts = np.array(bare_name.split(_sep_out))

    # bin_code is made of "len(conf_output_setup["used_quantities"])" decimals of "1" and "0"
    bin_code = [part for part in name_parts if (part.isdecimal()
                                                and len(part) == len(conf_output_setup["used_quantities"])
                                                and part.count("0") + part.count("1") == len(part)
                                                # and "1" in part  # at least one quantity must be present
                                                )]

    if len(bin_code) == 1:
        return str(bin_code[0])

    raise ValueError("Bin code is not unique")


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


def create_full_video_from_images(ar_number: int,
                                  output_filename: str = "video",
                                  output_format: Literal["avi", "mp4"] = "avi",
                                  fps: int = 1) -> None:
    list_of_figures = sorted(glob(path.join(_path_figures, "HMI_to_SOT", f"AR_{ar_number}", "*")))
    make_video_from_images(list_of_figures=list_of_figures, output_filename=output_filename,
                           output_format=output_format, fps=fps)


def create_full_video_from_fits(ar_number: int,
                                output_filename: str = "video",
                                output_format: Literal["avi", "mp4"] = "avi",
                                fps: int = 1) -> None:

    fits_all = sorted(glob(path.join(_result_dir, f"AR_{ar_number}_*")))

    # create the base video
    fits_name = fits_all[0]
    with fits.open(fits_name) as hdu:
        data = hdu[0].data

    filenames = make_video_from_arrays(data, output_filename=output_filename, output_format=output_format, fps=fps)

    # append other videos
    for fits_name in fits_all[1:]:
        with fits.open(fits_name) as hdu:
            data = hdu[0].data

        tmp_filenames = make_video_from_arrays(data, output_filename="tmp_video", output_format=output_format, fps=fps)

        # concatenate the video to the base one
        [concatenate_videos(list_of_video_names=[filenames[i], tmp_filenames[i]],
                            output_filename=filenames[i].split(".")[0],
                            output_format=output_format,
                            target_fps=fps) for i in range(len(filenames))]

        # remove temp videos
        [remove_if_exists(tmp_filename) for tmp_filename in tmp_filenames]


def disambigue_azimuth(azimuth: np.ndarray, disambig: np.ndarray, method: int = 1,
                       rotated_image: bool = False) -> np.ndarray:
    disambig = np.array(disambig, dtype=int)
    disambig_shape = np.shape(disambig)
    disambig = np.ravel(disambig)
    diasmbig_matrix = np.array([f"{value:03b}"[method] for value in disambig], dtype=float)
    diasmbig_matrix = np.reshape(diasmbig_matrix, disambig_shape)

    if rotated_image:
        # im_patch rotate the image 180 deg
        #  - add 180 deg to all pixels and add another 180 deg to those with disambig == 1
        #  - efficiently add 180 deg only to those with disambig == 0  =>  (1 - disambig) * 180
        return azimuth + (1. - diasmbig_matrix) * 180.
    else:
        return azimuth + diasmbig_matrix * 180.


def ar_info(ar_number: int) -> dict:
    excel = load_xlsx(path.join(_data_dir, "SKVRNY.xlsx"),
                      usecols=["ar_number", "t_start", "t_end", "t_ref", "location", "downloaded"],
                      dtype={"ar_number": int, "t_start": str, "t_end": str, "t_ref": str, "location": str, "downloaded": str})
    info = excel[ar_number == excel["ar_number"]]
    if np.size(info) > 0:
        info = info.to_dict("records")[0]

        info["t_start"] = datetime.strptime(info["t_start"], "%d.%m.%Y").strftime("%Y-%m-%d")
        info["t_end"] = datetime.strptime(info["t_end"], "%d.%m.%Y").strftime("%Y-%m-%d")
        if " " in info["t_ref"]:
            info["t_ref"] = info["t_ref"].split(" ")[0]

        if isinstance(info["location"], str):
            info["location"] = info["location"].replace(" ", "")
        info["downloaded"] = isinstance(info["downloaded"], str) and info["downloaded"].lower() == "yes"

        return info
    else:
        return {"ar_number": None, "t_start": None, "t_end": None, "t_ref": None, "location": None, "downloaded": None}


def str_coordinate_to_tuple(coordinate_string: str) -> list[float]:
    if "W" in coordinate_string:
        coordinate_string = coordinate_string.split("W")
        coordinate_string[1] = float(coordinate_string[1])
    else:
        coordinate_string = coordinate_string.split("E")
        coordinate_string[1] = -float(coordinate_string[1])
    if "N" in coordinate_string[0]:
        coordinate_string[0] = float(coordinate_string[0][1:])
    else:
        coordinate_string[0] = -float(coordinate_string[0][1:])

    return coordinate_string


def tuple_coordinate_to_str(coordinate_tuple: tuple[float]) -> str:
    if coordinate_tuple[0] >= 0.:
        coordinate_string = f"N{int(np.round(coordinate_tuple[0])):02d}"
    else:
        coordinate_string = f"S{int(np.round(coordinate_tuple[0])):02d}"

    if coordinate_tuple[1] >= 0.:
        coordinate_string = f"{coordinate_string}W{int(np.round(coordinate_tuple[1])):02d}"
    else:
        coordinate_string = f"{coordinate_string}E{int(np.round(-coordinate_tuple[1])):02d}"

    return coordinate_string


def filter_files(files: list[str]) -> list[str]:
    return sorted([file for file in files if path.isfile(file)
                   and path.isfile(file.replace("field", "inclination"))
                   and path.isfile(file.replace("field", "azimuth"))
                   and path.isfile(file.replace("field", "disambig"))
                   and path.isfile(file.replace("field", "1.continuum").replace("hmi.b", "hmi.ic"))])


def hmi_psf(target_shape: tuple[int, int] | int = 65, calc_new: bool = False) -> np.ndarray:
    hmi_psf_file = "/nfsscratch/david/NN/data/datasets/HMI_PSF.npz"

    if calc_new or not path.isfile(hmi_psf_file):
        file_orig = "/nfsscratch/david/NN/data/SDO_HMI_stat/20100501_190000/hmi.Ic_720s.20100501_190000_TAI.1.continuum.fits"
        file_dcon = "/nfsscratch/david/NN/data/SDO_HMI_stat/20100501_190000/hmi.Ic_720s_dconS.20100501_190000_TAI.1.continuum.fits"

        with fits.open(file_orig) as hdu:
            unsharp = hdu[1].data
            index_unsharp = hdu[1].header
        with fits.open(file_dcon) as hdu:
            sharp = hdu[1].data
            index_sharp = hdu[1].header

        # Observations are in N up, W right
        lon, lat = return_lonlat(index_unsharp)
        unsharp = rot_coordinates_to_NW(longitude=lon, latitude=lat, array_to_flip=np.expand_dims(unsharp, axis=0))[0]
        lon, lat = return_lonlat(index_sharp)
        sharp = rot_coordinates_to_NW(longitude=lon, latitude=lat, array_to_flip=np.expand_dims(sharp, axis=0))[0]

        N = 401
        [nr, nc] = np.shape(unsharp)
        cen_row, cen_col = nr // 2, nc // 2
        unsharp = unsharp[cen_row - N // 2:cen_row + N // 2 + 1, cen_col - N // 2:cen_col + N // 2 + 1]
        sharp = sharp[cen_row - N // 2:cen_row + N // 2 + 1, cen_col - N // 2:cen_col + N // 2 + 1]

        unsharp_fft = fft2(unsharp)
        shart_fft = fft2(sharp)

        psf = np.real(fftshift(ifft2(unsharp_fft / shart_fft)))
        psf /= np.sum(psf)

        check_dir(hmi_psf_file, is_file=True)
        with open(hmi_psf_file, "wb") as f:
            np.savez_compressed(f, HMI_PSF=psf)

    else:
        data = load_npz(hmi_psf_file)
        psf = data["HMI_PSF"]
        data.close()

    psf_target_shape = pad_zeros_or_crop(array=psf, target_shape=target_shape)

    return psf_target_shape / np.sum(psf_target_shape)


def hmi_noise(calc_new: bool = False) -> tuple[np.ndarray, ...]:
    hmi_noise_file = "/nfsscratch/david/NN/data/datasets/HMI_noise.npz"

    if calc_new or not path.isfile(hmi_noise_file):
        filename = "/nfsscratch/david/NN/data/SDO_HMI_stat/20100501_190000/hmi.B_720s.20100501_190000_TAI.ptr.sav"
        hmi_ptr = sio.readsav(filename)
        hmi_ptr, lonlat = hmi_ptr["bptr"], hmi_ptr["lonlat"]

        # Observations are in N up, W right
        hmi_ptr = rot_coordinates_to_NW(longitude=lonlat[0], latitude=lonlat[1], array_to_flip=hmi_ptr)

        noise_sample = hmi_ptr[0, 1486:2286, 1867:2446]
        noise_p = noise_sample[221:, :]

        mask = np.zeros_like(hmi_ptr[0], dtype=bool)
        mask[1707:2286, 1867:2446] = True
        filenames = [filename]
        masks = [mask]

        noise_sample = hmi_ptr[1, 1641:2276, 1896:2616]
        noise_t = noise_sample[:, 85:]

        mask = np.zeros_like(hmi_ptr[1], dtype=bool)
        mask[1641:2276, 1981:2616] = True
        filenames.append(filename)
        masks.append(mask)

        filename = "/nfsscratch/david/NN/data/SDO_HMI_stat/20100618_160006/hmi.b_720s.20100618_152400_TAI.ptr.sav"
        hmi_ptr = sio.readsav(filename)
        hmi_ptr, lonlat = hmi_ptr["bptr"], hmi_ptr["lonlat"]

        # Observations are in N up, W right
        hmi_ptr = rot_coordinates_to_NW(longitude=lonlat[0], latitude=lonlat[1], array_to_flip=hmi_ptr)

        noise_r = hmi_ptr[2, 163:244, 131:212]

        mask = np.zeros_like(hmi_ptr[2], dtype=bool)
        mask[163:244, 131:212] = True
        filenames.append(filename)
        masks.append(mask)

        # largest_square_mask = largest_square_below_threshold(np.abs(hmi_ptr[2]), 120., True)
        # _rows, _cols = np.where(largest_square_mask)
        # noise_sample_final = hmi_ptr[2, _rows.min():_rows.max() + 1, _cols.min():_cols.max() + 1]

        check_dir(hmi_noise_file, is_file=True)
        with open(hmi_noise_file, "wb") as f:
            np.savez_compressed(f, HMI_noise_p=noise_p, HMI_noise_t=noise_t, HMI_noise_r=noise_r,
                                masks=np.array(masks, dtype=object), filenames=filenames)

    else:
        data = load_npz(hmi_noise_file)
        noise_p, noise_t, noise_r = data["HMI_noise_p"], data["HMI_noise_t"], data["HMI_noise_r"]
        data.close()

    return noise_p, noise_t, noise_r


def hmi_psf_approx(kernel_size: int = 7, method: Literal["Couvidat", "Baso"] = "Baso",
                   lambda_obs_nn: float = 617.3, telescope_diameter_cm: float = 28.,
                   focal_length_mm: float = 4953., pixel_size_um: float = 12.,
                   gauss_lorentz_trade_off: float = 0.1, gauss_sigma: float = 1.8, lorentz_width: float = 3.,
                   lorentz_power: float = 3.) -> np.ndarray:

    # Create a grid of 2D coordinates
    x = np.arange(-(kernel_size // 2), (kernel_size + 1)//2)
    y = x
    X, Y = np.meshgrid(x, y)

    # Compute the distance from the origin to each point in the grid
    r = np.sqrt(X ** 2 + Y ** 2)

    # 10**4 to compensate units (-6-2+3+9)
    factor = np.pi * pixel_size_um * telescope_diameter_cm / (focal_length_mm * lambda_obs_nn) * 10**4
    r_prime = r * factor

    if method == "Baso":
        # Eq. (6) in https://www.aanda.org/articles/aa/pdf/2018/06/aa31344-17.pdf
        res = ((1. - gauss_lorentz_trade_off) * np.exp(-(r / gauss_sigma)**2)
               + gauss_lorentz_trade_off * (1 + (r_prime / lorentz_width)**lorentz_power)**(-1))

    else:
        # Eq (19) in https://link.springer.com/article/10.1007/s11207-016-0957-3
        res = (2. * j1(r_prime)) ** 2
        res = np.where(r_prime > 0., res / (r_prime**2), 1.)

    return res / np.sum(res)


def parse_datetime(date_str: str) -> datetime | None:
    """
    Try parsing the date string with multiple formats and return the parsed datetime object.
    Returns None if parsing fails.
    """

    # List of possible date formats
    formats = [
        "%Y-%m-%dT%H:%M:%S.%f",  # e.g., 2025-02-13T12:30:45.123456
        "%Y-%m-%dT%H:%M:%S",  # e.g., 2025-02-13T12:30:45
        "%Y-%m-%dT%H:%M",  # e.g., 2025-02-13T12:30
        "%Y-%m-%dT%H",  # e.g., 2025-02-13T12
        "%Y-%m-%d",  # e.g., 2025-02-13
        "%d/%m/%Y %H:%M:%S.%f",  # e.g., 13/02/2025 12:30:45.123456
        "%d/%m/%Y %H:%M:%S",  # e.g., 13/02/2025 12:30:45
        "%d/%m/%Y %H:%M",  # e.g., 13/02/2025 12:30
        "%d/%m/%Y %H",  # e.g., 13/02/2025 12
        "%d/%m/%Y",  # e.g., 13/02/2025
    ]

    for date_format in formats:
        try:
            return datetime.strptime(date_str, date_format)
        except ValueError:
            continue
    return None


def fill_header_for_wcs(header):
    """
    Fill in missing keywords and ensure the header is complete for WCS.
    """

    header.setdefault("RSUN_OBS", header.get("SOLAR_RA", 960.))  # In arcsec
    header.setdefault("DSUN_OBS", 1.0 * u.AU.to(u.m))  # Assume distance to Sun is 1 AU.

    header.setdefault("CRLN_OBS", 0.0)  # Default Carrington longitude
    header.setdefault("CRLT_OBS", header.get("B_ANGLE", 0.))  # Use B_ANGLE for CRLT_OBS.

    header.setdefault("CROTA2", -header.get("P_ANGLE", -0.))  # Translate P_ANGLE to CROTA2

    header.setdefault("CDELT1", header.get("XSCALE", None))  # Pixel scale in arcsec/pixel
    header.setdefault("CDELT2", header.get("YSCALE", None))  # Pixel scale in arcsec/pixel

    crota2 = np.deg2rad(header["CROTA2"])

    # Set reference point to [0, 0] arcsec by default
    header.setdefault("CRVAL1", 0.)
    header.setdefault("CRVAL2", 0.)

    # Convert center and scale information into the WCS-required format
    # http://jsoc.stanford.edu/doc/keywords/JSOC_Keywords_for_metadata.pdf

    crpix1, crpix2 = None, None
    # compute crpix1, crpix2
    if all(k not in header for k in ["CRPIX1", "CRPIX2"]) and all(k in header for k in ["NAXIS1", "NAXIS2", "XCEN", "YCEN"]):
        # Solved using determinants
        # y1 = a1 * crpix1 + b1 * crpix2  # XCEN = eq.
        # y2 = a2 * crpix1 + b2 * crpix2  # YCEN = eq.
        y1 = (header["XCEN"] - header["CRVAL1"]
              - header["CDELT1"] * np.cos(crota2) * (header["NAXIS1"] + 1.) / 2.
              + header["CDELT2"] * np.sin(crota2) * (header["NAXIS2"] + 1.) / 2.
              )
        y2 = (header["YCEN"] - header["CRVAL2"]
              - header["CDELT1"] * np.sin(crota2) * (header["NAXIS1"] + 1.) / 2.
              - header["CDELT2"] * np.cos(crota2) * (header["NAXIS2"] + 1.) / 2.
              )

        a1 = -header["CDELT1"] * np.cos(crota2)
        a2 = -header["CDELT1"] * np.sin(crota2)

        b1 = header["CDELT2"] * np.sin(crota2)
        b2 = -header["CDELT2"] * np.cos(crota2)

        det_main = a1 * b2 - a2 * b1
        if det_main != 0.:
            det_crpix1 = y1 * b2 - y2 * b1
            det_crpix2 = a1 * y2 - a2 * y1
            crpix1 = det_crpix1 / det_main
            crpix2 = det_crpix2 / det_main

    xcen, ycen = None, None
    # compute xcen, ycen
    if all(k not in header for k in ["XCEN", "YCEN"]) and all(k in header for k in ["NAXIS1", "NAXIS2", "CRPIX1", "CRPIX2"]):
        xcen = (header["CRVAL1"]
                + header["CDELT1"] * np.cos(crota2) * ((header["NAXIS1"] + 1.) / 2. - header["CRPIX1"])
                - header["CDELT2"] * np.sin(crota2) * ((header["NAXIS2"] + 1.) / 2. - header["CRPIX2"]))

        ycen = (header["CRVAL2"]
                + header["CDELT1"] * np.sin(crota2) * ((header["NAXIS1"] + 1.) / 2. - header["CRPIX1"])
                + header["CDELT2"] * np.cos(crota2) * ((header["NAXIS2"] + 1.) / 2. - header["CRPIX2"]))

    header.setdefault("CRPIX1", crpix1)
    header.setdefault("CRPIX2", crpix2)
    header.setdefault("XCEN", xcen)
    header.setdefault("YCEN", ycen)

    # Set default units and types
    header.setdefault("CUNIT1", "arcsec")
    header.setdefault("CUNIT2", "arcsec")
    header.setdefault("CTYPE1", "HPLN-TAN")
    header.setdefault("CTYPE2", "HPLT-TAN")

    # Ensure DATE-OBS or TSTART is available as obstime
    header.setdefault("DATE-OBS", header.get("TSTART", None))

    # Ensure T_OBS is available, using TSTART and TEND
    t_obs = None  # Default if TSTART and TEND aren't present or parsing fails
    if "TSTART" in header and "TEND" in header:
        # Convert to datetime objects
        tstart = try_parse_datetime(header["TSTART"])
        tend = try_parse_datetime(header["TEND"])

        if tstart and tend:
            # Compute T_OBS as the midpoint
            t_obs = tstart + (tend - tstart) / 2.

            # Convert back to string if needed
            t_obs = t_obs.strftime("%Y-%m-%dT%H:%M:%S.%f")

    header.setdefault("T_OBS", t_obs)

    return header


def read_cotemporal_fits(filename: str, check_uniqueness: bool = False) -> dict:
    fits_folder, fits_name = path.split(filename)
    timestamp = (".".join(fits_name.split(".")[1:3])).partition("_")[-1]
    filenames = glob(path.join(fits_folder, f"*{timestamp}*"))

    # Keywords to look for
    keywords = [".continuum.fits", ".field.fits", ".inclination.fits", ".azimuth.fits", ".disambig.fits"]
    fits_names = ["ic", "b", "inc", "azi", "disamb"]

    if check_uniqueness:
        for keyword in keywords:
            if sum(keyword in _filename for _filename in filenames) > 1:
                raise ValueError(f'Fits names are not unique: Multiple files contain the keyword "{keyword}".')

    # Create the dictionary
    result_dict = {}
    for index, keyword in enumerate(keywords):
        # Find the first string containing the keyword
        matching_strings = [s for s in filenames if keyword in s]
        result_dict[f"fits_{fits_names[index]}"] = matching_strings[0] if matching_strings else None

    return result_dict


def return_lonlat(header) -> tuple[np.ndarray, np.ndarray]:
    # Fill the header with necessary keywords for WCS
    header = fill_header_for_wcs(header)

    # Create WCS object directly from the complete header
    wcs = WCS(header)

    # Get observer location information from the header
    dsun_obs = header["DSUN_OBS"] * u.m
    crln_obs = header["CRLN_OBS"] * u.deg
    crlt_obs = header["CRLT_OBS"] * u.deg
    obstime = header["T_OBS"]

    # Assuming nx and ny are the dimensions of your image:
    nx, ny = header["NAXIS1"], header["NAXIS2"]
    y, x = np.indices((ny, nx))

    # Convert pixel coordinates to world coordinates (Helioprojective)
    hpc_coords = wcs.pixel_to_world(x, y)

    # Define observer's location in Heliographic Stonyhurst frame
    observer = SkyCoord(lon=crln_obs, lat=crlt_obs, radius=dsun_obs,
                        frame=HeliographicStonyhurst, obstime=obstime)

    # Attach observer information to helioprojective coordinates
    hpc_coords = SkyCoord(hpc_coords.Tx, hpc_coords.Ty,
                          frame=Helioprojective(observer=observer, obstime=obstime))

    # Transform to Heliographic Stonyhurst coordinates
    heliographic_coords = hpc_coords.transform_to(HeliographicStonyhurst(obstime=obstime))

    # Extract longitude and latitude
    lon = heliographic_coords.lon.to(u.deg).value
    lat = heliographic_coords.lat.to(u.deg).value

    # Apply the correction for the Carrington longitude; zero longitude in the central meridian
    lon = (lon + 360. - crln_obs.to(u.deg).value) % 360.
    lon[lon > 180.] -= 360.  # have if from -180 to +180

    return lon, lat


def data_b2ptr(index, bvec: np.ndarray, disambig: np.ndarray | None = None,
               return_coordinates: bool = False) -> np.ndarray | tuple[np.ndarray, ...]:
    # Example usage:
    """
    index = fits.open("/nfsscratch/david/NN/data/SDO_HMI/20100501_N23E55/hmi.b_720s.20100501_000000_TAI.field.fits")[1].header
    bvec = np.array([fits.open("/nfsscratch/david/NN/data/SDO_HMI/20100501_N23E55/hmi.b_720s.20100501_000000_TAI.field.fits")[1].data,
                     fits.open("/nfsscratch/david/NN/data/SDO_HMI/20100501_N23E55/hmi.b_720s.20100501_000000_TAI.inclination.fits")[1].data,
                     fits.open("/nfsscratch/david/NN/data/SDO_HMI/20100501_N23E55/hmi.b_720s.20100501_000000_TAI.azimuth.fits")[1].data])
    disambig = fits.open("/nfsscratch/david/NN/data/SDO_HMI/20100501_N23E55/hmi.b_720s.20100501_000000_TAI.disambig.fits")[1].data
    """
    # bptr, lonlat = data_b2ptr(index, bvec, disambig)

    # Fill the header with necessary keywords for WCS
    index = fill_header_for_wcs(header=index)

    # Check dimensions
    nq, ny, nx = np.shape(bvec)
    if nq != 3 or nx != index["NAXIS1"] or ny != index["NAXIS2"]:
        raise ValueError("Dimension of bvec incorrect")

    if disambig is not None:
        # disambiguate azimuth; add 180 to azimuth for rotated images came from im_patch
        bvec[2, :, :] = disambigue_azimuth(bvec[2, :, :], disambig, method=1,
                                           rotated_image="history" in index and "rotated" in str(index["history"]))

    # Convert bvec to B_xi, B_eta, B_zeta
    field = bvec[0, :, :]
    gamma = np.deg2rad(bvec[1, :, :])
    psi = np.deg2rad(bvec[2, :, :])

    b_xi = -field * np.sin(gamma) * np.sin(psi)
    b_eta = field * np.sin(gamma) * np.cos(psi)
    b_zeta = field * np.cos(gamma)

    lon, lat = return_lonlat(header=index)

    # Get matrix to convert
    b = np.deg2rad(index["CRLT_OBS"])  # b-angle, disk center latitude
    p = np.deg2rad(-index["CROTA2"])  # p-angle, negative of CROTA2

    phi = np.deg2rad(lon)
    lambda_ = np.deg2rad(lat)

    sinb, cosb = np.sin(b), np.cos(b)
    sinp, cosp = np.sin(p), np.cos(p)
    sinphi, cosphi = np.sin(phi), np.cos(phi)
    sinlam, coslam = np.sin(lambda_), np.cos(lambda_)

    k11 = coslam * (sinb * sinp * cosphi + cosp * sinphi) - sinlam * cosb * sinp
    k12 = -coslam * (sinb * cosp * cosphi - sinp * sinphi) + sinlam * cosb * cosp
    k13 = coslam * cosb * cosphi + sinlam * sinb
    k21 = sinlam * (sinb * sinp * cosphi + cosp * sinphi) + coslam * cosb * sinp
    k22 = -sinlam * (sinb * cosp * cosphi - sinp * sinphi) - coslam * cosb * cosp
    k23 = sinlam * cosb * cosphi - coslam * sinb
    k31 = -sinb * sinp * sinphi + cosp * cosphi
    k32 = sinb * cosp * sinphi + sinp * cosphi
    k33 = -cosb * sinphi

    # Output
    bptr = np.zeros_like(bvec)
    bptr[0, :, :] = k31 * b_xi + k32 * b_eta + k33 * b_zeta
    bptr[1, :, :] = k21 * b_xi + k22 * b_eta + k23 * b_zeta
    bptr[2, :, :] = k11 * b_xi + k12 * b_eta + k13 * b_zeta

    if return_coordinates:
        return bptr, lon, lat
    return bptr


def remove_limb_darkening_approx(image: np.ndarray,
                                 degree: int = 4,
                                 mask: np.ndarray | None = None,
                                 threshold: float = 0.3,
                                 normalised_thresh: bool = True) -> np.ndarray:
    """
    Removes approximate limb darkening from an input 2D image using a polynomial surface fit.

    Parameters:
        image (np.ndarray): A 2D array representing the image. Must not contain all NaNs.
        degree (int): A degree of the fitted polynomial surface.
        mask (np.ndarray | None): A 2D mask to filter the image. If not set, it is computed
                                  from threshold and normalised_thresh.
        threshold (float): A value used to mask out low-intensity pixels for fitting.
                           Only pixels with values above this threshold are used in the fit.
        normalised_thresh (bool): If True, normalise the image by dividing by its maximum
                                  (ignoring NaNs) before applying the threshold.

    Returns:
        np.ndarray: The input image divided by the fitted quadratic surface.

    Notes:
        - The input image is not modified.
        - The normalisation step does not alter the input image but works on a temporary copy if needed.
        - The limb darkening approximation uses a design matrix for a polynomial surface:
          z(x, y) = c0 + c1*y + c2*x + c3*y^2 + c4*x*y + c5*x^2 + ...

    """
    nx, ny = image.shape  # Dimensions of the input image

    # Create standardised grid indices for x and y
    x = np.linspace(-1., 1., nx, dtype=np.float32)  # Standardised x-coordinates
    y = np.linspace(-1., 1., ny, dtype=np.float32)  # Standardised y-coordinates
    x, y = np.meshgrid(x, y, indexing="ij")  # Create meshgrid
    x, y = x.ravel(), y.ravel()  # Flatten x and y into 1D arrays

    # Design matrix for the quadratic surface.
    # Compute `V` row-by-row to avoid peak memory usage.
    V = np.empty((len(x), ((degree + 1) * (2 + degree)) // 2), dtype=np.float32)

    index = 0
    for j in range(degree + 1):  # Iterate over the total degree
        for k in range(j + 1):  # Iterate over the individual terms x^k * y^(j-k)
            V[:, index] = x**k * y**(j-k)
            index += 1
    V = V[:, :index]  # just in case (e.g. needed if only even degrees are included (radial symmetry forcing)

    # Collect garbage (x and y are small but still of nonzero sizes)
    del x, y
    gc.collect()

    max_val = np.nanmax(image)  # Compute maximum value ignoring NaNs

    if mask is None:
        # Normalise the image if requested, working on a copy
        if normalised_thresh:
            # Create a mask to select pixels above the threshold
            mask = image > threshold * max_val if normalised_thresh else image > threshold

    mask[~np.isfinite(image)] = False
    mask = mask.ravel()
    image = image / max_val  # Operates on a temporary copy

    # Fit the quadratic surface to the selected pixels using least squares
    if V[mask, :].size < 2000 * 2000 * 6:
        coefficients, _, _, _ = np.linalg.lstsq(V[mask, :], image.ravel()[mask], rcond=None)
    else:
        coefficients = lsmr(V[mask, :], image.ravel()[mask])[0]

    # Compute the quadratic surface and divide the image by it
    return image / (V @ coefficients).reshape(nx, ny)


def compute_mu(header) -> np.ndarray:
    # Fill the header with necessary keywords for WCS
    header = fill_header_for_wcs(header)

    # Create WCS object directly from the complete header
    wcs = WCS(header)

    # Get observer location information from the header
    dsun_obs = header["DSUN_OBS"] * u.m
    crln_obs = header["CRLN_OBS"] * u.deg
    crlt_obs = header["CRLT_OBS"] * u.deg
    obstime = header["DATE-OBS"]

    # Assuming nx and ny are the dimensions of your image:
    nx, ny = header["NAXIS1"], header["NAXIS2"]
    y, x = np.indices((ny, nx))

    # Convert pixel coordinates to world coordinates (Helioprojective)
    hpc_coords = wcs.pixel_to_world(x, y)

    # Define observer's location in Heliographic Stonyhurst frame
    observer = SkyCoord(lon=crln_obs, lat=crlt_obs, radius=dsun_obs,
                        frame=HeliographicStonyhurst, obstime=obstime)

    # Attach observer information to helioprojective coordinates
    hpc_coords = SkyCoord(hpc_coords.Tx, hpc_coords.Ty,
                          frame=Helioprojective(observer=observer, obstime=obstime))

    l1 = np.array(hpc_coords.Tx) / header["RSUN_OBS"]
    l2 = np.array(hpc_coords.Ty) / header["RSUN_OBS"]

    """
    # Convert longitude and latitude from degrees to radians
    B0_deg = header["CRLT_OBS"]  # HERE IT GIVES BETTER RESULTS WITH MINUS SIGN. WHY?
    longitude_rad = np.deg2rad(longitude_deg)
    latitude_rad = np.deg2rad(latitude_deg)
    B0_rad = np.deg2rad(B0_deg)

    # Compute mu = cos(theta)
    mu = np.cos(latitude_rad) * np.cos(longitude_rad) * np.cos(B0_rad) - np.sin(latitude_rad) * np.sin(B0_rad)
    mu = np.where(mu > 0., mu, np.nan)
    """

    # r2 = l1 ** 2 - l2 ** 2
    # return np.where(r2 <= 1., np.sqrt(1. - r2), np.nan)
    return np.sqrt(1. - l1**2 - l2**2)  # this is faster


def remove_limb_darkening_exact(image: np.ndarray, header) -> np.ndarray:
    """
    Corrects a solar image for limb darkening using an empirical polynomial model.

    Parameters:
        image (2D array): Observed solar image (e.g., intensity values).
        header: image header file

    Returns:
        corrected_image (2D array): Solar image corrected for limb darkening.
    """

    # Define limb-darkening coefficients
    ld_coefs = [0.32519, 1.26432, -1.44591, 1.55723, -0.87415, 0.173333]
    mu = compute_mu(header)

    # Compute the limb-darkening model
    limb_darkening = sum(c_n * mu ** n for n, c_n in enumerate(ld_coefs))
    limb_darkening = np.clip(limb_darkening, a_min=_num_eps, a_max=1.)

    # Correct the observed image
    image = image / limb_darkening.astype(np.result_type(image))

    return image


def calculate_contrast(image: np.ndarray) -> float:
    """Calculate contrast as standard deviation divided by mean."""
    mean_value, std_dev = return_mean_std(image)

    return std_dev / mean_value if mean_value != 0. else 0.


def calculate_laplacian_variance(image: np.ndarray) -> float:
    """Calculate the variance of the Laplacian of the image."""
    laplacian = filters.laplace(image)
    return np.var(laplacian)


def calculate_gradient_magnitude(image: np.ndarray) -> float:
    """Calculate the mean gradient magnitude using Sobel filters."""
    grad_x = filters.sobel_h(image)
    grad_y = filters.sobel_v(image)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return np.mean(gradient_magnitude)


def calculate_fourier_sharpness(image: np.ndarray) -> float:
    """Calculate the proportion of high-frequency components in the Fourier transform."""
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)

    # Define high frequencies as those above a threshold (e.g., 10% of max frequency)
    threshold = 0.1 * np.max(magnitude_spectrum)
    high_freq_energy = np.sum(magnitude_spectrum[magnitude_spectrum > threshold])
    total_energy = np.sum(magnitude_spectrum)

    return high_freq_energy / total_energy if total_energy != 0 else 0


def calculate_edge_density(image: np.ndarray) -> float:
    """Calculate the edge density using the Canny edge detector."""
    edges = feature.canny(image, sigma=1.0)
    return np.sum(edges) / np.size(image)


def calculate_sharpness_matrics(image: np.ndarray) -> dict:
    # Load images (convert to grayscale for processing)
    if np.ndim(image) == 3:
        image = color.rgb2gray(image)

    metrics = {
        "Contrast": calculate_contrast,
        "Laplacian Variance": calculate_laplacian_variance,
        "Gradient Magnitude": calculate_gradient_magnitude,
        "Fourier Sharpness": calculate_fourier_sharpness,
        "Edge Density": calculate_edge_density,
    }

    for metric_name, metric_func in metrics.items():
        result = metric_func(image)
        # print(f"  {metric_name}: {result:.4f}")
        metrics[metric_name] = result
    # print()

    return metrics
