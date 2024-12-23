from modules.utilities import is_empty
from modules.utilities_data import if_no_test_data, load_npz, convert_unit

from modules._constants import _wp, _rnd_seed, _quiet, _observations_name, _label_name, _b_unit

# defaults only
from modules.NN_config import conf_output_setup, conf_filtering_setup, conf_data_split_setup, conf_grid_setup, p

import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from skimage import measure
from skimage.draw import polygon


def load_prepared_data(filename_data: str,
                       used_quantities: np.ndarray | None = None,
                       use_simulated_hmi: bool = False,
                       convert_b: bool = False,
                       subfolder_data: str = "",
                       out_type: type = _wp) -> tuple[np.ndarray, ...]:
    if not _quiet:
        print("\nLoading prepared data")

    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    data = load_npz(filename_data, subfolder=subfolder_data)
    y_data = np.array(data[_label_name][..., used_quantities], dtype=out_type)

    if not use_simulated_hmi:
        x_data = np.array(data[_observations_name][..., used_quantities], dtype=out_type)
    else:  # blurred Hinode observations (matching algorithm is not ideal and structures evolve in time)
        x_data = np.array(data[f"{_observations_name}_simulated"][..., used_quantities], dtype=out_type)

    if convert_b and "units" in data.files and len(data["units"]) > 1:
        b_unit = data["units"][1]
        x_data = convert_unit(x_data, initial_unit=b_unit, final_unit=_b_unit, used_quantities=used_quantities)
        y_data = convert_unit(y_data, initial_unit=b_unit, final_unit=_b_unit, used_quantities=used_quantities)

    metadata = data["patch_identification"] if "patch_identification" in data.files else None

    return x_data, y_data, metadata


def load_data(filename_data: str,
              used_quantities: np.ndarray | None = None,
              convert_b: bool = False,
              subfolder_data: str = "",
              out_type: type = _wp) -> tuple[np.ndarray, ...]:
    if not _quiet:
        print("\nLoading data")

    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    data = load_npz(filename_data, subfolder=subfolder_data)

    x_data = np.array(data[_observations_name][..., used_quantities], dtype=out_type)
    y_data = np.array(data[_label_name][..., used_quantities], dtype=out_type)

    if convert_b and "units" in data.files and len(data["units"]) > 1:
        b_unit = data["units"][1]
        x_data = convert_unit(x_data, initial_unit=b_unit, final_unit=_b_unit, used_quantities=used_quantities)
        y_data = convert_unit(y_data, initial_unit=b_unit, final_unit=_b_unit, used_quantities=used_quantities)

    return x_data, y_data


def data_generator(filename: str, batch_size: int | None = None, used_quantities: np.ndarray | None = None,
                   subfolder_data: str = ""):
    if batch_size is None: batch_size = p["batch_size"]
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]
    # This is a data generator for the neural network. The data are already prepared in patches and train-val parts.

    # Select training data
    data = load_npz(filename, subfolder=subfolder_data)
    batches_per_epoch = data["n_data"] // batch_size

    def generator():
        while 1:
            for i in range(batches_per_epoch):
                inputs = data[_observations_name][i * batch_size:(i + 1) * batch_size, :, :, used_quantities]
                labels = data[_label_name][i * batch_size:(i + 1) * batch_size, :, :, used_quantities]

                yield inputs, labels

    return generator


def split_data(x_data: np.ndarray,
               y_data: np.ndarray,
               metadata: pd.DataFrame | np.ndarray | None = None,
               val_portion: float | None = None,
               test_portion: float | None = None,
               use_random: bool | None = None,
               rnd_seed: int | None = _rnd_seed,
               out_type: type = _wp) -> (tuple[np.ndarray, ...] |
                                         tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray,
                                         pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]):

    if val_portion is None: val_portion = conf_data_split_setup["val_portion"]
    if test_portion is None: test_portion = conf_data_split_setup["test_portion"]
    if use_random is None: use_random = conf_data_split_setup["use_random"]

    ind_train = np.arange(len(x_data), dtype=int)
    ind_val = np.array([], dtype=int)
    ind_test = np.array([], dtype=int)

    if use_random:
        if val_portion > 0.:
            if not _quiet:
                print("Creating validation data")
                ind_train, ind_val, _, _ = train_test_split(ind_train, y_data,
                                                            test_size=val_portion,
                                                            stratify=None,
                                                            random_state=rnd_seed)
        if test_portion > 0.:
            if not _quiet:
                print("Creating test data")
            ind_train, ind_test, _, _ = train_test_split(ind_train, y_data[ind_train],
                                                         test_size=test_portion / (1. - val_portion),
                                                         stratify=None,
                                                         random_state=rnd_seed)
    else:  # not random but chronological
        print("Train-Validation-Test split chronologically...")
        test_len = int(np.round(len(ind_train) * test_portion))
        val_len = int(np.round(len(ind_train) * val_portion))
        ind_train, ind_val, ind_test = ind_train[:-val_len - test_len], ind_train[-val_len - test_len:-test_len], ind_train[-test_len:]

    x_train, y_train = x_data[ind_train], y_data[ind_train]
    x_val, y_val = x_data[ind_val], y_data[ind_val]
    x_test, y_test = x_data[ind_test], y_data[ind_test]

    # convert data to working precision
    x_train, y_train = np.array(x_train, dtype=out_type), np.array(y_train, dtype=out_type)
    x_val, y_val = np.array(x_val, dtype=out_type), np.array(y_val, dtype=out_type)
    x_test, y_test = np.array(x_test, dtype=out_type), np.array(y_test, dtype=out_type)

    # if test_portion == 0:
    if is_empty(x_test):
        x_test, y_test = if_no_test_data(x_train, y_train, x_val, y_val)

    if metadata is not None:
        if isinstance(metadata, pd.DataFrame):
            meta_train, meta_val, meta_test = metadata.iloc[ind_train], metadata.iloc[ind_val], metadata.iloc[ind_test]
        else:
            meta_train, meta_val, meta_test = metadata[ind_train], metadata[ind_val], metadata[ind_test]

        # if test_portion == 0:
        if is_empty(meta_test):
            _, meta_test = if_no_test_data(None, meta_train, None, meta_val)

        return x_train, y_train, meta_train, x_val, y_val, meta_val, x_test, y_test, meta_test

    return x_train, y_train, x_val, y_val, x_test, y_test


def clean_data(x_data: np.ndarray, y_data: np.ndarray, filtering_setup: dict | None = None,
               metadata: np.ndarray | None = None,
               used_quantities: np.ndarray | None = None,
               return_deleted: bool = False) -> tuple[np.ndarray, ...]:
    if filtering_setup is None: filtering_setup = conf_filtering_setup
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    # filtering based on HMI or SOT-SP data?
    inds_to_keep = index_clean_data(base_data=x_data if "HMI" in filtering_setup["base_data"].upper() else y_data,
                                    filtering_setup=filtering_setup, used_quantities=used_quantities)

    if not return_deleted:
        if metadata is None:
            return x_data[inds_to_keep], y_data[inds_to_keep]
        else:
            return x_data[inds_to_keep], y_data[inds_to_keep], metadata[inds_to_keep]
    else:
        inds_not_to_keep = np.setdiff1d(np.arange(len(x_data)), inds_to_keep)
        if metadata is None:
            return x_data[inds_to_keep], y_data[inds_to_keep], x_data[inds_not_to_keep], y_data[inds_not_to_keep]
        else:
            return x_data[inds_to_keep], y_data[inds_to_keep], metadata[inds_to_keep], x_data[inds_not_to_keep], y_data[inds_not_to_keep], metadata[inds_not_to_keep]


def index_clean_data(base_data: np.ndarray, filtering_setup: dict | None = None,
                     used_quantities: np.ndarray | None = None) -> np.ndarray:
    if filtering_setup is None: filtering_setup = conf_filtering_setup
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    if not filtering_setup:  # return all indices
        return np.arange(len(base_data), dtype=int)

    quantity_indices = np.cumsum(used_quantities) - 1  # to index from 0
    if used_quantities[3]:  # Br
        indices = np.array(np.unique(np.where(np.abs(base_data[..., quantity_indices[3]]) >= filtering_setup["Br"])[0]), dtype=int)
    elif used_quantities[1]:  # Bp
        indices = np.array(np.unique(np.where(np.abs(base_data[..., quantity_indices[1]]) >= filtering_setup["Bp"])[0]), dtype=int)
    elif used_quantities[2]:  # Bt
        indices = np.array(np.unique(np.where(np.abs(base_data[..., quantity_indices[2]]) >= filtering_setup["Bt"])[0]), dtype=int)
    else:  # I
        indices = np.array(np.unique(np.where(np.abs(base_data[..., quantity_indices[0]]) <= filtering_setup["I"])[0]), dtype=int)

    # return filtered indices
    return indices


def crop_image_on_contour(image: np.ndarray, level: float,
                          patch_size: int | None = None):
    if patch_size is None: patch_size = conf_grid_setup["patch_size"]

    contours = measure.find_contours(image, level=level)
    nrows, ncols = np.shape(image)
    y_grid, x_grid = np.mgrid[0:nrows, 0:ncols]

    contour_centers = np.zeros((len(contours), 2))
    contour_areas = np.zeros((len(contours)))

    for i, contour in enumerate(contours):
        mask = np.zeros_like(image, dtype=bool)

        # Fill the mask with the contour polygon
        rr, cc = polygon(contour[:, 0], contour[:, 1], np.shape(mask))
        mask[rr, cc] = True

        contour_centers[i, :] = np.mean(y_grid[mask]), np.mean(x_grid[mask])
        contour_areas[i] = np.sum(mask)

    # start with the biggest area and check if some others are too close to it

    while 1:
        biggest_centre = contour_centers[np.argmax(contour_areas)]
        contour_areas[np.argmax(contour_areas)] = -1.

        dists = np.array([math.dist(biggest_centre, center) for center in contour_centers])

        removed_indices = np.logical_and(dists < patch_size, contour_areas >= 0.)
        contour_areas[removed_indices] = 0.

        if not np.any(contour_areas > 0.):
            break

    selected_contours = contour_areas == -1.

    contour_centers = contour_centers[selected_contours]

    for contour_center in contour_centers:
        contour_center = np.array(np.round(contour_center, decimals=0), dtype=int)
        contour_center -= patch_size//2
        rows, cols = np.mgrid[contour_center[0]:contour_center[0] + patch_size,
                              contour_center[1]:contour_center[1] + patch_size]

        yield image[rows, cols]


def load_data_hmi_sot_like(filename_data: str,
                           return_meta: bool = False,
                           output_setup: dict | None = None,
                           grid_setup: dict | None = None,
                           filtering_setup: dict | None = None,
                           subfolder_data: str = "") -> tuple[np.ndarray, ...]:
    # This function loads data from a dataset
    if output_setup is None: output_setup = conf_output_setup
    if grid_setup is None: grid_setup = conf_grid_setup
    if filtering_setup is None: filtering_setup = conf_filtering_setup

    if not _quiet:
        print("\nLoading train file")

    # Select training data
    x_train = np.array(load_npz(filename_data, subfolder=subfolder_data)[_observations_name], dtype=_wp)

    # clean data
    # inds = index_clean_data(x_train, filtering_setup=filtering_setup)
    # x_train = x_train[inds]

    # remove unwanted quantities
    x_train = x_train[..., output_setup["used_quantities"]]

    # NO TARGET DATA, CREATING IT HERE
    y_train = target_data(x_train)

    # to mimic lower resolution
    sigma = 1.
    x_train = blur_data(x_train, sigma=sigma)

    # convert data to working precision
    x_train, y_train = np.array(x_train, dtype=_wp), np.array(y_train, dtype=_wp)

    return x_train, y_train


def target_data(x_data: np.ndarray) -> np.ndarray:
    nz, ny, nx, nq = np.shape(x_data)
    y_data = np.zeros((nz, 2 * ny, 2 * nx, nq), dtype=_wp)

    y, x = np.linspace(0, 100, ny), np.linspace(0, 100, nx)
    yv, xv = np.linspace(0, 100, 2 * ny), np.linspace(0, 100, 2 * nx)

    xv, yv = np.meshgrid(xv, yv)

    for i in range(nq):
        for j in range(nz):
            y_data[j, :, :, i] = RegularGridInterpolator(points=(y, x), values=x_data[j, :, :, i], method="linear")((yv, xv))

    return y_data


def blur_data(x_data: np.ndarray, sigma: float = 1.) -> np.ndarray:
    blurred_data = gaussian_filter(x_data, sigma=sigma, axes=(1, 2))
    correction = gaussian_filter(np.ones(np.shape(x_data)), sigma=sigma, axes=(1, 2))
    return blurred_data / correction
