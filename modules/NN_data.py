from typing import Literal
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.special import j1

from modules.utilities import is_empty, stack
from modules.utilities_data import if_no_test_data, load_npz, center_crop_to_patch_size

from modules._constants import _wp, _rnd_seed, _quiet, _observations_name, _label_name

# defaults only
from modules.NN_config import conf_output_setup, conf_filtering_setup, conf_data_split_setup, conf_grid_setup, p


def load_prepared_data(filename_data: str,
                       used_quantities: np.ndarray | None = None,
                       use_simulated_hmi: bool = False,
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
               rnd_seed: int | None = _rnd_seed,
               out_type: type = _wp) -> (tuple[np.ndarray, ...] |
                                         tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray,
                                         pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]):

    if val_portion is None: val_portion = conf_data_split_setup["val_portion"]
    if test_portion is None: test_portion = conf_data_split_setup["test_portion"]

    ind_train = np.arange(len(x_data), dtype=int)
    ind_val = np.array([], dtype=int)
    ind_test = np.array([], dtype=int)

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
               used_quantities: np.ndarray | None = None,
               return_deleted: bool = False) -> tuple[np.ndarray, ...]:
    if filtering_setup is None: filtering_setup = conf_filtering_setup
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    inds_to_keep = index_clean_data(x_data, filtering_setup, used_quantities)

    if not return_deleted:
        return x_data[inds_to_keep], y_data[inds_to_keep]
    else:
        inds_not_to_keep = np.setdiff1d(np.arange(len(x_data)), inds_to_keep)
        return x_data[inds_to_keep], y_data[inds_to_keep], x_data[inds_not_to_keep], y_data[inds_not_to_keep]


def index_clean_data(x_data: np.ndarray, filtering_setup: dict | None = None,
                     used_quantities: np.ndarray | None = None) -> np.ndarray:
    if filtering_setup is None: filtering_setup = conf_filtering_setup
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    if not filtering_setup:  # return all indices
        return np.arange(len(x_data), dtype=int)

    quantity_indices = np.cumsum(used_quantities) - 1  # to index from 0
    if used_quantities[3]:  # Br
        indices = np.array(np.unique(np.where(np.abs(x_data[..., quantity_indices[3]]) >= filtering_setup["Br"])[0]), dtype=int)
    elif used_quantities[1]:  # Bp
        indices = np.array(np.unique(np.where(np.abs(x_data[..., quantity_indices[1]]) >= filtering_setup["Bp"])[0]), dtype=int)
    elif used_quantities[2]:  # Bt
        indices = np.array(np.unique(np.where(np.abs(x_data[..., quantity_indices[2]]) >= filtering_setup["Bt"])[0]), dtype=int)
    else:  # I
        indices = np.array(np.unique(np.where(np.abs(x_data[..., quantity_indices[0]]) >= filtering_setup["I"])[0]), dtype=int)

    # return filtered indices
    return indices


def remove_constant_patches(x_data: np.ndarray) -> np.ndarray:
    if np.size(x_data) == 0:
        return x_data

    var = np.array([np.var(patch, axis=(0, 1)) for patch in x_data])

    return x_data[np.all(var > 0., axis=1)]


def split_data_to_patches(x_data: np.ndarray, patch_size: int | None = None,
                          crop_edges_to_patch: bool = True) -> np.ndarray:
    if patch_size is None: patch_size = conf_grid_setup["patch_size"]

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


def hmi_psf(kernel_size: int = 7, method: Literal["Couvidat", "Baso"] = "Baso",
            lambda_obs_nn: float = 617.3, telescope_diameter_cm: float = 14.,
            focal_length_mm: float = 4953., pixel_size_um: float = 12.,
            gauss_lorentz_trade_off: float = 0.3, gauss_sigma: float = 2.5, lorentz_width: float = 3.4,
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
