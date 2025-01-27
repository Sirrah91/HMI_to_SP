from modules.NN_config import conf_grid_setup
from modules.utilities_data import (save_data, load_npz, disambigue_azimuth, hmi_psf, data_b2ptr,
                                    remove_limb_darkening_approx, normalise_intensity,
                                    read_cotemporal_fits, convert_unit, hmi_noise, split_data_to_patches)
from modules.utilities import (rmse, remove_outliers_sliding_window, check_dir, stack, interpolate_mask, remove_nan,
                               apply_psf, is_empty, pad_zeros_or_crop, return_mean_std, plot_me, create_circular_mask,
                               filter_fft_amplitude)
from modules._constants import (_path_sp, _path_hmi, _data_dir,  _rnd_seed, _path_sp_hmi, _observations_name,
                                _label_name, _path_data, _path_sp_hmilike, _num_eps, _b_unit, _wp)

import os
from os import path
from glob import glob
import numpy as np
import scipy.io as sio
from copy import deepcopy
from scipy.fft import fft2, ifft2, ifftshift, fftshift, fftfreq
from scipy.interpolate import RegularGridInterpolator
import traceback
import concurrent.futures
import time
from datetime import datetime, timedelta
import warnings
import drms
from tqdm import tqdm
from typing import Literal

from astropy.io import fits
from sunpy.time import TimeRange

from skimage.measure import ransac
from skimage.feature import ORB, match_descriptors
from skimage.exposure import match_histograms
from skimage.transform import (resize, SimilarityTransform, EuclideanTransform, AffineTransform, ProjectiveTransform,
                               warp)
from skimage.registration import optical_flow_tvl1

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

hmi_dy_median, hmi_dx_median = 0.504310727, 0.504310727
sp_dy_median, sp_dx_median = 0.319978, 0.29714


def jsoc_query_from_sp_name(SP_filename: str, quantity: str, data_type: Literal["", "_dcon", "_dconS"] = "",
                            integration_time: int | None = None) -> None:
    warnings.filterwarnings("ignore")

    # E-mail registered in JSOC
    email = "d.korda@seznam.cz"

    margin_box = 0.  # margin of box width/height in "boxunits" (0. to keep scale)
    # margin of timespan in minutes
    margin_time = 10

    # reference SP level 2 Hinode observation
    hdu = fits.open(path.join(_path_sp, SP_filename))

    obs_start = hdu[0].header["TSTART"]
    obs_start = datetime.strptime(obs_start, "%Y-%m-%dT%H:%M:%S.%f")
    obs_start -= timedelta(minutes=margin_time)

    obs_end = hdu[0].header["TEND"]
    obs_end = datetime.strptime(obs_end, '%Y-%m-%dT%H:%M:%S.%f')
    obs_end += timedelta(minutes=margin_time)

    obs_length = obs_end - obs_start
    # cannot be (obs_start + obs_end) / 2 because datetime does not support "+" but timedelta does
    obs_centre = obs_start + obs_length / 2

    date_str_start = obs_start.strftime("%Y.%m.%d")
    time_str_start = obs_start.strftime("%H:%M:%S")

    date_str_centre = obs_centre.strftime("%Y-%m-%d")  # different format from the start
    time_str_centre = obs_centre.strftime("%H:%M:%S")

    obs_length = int(np.ceil(obs_length.total_seconds() / 3600.))  # in hours

    if quantity in ["I", "continuum", "intensity", "Ic"]:  # Ic data duration@lagImages
        if integration_time is None: integration_time = 45  # in seconds
        query_str = f"hmi.Ic_{integration_time}s{data_type}[{date_str_start}_{time_str_start}_TAI/{obs_length}h@{integration_time}s]{{continuum}}"
    elif quantity in ["B"]:  # magnetic field vector data
        if integration_time is None: integration_time = 720  # in seconds
        query_str = f"hmi.B_{integration_time}s{data_type}[{date_str_start}_{time_str_start}_TAI/{obs_length}h@{integration_time}s]{{field,inclination,azimuth,disambig}}"
    else:  # magnetic field component
        if integration_time is None: integration_time = 720  # in seconds
        query_str = f"hmi.B_{integration_time}s{data_type}[{date_str_start}_{time_str_start}_TAI/{obs_length}h@{integration_time}s]{{{quantity}}}"
    print(f"Data export query:\n\t{query_str}\n")

    process = {"im_patch": {
        "t_ref": f"{date_str_centre}T{time_str_centre}",
        # there must be a non-missing image within +- 2 hours of t_ref
        "t": 0,  # tracking?
        "r": 0,  # register the images to the first frame?
        "c": 0,  # cropping?
        "locunits": "arcsec",  # units for x and y
        "x": hdu[0].header["XCEN"],  # center_x in locunits
        "y": hdu[0].header["YCEN"],  # center_y in locunits
        "boxunits": "arcsec",  # units for width and height
        "width": np.max(hdu[38].data) - np.min(hdu[38].data) + margin_box,
        "height": np.max(hdu[39].data) - np.min(hdu[39].data) + margin_box,
    }}

    hdu.close()

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

    out_dir = path.join(_data_dir, f"SDO_HMI{data_type}", SP_filename.replace(".fits", ""))
    check_dir(out_dir)

    # Skip existing files.
    stored_files = os.listdir(out_dir)
    new_file_indices = np.where([file not in stored_files for file in result.data["filename"]])[0]
    print(f"{len(new_file_indices)} file(s) haven't been downloaded yet.\n")

    # Download selected files.
    result.wait()
    result.download(out_dir, index=new_file_indices)
    print("Download finished.")
    print(f'Download directory:\n\t"{path.abspath(out_dir)}"\n')

    print("Pausing the code for 10 seconds to avoid errors caused by pending requests.\n")
    time.sleep(10)


def return_sp_shape(SP_filename: str) -> tuple[int, ...]:
    hdu = fits.open(path.join(_path_sp, SP_filename))  # SP level 2 Hinode

    shape = np.shape(hdu[33])
    hdu.close()

    return shape


def return_hmi_shape(SP_filename: str) -> tuple[int, ...]:
    hmi_data = glob(path.join(_path_hmi, SP_filename.replace(".fits", ""), f"*.field.fits"))
    if hmi_data:
        return np.shape(fits.getdata(hmi_data[0], 1))

    # if the file does not exist, compute the shape from SP data
    print("HMI data does not exist. Computing the shape from SP data...")
    return sp_to_hmi_shape(SP_filename=SP_filename)


def sp_to_hmi_shape(SP_filename: str, input_shape: tuple[int, int] | None = None) -> tuple[int, ...]:
    if input_shape is None:
        input_shape = return_sp_shape(SP_filename=SP_filename)
    input_shape = np.array(input_shape)

    if "mean_hmi_resolution" in locals() or "mean_hmi_resolution" in globals():
        sp_to_hmi = np.array(return_sp_resolution(SP_filename)) / mean_hmi_resolution
    else:
        sp_to_hmi = np.array(return_sp_resolution(SP_filename)) / np.array([0.504310727, 0.504310727])

    return tuple(np.array(np.round(input_shape * sp_to_hmi), dtype=int))


def hmi_to_sp_shape(SP_filename: str, input_shape: tuple[int, int]) -> tuple[int, ...]:
    input_shape = np.array(input_shape)

    if "mean_hmi_resolution" in locals() or "mean_hmi_resolution" in globals():
        hmi_to_sp = mean_hmi_resolution / np.array(return_sp_resolution(SP_filename))
    else:
        hmi_to_sp = np.array([0.504310727, 0.504310727]) / np.array(return_sp_resolution(SP_filename))

    return tuple(np.array(np.round(input_shape * hmi_to_sp), dtype=int))


def return_sp_resolution(SP_filename: str) -> tuple[float, ...]:
    filename = path.join(_path_sp, SP_filename)
    if path.isfile(filename):
        with fits.open(filename) as hdu:  # SP level 2 Hinode
            x_scale = hdu[0].header["XSCALE"]
            y_scale = hdu[0].header["YSCALE"]

        return y_scale, x_scale
    return np.nan, np.nan


def return_hmi_resolution(SP_filename: str) -> tuple[float, ...]:
    hmi_data = glob(path.join(_path_hmi, SP_filename.replace(".fits", ""), f"*.field.fits"))
    if hmi_data:
        header = fits.getheader(hmi_data[0], 1)
        return header["CDELT2"], header["CDELT1"]
    return np.nan, np.nan


def check_resolution(instrument: Literal["SP", "HMI"], threshold: float = 0.0003) -> np.ndarray | None:
    SP_filenames = sorted([sp_file for sp_file in os.listdir(_path_sp) if "L2.1" not in sp_file])

    if instrument.upper() == "SP":
        sp = np.array([return_sp_resolution(_sp) for _sp in SP_filenames])
        # sp_resolution = calc_sp_resolution(method="median")
        sp_resolution = np.array([sp_dy_median, sp_dx_median])
        return np.where(np.logical_or.reduce((np.abs(sp[:, 0] - sp_resolution[0]) > threshold,
                                              np.abs(sp[:, 1] - sp_resolution[1]) > threshold)))[0]

    elif instrument.upper() == "HMI":
        hmi = np.array([return_hmi_resolution(_hmi) for _hmi in SP_filenames])
        # hmi_resolution = calc_hmi_resolution(method="median")
        hmi_resolution = np.array([hmi_dy_median, hmi_dx_median])
        return np.where(np.logical_or.reduce((np.abs(hmi[:, 0] - hmi_resolution[0]) > threshold,
                                              np.abs(hmi[:, 1] - hmi_resolution[1]) > threshold)))[0]

    else:
        return np.array([], dtype=int)


def check_SP_completeness(sp_path: str = _path_sp) -> bool:
    # there must be 1-1 correspondence of L2.0 and L2.1 data
    # the datasets are 1-1 complete if t(1) here are the same numbers of both parts AND
    # (2) for each in one part exists corresponding one in the other part

    SP20_filenames = sorted([sp_file for sp_file in os.listdir(sp_path) if "L2.1" not in sp_file])
    SP21_filenames = sorted([sp_file for sp_file in os.listdir(sp_path) if "L2.1" in sp_file])

    if len(SP20_filenames) != len(SP21_filenames):
        return False

    # The datasets are if the same size. 1-1 <=> ∀ x∈SP20 ∃ y∈SP21: y = x.replace(".fits", "_L2.1.fits")
    SP21_like = [sp_file.replace(".fits", "_L2.1.fits") for sp_file in SP20_filenames]

    return SP21_like == SP21_filenames


def calc_hmi_to_sp_resolution(method: Literal["mean", "median"] = "median", fast: bool = False) -> np.ndarray:
    if fast:
        return np.array([hmi_dy_median, hmi_dx_median]) / np.array([sp_dy_median, sp_dx_median])

    SP_filenames = sorted([sp_file for sp_file in os.listdir(_path_sp) if "L2.1" not in sp_file])
    sp = np.array([return_sp_resolution(_sp) for _sp in SP_filenames])
    hmi = np.array([return_hmi_resolution(_hmi) for _hmi in SP_filenames])

    if method == "mean":
        result = np.nanmean(hmi / sp, axis=0)
    elif method == "median":
        result = np.nanmedian(hmi / sp, axis=0)
    else:
        raise ValueError(f'method must be in ["mean", "median"] but is {method}')

    if np.any(~np.isfinite(result)):
        result = np.array([hmi_dy_median, hmi_dx_median]) / np.array([sp_dy_median, sp_dx_median])
    return result


def calc_hmi_resolution(method: Literal["mean", "median"] = "median") -> np.ndarray:
    SP_filenames = sorted([sp_file for sp_file in os.listdir(_path_sp) if "L2.1" not in sp_file])
    if not SP_filenames:
        return np.array([hmi_dy_median, hmi_dx_median])

    hmi = np.array([return_hmi_resolution(_hmi) for _hmi in SP_filenames])

    if method == "mean":
        result = np.nanmean(hmi, axis=0)
    elif method == "median":
        result = np.nanmedian(hmi, axis=0)
    else:
        raise ValueError(f'method must be in ["mean", "median"] but is {method}')

    if np.any(~np.isfinite(result)):
        result = np.array([hmi_dy_median, hmi_dx_median])
    return result


def calc_sp_resolution(method: Literal["mean", "median"] = "median") -> np.ndarray:
    SP_filenames = sorted([sp_file for sp_file in os.listdir(_path_sp) if "L2.1" not in sp_file])
    if not SP_filenames:
        return np.array([sp_dy_median, sp_dx_median])

    sp = np.array([return_sp_resolution(_sp) for _sp in SP_filenames])

    if method == "mean":
        result = np.nanmean(sp, axis=0)
    elif method == "median":
        result = np.nanmedian(sp, axis=0)
    else:
        raise ValueError(f'method must be in ["mean", "median"] but is {method}')

    if np.any(~np.isfinite(result)):
        result = np.array([sp_dy_median, sp_dx_median])
    return result


def read_hmi_B(SP_filename: str, coordinates: Literal["ptr", "ptr_native"] = "ptr_native") -> np.ndarray:
    def _collect_from_sav(sav_file: str) -> np.ndarray:
        return sio.readsav(sav_file)["bptr"]

    def _collect_from_fits(fits_file: str, print_warn: bool = False) -> np.ndarray:
        if print_warn:
            warnings.warn(f"The sav file does not exist.\n\t{fits_file}\nCalculating B_ptr from fits.")

        cotemporal_fits = read_cotemporal_fits(fits_file, check_uniqueness=True)

        field_fits = cotemporal_fits["fits_b"]
        if field_fits is None:
            raise ValueError("(At least) one of field fits is missing. Check your data.")
        index = fits.getheader(field_fits, 1)
        b = fits.getdata(field_fits, 1)

        inclination_fits = cotemporal_fits["fits_inc"]
        if inclination_fits is None:
            raise ValueError("(At least) one of inclination fits is missing. Check your data.")
        bi = fits.getdata(inclination_fits, 1)

        azimuth_fits = cotemporal_fits["fits_azi"]
        if azimuth_fits is None:
            raise ValueError("(At least) one of azimuth fits is missing. Check your data.")
        bg = fits.getdata(azimuth_fits, 1)

        disambig_fits = cotemporal_fits["fits_disamb"]
        if disambig_fits is None:
            raise ValueError("(At least) one of disambig fits is missing. Check your data.")
        bgd = np.array(fits.getdata(disambig_fits, 1), dtype=int)

        bg = disambigue_azimuth(bg, bgd, method=1,
                                rotated_image="history" in index and "rotated" in str(index["history"]))

        return data_b2ptr(index=index, bvec=np.array([b, bi, bg]))

    # list of all files of one type
    hmi_b = sorted(glob(path.join(_path_hmi, SP_filename.replace(".fits", ""), f"*.field.fits")))

    if coordinates == "ptr_native":
        hmi_sav = [_hmi_b.replace(".field.fits", ".ptr.sav") for _hmi_b in hmi_b]

        return np.array([_collect_from_sav(sav_file=sav_file) if path.isfile(sav_file) else
                         _collect_from_fits(fits_file=sav_file, print_warn=True)
                         for sav_file in hmi_sav])

    return np.array([_collect_from_fits(fits_file=fits_file) for fits_file in hmi_b])


def read_sp(SP_filename: str, coordinates: Literal["ptr", "ptr_native"] = "ptr_native") -> np.ndarray:
    hdu = fits.open(path.join(_path_sp, SP_filename))  # SP level 2 Hinode
    hdu21 = fits.open(path.join(_path_sp, SP_filename.replace(".fits", "_L2.1.fits")))  # SP level 2.1 Hinode

    if coordinates == "ptr_native":
        # -hdu21[3].data to follow HMI Bt +S direction
        # there are chess-like maps if B is not strong enough
        sp_observation = np.array([hdu[33].data, hdu21[2].data, -hdu21[3].data, hdu21[4].data])

    else:
        # - 90 to have the same zero point as HMI data (i.e. +S and +W)
        # there are chess-like maps if B is not strong enough
        azimuth = np.mod(hdu21[1].data - 90., 360.)  # disambiguated

        # filling factor correction for level 2 data
        f = hdu[12].data
        f = np.clip(interpolate_mask(image=f, mask=np.logical_or(f <= 0., f > 1.),
                                     interp_nans=True, fill_value=_num_eps), _num_eps, 1.)

        # https://darts.isas.jaxa.jp/solar/hinode/data/sotsp_level2.html
        # continuum (arb.), |B| (G), B_inclination (deg), B_azimuth (deg)
        sp_observation = np.array([hdu[33].data, f * hdu[1].data, hdu[2].data, azimuth])

        # now convert B to ptr coordinates
        sp_observation[1:] = data_b2ptr(index=hdu[0].header, bvec=sp_observation[1:])

    hdu.close()
    hdu21.close()

    return sp_observation


def resize_data(data: np.ndarray, final_shape: tuple[int, ...]) -> np.ndarray:
    if np.ndim(data) > 2:
        return np.array([resize_data(data=data_part, final_shape=final_shape) for data_part in data])
    return resize(data, final_shape, anti_aliasing=True)


def blur_sp(sp_observation: np.ndarray, psf: np.ndarray | None = None) -> np.ndarray:
    if psf is None:
        psf = hmi_psf(target_shape=97, calc_new=False)

    # blur the downscaled SP data
    return np.array([apply_psf(sp_obs, psf, use_fft=True) for sp_obs in sp_observation])


def noise_realisation(component: int | str,
                      output_shape: tuple[int, int] | None = None,
                      noise_sample: np.ndarray | None = None,
                      ) -> np.ndarray:
    if noise_sample is None:
        noise_sample = hmi_noise(calc_new=False)

    rng = np.random.default_rng(seed=None)

    # Generate a new noise sample with the same power spectrum
    def generate_noise_sample(fft_amplitude: np.ndarray) -> np.ndarray:
        """
        Generates a full-sized noise sample with translation-invariant correlation structure
        based on the provided power spectrum.
        """
        for i in range(10):  # max 10 attempts
            # Generate random phases for 2D data
            random_angle = 2. * np.pi * rng.random(np.shape(fft_amplitude)) - np.pi  # from -pi to pi

            # Original signal is real values -> force FT symmetries
            fft_amplitude, random_angle = force_fft_sym(amplitude=fft_amplitude, angle=random_angle)

            # Create new noise in the frequency domain and transform back to the spatial domain
            noise = ifft2(fft_amplitude * np.exp(1j * random_angle))

            max_imag = np.max(np.abs(np.imag(noise)))
            if np.max(np.abs(np.imag(noise))) > 0.1:
                if i < 9:
                    warnings.warn(f"Maximum imaginary part of generated noise is {max_imag}! "
                                  f"Trying to generate a different one...")
                else:
                    warnings.warn(f"Maximum imaginary part of generated noise is {max_imag}!")
            else:
                break

        return np.real(noise)

    def force_fft_sym(amplitude: np.ndarray, angle: np.ndarray) -> tuple[np.ndarray, ...]:
        nr, nc = np.shape(amplitude)
        _nr = nr

        amplitude, angle = fftshift(amplitude), fftshift(angle)

        if nr % 2 == 0:  # add a row on top to shift zero frequency to centre
            amplitude = np.r_[amplitude, np.nan * np.zeros((1, nc))]
            angle = np.r_[angle, np.nan * np.zeros((1, nc))]
            _nr = nr + 1
        if nc % 2 == 0:  # add a column on top to shift zero frequency to centre
            amplitude = np.c_[amplitude, np.nan * np.zeros(_nr)]
            angle = np.c_[angle, np.nan * np.zeros(_nr)]

        lower_triangle_amp, upper_triangle_amp = np.tril(amplitude, k=-1), np.rot90(np.tril(amplitude, k=-1), k=2)
        lower_triangle_ang, upper_triangle_ang = np.tril(angle, k=-1), np.rot90(np.tril(angle, k=-1), k=2)
        diag_amp, diag_ang = np.diag(amplitude), np.diag(angle)

        amp = lower_triangle_amp + upper_triangle_amp  # diagonal is missing
        lower_diag = diag_amp[:nr // 2]  # symmetrical part
        diag = stack((lower_diag, diag_amp[nr // 2], np.flip(lower_diag)))
        amp += np.diag(diag)
        amp[~np.isfinite(amp)] = amplitude[~np.isfinite(amp)]  # if even dimensions

        ang = lower_triangle_ang - upper_triangle_ang  # diagonal is missing
        lower_diag = diag_ang[:nr // 2]  # antisymmetrical part (zero in centre)
        diag = stack((lower_diag, 0, -np.flip(lower_diag)))
        ang += np.diag(diag)
        ang[~np.isfinite(ang)] = angle[~np.isfinite(ang)]  # if even dimensions

        return ifftshift(amp[:nr, :nc]), ifftshift(ang[:nr, :nc])

    def calc_total_energy(fft_amplitude: np.ndarray) -> np.ndarray:
        # Compute the total energy (sum of squared amplitudes)
        return np.sum(np.abs(fft_amplitude) ** 2.)

    def resize_fft_amplitude(fft_amplitude: np.ndarray, target_shape: tuple[int, ...],
                             method: str = "linear") -> np.ndarray:
        """
        Resize the amplitude spectrum in the Fourier domain.
        """
        print(f"Spectrum amplitude interpolation from {np.shape(fft_amplitude)} to {target_shape}.")
        nr, nc = np.shape(fft_amplitude)
        freq_r, freq_c = fftfreq(nr), fftfreq(nc)
        fun = RegularGridInterpolator(points=(fftshift(freq_r), fftshift(freq_c)), values=fftshift(fft_amplitude),
                                      method=method, bounds_error=False, fill_value=None)
        nr, nc = target_shape
        freq_r, freq_c = fftfreq(nr), fftfreq(nc)
        freq_r, freq_c = np.meshgrid(freq_r, freq_c, indexing='ij', sparse=True)
        fft_amplitude = fun((freq_r, freq_c))

        return fft_amplitude

    if component in ["Bp", "p"]:
        component = 0
    elif component in ["Bt", "t"]:
        component = 1
    elif component in ["Br", "r"]:
        component = 2
    else:
        component = int(component)

    if output_shape is None:
        output_shape = np.shape(noise_sample[component])

    # it is better to crop the data to (patch x patch)
    # noise = pad_zeros_or_crop(noise_sample[component], target_shape=output_shape)
    noise = noise_sample[component]

    # Fourier transform the noise sample and extract its amplitude spectrum
    noise_fft = fft2(noise)
    fft_amplitude = np.abs(noise_fft)

    if np.shape(fft_amplitude) != output_shape:
        # Resize the FFT amplitude to the target size if needed
        fft_amplitude = resize_fft_amplitude(fft_amplitude, target_shape=output_shape)

    # Apply a high-pass filter to remove low frequencies (retain zero-mean noise)
    fft_amplitude = filter_fft_amplitude(fft_amplitude, min_fraction_nyquist=0.04)

    # Generate translation-invariant noise for the full image size
    full_noise_sample = generate_noise_sample(fft_amplitude)

    return full_noise_sample


def add_noise_realisation(array: np.ndarray, thresh: float = 100.) -> np.ndarray:
    for component in range(len(array)):
        noise = noise_realisation(component, np.shape(array[component]))

        # Add noise where |array| < thresh
        mask = np.abs(array[component]) < thresh
        array[component][mask] += noise[mask]

    return array


def create_noise_patch(SP_filename: str, patch_size: int | None = None) -> np.ndarray:
    if patch_size is None: patch_size = conf_grid_setup["patch_size"]

    noise = [noise_realisation(component=c) for c in range(3)]
    noise = [resize_data(noise_part, final_shape=hmi_to_sp_shape(SP_filename=SP_filename,
                                                                 input_shape=np.shape(noise_part)))
             for noise_part in noise]
    return np.array([pad_zeros_or_crop(noise_part, target_shape=(patch_size, patch_size)) for noise_part in noise])


def add_noise_realisation_old(array: np.ndarray, noise_sample: np.ndarray | None = None,
                              thresh: float = 100.) -> np.ndarray:
    if noise_sample is None:
        noise_sample = hmi_noise(calc_new=False)

    rng = np.random.default_rng(seed=None)

    # Generate a new noise sample with the same power spectrum
    def generate_noise_sample(fft_amplitude: np.ndarray) -> np.ndarray:
        # Generate random phases for 2D data
        random_phases = np.exp(2j * np.pi * rng.random(np.shape(fft_amplitude)))

        # Create new noise in the frequency domain and transform back to the spatial domain
        return np.real(ifft2(fft_amplitude * random_phases))

    for component in range(len(array)):
        # Compute the 2D Fourier transform to get the power spectrum
        noise_fft = fft2(noise_sample[component])
        fft_amplitude = np.abs(noise_fft)

        # apply high-pass filter to keep zero-mean noise only
        high_pass = ifftshift(1. - create_circular_mask(shape=np.shape(fft_amplitude), radius=7.,
                                                        smooth=True, steepness=1.))
        fft_amplitude *= high_pass

        # Dimensions of the image and noise sample
        image_shape = np.shape(array[component])
        noise_shape = np.shape(noise_fft)

        # Calculate how many times we need to tile the noise
        # tiles_x = np.ceil(image_shape[1], noise_shape[1])
        # tiles_y = np.ceil(image_shape[0], noise_shape[0])
        tiles_x = (image_shape[1] // noise_shape[1]) + 1
        tiles_y = (image_shape[0] // noise_shape[0]) + 1

        new_noise_sample = stack([stack([generate_noise_sample(fft_amplitude=fft_amplitude)
                                         for _ in range(tiles_y)], axis=0)
                                  for _ in range(tiles_x)], axis=1)
        new_noise_sample = new_noise_sample[:image_shape[0], :image_shape[1]]

        mask = np.abs(array[component]) < thresh
        array[component][mask] = array[component][mask] + new_noise_sample[mask]

    return array


def get_hmi_sectors(quantity: str, SP_filename: str, hmi_b: np.ndarray | None = None):
    hdu = fits.open(path.join(_path_sp, SP_filename))  # SP level 2 Hinode)

    if quantity != "continuum":
        index_b = int(quantity[-1])
        quantity = "field"

    # only used for continuum (B uses index variable)
    files_hmi = sorted(glob(path.join(_path_hmi, SP_filename.replace(".fits", ""), f"*.{quantity}.fits")))

    # observational time of SP and HMI
    times_sp = [date[0] for date in hdu[40].data]
    times_hmi = [fits.getheader(file_hmi, 1)["date-obs"] for file_hmi in files_hmi]

    # indices of common times (for HMI)
    ind_i = np.argmin([np.abs(TimeRange(time_hmi, times_sp[0]).seconds.value) for time_hmi in times_hmi])
    ind_f = np.argmin([np.abs(TimeRange(time_hmi, times_sp[-1]).seconds.value) for time_hmi in times_hmi])
    ind_time_hmi = np.array([np.argmin([np.abs(TimeRange(time_hmi, time_sp).seconds.value)
                                        for time_hmi in times_hmi[ind_i:ind_f + 1]])
                             for time_sp in times_sp])
    unique_ind_time_hmi, widths = np.unique(ind_time_hmi, return_counts=True)

    num_sectors = len(unique_ind_time_hmi)
    width_mean = int(np.ceil(np.mean(widths)))

    # merge HMI observations and reshape them to SP shape
    hmi_sectors_quantity = np.zeros_like(hdu[33].data)
    hdu.close()

    for i in range(num_sectors):
        index0 = np.clip(unique_ind_time_hmi[i] - 1, 0, num_sectors - 1)
        index1 = np.clip(unique_ind_time_hmi[i], 0, num_sectors - 1)  # this clip is not necessary
        index2 = np.clip(unique_ind_time_hmi[i] + 1, 0, num_sectors - 1)

        if quantity == "continuum":
            data = (fits.getdata(files_hmi[index0], 1) * 0.25
                    + fits.getdata(files_hmi[index1], 1) * 0.5
                    + fits.getdata(files_hmi[index2], 1) * 0.25)
        else:
            # no mean for B (HMI has cadence of 12 minutes...)
            data = hmi_b[index1, index_b]

        ind_sector = np.where(ind_time_hmi == unique_ind_time_hmi[i])[0]
        hmi_sectors_quantity[:, ind_sector] = resize(data, np.shape(hmi_sectors_quantity), anti_aliasing=True)[:, ind_sector]

    return hmi_sectors_quantity, width_mean


def orbAlign(image: np.ndarray, reference: np.ndarray, align_rules: dict, mode: str = "affine") -> np.ndarray:
    if mode == "euclidean":
        mode = EuclideanTransform

    elif mode == "similarity":
        mode = SimilarityTransform

    elif mode == "affine":
        mode = AffineTransform

    elif mode == "projective":
        mode = ProjectiveTransform

    else:
        list_modes = ["similarity", "euclidean", "affine", "projective"]
        raise ValueError(f"Mode {mode} is not supported, please use one of the following modes: {list_modes}")

    original, distorted = np.array(reference, copy=True), np.array(image, copy=True)
    warped_image = np.zeros_like(original)

    for index_aligned, index_master in align_rules.items():
        # extract key features
        orb = ORB(n_keypoints=2000)

        original_master, distorted_master = original[index_master], np.array(distorted[index_master], copy=True)
        # distorted_master *= (np.max(original_master) / np.max(distorted_master))

        orb.detect_and_extract(original_master)
        keypoints1 = orb.keypoints
        descriptors1 = orb.descriptors

        orb.detect_and_extract(distorted_master)
        keypoints2 = orb.keypoints
        descriptors2 = orb.descriptors

        # descriptor matching
        matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)

        src = keypoints2[matches12[:, 1]][:, ::-1]
        dst = keypoints1[matches12[:, 0]][:, ::-1]

        affine, inliers = ransac((src, dst), mode, min_samples=3, residual_threshold=2, max_trials=1000, rng=_rnd_seed)
        warped_image[index_aligned] = warp(distorted[index_aligned], inverse_map=affine.inverse,
                                           output_shape=np.shape(original_master), order=1, mode="constant", cval=np.nan)

    return warped_image


def shift_pixels(reference: np.ndarray, image: np.ndarray, align_rules: dict, match_hist: bool = False) -> np.ndarray:
    # shift the pixels using optical flow (fine alignment); based on continuum images

    image_shifted = np.zeros_like(image)

    _, n_rows, n_cols = np.shape(image)
    row_coords, col_coords = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")

    for index_aligned, index_master in align_rules.items():
        # measured in opposite direction; (more stable if the shifted figure is sharp)
        if match_hist:
            moving_image = match_histograms(image=reference[index_master], reference=image[index_master])
        else:
            moving_image = reference[index_master]

        v_row, v_col = -optical_flow_tvl1(reference_image=image[index_master], moving_image=moving_image)

        image_shifted[index_aligned] = warp(image[index_aligned], np.array([row_coords + v_row, col_coords + v_col]),
                                            order=1, mode="constant", cval=np.nan)

    return image_shifted


def align_SP_file(SP_filename: str, coordinates: Literal["ptr", "ptr_native"] = "ptr_native",
                  index_continuum: int = 0, align_rules: dict | None = None,
                  subpixel_shift: bool = False, interpolate_outliers: bool = True,
                  control_plots: bool = False, quiet: bool = False) -> tuple[np.ndarray, ...]:
    # B orientation:
    #  - Bp +W
    #  - Bt +S  !!!!
    #  - Br -grav

    # SP_filename = sorted([sp_file for sp_file in os.listdir(_path_sp) if "L2.1" not in sp_file])[0]

    if align_rules is None:
        # {aligned_image: reference_image} in order [I, Bp, Bt, Br]
        # good to keep reference_image constant (same transformations for all quantities)
        align_rules = {0: 0, 1: 0, 2: 0, 3: 0}

    if not quiet:
        print(SP_filename)

    # hmi magnetic field in the given coordinates and G
    hmi_ptr = read_hmi_B(SP_filename, coordinates=coordinates)

    # continuum and B in the given coordinates and G
    sp_iptr = read_sp(SP_filename=SP_filename, coordinates=coordinates)

    # downscale SP observation to pixel size of HMI (SP and HMI both observe the same box)
    _, _, *hmi_shape = np.shape(hmi_ptr)
    sp_iptr_hmilike = resize_data(data=sp_iptr, final_shape=hmi_shape)

    # allocate space for upscaled HMI
    hmi_iptr = np.zeros_like(sp_iptr)

    # the "x: int" in "Bx" codes index in hmi_ptr, do not change!
    for i, quantity in enumerate(["continuum", "B0", "B1", "B2"]):
        # here is the first interpolation. Inclination/azimuth must be converted before!
        hmi_iptr[i], _ = get_hmi_sectors(quantity, SP_filename, hmi_ptr)

    sp_iptr[index_continuum] = normalise_intensity(remove_limb_darkening_approx(sp_iptr[index_continuum]))
    sp_iptr_hmilike[index_continuum] = normalise_intensity(remove_limb_darkening_approx(sp_iptr_hmilike[index_continuum]))
    hmi_iptr[index_continuum] = normalise_intensity(remove_limb_darkening_approx(hmi_iptr[index_continuum]))

    sp_iptr[index_continuum] = np.clip(interpolate_mask(image=sp_iptr[index_continuum],
                                                        mask=sp_iptr[index_continuum] <= 0.,
                                                        interp_nans=True, fill_value=_num_eps),
                                       _num_eps, None)
    sp_iptr_hmilike[index_continuum] = np.clip(interpolate_mask(image=sp_iptr_hmilike[index_continuum],
                                                                mask=sp_iptr_hmilike[index_continuum] <= 0.,
                                                                interp_nans=True, fill_value=_num_eps),
                                                  _num_eps, None)
    hmi_iptr[index_continuum] = np.clip(interpolate_mask(image=hmi_iptr[index_continuum],
                                                         mask=hmi_iptr[index_continuum] <= 0.,
                                                         interp_nans=True, fill_value=_num_eps),
                                        _num_eps, None)

    if control_plots:
        mplbackend = "TkAgg"  # must be done like this to confuse PyInstaller
        mpl.use(backend=mplbackend)
        import matplotlib.pyplot as plt

        figures = {}
        axes = {}
        for i in range(len(sp_iptr)):
            figures[f"fig{i}"], axes[f"ax{i}"] = plt.subplots(3, 3)

        for i, ax in enumerate(axes.values()):
            add_subplot(ax[0, 0], hmi_iptr[i])  # title="HMI sectors"
            add_subplot(ax[1, 0], sp_iptr_hmilike[i])  # title="HMI like"
            add_subplot(ax[2, 0], sp_iptr[i])  # title="SP original"

    # blur the sp_downscaled data
    sp_iptr_hmilike = blur_sp(sp_observation=sp_iptr_hmilike)
    # adding noise to B (mostly horizontal to mimic chess-like structure in low-B pixels
    sp_iptr_hmilike[1:] = add_noise_realisation_old(array=sp_iptr_hmilike[1:], thresh=100.)
    # upscale again
    sp_iptr_hmilike = resize_data(sp_iptr_hmilike, final_shape=np.shape(sp_iptr[0]))

    # align HMI and SP data based on align_rules, remove NaNs, and rescale intensities
    # use original SP or blurred SP data to align original HMI with SP?
    hmi_iptr = orbAlign(image=hmi_iptr, reference=sp_iptr_hmilike, mode="affine", align_rules=align_rules)

    _, sp_iptr = remove_nan(corrupted=hmi_iptr, clear=sp_iptr)
    hmi_iptr, sp_iptr_hmilike = remove_nan(corrupted=hmi_iptr, clear=sp_iptr_hmilike)

    if control_plots:
        for i, ax in enumerate(axes.values()):
            add_subplot(ax[0, 1], hmi_iptr[i])  # title="HMI sectors"
            add_subplot(ax[1, 1], sp_iptr_hmilike[i])  # title="HMI like"
            add_subplot(ax[2, 1], sp_iptr[i])  # title="SP original"

    if subpixel_shift:
        # subpixel shift of the HMI images based on align_rules, remove NaNs, and rescale intensities
        hmi_iptr = shift_pixels(reference=sp_iptr_hmilike, image=hmi_iptr, align_rules=align_rules)
        _, sp_iptr = remove_nan(corrupted=hmi_iptr, clear=sp_iptr)
        hmi_iptr, sp_iptr_hmilike = remove_nan(corrupted=hmi_iptr, clear=sp_iptr_hmilike)

        sp_iptr[index_continuum] = normalise_intensity(sp_iptr[index_continuum])
        sp_iptr_hmilike[index_continuum] = normalise_intensity(sp_iptr_hmilike[index_continuum])
        hmi_iptr[index_continuum] = normalise_intensity(hmi_iptr[index_continuum])

    if interpolate_outliers:  # remove outliers (slow...)
        print(f"Outliers are{'' if interpolate_outliers else ' not'} interpolated.")
        sp_iptr = np.array([remove_outliers_sliding_window(image=data_part, kernel_size=7, n_std=3.) for data_part in sp_iptr])
        sp_iptr_hmilike = np.array([remove_outliers_sliding_window(image=data_part, kernel_size=7, n_std=3.) for data_part in sp_iptr_hmilike])
        hmi_iptr = np.array([remove_outliers_sliding_window(image=data_part, kernel_size=7, n_std=3.) for data_part in hmi_iptr])

        sp_iptr[index_continuum] = normalise_intensity(sp_iptr[index_continuum])
        sp_iptr_hmilike[index_continuum] = normalise_intensity(sp_iptr_hmilike[index_continuum])
        hmi_iptr[index_continuum] = normalise_intensity(hmi_iptr[index_continuum])

    """
    # adjust the histgram of sp_hmilike
    # MAY CAUSE UNWANTED EFFECTS
    sp_iptr_hmilike = np.array([match_histograms(image=sp_iptr_hmilike[i], reference=hmi_iptr[i])
                                for i in range(len(sp_iptr_hmilike))])
    """

    if control_plots:
        for i, ax in enumerate(axes.values()):
            add_subplot(ax[0, 2], hmi_iptr[i])  # title="HMI sectors"
            add_subplot(ax[1, 2], sp_iptr_hmilike[i])  # title="HMI like"
            add_subplot(ax[2, 2], sp_iptr[i])  # title="SP original"

        for fig in figures.values():
            mng = fig.canvas.manager
            mng.resize(*mng.window.maxsize())
            fig.tight_layout()

        plt.show()

    if not quiet:
        print(" I      Bp     Bt     Br")
        print(", ".join([f"{rmse(hmi_iptr[i], sp_iptr_hmilike[i]):.2f}" for i in range(len(sp_iptr))]))

    # there is one observation of 4 quantities (nq, nx, ny) -> (1, nx, ny, nq)
    hmi_iptr = np.expand_dims(np.transpose(hmi_iptr, (1, 2, 0)), axis=0)
    sp_iptr = np.expand_dims(np.transpose(sp_iptr, (1, 2, 0)), axis=0)
    sp_iptr_hmilike = np.expand_dims(np.transpose(sp_iptr_hmilike, (1, 2, 0)), axis=0)

    save_data(SP_filename.replace(".fits", ".npz"), observations=hmi_iptr, labels=sp_iptr,
              subfolder=_path_sp_hmi, other_info={f"{_observations_name}_simulated": np.array(sp_iptr_hmilike, dtype=_wp),
                                                  "units": ["quiet-Sun normalised", "G", "G", "G"],
                                                  "direction": [None, "+W", "+S", "-grav"]})

    return hmi_iptr, sp_iptr, sp_iptr_hmilike


def sp_to_hmilike_patches(SP_filename: str, coordinates: Literal["ptr", "ptr_native"] = "ptr_native",
                          index_continuum: int = 0, interpolate_outliers: bool = True,
                          patch_size: int | None = None, cut_edge_px: int = 0,
                          control_plots: bool = False, quiet: bool = False) -> tuple[np.ndarray, ...]:
    # B orientation:
    #  - Bp +W
    #  - Bt +S  !!!!
    #  - Br -grav

    # SP_filename = sorted([sp_file for sp_file in os.listdir(_path_sp) if "L2.1" not in sp_file])[0]
    if patch_size is None: patch_size = conf_grid_setup["patch_size"]

    if not quiet:
        print(SP_filename)

    # continuum and B in the given coordinates and G
    sp_iptr = read_sp(SP_filename=SP_filename, coordinates=coordinates)
    sp_iptr[index_continuum] = remove_limb_darkening_approx(sp_iptr[index_continuum])

    if interpolate_outliers:  # remove outliers (slow...)
        print(f"Outliers are{'' if interpolate_outliers else ' not'} interpolated.")
        sp_iptr = np.array([remove_outliers_sliding_window(image=data_part, kernel_size=7, n_std=2.) for data_part in sp_iptr])
        sp_iptr[index_continuum] = remove_limb_darkening_approx(sp_iptr[index_continuum])

    # downscale SP observation to pixel size of HMI (SP and HMI both observe the same box)
    sp_iptr_hmilike = resize_data(data=sp_iptr, final_shape=sp_to_hmi_shape(SP_filename))
    sp_iptr_hmilike[index_continuum] = remove_limb_darkening_approx(sp_iptr_hmilike[index_continuum])

    sp_iptr[index_continuum] = np.clip(interpolate_mask(image=sp_iptr[index_continuum],
                                                        mask=sp_iptr[index_continuum] <= 0.,
                                                        interp_nans=True, fill_value=_num_eps),
                                       _num_eps, None)
    sp_iptr_hmilike[index_continuum] = np.clip(interpolate_mask(image=sp_iptr_hmilike[index_continuum],
                                                                mask=sp_iptr_hmilike[index_continuum] <= 0.,
                                                                interp_nans=True, fill_value=_num_eps),
                                               _num_eps, None)

    sp_iptr[index_continuum] = remove_limb_darkening_approx(sp_iptr[index_continuum])
    sp_iptr_hmilike[index_continuum] = remove_limb_darkening_approx(sp_iptr_hmilike[index_continuum])

    if control_plots:
        mplbackend = "TkAgg"  # must be done like this to confuse PyInstaller
        mpl.use(backend=mplbackend)
        import matplotlib.pyplot as plt

        figures = {}
        axes = {}
        for i in range(len(sp_iptr)):
            figures[f"fig{i}"], axes[f"ax{i}"] = plt.subplots(2, 3)

        for i, ax in enumerate(axes.values()):
            add_subplot(ax[0, 0], sp_iptr_hmilike[i])  # title="HMI like"
            add_subplot(ax[1, 0], sp_iptr[i])  # title="SP original"

    # blur the sp_downscaled data
    sp_iptr_hmilike = blur_sp(sp_observation=sp_iptr_hmilike)

    # upscale again
    sp_iptr_hmilike = resize_data(sp_iptr_hmilike, final_shape=np.shape(sp_iptr[0]))

    #
    # HERE MUST BE SPLITTING TO PATCHES
    # adding noise to B (mostly horizontal to mimic chess-like structure in low-B pixels
    #
    sp_iptr = np.expand_dims(np.transpose(sp_iptr, (1, 2, 0)), axis=0)
    sp_iptr_hmilike = np.expand_dims(np.transpose(sp_iptr_hmilike, (1, 2, 0)), axis=0)

    hmi_iptr, sp_iptr_hmilike, sp_iptr = split_to_patches(hmi_full=None,
                                                          hmilike_full=sp_iptr_hmilike,
                                                          sp_full=sp_iptr,
                                                          patch_size=patch_size,
                                                          cut_edge_px=cut_edge_px)

    if control_plots:
        for i, ax in enumerate(axes.values()):
            add_subplot(ax[0, 1], sp_iptr_hmilike[0, :, :, i])  # title="HMI like"
            add_subplot(ax[1, 1], sp_iptr[0, :, :, i])  # title="SP original"

    noise_patch = np.array([create_noise_patch(SP_filename=SP_filename, patch_size=patch_size)
                            for _ in range(len(sp_iptr_hmilike))])
    noise_patch = np.transpose(noise_patch, axes=(0, 2, 3, 1))
    mask = np.abs(sp_iptr_hmilike[:, :, :, 1:]) < 100.
    sp_iptr_hmilike[:, :, :, 1:][mask] += noise_patch[mask]

    if control_plots:
        for i, ax in enumerate(axes.values()):
            add_subplot(ax[0, 2], sp_iptr_hmilike[0, :, :, i])  # title="HMI like"
            add_subplot(ax[1, 2], sp_iptr[0, :, :, i])  # title="SP original"

        for fig in figures.values():
            mng = fig.canvas.manager
            mng.resize(*mng.window.maxsize())
            fig.tight_layout()

        plt.show()

    _, nr, nc, nq = np.shape(sp_iptr)
    save_data(SP_filename.replace(".fits", "_patched.npz"), observations=np.zeros((0, nr, nc, nq)),
              labels=sp_iptr, subfolder=path.join(_path_sp_hmilike, "patches"),
              other_info={f"{_observations_name}_simulated": np.array(sp_iptr_hmilike, dtype=_wp),
                          "units": ["quiet-Sun normalised", "G", "G", "G"],
                          "direction": [None, "+W", "+S", "-grav"]})

    return sp_iptr, sp_iptr_hmilike


def sp_to_hmilike(SP_filename: str, coordinates: Literal["ptr", "ptr_native"] = "ptr_native",
                  index_continuum: int = 0, interpolate_outliers: bool = True,
                  control_plots: bool = False, quiet: bool = False) -> tuple[np.ndarray, ...]:
    # does not perform splitting to patches (problems with noise statistics)

    # B orientation:
    #  - Bp +W
    #  - Bt +S  !!!!
    #  - Br -grav

    # SP_filename = sorted([sp_file for sp_file in os.listdir(_path_sp) if "L2.1" not in sp_file])[0]

    if not quiet:
        print(SP_filename)

    # continuum and B in the given coordinates and G
    sp_iptr = read_sp(SP_filename=SP_filename, coordinates=coordinates)

    # downscale SP observation to pixel size of HMI (SP and HMI both observe the same box)
    sp_iptr_hmilike = resize_data(data=sp_iptr, final_shape=sp_to_hmi_shape(SP_filename))

    sp_iptr[index_continuum] = remove_limb_darkening_approx(sp_iptr[index_continuum])
    sp_iptr_hmilike[index_continuum] = remove_limb_darkening_approx(sp_iptr_hmilike[index_continuum])

    sp_iptr[index_continuum] = np.clip(interpolate_mask(image=sp_iptr[index_continuum],
                                                        mask=sp_iptr[index_continuum] <= 0.,
                                                        interp_nans=True, fill_value=_num_eps),
                                       _num_eps, None)
    sp_iptr_hmilike[index_continuum] = np.clip(interpolate_mask(image=sp_iptr_hmilike[index_continuum],
                                                                mask=sp_iptr_hmilike[index_continuum] <= 0.,
                                                                interp_nans=True, fill_value=_num_eps),
                                               _num_eps, None)

    sp_iptr[index_continuum] = remove_limb_darkening_approx(sp_iptr[index_continuum])
    sp_iptr_hmilike[index_continuum] = remove_limb_darkening_approx(sp_iptr_hmilike[index_continuum])

    if control_plots:
        mplbackend = "TkAgg"  # must be done like this to confuse PyInstaller
        mpl.use(backend=mplbackend)
        import matplotlib.pyplot as plt

        figures = {}
        axes = {}
        for i in range(len(sp_iptr)):
            figures[f"fig{i}"], axes[f"ax{i}"] = plt.subplots(2, 3)

        for i, ax in enumerate(axes.values()):
            add_subplot(ax[0, 0], sp_iptr_hmilike[i])  # title="HMI like"
            add_subplot(ax[1, 0], sp_iptr[i])  # title="SP original"

    # blur the sp_downscaled data
    sp_iptr_hmilike = blur_sp(sp_observation=sp_iptr_hmilike)
    # adding noise to B (mostly horizontal to mimic chess-like structure in low-B pixels
    sp_iptr_hmilike[1:] = add_noise_realisation_old(array=sp_iptr_hmilike[1:], thresh=100.)
    # upscale again
    sp_iptr_hmilike = resize_data(sp_iptr_hmilike, final_shape=np.shape(sp_iptr[0]))
    sp_iptr_hmilike[index_continuum] = remove_limb_darkening_approx(sp_iptr_hmilike[index_continuum])

    if control_plots:
        for i, ax in enumerate(axes.values()):
            add_subplot(ax[0, 1], sp_iptr_hmilike[i])  # title="HMI like"
            add_subplot(ax[1, 1], sp_iptr[i])  # title="SP original"

    if interpolate_outliers:  # remove outliers (slow...)
        print(f"Outliers are{'' if interpolate_outliers else ' not'} interpolated.")
        sp_iptr = np.array([remove_outliers_sliding_window(image=data_part, kernel_size=7, n_std=3.) for data_part in sp_iptr])
        sp_iptr_hmilike = np.array([remove_outliers_sliding_window(image=data_part, kernel_size=7, n_std=3.) for data_part in sp_iptr_hmilike])

        sp_iptr[index_continuum] = remove_limb_darkening_approx(sp_iptr[index_continuum])
        sp_iptr_hmilike[index_continuum] = remove_limb_darkening_approx(sp_iptr_hmilike[index_continuum])

    if control_plots:
        for i, ax in enumerate(axes.values()):
            add_subplot(ax[0, 2], sp_iptr_hmilike[i])  # title="HMI like"
            add_subplot(ax[1, 2], sp_iptr[i])  # title="SP original"

        for fig in figures.values():
            mng = fig.canvas.manager
            mng.resize(*mng.window.maxsize())
            fig.tight_layout()

        plt.show()

    sp_iptr = np.expand_dims(np.transpose(sp_iptr, (1, 2, 0)), axis=0)
    sp_iptr_hmilike = np.expand_dims(np.transpose(sp_iptr_hmilike, (1, 2, 0)), axis=0)

    _, nr, nc, nq = np.shape(sp_iptr)
    save_data(SP_filename.replace(".fits", ".npz"), observations=np.zeros((0, nr, nc, nq)),
              labels=sp_iptr, subfolder=_path_sp_hmilike,
              other_info={f"{_observations_name}_simulated": np.array(sp_iptr_hmilike, dtype=_wp),
                          "units": ["quiet-Sun normalised", "G", "G", "G"],
                          "direction": [None, "+W", "+S", "-grav"]})

    return sp_iptr, sp_iptr_hmilike


def add_subplot(axis, x, title: str | None = None) -> None:
    import matplotlib.pyplot as plt

    y_max, x_max = np.shape(x)
    im = axis.imshow(x, origin="lower", extent=[0, x_max, 0, y_max], aspect="auto")

    if title is not None:
        axis.set_title(title)

    divider = make_axes_locatable(axis)
    cax = divider.append_axes(position="right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)


def split_to_patches(hmi_full: np.ndarray | None,
                     hmilike_full: np.ndarray,
                     sp_full: np.ndarray,
                     patch_size: int | None = None,
                     cut_edge_px: int = 0) -> tuple[np.ndarray, ...]:
    if patch_size is None: patch_size = conf_grid_setup["patch_size"]
    used_quantities = np.array([True, True, True, True])

    _, nr, nc, _ = np.shape(sp_full)

    if hmi_full is None:
        hmi_full = np.zeros((0, nr, nc, np.sum(used_quantities)))

    hmi_patchs = split_data_to_patches(hmi_full[:, cut_edge_px:nr-cut_edge_px, cut_edge_px:nc-cut_edge_px, :],
                                       patch_size=patch_size, used_quantities=used_quantities)
    hmilike_patches = split_data_to_patches(hmilike_full[:, cut_edge_px:nr-cut_edge_px, cut_edge_px:nc-cut_edge_px, :],
                                            patch_size=patch_size, used_quantities=used_quantities)
    sp_patches = split_data_to_patches(sp_full[:, cut_edge_px:nr-cut_edge_px, cut_edge_px:nc-cut_edge_px, :],
                                       patch_size=patch_size, used_quantities=used_quantities)

    return hmi_patchs, hmilike_patches, sp_patches


def combine_data_to_patches(data_folder: str, patch_size: int | None = None,
                            output_name: str | None = None,
                            cut_edge_px: int = 0) -> None:
    print(f"Combining aligned HMI and SP data and splitting them into the final patches...")
    if patch_size is None: patch_size = conf_grid_setup["patch_size"]
    if output_name is None:
        if data_folder == _path_sp_hmi:
            output_name = "SP_HMI_aligned.npz"
        elif _path_sp_hmilike in data_folder:
            output_name = "SP_HMI-like.npz"
        else:
            output_name = "patched_data.npz"

    patched = True if "patches" in data_folder else False

    stored_files = sorted(glob(f"{path.join(data_folder, '')}*.npz"))

    hmi = len(stored_files) * [0]
    sp_hmi_like = len(stored_files) * [0]
    sp = len(stored_files) * [0]
    final_files = []
    used_quantities = np.array([True, True, True, True])

    for i, file in tqdm(enumerate(stored_files)):
        data = load_npz(file)
        if i == 0:
            initial_unit = data["units"][1]

        if patched:
            hmi[i], sp_hmi_like[i], sp[i] = data[_observations_name], data[f"{_observations_name}_simulated"], data[_label_name]
        else:
            if i == 0:
                print(f"Patch size: {patch_size} px")
                print(f"Cut edge: {cut_edge_px} px")

            hmi[i], sp_hmi_like[i], sp[i] = split_to_patches(hmi_full=data[_observations_name],
                                                             hmilike_full=data[f"{_observations_name}_simulated"],
                                                             sp_full=data[_label_name],
                                                             patch_size=patch_size,
                                                             cut_edge_px=cut_edge_px)
        data.close()

        final_files.extend([{"filename": file, "patch_index": j} for j in range(len(sp[i]))])

    hmi, sp, sp_hmi_like = np.array(stack(hmi, axis=0)), np.array(stack(sp, axis=0)), np.array(stack(sp_hmi_like, axis=0))
    final_files = np.array(final_files, dtype=object)

    # HMI may be empty which causes problems during saving (n * (0, 72, 72, 4) is (1, 0) after stack)
    if is_empty(hmi):
        hmi = np.zeros((0, patch_size, patch_size, np.sum(used_quantities)))

    # G -> _b_unit (kG) to have values close to 1
    hmi = convert_unit(array=hmi, initial_unit=initial_unit, final_unit=_b_unit, used_quantities=used_quantities)
    sp = convert_unit(array=sp, initial_unit=initial_unit, final_unit=_b_unit, used_quantities=used_quantities)
    sp_hmi_like = convert_unit(array=sp_hmi_like, initial_unit=initial_unit, final_unit=_b_unit, used_quantities=used_quantities)

    check_dir(output_name, is_file=True)
    save_data(final_name=output_name, observations=hmi, labels=sp, order="k", subfolder=_path_data,
              other_info={f"{_observations_name}_simulated": np.array(sp_hmi_like, dtype=_wp),
                          "units": ["quiet-Sun normalised", _b_unit, _b_unit, _b_unit],
                          "direction": [None, "+W", "+S", "-grav"],
                          "patch_size": patch_size,
                          "cut_edge_px": cut_edge_px,
                          "patch_identification": final_files})

    print(f"Total number of patches: {len(sp)}")


def remove_sp_data_with_different_resolution(SP_filenames_to_delete: np.ndarray) -> None:
    for name_to_delete in SP_filenames_to_delete:
        os.remove(path.join(_path_sp, name_to_delete))
        os.remove(path.join(_path_sp, name_to_delete.replace(".fits", "_L2.1.fits")))


def plot_sp_to_check(SP_filenames: list[str] | None = None,
                     image_start: int = 0,
                     image_num: int | None = None,
                     backend: str = "Agg") -> None:
    mpl.use(backend=backend)
    from matplotlib import pyplot as plt
    from modules._constants import _path_figures

    if SP_filenames is None:
        SP_filenames = sorted([sp_file for sp_file in os.listdir(_path_sp) if "L2.1" not in sp_file])

    if image_num is None:  # plot all
        image_num = len(SP_filenames) - image_start

    savefig_kwargs = {"bbox_inches": "tight",
                      "pad_inches": 0.05,
                      "dpi": 100}

    for ii in range(image_start, image_start + image_num):
        sp = read_sp(SP_filenames[ii], coordinates="ptr_native")

        fig, ax = plt.subplots(2, 2, figsize=(16, 12))
        ax = np.ravel(ax)

        for iax in range(len(ax)):
            if iax == 0:
                x = remove_limb_darkening_approx(sp[iax])
            else:
                x = sp[iax]
            y_max, x_max = np.shape(x)
            im = ax[iax].imshow(x, origin="lower", extent=[0, x_max, 0, y_max], aspect="auto")
            divider = make_axes_locatable(ax[iax])
            cax = divider.append_axes(position="right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax)

        plt.suptitle(SP_filenames[ii])

        if backend in ["TkAgg", "Qt5Agg", "GTK3Agg", "WXAgg"]:
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
        plt.tight_layout()

        if backend in ["TkAgg", "Qt5Agg", "GTK3Agg", "WXAgg"]:
            plt.show()

        fig_name = SP_filenames[ii].replace(".fits", ".jpg")
        fig_full_name = path.join(_path_figures, "HMI_to_SOT", "SP_check", fig_name)
        check_dir(fig_full_name, is_file=True)
        fig.savefig(fig_full_name, format="jpg", **savefig_kwargs)
        plt.close(fig)


def pipeline_alignment(skip_jsoc_query: bool = False,
                       coordinates: Literal["ptr", "ptr_native"] = "ptr_native",
                       interpolate_outliers: bool = True,
                       subpixel_shift: bool = True,
                       patch_size: int | None = None,
                       cut_edge_px: int = 0,
                       output_name: str = "SP_HMI_aligned.npz",
                       control_plots: bool = False,
                       alignment_plots: bool = True,
                       do_parallel: bool = False) -> None:
    print("Registering HMI data to SP")

    if patch_size is None or patch_size <= 0: patch_size = conf_grid_setup["patch_size"]

    completeness_check = check_SP_completeness(sp_path=_path_sp)
    if not completeness_check:
        raise ValueError("L2.0 and L2.1 datasets for Hinode/SOT-SP are not one-to-one.")

    SP_filenames = sorted([sp_file for sp_file in os.listdir(_path_sp) if "L2.1" not in sp_file])

    if not skip_jsoc_query:
        for SP_filename in tqdm(SP_filenames):
            try:
                jsoc_query_from_sp_name(SP_filename, quantity="I", data_type="")
                jsoc_query_from_sp_name(SP_filename, quantity="B", data_type="")
            except Exception:  # sometimes got drms.exceptions.DrmsExportError:  [status=4]
                print(traceback.format_exc())
                continue

    resolution_check = np.unique(stack((check_resolution("SP"), check_resolution("HMI"))))
    if np.size(resolution_check) > 0:
        print(np.array(SP_filenames)[resolution_check])
        raise ValueError("Remove SP and/or HMI data with different resolution")

    #
    # HERE SHOULD BE IDL RUN FOR coordinates="ptr_native"
    #

    def process_file(SP_filename: str) -> None:
        # necessary function for parallelization
        try:
            hmi_iptr, sp_iptr, sp_iptr_hmilike = align_SP_file(SP_filename=SP_filename, coordinates=coordinates,
                                                               index_continuum=0, align_rules={0: 0, 1: 0, 2: 0, 3: 0},
                                                               subpixel_shift=subpixel_shift, interpolate_outliers=interpolate_outliers,
                                                               control_plots=control_plots, quiet=False)
            if alignment_plots:
                from modules.control_plots import plot_alignment
                plot_alignment(hmi_iptr, sp_iptr_hmilike, sp_iptr, suf=f"_{SP_filename.replace('.fits', '')}",
                               subfolder="HMI_aligned")

        except Exception:
            # Handle exceptions and print the traceback
            # alignment of the SP and HMI data is not perfect (coordinates in header are often not precise enough)
            # possible missing HMI data due to the drms.exceptions.DrmsExportError:  [status=4]
            print(traceback.format_exc())

    if do_parallel:
        # !!!
        # NEFUNGUJE
        # AttributeError: Can't pickle local object 'pipeline_alignment.<locals>.process_file'
        max_workers = 4
        # Choose ThreadPoolExecutor for I/O-bound tasks or ProcessPoolExecutor for CPU-bound tasks
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(process_file, SP_filenames), total=len(SP_filenames)))
    else:
        for SP_filename in tqdm(SP_filenames):
            process_file(SP_filename)

    # you should check the control plots and remove the cubes that are poorly matched or with other artifacts...
    combine_data_to_patches(data_folder=_path_sp_hmi, patch_size=patch_size, cut_edge_px=cut_edge_px,
                            output_name=output_name)

    """
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    index_to_show = 0

    imagelist = [hmi_iptr[index_to_show], sp_iptr[index_to_show]]

    rate = 1 * 1000

    fig = plt.figure()
    im = plt.imshow(imagelist[0], origin="lower", cmap="gray")

    # function to make blinking figure
    def updatefig(j: int):
        im.set_array(imagelist[j])
        return [im]

    ani = animation.FuncAnimation(fig, updatefig, frames=len(imagelist), interval=rate, blit=True)
    plt.show()
    """


def pipeline_sp_to_hmilike(coordinates: Literal["ptr", "ptr_native"] = "ptr_native",
                           interpolate_outliers: bool = True,
                           patch_size: int | None = None,
                           cut_edge_px: int = 0,
                           output_name: str = "SP_HMI-like.npz",
                           control_plots: bool = False,
                           alignment_plots: bool = True,
                           do_parallel: bool = False) -> None:
    print("Blurring SP data to match HMI's resolution")

    if patch_size is None or patch_size <= 0: patch_size = conf_grid_setup["patch_size"]

    completeness_check = check_SP_completeness(sp_path=_path_sp)
    if not completeness_check:
        raise ValueError("L2.0 and L2.1 datasets for Hinode/SOT-SP are not one-to-one.")

    SP_filenames = sorted([sp_file for sp_file in os.listdir(_path_sp) if "L2.1" not in sp_file])

    resolution_check = np.unique(check_resolution("SP"))
    if np.size(resolution_check) > 0:
        sp_to_delete = np.array(SP_filenames)[resolution_check]
        print(sp_to_delete)
        raise ValueError("Remove SP data with different resolution")
        # remove_sp_data_with_different_resolution(SP_filenames_to_delete=sp_to_delete)

    def process_file(SP_filename: str) -> None:
        # necessary function for parallelization
        try:
            sp_iptr, sp_iptr_hmilike = sp_to_hmilike(SP_filename=SP_filename, coordinates=coordinates,
                                                     index_continuum=0, interpolate_outliers=interpolate_outliers,
                                                     control_plots=control_plots, quiet=False)
            if alignment_plots:
                from modules.control_plots import plot_alignment
                plot_alignment(sp_iptr_hmilike, sp_iptr, suf=f"_{SP_filename.replace('.fits', '')}",
                               subfolder="HMI-like")
        except Exception:
            # Handle exceptions and print the traceback
            print(traceback.format_exc())

    if do_parallel:
        # !!!
        # NEFUNGUJE
        # AttributeError: Can't pickle local object 'pipeline_alignment.<locals>.process_file'
        max_workers = 4
        # Choose ThreadPoolExecutor for I/O-bound tasks or ProcessPoolExecutor for CPU-bound tasks
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(process_file, SP_filenames), total=len(SP_filenames)))
    else:
        for SP_filename in tqdm(SP_filenames):
            process_file(SP_filename)

    # you should check the control plots and remove the cubes that contain various artifacts...
    combine_data_to_patches(data_folder=_path_sp_hmilike, patch_size=patch_size, cut_edge_px=cut_edge_px,
                            output_name=output_name)


def pipeline_sp_to_hmilike_v2(coordinates: Literal["ptr", "ptr_native"] = "ptr_native",
                              interpolate_outliers: bool = True,
                              patch_size: int | None = None,
                              cut_edge_px: int = 0,
                              output_name: str = "SP_HMI-like.npz",
                              control_plots: bool = False,
                              alignment_plots: bool = True,
                              do_parallel: bool = False) -> None:
    print("Blurring SP data to match HMI's resolution and add noise to each patch separately")

    if patch_size is None or patch_size <= 0: patch_size = conf_grid_setup["patch_size"]

    completeness_check = check_SP_completeness(sp_path=_path_sp)
    if not completeness_check:
        raise ValueError("L2.0 and L2.1 datasets for Hinode/SOT-SP are not one-to-one.")

    SP_filenames = sorted([sp_file for sp_file in os.listdir(_path_sp) if "L2.1" not in sp_file])

    resolution_check = np.unique(check_resolution("SP"))
    if np.size(resolution_check) > 0:
        sp_to_delete = np.array(SP_filenames)[resolution_check]
        print(sp_to_delete)
        raise ValueError("Remove SP data with different resolution")
        # remove_sp_data_with_different_resolution(SP_filenames_to_delete=sp_to_delete)

    def process_file(SP_filename: str) -> None:
        # necessary function for parallelization
        try:
            sp_iptr, sp_iptr_hmilike = sp_to_hmilike_patches(SP_filename=SP_filename,
                                                             coordinates=coordinates,
                                                             index_continuum=0,
                                                             interpolate_outliers=interpolate_outliers,
                                                             patch_size=patch_size,
                                                             cut_edge_px=cut_edge_px,
                                                             control_plots=control_plots,
                                                             quiet=False)
            if alignment_plots:
                from modules.control_plots import plot_alignment
                plot_alignment(sp_iptr_hmilike, sp_iptr, suf=f"_{SP_filename.replace('.fits', '')}",
                               subfolder="HMI-like")
        except Exception:
            # Handle exceptions and print the traceback
            print(traceback.format_exc())

    for SP_filename in tqdm(SP_filenames):
        process_file(SP_filename)

    combine_data_to_patches(data_folder=path.join(_path_sp_hmilike, "patches"), output_name=output_name)


if __name__ == "__main__":
    exact_hmi = False
    mean_hmi_resolution = calc_hmi_resolution(method="median")

    if exact_hmi:
        pipeline_alignment(skip_jsoc_query=False,
                           coordinates="ptr_native",
                           interpolate_outliers=True,
                           subpixel_shift=True,
                           cut_edge_px=8,
                           control_plots=False,
                           alignment_plots=False,
                           do_parallel=False)
    else:
        pipeline_sp_to_hmilike_v2(coordinates="ptr_native",
                                  interpolate_outliers=True,
                                  cut_edge_px=8,
                                  control_plots=False,
                                  alignment_plots=False,
                                  do_parallel=False)
