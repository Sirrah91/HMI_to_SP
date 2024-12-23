from modules.align_data import calc_hmi_to_sp_resolution, normalize_intensity as ic_norm
from modules.utilities import check_dir, stack, interpolate_outliers_median, is_empty
from modules.utilities_data import (rot_coordinates_to_NW, disambigue_azimuth, data_b2ptr, return_lonlat,
                                    read_cotemporal_fits)
from modules.NN_evaluate import process_patches
from modules._constants import _wp, _model_config_file
from modules._base_models import load_base_models

from glob import glob
from os import path
from typing import Literal
import re
import numpy as np
import scipy.io as sio
from skimage.transform import resize
from astropy.io import fits
import argparse
import socket


def read_sav_data(filename: str, used_quantities: list[bool]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def return_coordinates(data) -> tuple[np.ndarray, np.ndarray]:
        lon = np.array([], dtype=_wp)
        lat = np.array([], dtype=_wp)

        if "lon" in data:
            lon = np.array(data["lon"], dtype=_wp)
        if "lat" in data:
            lat = np.array(data["lat"], dtype=_wp)
        if "lonlat" in data:
            lon, lat = np.array(data["lonlat"], dtype=_wp)
        if "latlon" in data:
            lat, lon = np.array(data["latlon"], dtype=_wp)

        return lon, lat

    if not np.any(used_quantities):
        raise ValueError("No quantity is used.")

    data = sio.readsav(filename)
    if used_quantities[0]:
        i = np.array(data["i"], dtype=_wp)
    else:
        i = np.array([], dtype=_wp)
    if np.any(used_quantities[1:]):
        bptr = np.array(data["bptr"], dtype=_wp)[used_quantities[1:]]
    else:
        bptr = np.array([], dtype=_wp)

    obs = stack((i, bptr), axis=0)

    lon, lat = return_coordinates(data)

    return obs, lon, lat


def read_fits_data(filename: str, used_quantities: list[bool],
                   disambiguate: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    cotemporal_fits = read_cotemporal_fits(filename=filename, check_uniqueness=True)

    if not np.any(used_quantities):
        raise ValueError("No quantity is used.")

    if used_quantities[0]:
        intensity_fits = [fits_name for fits_name in cotemporal_fits if ".continuum" in fits_name]
        if not intensity_fits:
            raise ValueError("(At least) one of continuum fits is missing. Check your data.")
        index = fits.getheader(intensity_fits[0], 1)
        i = fits.getdata(intensity_fits[0])
    else:
        i = np.array([], dtype=_wp)

    if np.any(used_quantities[1:]):
        field_fits = [fits_name for fits_name in cotemporal_fits if ".field" in fits_name]
        if not field_fits:
            raise ValueError("(At least) one of field fits is missing. Check your data.")
        index = fits.getheader(field_fits[0], 1)
        b = fits.getdata(field_fits[0], 1)

        inclination_fits = [fits_name for fits_name in cotemporal_fits if ".inclination" in fits_name]
        if not inclination_fits:
            raise ValueError("(At least) one of inclination fits is missing. Check your data.")
        bi = fits.getdata(inclination_fits[0])

        azimuth_fits = [fits_name for fits_name in cotemporal_fits if ".azimuth" in fits_name]
        if not azimuth_fits:
            raise ValueError("(At least) one of azimuth fits is missing. Check your data.")
        bg = fits.getdata(azimuth_fits[0], 1)

        if disambiguate:
            disambig_fits = [fits_name for fits_name in cotemporal_fits if ".disambig" in fits_name]
            if not disambig_fits:
                raise ValueError("(At least) one of disambig fits is missing. Check your data.")
            bgd = fits.getdata(disambig_fits[0], 1)
            bg = disambigue_azimuth(bg, bgd, method=1,
                                    rotated_image="history" in index and "rotated" in str(index["history"]))

        bptr = data_b2ptr(index=index, bvec=np.array([b, bi, bg], dtype=_wp))[used_quantities[1:]]
    else:
        bptr = np.array([], dtype=_wp)

    obs = stack((i, bptr), axis=0)
    lon, lat = return_lonlat(header=index)

    return obs, lon, lat


def check_hmi_format(string: str) -> bool:
    # Define the regex pattern based on your format
    pattern = r"^hmi\.[a-zA-Z]+_\d+s\.\d{8}_\d{6}_TAI.*$"

    # Match the string against the pattern
    match = re.match(pattern, string)

    # Return True if it matches, False otherwise
    return bool(match)


def modify_hmi_string(string: str, used_quantities_str: str) -> str:
    # Step 1: Replace the [a-zA-Z]+ part with used_quantities_str
    modified_string = re.sub(r"(?<=hmi\.)[a-zA-Z]+", used_quantities_str, string)

    # Step 2: Replace everything after "TAI" with ".fits"
    modified_string = re.sub(r"TAI.*$", "TAI.fits", modified_string)

    return modified_string


def end_to_end_evaluate(data_dir: str,
                        output_dir: str | None = None,
                        disambiguate: bool = False,
                        b_unit: Literal["kG", "G", "T", "mT"] = "G",
                        normalize_intensity: bool = True,
                        used_quantities_str: str = "ptr",
                        data_type: Literal["fits", "sav", "auto"] = "auto",
                        interpolate_outliers: bool = True,
                        max_valid_size: int = 256,  # px x px
                        ) -> None:
    _model_names = load_base_models(_model_config_file)
    used_quantities = 4 * [False]

    _used_quantities_str = ""
    if "i" in used_quantities_str:
        _used_quantities_str = _used_quantities_str + "i"
        used_quantities[0] = True
        regex = "*.continuum.fits"
    if "p" in used_quantities_str:
        _used_quantities_str = _used_quantities_str + "p"
        used_quantities[1] = True
        regex = "*.field.fits"
    if "t" in used_quantities_str:
        _used_quantities_str = _used_quantities_str + "t"
        used_quantities[2] = True
        regex = "*.field.fits"
    if "r" in used_quantities_str:
        _used_quantities_str = _used_quantities_str + "r"
        used_quantities[3] = True
        regex = "*.field.fits"

    model_names = [model_name for i, model_name in enumerate(_model_names) if used_quantities[i]]
    hmi_to_sp_rows, hmi_to_sp_cols = calc_hmi_to_sp_resolution(fast=True)

    if data_type == "sav":
        filenames = sorted(glob(path.join(data_dir, "*.sav")))
    elif data_type == "fits":
        filenames = sorted(glob(path.join(data_dir, regex)))
    elif data_type == "auto":  # fits first
        data_type = "fits"
        filenames = sorted(glob(path.join(data_dir, regex)))
        if not filenames:
            data_type = "sav"
            filenames = sorted(glob(path.join(data_dir, "*.sav")))

    for filename in filenames:
        if data_type == "sav":
            data, *lonlat = read_sav_data(filename=filename, used_quantities=used_quantities)
        else:
            data, *lonlat = read_fits_data(filename=filename, used_quantities=used_quantities, disambiguate=disambiguate)

        if used_quantities[0] and normalize_intensity:
            data[0] = ic_norm(data[0])
        if np.any(used_quantities[1:]) and interpolate_outliers:
            # field above 10 kG is not allowed
            data[int(used_quantities[0]):] = np.array([interpolate_outliers_median(data_part,
                                                                                   kernel_size=3,
                                                                                   threshold_type="amplitude",
                                                                                   threshold=10000.)
                                                       for data_part in data[int(used_quantities[0]):]])

        _, nrows, ncols = np.shape(data)
        output_shape = np.array(np.round((hmi_to_sp_rows * nrows, hmi_to_sp_cols * ncols)), dtype=int)
        data = np.array([resize(data_part, output_shape, anti_aliasing=True) for data_part in data])

        if used_quantities[0] and normalize_intensity:
            data[0] = ic_norm(data[0])
        if np.any(used_quantities[1:]) and interpolate_outliers:
            # field above 10 kG is not allowed
            data[int(used_quantities[0]):] = np.array([interpolate_outliers_median(data_part,
                                                                                   kernel_size=3,
                                                                                   threshold_type="amplitude",
                                                                                   threshold=10000.)
                                                       for data_part in data[int(used_quantities[0]):]])

        data = np.expand_dims(np.transpose(data, axes=(1, 2, 0)), axis=0)

        if np.size(lonlat[0]) > 0 and np.size(lonlat[1]) > 0:
            lonlat = np.clip(lonlat, a_min=-90., a_max=90.)
            lonlat = np.array([resize(data_part, output_shape, anti_aliasing=True) for data_part in lonlat])
            lonlat = np.clip(lonlat, a_min=-90., a_max=90.)

        predictions = process_patches(image_4d=data, model_names=model_names,
                                      initial_b_unit=b_unit, final_b_unit=b_unit,
                                      max_valid_size=max_valid_size, kernel_size="auto", subfolder_model="HMI_to_SOT")

        if np.size(lonlat[0]) > 0 and np.size(lonlat[1]) > 0:
            # rotate/flip the frames if needed (N in up and W in right)
            predictions = rot_coordinates_to_NW(longitude=lonlat[0], latitude=lonlat[1], array_to_flip=predictions)
            lonlat = rot_coordinates_to_NW(longitude=lonlat[0], latitude=lonlat[1], array_to_flip=lonlat)

        primary_hdu = fits.PrimaryHDU()
        hdu_list = [primary_hdu]
        index = 0

        if used_quantities[0]:
            image_hdu_i = fits.ImageHDU(predictions[:, :, :, index], name="Ic")
            image_hdu_i.header["QUANTITY"] = "Continuum intensity"
            image_hdu_i.header["UNIT"] = "quiet-Sun normalised"
            hdu_list.append(image_hdu_i)
            index += 1

        if used_quantities[1]:
            image_hdu_bp = fits.ImageHDU(predictions[:, :, :, index], name="Bp")
            image_hdu_bp.header["QUANTITY"] = "Zonal magnetic field"
            image_hdu_bp.header["UNIT"] = b_unit
            image_hdu_bp.header["DIRECT"] = "+W"
            hdu_list.append(image_hdu_bp)
            index += 1

        if used_quantities[2]:
            image_hdu_bt = fits.ImageHDU(-predictions[:, :, :, index], name="Bt")  # - to have +N orientation
            image_hdu_bt.header["QUANTITY"] = "Azimuthal magnetic field"
            image_hdu_bt.header["UNIT"] = b_unit
            image_hdu_bt.header["DIRECT"] = "+N"
            hdu_list.append(image_hdu_bt)
            index += 1

        if used_quantities[3]:
            image_hdu_br = fits.ImageHDU(predictions[:, :, :, index], name="Br")
            image_hdu_br.header["QUANTITY"] = "Radial magnetic field"
            image_hdu_br.header["UNIT"] = b_unit
            image_hdu_br.header["DIRECT"] = "-grav"
            hdu_list.append(image_hdu_br)
            index += 1

        if np.size(lonlat[0]) > 0 and np.size(lonlat[1]) > 0:
            image_hdu_lat = fits.ImageHDU(lonlat[1], name="Latitude")
            image_hdu_lon = fits.ImageHDU(lonlat[0], name="Longitude")

            image_hdu_lat.header["QUANTITY"] = "Latitude"
            image_hdu_lat.header["UNIT"] = "deg"
            image_hdu_lat.header["DIRECT"] = "+N"
            image_hdu_lat.header["WCSNAME"] = "Stonyhurst"

            image_hdu_lon.header["QUANTITY"] = "Longitude"
            image_hdu_lon.header["UNIT"] = "deg"
            image_hdu_lon.header["DIRECT"] = "+W"
            image_hdu_lon.header["WCSNAME"] = "Stonyhurst"

            hdu_list.append(image_hdu_lat)
            hdu_list.append(image_hdu_lon)

        hdul = fits.HDUList(hdu_list)

        output_filename = path.split(filename)[1].replace(".sav", "").replace(".field.fits", "")
        if check_hmi_format(output_filename):
            output_filename = modify_hmi_string(output_filename, _used_quantities_str)
        else:
            output_filename = f"{output_filename}.{_used_quantities_str}.fits"

        if is_empty(output_dir) is None:
            # output_dir = path.join(data_dir, "NN_predictions")
            output_dir = data_dir
        else:
            # this allows both absolute path and relative path to the data_dir
            output_dir = path.join(data_dir, output_dir)

        filename_fits = path.join(output_dir, output_filename)
        try:
            check_dir(filename_fits)
        except Exception:
            pass

        try:
            print(f"Saving predictions to\n\t{filename_fits}")
            hdul.writeto(filename_fits, overwrite=True)
        except Exception:
            print(f"Couldn't save predictions to {filename_fits}.")
            filename_fits = path.join("/nfsscratch/david/NN/results/temp", output_filename)
            check_dir(filename_fits)

            print(f"Saving predictions to\n\t{filename_fits}")
            hdul.writeto(filename_fits, overwrite=True)


if __name__ == "__main__":
    hostname = socket.gethostname()
    print(f"Running on: {hostname}\n")

    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--data_type", type=str, default="auto")
    parser.add_argument("--normalize_intensity", action="store_true")
    parser.add_argument("--disambiguate", action="store_true")
    parser.add_argument("--interp_outliers", action="store_true")
    parser.add_argument("--used_quantities", type=str, default="ptr")
    parser.add_argument("--used_B_units", type=str, default="G")
    parser.add_argument("--max_valid_size", type=int, default=256)

    args, _ = parser.parse_known_args()
    if not args.output_dir:
        args.output_dir = None

    end_to_end_evaluate(data_dir=args.data_dir,
                        data_type=args.data_type,
                        disambiguate=args.disambiguate,
                        b_unit=args.used_B_units,
                        normalize_intensity=args.normalize_intensity,
                        used_quantities_str=args.used_quantities,
                        interpolate_outliers=args.interp_outliers,
                        max_valid_size=args.max_valid_size,
                        output_dir=args.output_dir)
