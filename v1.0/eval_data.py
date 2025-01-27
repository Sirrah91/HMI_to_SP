from modules.utilities import check_dir, is_empty
from modules.utilities_data import read_cotemporal_fits, prepare_hmi_data
from modules.NN_evaluate import process_patches
from modules._constants import _model_config_file
from modules._base_models import load_base_models

from glob import glob
from os import path
import os
from typing import Literal
import re
import numpy as np
from astropy.io import fits
import argparse
import socket


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

    # Step 1: Add _dconANN to integration time
    modified_string = re.sub(r"(hmi\.[a-zA-Z]+_\d+)s\.", r"\1s_dconANN.", modified_string)

    # Step 3: Replace everything after "TAI" with ".fits"
    modified_string = re.sub(r"TAI.*$", "TAI.fits", modified_string)

    return modified_string


def generate_output_filename(filename: str,
                             used_quantities_str: str) -> str:
    """
    Generate the output filename based on the given path, filename, data type, and regex.

    Parameters:
        filename (str): The name of the file.
        used_quantities_str (str): The ordered string of used quantities.

    Returns:
        str: The generated output filename.
    """
    output_filename = path.split(filename)[1]

    output_filename = output_filename.replace(".ptr.sav", "")
    output_filename = output_filename.replace(".field.fits", "")
    output_filename = re.sub(r"\.\d+\.continuum\.fits", "", output_filename)

    if check_hmi_format(output_filename):
        output_filename = modify_hmi_string(output_filename, used_quantities_str)
    else:
        output_filename = f"{output_filename}.{used_quantities_str}_dconANN.fits"

    return output_filename


def write_to_fits(predictions: np.ndarray,
                  lon: np.ndarray,
                  lat: np.ndarray,
                  used_quantities: list[bool],
                  b_unit: Literal["kG", "G", "T", "mT"],
                  output_file: str = "output.fits") -> None:
    """
    Write model predictions and coordinate data to a FITS file efficiently.

    Parameters:
        predictions (ndarray): The model predictions with shape (1, variable, variable, 1-4).
        lon (ndarray): Longitude data.
        lat (ndarray): Latitude data.
        used_quantities (list): A list of booleans indicating which quantities to include.
        b_unit (str): The unit for the magnetic field quantities.
        output_file (str): Path to the output FITS file.
    """
    # Write the primary HDU to initialise the file
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.writeto(output_file, overwrite=True)  # Overwrite if the file exists

    # Metadata for quantities
    quantities_metadata = [
        {
            "used": used_quantities[0],
            "name": "Ic",
            "quantity": "Continuum intensity",
            "unit": "quiet-Sun normalised",
            "direct": None,
            "sign": 1.,
        },
        {
            "used": used_quantities[1],
            "name": "Bp",
            "quantity": "Zonal magnetic field",
            "unit": b_unit,
            "direct": "+W",
            "sign": 1.,
        },
        {
            "used": used_quantities[2],
            "name": "Bt",
            "quantity": "Azimuthal magnetic field",
            "unit": b_unit,
            "direct": "+N",
            "sign": -1.,  # -1 to have +N orientation
        },
        {
            "used": used_quantities[3],
            "name": "Br",
            "quantity": "Radial magnetic field",
            "unit": b_unit,
            "direct": "-grav",
            "sign": 1.,
        },
    ]

    # Append quantity HDUs to the FITS file
    index = 0
    with fits.open(output_file, mode="append") as hdul:
        for metadata in quantities_metadata:
            if metadata["used"]:
                image_hdu = fits.ImageHDU(metadata["sign"] * predictions[:, :, :, index], name=metadata["name"])
                image_hdu.header["QUANTITY"] = metadata["quantity"]
                image_hdu.header["UNIT"] = metadata["unit"]
                if metadata["direct"]:
                    image_hdu.header["DIRECT"] = metadata["direct"]
                hdul.append(image_hdu)  # Append the HDU to the file
                index += 1

    # Metadata for coordinates
    coordinates_metadata = [
        {
            "data": lat,
            "name": "Latitude",
            "quantity": "Latitude",
            "unit": "deg",
            "direct": "+N",
            "WCSNAME": "Stonyhurst",
        },
        {
            "data": lon,
            "name": "Longitude",
            "quantity": "Longitude",
            "unit": "deg",
            "direct": "+W",
            "WCSNAME": "Stonyhurst",
        },
    ]

    # Append latitude and longitude HDUs to the FITS file
    with fits.open(output_file, mode="append") as hdul:
        for coord in coordinates_metadata:
            image_hdu = fits.ImageHDU(coord["data"], name=coord["name"])
            image_hdu.header["QUANTITY"] = coord["quantity"]
            image_hdu.header["UNIT"] = coord["unit"]
            image_hdu.header["DIRECT"] = coord["direct"]
            image_hdu.header["WCSNAME"] = coord["WCSNAME"]
            hdul.append(image_hdu)  # Append the HDU to the file
            
    # Ensure the file has 644 permissions (read-write for owner, read-only for group/others)
    os.chmod(output_file, mode=0o644)


"""
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
        intensity_fits = cotemporal_fits["fits_ic"]
        if intensity_fits is None:
            raise ValueError("(At least) one of continuum fits is missing. Check your data.")
        index = fits.getheader(intensity_fits, 1)
        i = fits.getdata(intensity_fits)
    else:
        i = np.array([], dtype=_wp)

    if np.any(used_quantities[1:]):
        field_fits = cotemporal_fits["fits_b"]
        if field_fits is None:
            raise ValueError("(At least) one of field fits is missing. Check your data.")
        index = fits.getheader(field_fits, 1)
        b = fits.getdata(field_fits, 1)

        inclination_fits = cotemporal_fits["fits_inc"]
        if inclination_fits is None:
            raise ValueError("(At least) one of inclination fits is missing. Check your data.")
        bi = fits.getdata(inclination_fits)

        azimuth_fits = cotemporal_fits["fits_azi"]
        if azimuth_fits is None:
            raise ValueError("(At least) one of azimuth fits is missing. Check your data.")
        bg = fits.getdata(azimuth_fits, 1)

        if disambiguate:
            disambig_fits = cotemporal_fits["fits_disamb"]
            if disambig_fits is None:
                raise ValueError("(At least) one of disambig fits is missing. Check your data.")
            bgd = fits.getdata(disambig_fits, 1)
            bg = disambigue_azimuth(bg, bgd, method=1,
                                    rotated_image="history" in index and "rotated" in str(index["history"]))

        bptr = data_b2ptr(index=index, bvec=np.array([b, bi, bg], dtype=_wp))[used_quantities[1:]]
    else:
        bptr = np.array([], dtype=_wp)

    obs = stack((i, bptr), axis=0)
    lon, lat = return_lonlat(header=index)

    return obs, lon, lat


def end_to_end_evaluate(data_dir: str,
                        output_dir: str | None = None,
                        disambiguate: bool = False,
                        b_unit: Literal["kG", "G", "T", "mT"] = "G",
                        normalize_intensity: bool = True,
                        used_quantities_str: str = "iptr",
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

        if np.size(lonlat[0]) > 0 and np.size(lonlat[1]) > 0:
            # rotate/flip the frames if needed (N in up and W in left)
            data = rot_coordinates_to_NW(longitude=lonlat[0], latitude=lonlat[1], array_to_flip=data)
            lonlat = rot_coordinates_to_NW(longitude=lonlat[0], latitude=lonlat[1], array_to_flip=lonlat)

        predictions = process_patches(model_names=model_names, image_4d=data, kernel_size="auto", initial_b_unit=b_unit,
                                      final_b_unit=b_unit, max_valid_size=max_valid_size, subfolder_model="HMI_to_SOT")

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
            check_dir(filename_fits, is_file=True)
        except Exception:
            pass

        try:
            print(f"Saving predictions to\n\t{filename_fits}")
            hdul.writeto(filename_fits, overwrite=True)
        except Exception:
            print(f"Couldn't save predictions to {filename_fits}.")
            filename_fits = path.join("/nfsscratch/david/NN/results/temp", output_filename)
            check_dir(filename_fits, is_file=True)

            print(f"Saving predictions to\n\t{filename_fits}")
            hdul.writeto(filename_fits, overwrite=True)
"""


def end_to_end_evaluate(data_dir: str,
                        output_dir: str | None = None,
                        data_type: Literal["fits", "sav", "auto"] = "auto",
                        used_quantities_str: str = "iptr",
                        remove_limb_dark: bool = True,
                        disambiguate: bool = True,
                        interpolate_outliers: bool = True,
                        b_unit: Literal["kG", "G", "T", "mT"] = "G",
                        max_valid_size: int = 256,  # px x px
                        ) -> None:
    _model_names = load_base_models(_model_config_file)
    used_quantities = 4 * [False]

    used_quantities_str_ordered = ""
    if "i" in used_quantities_str:
        used_quantities_str_ordered = used_quantities_str_ordered + "i"
        used_quantities[0] = True
        regex = "*.continuum.fits"
    if "p" in used_quantities_str:
        used_quantities_str_ordered = used_quantities_str_ordered + "p"
        used_quantities[1] = True
        regex = "*.field.fits"
    if "t" in used_quantities_str:
        used_quantities_str_ordered = used_quantities_str_ordered + "t"
        used_quantities[2] = True
        regex = "*.field.fits"
    if "r" in used_quantities_str:
        used_quantities_str_ordered = used_quantities_str_ordered + "r"
        used_quantities[3] = True
        regex = "*.field.fits"

    if not any(used_quantities):
        raise ValueError("Invalid input: No quantities to process.")

    model_names = [model_name for i, model_name in enumerate(_model_names) if used_quantities[i]]

    filenames = sorted(glob(path.join(data_dir, regex)))

    for filename in filenames:
        fits_dict = read_cotemporal_fits(filename, check_uniqueness=True)

        # filter fits_dict using used_quantities
        if not used_quantities[0]:
            fits_dict["fits_ic"] = None
        if not any(used_quantities[1:]):
            fits_dict["fits_b"] = None
            fits_dict["fits_inc"] = None
            fits_dict["fits_azi"] = None
            fits_dict["fits_disamb"] = None

        data, lon, lat = prepare_hmi_data(**fits_dict,
                                          remove_limb_dark=remove_limb_dark,
                                          disambiguate=disambiguate,
                                          interpolate_outliers=interpolate_outliers)

        # Cut data to desired magnetic components
        if used_quantities[0] and any(used_quantities[1:]):  # [..., [ic, bp, bt, br]]
            data = data[..., np.array(used_quantities)]
        elif not used_quantities[0] and any(used_quantities[1:]):  # [..., [bp, bt, br]]
            data = data[..., np.array(used_quantities[1:])]

        data = process_patches(model_names=model_names, image_4d=data, kernel_size="auto", initial_b_unit=b_unit,
                               final_b_unit=b_unit, max_valid_size=max_valid_size, subfolder_model="HMI_to_SOT")

        output_filename = generate_output_filename(filename=filename, used_quantities_str=used_quantities_str_ordered)

        if is_empty(output_dir) is None:
            output_dir = data_dir
        else:
            # this allows both absolute path and relative path to the data_dir
            output_dir = path.join(data_dir, output_dir)

        filename_fits = path.join(output_dir, output_filename)
        try:
            check_dir(filename_fits, is_file=True)
        except Exception:
            pass

        try:
            print(f"Saving predictions to\n\t{filename_fits}")
            write_to_fits(predictions=data,
                          lon=lon,
                          lat=lat,
                          used_quantities=used_quantities,
                          b_unit=b_unit,
                          output_file=filename_fits)

        except Exception:
            print(f"Couldn't save predictions to {filename_fits}.")

            if path.isfile(output_filename):
                try:
                    print(f"Removing existing file in {filename_fits}.")
                    os.remove(output_filename)
                except Exception:
                    print(f"Couldn't remove existing file in {filename_fits}.")

            filename_fits = path.join("/nfsscratch/david/NN/results/temp", output_filename)
            check_dir(filename_fits, is_file=True)

            print(f"Saving predictions to\n\t{filename_fits}")
            write_to_fits(predictions=data,
                          lon=lon,
                          lat=lat,
                          used_quantities=used_quantities,
                          b_unit=b_unit,
                          output_file=filename_fits)

        """
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

        image_hdu_lat = fits.ImageHDU(lat, name="Latitude")
        image_hdu_lon = fits.ImageHDU(lon, name="Longitude")

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

        output_filename = path.split(filename)[1]

        if data_type == "sav":
            output_filename = output_filename.replace(".ptr.sav", "")
        elif data_type == "fits":
            if "field" in regex:
                output_filename = output_filename.replace(".field.fits", "")
            elif "continuum" in regex:
                output_filename = re.sub(r"\.\d+\.continuum\.fits", "", output_filename)

        if check_hmi_format(output_filename):
            output_filename = modify_hmi_string(output_filename, used_quantities_str_ordered)
        else:
            output_filename = f"{output_filename}.{used_quantities_str_ordered}.fits"

        if is_empty(output_dir) is None:
            output_dir = data_dir
        else:
            # this allows both absolute path and relative path to the data_dir
            output_dir = path.join(data_dir, output_dir)

        filename_fits = path.join(output_dir, output_filename)
        try:
            check_dir(filename_fits, is_file=True)
        except Exception:
            pass

        try:
            print(f"Saving predictions to\n\t{filename_fits}")
            hdul.writeto(filename_fits, overwrite=True)
        except Exception:
            print(f"Couldn't save predictions to {filename_fits}.")
            filename_fits = path.join("/nfsscratch/david/NN/results/temp", output_filename)
            check_dir(filename_fits, is_file=True)

            print(f"Saving predictions to\n\t{filename_fits}")
            hdul.writeto(filename_fits, overwrite=True)
        """


if __name__ == "__main__":
    hostname = socket.gethostname()
    print(f"Running on: {hostname}\n")

    """
    from argparse import Namespace
    args = Namespace(data_dir="/nfsscratch/david/NN/data/SDO_HMI_Marta/20210121",
                     output_dir="/nfsscratch/david/NN/results_Marta",
                     data_type="fits",
                     used_quantities="iptr",
                     remove_limb_dark=True,
                     disambiguate=True,
                     interp_outliers=True,
                     used_B_units="G",
                     max_valid_size=256)
    """

    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--data_type", type=str, default="auto")
    parser.add_argument("--used_quantities", type=str, default="ptr")
    parser.add_argument("--remove_limb_dark", action="store_true")
    parser.add_argument("--disambiguate", action="store_true")
    parser.add_argument("--interp_outliers", action="store_true")
    parser.add_argument("--used_B_units", type=str, default="G")
    parser.add_argument("--max_valid_size", type=int, default=256)

    args, _ = parser.parse_known_args()
    if not args.output_dir:
        args.output_dir = None

    end_to_end_evaluate(data_dir=args.data_dir,
                        output_dir=args.output_dir,
                        data_type=args.data_type,
                        used_quantities_str=args.used_quantities,
                        remove_limb_dark=args.remove_limb_dark,
                        disambiguate=args.disambiguate,
                        interpolate_outliers=args.interp_outliers,
                        b_unit=args.used_B_units,
                        max_valid_size=args.max_valid_size,
                        )
