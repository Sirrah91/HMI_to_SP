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
                  header,
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
    primary_hdu.header = header
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
            "WCSNAME": "Stonyhurst heliographic",
        },
        {
            "data": lon,
            "name": "Longitude",
            "quantity": "Longitude",
            "unit": "deg",
            "direct": "+W",
            "WCSNAME": "Stonyhurst heliographic",
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


def end_to_end_evaluate(data_dir: str,
                        output_dir: str | None = None,
                        data_type: Literal["fits", "sav", "auto"] = "auto",
                        used_quantities_str: str = "iptr",
                        remove_limb_dark: bool = True,
                        disambiguate: bool = True,
                        interp_outliers: bool = False,
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

        data, lon, lat, header = prepare_hmi_data(**fits_dict,
                                                  remove_limb_dark=remove_limb_dark,
                                                  disambiguate=disambiguate,
                                                  interpolate_outliers=interp_outliers)

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
                          header=header,
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
                          header=header,
                          used_quantities=used_quantities,
                          b_unit=b_unit,
                          output_file=filename_fits)


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
                     used_B_units="G",
                     max_valid_size=256)
    """

    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--data_type", type=str, default="auto")
    parser.add_argument("--used_quantities", type=str, default="iptr")
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
                        interp_outliers=args.interp_outliers,
                        b_unit=args.used_B_units,
                        max_valid_size=args.max_valid_size,
                        )
