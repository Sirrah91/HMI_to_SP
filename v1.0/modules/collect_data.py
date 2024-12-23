from modules.NN_config import conf_model_setup
from modules.NN_evaluate import process_patches
from modules.utilities_data import (filter_files, load_npz, disambigue_azimuth, ar_info, convert_unit,
                                    tuple_coordinate_to_str, str_coordinate_to_tuple, data_b2ptr,
                                    gimme_used_from_name, rot_coordinates_to_NW, read_cotemporal_fits)
from modules.utilities import check_dir, interpolate_outliers_median, remove_if_exists, plot_me
from modules.align_data import calc_hmi_to_sp_resolution, normalize_intensity
from modules._constants import _path_hmi, _path_data, _wp, _sep_out, _sep_in, _data_dir, _model_config_file, _b_unit
from modules._base_models import load_base_models

from copy import deepcopy
import ast
from os import path, listdir
import numpy as np
import scipy.io as sio
import traceback
# import concurrent.futures
from functools import partial
import time
from datetime import datetime, timedelta
import warnings
import drms
from tqdm import tqdm
from typing import Literal
from glob import glob
from astropy.io import fits
from skimage.transform import resize


def jsoc_query_simple(obs_date: str, quantity: str,
                      locunits: Literal["arcsec", "pixels", "stony", "carrlong"] = "stony",
                      locref: tuple[float, float] | list[float] | np.ndarray | str = (0., 0.),
                      t_ref: str | None = None,
                      boxunits: Literal["pixels", "arcsec", "degrees"] = "pixels",
                      boxsize: tuple[int, int] | list[int] | np.ndarray | int = (512, 512),
                      outdir_prefix: str = "") -> str:

    warnings.filterwarnings("ignore")

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
    margin_time = margin_frames * 720. / 60. if quantity in ["I", "continuum", "intensity"] else margin_frames * 720. / 60.

    if "T" not in obs_date:
        start_obs = f"{obs_date}T00:00:00.000"
    else:
        start_obs = obs_date
        obs_date = obs_date.split("T")[0]
    start_obs = datetime.strptime(start_obs, "%Y-%m-%dT%H:%M:%S.%f")
    start_obs -= timedelta(minutes=margin_time)

    end_obs = start_obs + timedelta(days=1, minutes=2*margin_time)

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

    if quantity in ["I", "continuum", "intensity"]:  # Ic data duration@lagImages
        query_str = f"hmi.ic_45s[{date_str_start}_{time_str_start}_TAI/{obs_length}h@45s]{{continuum}}"
    elif quantity in ["B"]:  # magnetic field vector data
        query_str = f"hmi.B_720s[{date_str_start}_{time_str_start}_TAI/{obs_length}h@720s]{{field,inclination,azimuth,disambig}}"
        # query_str = f"hmi.m_45s[{date_str_start}_{time_str_start}_TAI/{obs_length}h@45s]{{magnetogram}}"  # LOS magnetic field data
    else:  # magnetic field component
        query_str = f"hmi.B_720s[{date_str_start}_{time_str_start}_TAI/{obs_length}h@720s]{{{quantity}}}"
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


def download_hmi_data(ar_number: int | None, obs_start: str | None = None, obs_end: str | None = None,
                      stony_ref: tuple[float, float] | str | None = None, t_ref: str | None = None,
                      size_px: tuple[int, int] | int = (512, 512), skip_already_done: bool = False) -> None:

    already_done = False  # can be changed if ar_number is set

    if ar_number is not None:
        ar_dict = ar_info(ar_number=ar_number)

        if obs_start is None:
            obs_start = ar_dict["t_start"]

        if obs_end is None:
            obs_end = ar_dict["t_end"]

        if stony_ref is None:
            stony_ref = ar_dict["location"]

        if t_ref is None:
            t_ref = ar_dict["t_ref"]

        already_done = ar_dict["downloaded"]

    if skip_already_done and already_done:
        print(f"Active region {ar_number} was already processed. Skipping it.")
        return

    # Convert the string dates to datetime objects
    start_date_dt = datetime.strptime(obs_start, "%Y-%m-%d")
    if obs_end is None:
        end_date_dt = start_date_dt + timedelta(days=1)
    else:
        end_date_dt = datetime.strptime(obs_end, "%Y-%m-%d")

    date_list = []
    current_date = start_date_dt
    while current_date <= end_date_dt:
        date_list.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)

    if isinstance(stony_ref, str):
        stony_ref = str_coordinate_to_tuple(stony_ref)

    stony_ref = np.array(stony_ref)

    for obs_date in date_list:
        print(f"Downloading data for {obs_date}...")

        for quantity in ["I", "B"]:
            jsoc_query_simple(obs_date=obs_date, quantity=quantity,
                              locunits="stony", locref=stony_ref, t_ref=t_ref,
                              boxunits="pixels", boxsize=size_px,
                              outdir_prefix=f"AR{_sep_in}{ar_number}{_sep_out}")


def process_hmi_data(ar_number: int | None, obs_start: str | None = None, obs_end: str | None = None,
                     stony_ref: tuple[float, float] | str | None = None, t_ref: str | None = None,
                     coordinates: Literal["ptr", "ptr_native"] = "ptr_native",
                     model_names: list[str] | None = None,
                     skip_already_done: bool = False) -> None:

    if model_names is None:
        model_names = load_base_models(_model_config_file)

    already_done = False  # can be changed if ar_number is set

    if ar_number is not None:
        ar_dict = ar_info(ar_number=ar_number)

        if obs_start is None:
            obs_start = ar_dict["t_start"]

        if obs_end is None:
            obs_end = ar_dict["t_end"]

        if stony_ref is None:
            stony_ref = ar_dict["location"]

        if t_ref is None:
            t_ref = ar_dict["t_ref"]

        already_done = ar_dict["downloaded"]

    if skip_already_done and already_done:
        print(f"Active region {ar_number} was already processed. Skipping it.")
        return

    # Convert the string dates to datetime objects
    start_date_dt = datetime.strptime(obs_start, "%Y-%m-%d")
    if obs_end is None:
        end_date_dt = start_date_dt + timedelta(days=1)
    else:
        end_date_dt = datetime.strptime(obs_end, "%Y-%m-%d")

    date_list = []
    current_date = start_date_dt
    while current_date <= end_date_dt:
        date_list.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)

    stony_ref = stony_ref if isinstance(stony_ref, str) else tuple_coordinate_to_str(stony_ref)

    for obs_date in date_list:
        print(f"Processing data for {obs_date}...")
        hmi_fits_dir = path.join(_path_hmi, f"AR{_sep_in}{ar_number}{_sep_out}{obs_date.replace('-', '')}{_sep_out}{stony_ref}")
        filename = read_hmi_original(hmi_fits_dir, ar_number=ar_number, coordinates=coordinates)
        header_info = {"ar_num": ar_number,
                       "t_obs": obs_date,
                       "t_ref": t_ref,
                       "locunits": "stony",
                       "loc_ref": stony_ref,
                       "HMI_file": filename}

        resave_predictions_to_fits(filenames=[filename], info_data=[header_info], model_names=model_names,
                                   remove_npz_data=True)


def resave_predictions_to_fits(filenames: list[str], info_data: list[dict],
                               model_names: list[str] | None = None,
                               max_valid_size: int = 256,
                               remove_npz_data: bool = True) -> None:

    if model_names is None:
        model_names = load_base_models(_model_config_file)

    b_unit: Literal["kG", "G", "T", "mT"] = "G"

    outdir = "/nfsscratch/david/NN/results"
    check_dir(outdir)

    for filename, info in zip(filenames, info_data):
        print(f"Data file: {filename}")
        data = load_npz(filename)

        filename_fits = path.join(outdir, path.split(filename)[1].replace(".npz", ".fits"))

        obs_time = data["obs_time"]
        initial_b_unit = data["units"][-1]

        lat = np.clip(data["lat"], a_min=-90., a_max=90.) if "lat" in data.files else None
        lon = np.clip(data["lon"], a_min=-90., a_max=90.) if "lon" in data.files else None

        results = process_patches(image_4d=data["HMI"],
                                  model_names=model_names,
                                  initial_b_unit=initial_b_unit,
                                  final_b_unit=b_unit,
                                  max_valid_size=max_valid_size,
                                  kernel_size="auto",
                                  subfolder_model=conf_model_setup["model_subdir"])

        # Create a table HDU for the string array
        cols = [fits.Column(name="OBS_TIME", array=obs_time, format="27A")]
        hdu_table = fits.BinTableHDU.from_columns(cols)

        primary_hdu = fits.PrimaryHDU()

        image_hdu_i = fits.ImageHDU(results[:, :, :, 0], name="Ic")
        image_hdu_bp = fits.ImageHDU(results[:, :, :, 1], name="Bp")
        image_hdu_bt = fits.ImageHDU(-results[:, :, :, 2], name="Bt")  # - to have +N orientation
        image_hdu_br = fits.ImageHDU(results[:, :, :, 3], name="Br")

        image_hdu_lat = fits.ImageHDU(lat, name="Latitude")
        image_hdu_lon = fits.ImageHDU(lon, name="Longitude")

        hdu_table.header["EXTNAME"] = "OBS_TIME"

        for key, value in info.items():
            if key == "ar_number":
                key = "AR_num"

            primary_hdu.header[key] = str(value)
            hdu_table.header[key] = str(value)
            image_hdu_i.header[key] = str(value)
            image_hdu_bp.header[key] = str(value)
            image_hdu_bt.header[key] = str(value)
            image_hdu_br.header[key] = str(value)
            image_hdu_lat.header[key] = str(value)
            image_hdu_lon.header[key] = str(value)

        used_quantities = np.array([gimme_used_from_name(model_name) for model_name in model_names])
        model_indices, quantity_indices = np.where(used_quantities)

        hdu_table.header["QUANTITY"] = "Observation time"

        image_hdu_i.header["QUANTITY"] = "Continuum intensity"
        image_hdu_i.header["UNIT"] = "quiet sun normalised"
        image_hdu_i.header["NN_MODEL"] = model_names[model_indices[np.where(quantity_indices == 0)[0][-1]]]

        image_hdu_bp.header["QUANTITY"] = "Zonal magnetic field"
        image_hdu_bp.header["UNIT"] = b_unit
        image_hdu_bp.header["DIRECT"] = "+W"
        image_hdu_bp.header["NN_MODEL"] = model_names[model_indices[np.where(quantity_indices == 1)[0][-1]]]

        image_hdu_bt.header["QUANTITY"] = "Azimuthal magnetic field"
        image_hdu_bt.header["UNIT"] = b_unit
        image_hdu_bt.header["DIRECT"] = "+N"
        image_hdu_bt.header["NN_MODEL"] = model_names[model_indices[np.where(quantity_indices == 2)[0][-1]]]

        image_hdu_br.header["QUANTITY"] = "Radial magnetic field"
        image_hdu_br.header["UNIT"] = b_unit
        image_hdu_br.header["DIRECT"] = "-grav"
        image_hdu_br.header["NN_MODEL"] = model_names[model_indices[np.where(quantity_indices == 3)[0][-1]]]

        image_hdu_lat.header["QUANTITY"] = "Latitude"
        image_hdu_lat.header["UNIT"] = "deg"
        image_hdu_lat.header["DIRECT"] = "+N"
        image_hdu_lat.header["WCSNAME"] = "Stonyhurst"

        image_hdu_lon.header["QUANTITY"] = "Longitude"
        image_hdu_lon.header["UNIT"] = "deg"
        image_hdu_lon.header["DIRECT"] = "+W"
        image_hdu_lon.header["WCSNAME"] = "Stonyhurst"

        hdul = fits.HDUList([primary_hdu, image_hdu_i, image_hdu_bp, image_hdu_bt, image_hdu_br,
                             image_hdu_lat, image_hdu_lon, hdu_table])

        check_dir(filename_fits)
        print(f"Saving predictions to\n\t{filename_fits}")
        hdul.writeto(filename_fits, overwrite=True)

        if remove_npz_data:
            remove_if_exists(filename)


def read_hmi_original(subfolder_or_SP_filename: str, ar_number: int | None = None,
                      coordinates: Literal["ptr", "ptr_native"] = "ptr_native") -> str:
    hmi_dir = subfolder_or_SP_filename.replace(".fits", "")
    suffix = next((s for s in hmi_dir.split(path.sep)[::-1] if s), "unknown")
    if ar_number is not None and f"AR{_sep_in}{ar_number}{_sep_out}" not in suffix:
        prefix = f"AR{_sep_in}{ar_number}{_sep_out}"
    else:
        prefix = ""

    if ar_number is not None:
        final_name = path.join(_data_dir, "active_regions", f"{prefix}{suffix}.npz")
    else:
        final_name = path.join(_path_data, f"{prefix}{suffix}.npz")

    print(f"Resaving data to\n\t{final_name}")

    files = sorted([path.join(_path_hmi, hmi_dir, file) for file in listdir(path.join(_path_hmi, hmi_dir)) if "field" in file])
    files = filter_files(files)

    obs_time = np.array([fits.getheader(file, 1)["T_REC"] for file in files])

    nrows, ncols = np.shape(fits.getdata(files[0], 1))

    # resize to real HMI-to-Hinode fraction
    hmi_to_sp_rows, hmi_to_sp_cols = calc_hmi_to_sp_resolution(fast=True)
    output_shape = np.array(np.round((hmi_to_sp_rows * nrows, hmi_to_sp_cols * ncols)), dtype=int)

    hmi_b = np.zeros((3, nrows, ncols))
    HMI = np.zeros((len(files), *output_shape, 4))

    lon = np.zeros((len(files), *output_shape), dtype=np.float32)
    lat = np.zeros_like(lon)

    for ifile, file in enumerate(files):
        cotemporal_fits = read_cotemporal_fits(filename=file, check_uniqueness=True)

        sav_file = file.replace("field.fits", "ptr.sav")
        if coordinates == "ptr_native" and path.isfile(sav_file):
            data = sio.readsav(sav_file)
            hmi_b = data["bptr"]
            lonpart, latpart = data["lonlat"]

        else:
            if coordinates == "ptr_native":
                warnings.warn(f"The sav file for {sav_file} does not exist. Calculating B_ptr from fits.")

            field_fits = [fits_name for fits_name in cotemporal_fits if ".field" in fits_name]
            if not field_fits:
                raise ValueError("(At least) one of field fits is missing. Check your data.")
            index = fits.getheader(field_fits[0], 1)
            hmi_b[0] = fits.getdata(field_fits[0], 1)

            inclination_fits = [fits_name for fits_name in cotemporal_fits if ".inclination" in fits_name]
            if not inclination_fits:
                raise ValueError("(At least) one of inclination fits is missing. Check your data.")
            hmi_b[1] = fits.getdata(inclination_fits[0], 1)

            azimuth_fits = [fits_name for fits_name in cotemporal_fits if ".azimuth" in fits_name]
            if not azimuth_fits:
                raise ValueError("(At least) one of azimuth fits is missing. Check your data.")
            hmi_b[2] = fits.getdata(azimuth_fits[0], 1)

            disambig_fits = [fits_name for fits_name in cotemporal_fits if ".disambig" in fits_name]
            if not disambig_fits:
                raise ValueError("(At least) one of disambig fits is missing. Check your data.")
            disambig = np.array(fits.getdata(disambig_fits[0], 1), dtype=int)
            hmi_b[2] = disambigue_azimuth(hmi_b[2], disambig, method=1,
                                          rotated_image="history" in index and "rotated" in str(index["history"]))

            hmi_b, lonpart, latpart = data_b2ptr(index=index, bvec=hmi_b)

        lonpart = np.clip(lonpart, a_min=-90., a_max=90.)
        latpart = np.clip(latpart, a_min=-90., a_max=90.)
        lonpart = np.array(resize(lonpart, output_shape, anti_aliasing=True), dtype=_wp)
        latpart = np.array(resize(latpart, output_shape, anti_aliasing=True), dtype=_wp)
        lonpart = np.clip(lonpart, a_min=-90., a_max=90.)
        latpart = np.clip(latpart, a_min=-90., a_max=90.)

        lon_final = deepcopy(lonpart)
        lat_final = deepcopy(latpart)

        # rotate/flip the frames if needed (N in up and W in right)
        hmi_b = rot_coordinates_to_NW(longitude=lonpart, latitude=latpart, array_to_flip=hmi_b)
        lon[ifile] = rot_coordinates_to_NW(longitude=lonpart, latitude=latpart, array_to_flip=lon_final)
        lat[ifile] = rot_coordinates_to_NW(longitude=lonpart, latitude=latpart, array_to_flip=lat_final)

        # remove outliers (especially ptr_native sometimes diverges near the limb)
        # field above 10 kG is not allowed
        hmi_b = np.array([interpolate_outliers_median(b_part, kernel_size=3,
                                                      threshold_type="amplitude", threshold=10000.)
                          for b_part in hmi_b])

        # B in _b_unit (kG)
        hmi_b = np.transpose(np.array([resize(b_part, output_shape, anti_aliasing=True) for b_part in hmi_b]),
                             axes=(1, 2, 0))
        HMI[ifile, :, :, :] = convert_unit(hmi_b, initial_unit="G", final_unit=_b_unit)

        intensity_fits = [fits_name for fits_name in cotemporal_fits if ".continuum" in fits_name][0]
        intensity = normalize_intensity(resize(fits.getdata(intensity_fits, 1), output_shape, anti_aliasing=True))
        intensity[~np.isfinite(HMI[ifile, :, :, 1])] = np.nan  # remove disk edge
        HMI[ifile, :, :, 0] = intensity

    check_dir(final_name)
    with open(final_name, "wb") as f:
        np.savez_compressed(f, obs_time=obs_time, HMI=np.array(HMI, dtype=_wp), lon=lon, lat=lat,
                            units=np.array(["quiet-Sun normalised", _b_unit, _b_unit, _b_unit]),
                            direction=np.array([None, "+W", "+S", "-grav"]))

    return final_name


def pipeline_hmi_ar(ar_numbers: tuple[int, ...] = (11067, 11082, 11095, 11096, 11098, 11103, 11116, 11122, 11125, 11132,
                                                   11134, 11137, 11139, 11142, 11143, 11144, 11145, 11146, 11152, 11154,
                                                   11155, 11156, 11159, 11167, 11173, 11179, 11182, 11192, 11194, 11206,
                                                   11207, 11209, 11211, 11221, 11223, 11229, 11231, 11237, 11239, 11241,
                                                   11246, 11247, 11249, 11256, 11258, 11262, 11266, 11267, 11268, 11269,
                                                   11270, 11272, 11273, 11276, 11278, 11280, 11281, 11285, 11288, 11291,
                                                   11293, 11294),
                    skip_jsoc_query: bool = False,
                    coordinates: Literal["ptr", "ptr_native"] = "ptr_native",
                    model_names: list[str] | None = None) -> None:

    if model_names is None:
        model_names = load_base_models(_model_config_file)

    if not skip_jsoc_query:
        for ar_number in tqdm(ar_numbers):
            try:
                download_hmi_data(ar_number=ar_number)
            except Exception:
                print(traceback.format_exc())
                continue

    #
    # HERE SHOULD BE IDL RUN FOR coordinates="ptr_native"
    #

    for ar_number in tqdm(ar_numbers):
        try:
            process_hmi_data(ar_number=ar_number, coordinates=coordinates, model_names=model_names)
        except Exception:
            print(traceback.format_exc())
            continue


if __name__ == "__main__":
    ar_nums = (11067, 11082, 11095, 11096, 11098, 11103, 11116, 11122, 11125, 11132,
               11134, 11137, 11139, 11142, 11143, 11144, 11145, 11146, 11152, 11154,
               11155, 11156, 11159, 11167, 11173, 11179, 11182, 11192, 11194, 11206,
               11207, 11209, 11211, 11221, 11223, 11229, 11231, 11237, 11239, 11241,
               11246, 11247, 11249, 11256, 11258, 11262, 11266, 11267, 11268, 11269,
               11270, 11272, 11273, 11276, 11278, 11280, 11281, 11285, 11288, 11291,
               11293, 11294)

    model_names = load_base_models(_model_config_file)

    pipeline_hmi_ar(ar_numbers=ar_nums,
                    skip_jsoc_query=True,
                    coordinates="ptr_native",
                    model_names=model_names)
