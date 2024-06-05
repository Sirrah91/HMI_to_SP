import ast
from os import path, listdir
from glob import glob
import shutil
import numpy as np
from scipy.signal import convolve2d
import scipy.io as sio
import traceback
import time
from datetime import datetime, timedelta
import warnings
import drms
from tqdm import tqdm
from typing import Literal

from astropy.io import fits
from sunpy.time import TimeRange

from sklearn.model_selection import KFold
from skimage.measure import ransac
from skimage.feature import ORB, match_descriptors
from skimage.exposure import match_histograms
from skimage.transform import (resize, SimilarityTransform, EuclideanTransform, AffineTransform, ProjectiveTransform,
                               warp)
from skimage.registration import optical_flow_tvl1

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from modules.NN_data import split_data_to_patches, hmi_psf
from modules.NN_config import conf_grid_setup, conf_model_setup
from modules.NN_evaluate import evaluate
from modules.control_plots import plot_alignment
from modules.utilities_data import save_data, load_npz, rescale_intensity, load_xlsx
from modules.utilities import rmse, crop_nan, check_dir, stack, interpolate_mask, sliding_window
from modules.utilities import plot_me, check_dir
from modules._constants import (_path_sp, _path_hmi, _rnd_seed, _path_sp_hmi, _observations_name,
                                _label_name, _path_data, _num_eps, _wp, _sep_out, _data_dir)


def jsoc_query_from_sp_name(SP_filename: str, quantity: str) -> None:
    warnings.filterwarnings("ignore")

    # E-mail registered in JSOC
    email = "d.korda@seznam.cz"

    margin_box = 0.  # margin of box width/height in "boxunits" (0. to keep scale)
    # margin of timespan in minutes
    margin_frames = 5
    margin_time = margin_frames * 45. / 60. if quantity in ["I", "continuum", "intensity"] else margin_frames * 720. / 60.

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

    if quantity in ["I", "continuum", "intensity"]:
        query_str = f"hmi.ic_45s[{date_str_start}_{time_str_start}_TAI/{obs_length}h@45s]{{continuum}}"  # Ic data duration@lagImages
    else:
        query_str = f"hmi.B_720s[{date_str_start}_{time_str_start}_TAI/{obs_length}h@720s]{{field,inclination,azimuth,disambig}}"  # magnetic field vector data
        # query_str = f"hmi.m_45s[{date_str_start}_{time_str_start}_TAI/{obs_length}h@45s]{{magnetogram}}"  # LOS magnetic field data

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

    out_dir = path.join(_path_hmi, SP_filename.replace(".fits", ""))
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


def prepare_hmi_data(ar_number: int | None, obs_start: str | None = None, obs_end: str | None = None,
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

    # it is better to use fixed t_ref and coordinates at the t_ref...
    tracking_speed = 0.  # degrees per day

    if isinstance(stony_ref, str):
        stony_ref = str_coordinate_to_tuple(stony_ref)

    stony_ref = np.array(stony_ref)

    for obs_date in date_list:
        print(f"Processing data for {obs_date}...")

        jsoc_query_simple(obs_date=obs_date, quantity="I",
                          locunits="stony", locref=stony_ref, t_ref=t_ref,
                          boxunits="pixels", boxsize=size_px)
        outdir = jsoc_query_simple(obs_date=obs_date, quantity="B",
                                   locunits="stony", locref=stony_ref, t_ref=t_ref,
                                   boxunits="pixels", boxsize=size_px)

        # rescale HMI to Hinode pixel size and resave to npz
        read_hmi_original(outdir, ar_number=ar_number)

        # delete the fits
        shutil.rmtree(outdir)

        stony_ref[1] += tracking_speed

    # make predictions on the HMI data and save them to fits
    resave_predictions_to_fits(ar_numbers=[ar_number])


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


def jsoc_query_simple(obs_date: str, quantity: str,
                      locunits: Literal["arcsec", "pixels", "stony", "carrlong"] = "stony",
                      locref: tuple[float, float] | list[float] | np.ndarray | str = (0., 0.),
                      t_ref: str | None = None,
                      boxunits: Literal["pixels", "arcsec", "degrees"] = "pixels",
                      boxsize: tuple[int, int] | list[int] | np.ndarray | int = (512, 512)) -> str:

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
    margin_time = margin_frames * 45. / 60. if quantity in ["I", "continuum", "intensity"] else margin_frames * 720. / 60.

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

    if quantity in ["I", "continuum", "intensity"]:
        query_str = f"hmi.ic_720s[{date_str_start}_{time_str_start}_TAI/{obs_length}h@720s]{{continuum}}"  # Ic data duration@lagImages
    else:
        query_str = f"hmi.B_720s[{date_str_start}_{time_str_start}_TAI/{obs_length}h@720s]{{field,inclination,azimuth,disambig}}"  # magnetic field vector data

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

    out_dir = path.join(_path_hmi, f"{obs_date.replace('-', '')}{_sep_out}{locref_str}")
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


def resave_predictions_to_fits(ar_numbers: list[int], model_name: str = "weights_1111_20240513105441.h5"):

    n_splits = 5

    data_dir = "/nfsscratch/david/NN/data/datasets"
    outdir = "/nfsscratch/david/NN/results"
    check_dir(outdir)

    for AR_number in ar_numbers:
        info = ar_info(AR_number)
        del info["downloaded"]

        filenames = sorted(glob(path.join(data_dir, f"AR_{AR_number}*")))

        for filename in filenames:
            print(f"Data file: {filename}")
            data = load_npz(filename)

            filename = path.join(outdir, path.split(filename)[1].replace("npz", "fits"))

            obs_time = data["obs_time"]
            results = np.zeros((len(obs_time), 1024, 1024, 4), dtype=np.float32)

            indices = list(KFold(n_splits=n_splits).split(obs_time))

            for i_indices in range(n_splits):
                results[indices[i_indices][1]] = evaluate(model_names=[model_name],
                                                          filename_or_data=data["HMI"][indices[i_indices][1]],
                                                          proportiontocut=conf_model_setup["trim_mean_cut"],
                                                          params=conf_model_setup["params"],
                                                          subfolder_model=conf_model_setup["model_subdir"],
                                                          b_unit="G")

            # Create a table HDU for the string array
            cols = [fits.Column(name="OBSERVATION_TIME", array=obs_time, format="27A")]
            hdu_table = fits.BinTableHDU.from_columns(cols)
            primary_hdu = fits.PrimaryHDU(results)

            for key, value in info.items():
                if key == "ar_number": key = "AR_num"

                hdu_table.header[key] = str(value)
                primary_hdu.header[key] = str(value)

            hdu_table.header["QUANTITY"] = "[intensity, zonal field, meridional field, radial field]"
            hdu_table.header["UNITS"] = "[cont. norm., G, G, G]"
            primary_hdu.header["QUANTITY"] = "[intensity, zonal field, meridional field, radial field]"
            primary_hdu.header["UNITS"] = "[cont. norm., G, G, G]"

            hdul = fits.HDUList([primary_hdu, hdu_table])
            hdul.writeto(filename, overwrite=True)


def spherical_to_cartesian(r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([y, x, z])  # "poloidal" first


def cartesian_to_spherical(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(z / r)
    theta[~np.isfinite(theta)] = 0.
    phi = np.sign(y) * np.arccos(x / np.sqrt(x*x + y*y))
    phi[~np.isfinite(phi)] = 0.

    return np.array([r, theta, phi])


def hmi_sp_b2ptr(index, bvec: np.ndarray) -> np.ndarray:
    # Example usage:
    # index = fits.open('your_index_file.fits')[0].header
    # bvec = np.array([fits.open('field.fits')[0].data,
    #                  fits.open('inclination.fits')[0].data,
    #                  fits.open('azimuth.fits')[0].data])
    # bptr, lonlat = hmi_b2ptr(index, bvec)

    hmi = "RSUN_OBS" in index.keys()  # HMI data

    # Check dimensions
    nq, ny, nx = np.shape(bvec)
    if nq != 3 or nx != index["NAXIS1"] or ny != index["NAXIS2"]:
        raise ValueError("Dimension of bvec incorrect")

    # Convert bvec to B_xi, B_eta, B_zeta
    field = bvec[0, :, :]
    gamma = np.deg2rad(bvec[1, :, :])
    psi = np.deg2rad(bvec[2, :, :])

    b_xi = -field * np.sin(gamma) * np.sin(psi)
    b_eta = field * np.sin(gamma) * np.cos(psi)
    b_zeta = field * np.cos(gamma)

    # WCS conversion
    if hmi:
        origin_1, origin_2 = index["CRVAL1"], index["CRVAL2"]
        grid_1 = index["CDELT1"] * ((index["NAXIS1"] + 1) / 2 - index["CRPIX1"])
        grid_2 = index["CDELT2"] * ((index["NAXIS2"] + 1) / 2 - index["CRPIX2"])
        angle = np.deg2rad(index["CROTA2"])

        xcen = origin_1 + grid_1 * np.cos(angle) - (origin_2 + grid_2 * np.sin(angle))
        ycen = origin_1 + grid_1 * np.sin(angle) + (origin_2 + grid_2 * np.cos(angle))

        lon = np.arange(-index["NAXIS1"] // 2, index["NAXIS1"] // 2) * index["CDELT1"] + xcen
        lat = np.arange(-index["NAXIS2"] // 2, index["NAXIS2"] // 2) * index["CDELT2"] + ycen

        lon = np.rad2deg(np.arcsin(lon / index["RSUN_OBS"]))
        lat = np.rad2deg(np.arcsin(lat / index["RSUN_OBS"]))

    else:
        lon = np.arange(-index["NAXIS1"] // 2, index["NAXIS1"] // 2) * index["XSCALE"] + index["XCEN"]
        lat = np.arange(-index["NAXIS2"] // 2, index["NAXIS2"] // 2) * index["YSCALE"] + index["YCEN"]

        lon = np.rad2deg(np.arcsin(lon / index["SOLAR_RA"]))
        lat = np.rad2deg(np.arcsin(lat / index["SOLAR_RA"]))

    lon, lat = np.tile(lon, (len(lat), 1)), np.transpose(np.tile(lat, (len(lon), 1)))

    """
    from astropy.wcs import WCS
    wcs = WCS(index)
    y_coords, x_coords = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    lonlat = np.array(wcs.pixel_to_world(x_coords, y_coords))
    lonlat = np.rad2deg(np.arcsin(lonlat * 3600 / index["RSUN_OBS"]))
    """

    # Get matrix to convert
    if hmi:
        b = np.deg2rad(index["CRLT_OBS"])  # b-angle, disk center latitude
        p = np.deg2rad(-index["CROTA2"])  # p-angle, negative of CROTA2
    else:
        b = np.deg2rad(index["B_ANGLE"])
        p = np.deg2rad(-index["P_ANGLE"])

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
    bptr = np.zeros(np.shape(bvec))
    bptr[0, :, :] = k31 * b_xi + k32 * b_eta + k33 * b_zeta
    bptr[1, :, :] = k21 * b_xi + k22 * b_eta + k23 * b_zeta
    bptr[2, :, :] = k11 * b_xi + k12 * b_eta + k13 * b_zeta

    return bptr


def read_hmi_B(SP_filename: str, coordinates: Literal["cartesian", "ptr", "ptr_original"] = "cartesian") -> np.ndarray:
    if coordinates == "ptr_original":
        hmi_bptr = sorted(glob(path.join(_path_hmi, SP_filename.replace(".fits", ""), "*.bptr.sav")))
        return np.array([sio.readsav(hmi_bptr_part)["bptr"] for hmi_bptr_part in hmi_bptr])

    # inefficient (but correct in the sense of azimuth)
    hmi_b = sorted(glob(path.join(_path_hmi, SP_filename.replace(".fits", ""), f"*.field.fits")))
    hmi_bi = sorted(glob(path.join(_path_hmi, SP_filename.replace(".fits", ""), f"*.inclination.fits")))
    hmi_bg = sorted(glob(path.join(_path_hmi, SP_filename.replace(".fits", ""), f"*.azimuth.fits")))
    hmi_bgd = sorted(glob(path.join(_path_hmi, SP_filename.replace(".fits", ""), f"*.disambig.fits")))

    b = np.array([fits.getdata(hmi_b_part, 1) for hmi_b_part in hmi_b])
    bi = np.array([fits.getdata(hmi_bi_part, 1) for hmi_bi_part in hmi_bi])
    bg = np.array([fits.getdata(hmi_bg_part, 1) for hmi_bg_part in hmi_bg])
    bgd = np.array([fits.getdata(hmi_bgd_part, 1) for hmi_bgd_part in hmi_bgd], dtype=int)
    bgd = np.array([[[f"{num:03b}"[1] for num in row] for row in obs] for obs in bgd], dtype=float)

    bg += 180. * bgd

    if coordinates.lower() == "ptr":
        indices = [fits.getheader(hmi_b_part, 1) for hmi_b_part in hmi_b]
        return np.array([hmi_sp_b2ptr(indices[i], np.array([b[i], bi[i], bg[i]])) for i in range(len(hmi_b))])
    else:
        return np.array([spherical_to_cartesian(r=b[i], theta=np.deg2rad(bi[i]), phi=np.deg2rad(bg[i])) for i in range(len(hmi_b))])


def read_sp(SP_filename: str, coordinates: Literal["cartesian", "ptr", "ptr_original"] = "cartesian") -> np.ndarray:
    hdu = fits.open(path.join(_path_sp, SP_filename))  # SP level 2 Hinode
    hdu21 = fits.open(path.join(_path_sp, SP_filename.replace(".fits", "_L2.1.fits")))  # SP level 2.1 Hinode

    if coordinates == "ptr_original":
        return np.array([hdu[33].data, hdu21[2].data, -hdu21[3].data, hdu21[4].data])

    # + 90 to have the same zero point as HMI data
    azimuth = np.mod(hdu21[1].data + 90., 360.)

    # filling factor correction for level 2 data
    f = hdu[12].data
    f = np.clip(interpolate_mask(image=f, mask=np.logical_or(f <= 0., f > 1.),
                                 interp_nans=True, fill_value=_num_eps), _num_eps, 1.)

    # https://darts.isas.jaxa.jp/solar/hinode/data/sotsp_level2.html
    # continuum (arb.), |B| (G), B_inclination (deg), B_azimuth (deg)
    sp_observation = np.array([hdu[33].data, f * hdu[1].data, hdu[2].data, azimuth])

    if coordinates.lower() == "ptr":
        sp_observation[1:] = hmi_sp_b2ptr(hdu[0].header, sp_observation[1:])
    else:
        sp_observation[1:] = spherical_to_cartesian(r=sp_observation[1],
                                                    theta=np.deg2rad(sp_observation[2]),
                                                    phi=np.deg2rad(sp_observation[3]))

    return np.array(sp_observation)


def read_sp_raw(SP_filename: str) -> np.ndarray:
    hdu = fits.open(path.join(_path_sp, SP_filename))  # SP level 2 Hinode
    hdu21 = fits.open(path.join(_path_sp, SP_filename.replace(".fits", "_L2.1.fits")))  # SP level 2.1 Hinode

    # + 90 to have the same zero point as HMI data
    azimuth = np.mod(hdu[3].data + 90., 180.)
    azimuth_disambig = np.mod(hdu21[1].data + 90., 360.)

    # filling factor correction for level 2 data
    f = hdu[12].data
    f = np.clip(interpolate_mask(image=f, mask=np.logical_or(f <= 0., f > 1.),
                                 interp_nans=True, fill_value=_num_eps), _num_eps, 1.)

    # https://darts.isas.jaxa.jp/solar/hinode/data/sotsp_level2.html
    # continuum (arb.), |B| (G), B_inclination (deg), B_azimuth (deg)
    sp_observation = np.array([hdu[33].data, f * hdu[1].data, hdu[2].data, azimuth, azimuth_disambig])

    return np.array(sp_observation)


def convert_sp_to_Bptr(sp_observation: np.ndarray, hmi_sectors: np.ndarray, SP_filename: str,
                       coordinates: Literal["cartesian", "ptr", "ptr_original"] = "cartesian",
                       thresh: float = 200.) -> np.ndarray:
    result = np.array(sp_observation[:-1], copy=True)  # last index is disambiguated azimuth

    if coordinates == "ptr":  # cannot use the same inversion as for cartesian (non-trivial inversion problem)
        hdu = fits.open(path.join(_path_sp, SP_filename))  # SP level 2 Hinode

        # to disentangle azimuth ambiguity
        sp1 = np.array(sp_observation[:-1], copy=True)  # last index is disambiguated azimuth
        sp2 = np.array(sp_observation[:-1], copy=True)  # last index is disambiguated azimuth
        sp2[3] += 180.

        azimuth_disambig = sp_observation[-1]

        sp1[1:] = hmi_sp_b2ptr(hdu[0].header, sp1[1:])
        sp2[1:] = hmi_sp_b2ptr(hdu[0].header, sp2[1:])
        sp = hmi_sp_b2ptr(hdu[0].header, np.array([sp_observation[1], sp_observation[2], azimuth_disambig]))

        difference1 = np.abs(hmi_sectors[1:] - sp1[1:])
        difference2 = np.abs(hmi_sectors[1:] - sp2[1:])
        amb_solved = np.where(difference1 < difference2, sp1[1:], sp2[1:])  # for low-B regions

        result[1:] = np.where(np.logical_or(np.abs(sp1[1:] >= thresh), np.abs(sp2[1:] >= thresh)), sp, amb_solved)

    else:  # cartesian
        hmi_azimuth = np.mod(90. -
                             np.rad2deg(np.sign(hmi_sectors[2])
                                        * np.arccos(hmi_sectors[1] / (np.sqrt(hmi_sectors[1] ** 2 + hmi_sectors[2] ** 2)))),
                      360.)

        result[3][np.logical_and(90. <= np.abs(result[3] - hmi_azimuth), np.abs(result[3] - hmi_azimuth) <= 270.)] += 180.

        result[1:] = spherical_to_cartesian(r=result[1], theta=np.deg2rad(result[2]), phi=np.deg2rad(result[3]))

    return result


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
    hmi_sectors_quantity = np.empty_like(hdu[33].data)
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


def orbAlign(image1: np.ndarray, image2: np.ndarray, align_rules: dict, mode: str = "affine") -> np.ndarray:
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

    original, distorted = np.array(image1, copy=True), np.array(image2, copy=True)
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


def shift_pixels(image1: np.ndarray, image2: np.ndarray, align_rules: dict) -> np.ndarray:
    # shift the pixels using optical flow (fine alignment); based on continuum images
    # measured in opposite direction; (more stable if the shifted figure is sharp)

    image_shifted = np.zeros_like(image2)

    _, n_rows, n_cols = np.shape(image2)
    row_coords, col_coords = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")

    for index_aligned, index_master in align_rules.items():
        v_row, v_col = -optical_flow_tvl1(image2[index_master],
                                          match_histograms(image1[index_master], image2[index_master])
                                          )

        image_shifted[index_aligned] = warp(image2[index_aligned], np.array([row_coords + v_row, col_coords + v_col]),
                                            order=1, mode="constant", cval=np.nan)

    return image_shifted


def remove_nan(corrupted: np.ndarray, clear: np.ndarray) -> tuple[np.ndarray, ...]:
    # fill in NaNs first
    mask = ~np.all(np.isfinite(corrupted), axis=0)
    corrupted[:, mask] = np.nan
    clear[:, mask] = np.nan

    corrupted = np.array([crop_nan(corrupted_part) for corrupted_part in corrupted])
    clear = np.array([crop_nan(clear_part) for clear_part in clear])

    return corrupted, clear


def add_subplot(axis, x, title: str | None = None) -> None:
    y_max, x_max = np.shape(x)
    im = axis.imshow(x, origin="lower", extent=[0, x_max, 0, y_max], aspect="auto")

    if title is not None:
        axis.set_title(title)

    divider = make_axes_locatable(axis)
    cax = divider.append_axes(position="right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)


def align_SP_file(SP_filename: str, coordinates: Literal["cartesian", "ptr", "ptr_original"] = "cartesian",
                  index_continuum: int = 0, align_rules: dict | None = None,
                  control_plots: bool = False, quiet: bool = False) -> tuple[np.ndarray, np.ndarray]:
    # SP_filename = sorted([sp_file for sp_file in listdir(_path_sp) if "L2.1" not in sp_file])[0]

    if align_rules is None:
        # align zeroth with zeroth, align first with zeroth, ...
        # in order [I, Bp, Bt, Br]
        align_rules = {0: 0, 1: 0, 2: 0, 3: 0}

    index_plot = 3

    if not quiet:
        print(SP_filename)

    if coordinates == "ptr_original":  # continuum and B in cartesian and G
        sp_iptr = read_sp(SP_filename=SP_filename, coordinates=coordinates)
    else:  # read data in B, Bi, Bg and disentangle azimuth later
        sp_iBig = read_sp_raw(SP_filename=SP_filename)
        # convert it to Bptr (only r is correct at this moment; necessary for alignment)
        sp_iptr = np.array(sp_iBig[:-1], copy=True)
        sp_iptr[1:] = spherical_to_cartesian(r=sp_iBig[1], theta=np.deg2rad(sp_iBig[2]), phi=np.deg2rad(sp_iBig[3]))

    _, n_rows, n_cols = np.shape(sp_iptr)
    hmi_iptr = np.zeros((4, n_rows, n_cols))
    # width_mean = np.zeros(len(hmi_iptr), dtype=int)

    # hmi magnetic field in cartesian coordinates and G
    hmi_ptr = read_hmi_B(SP_filename, coordinates=coordinates)

    # here is the first interpolation. Conversion from spherical to cartesian must be before it (problems with azimuth)
    for i, quantity in enumerate(["continuum", "B0", "B1", "B2"]):
        hmi_iptr[i], _ = get_hmi_sectors(quantity, SP_filename, hmi_ptr)

    sp_iptr[index_continuum] = rescale_intensity(sp_iptr[index_continuum], thresh=0.9)
    hmi_iptr[index_continuum] = rescale_intensity(hmi_iptr[index_continuum], thresh=0.9)

    sp_iptr[index_continuum] = np.clip(interpolate_mask(image=sp_iptr[index_continuum],
                                                        mask=sp_iptr[index_continuum] <= 0,
                                                        interp_nans=True, fill_value=_num_eps),
                                       _num_eps, None)
    hmi_iptr[index_continuum] = np.clip(interpolate_mask(image=hmi_iptr[index_continuum],
                                                         mask=hmi_iptr[index_continuum] <= 0,
                                                         interp_nans=True, fill_value=_num_eps),
                                        _num_eps, None)

    if control_plots:
        mpl.use("TkAgg")
        fig, ax = plt.subplots(2, 3)

        add_subplot(ax[0, 0], hmi_iptr[index_plot], title="HMI sectors")
        add_subplot(ax[1, 0], sp_iptr[index_plot], title="SP original")

    # align HMI and SP data based on continuum images
    hmi_iptr = orbAlign(sp_iptr, hmi_iptr, mode="affine", align_rules=align_rules)

    # remove NaNs
    if coordinates != "ptr_original":
        _, sp_iBig = remove_nan(hmi_iptr, sp_iBig)
    hmi_iptr, sp_iptr = remove_nan(hmi_iptr, sp_iptr)

    sp_iptr[index_continuum] = rescale_intensity(sp_iptr[index_continuum], thresh=0.9)
    hmi_iptr[index_continuum] = rescale_intensity(hmi_iptr[index_continuum], thresh=0.9)

    if control_plots:
        add_subplot(ax[0, 1], hmi_iptr[index_plot], title="HMI aligned")
        add_subplot(ax[1, 1], sp_iptr[index_plot], title="SP aligned")

    # hmi_iptr = shift_pixels(sp_iptr, hmi_iptr, align_rules=align_rules)

    # remove NaNs
    if coordinates != "ptr_original":
        _, sp_iBig = remove_nan(hmi_iptr, sp_iBig)
    hmi_iptr, sp_iptr = remove_nan(hmi_iptr, sp_iptr)

    sp_iptr[index_continuum] = rescale_intensity(sp_iptr[index_continuum], thresh=0.9)
    hmi_iptr[index_continuum] = rescale_intensity(hmi_iptr[index_continuum], thresh=0.9)

    if coordinates != "ptr_original":
        # convert Hinode observation to Bptr and solve azimuth ambiguity
        sp_iBig[index_continuum] = rescale_intensity(sp_iBig[index_continuum], thresh=0.9)
        sp_iptr = convert_sp_to_Bptr(sp_iBig, hmi_iptr, SP_filename=SP_filename, coordinates=coordinates)

    if control_plots:
        add_subplot(ax[0, 2], hmi_iptr[index_plot], title="HMI final")
        add_subplot(ax[1, 2], sp_iptr[index_plot], title="SP final")

        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.tight_layout()
        plt.show()

    if not quiet:
        print(" I      Bp     Bt     Br")
        print(", ".join([f"{rmse(hmi_iptr[i], sp_iptr[i]):.2f}" for i in range(len(sp_iptr))]))

    # there is one observation of 4 quantities (nq, nx, ny) -> (1, nx, ny, nq)
    hmi_iptr = np.expand_dims(np.transpose(hmi_iptr, (1, 2, 0)), axis=0)
    sp_iptr = np.expand_dims(np.transpose(sp_iptr, (1, 2, 0)), axis=0)

    save_data(SP_filename.replace(".fits", ".npz"), observations=hmi_iptr, labels=sp_iptr,
              subfolder=_path_sp_hmi)

    return hmi_iptr, sp_iptr


def filter_files(files: list[str]) -> list[str]:
    return sorted([file for file in files if path.isfile(file)
                   and path.isfile(file.replace("field", "inclination"))
                   and path.isfile(file.replace("field", "azimuth"))
                   and path.isfile(file.replace("field", "disambig"))
                   and path.isfile(file.replace("field", "1.continuum").replace("hmi.b", "hmi.ic"))])


def read_hmi_original(subfolder_or_SP_filename: str, ar_number: str | None = None) -> None:
    hmi_dir = subfolder_or_SP_filename.replace(".fits", "")
    if ar_number is None:
        final_name = path.join(_path_data, f"{next((s for s in hmi_dir.split(path.sep)[::-1] if s), 'unknown')}.npz")
    else:
        final_name = path.join(_path_data, f"AR_{ar_number}_{next((s for s in hmi_dir.split(path.sep)[::-1] if s), 'unknown')}.npz")

    print(f"Resaving data to\n\t{final_name}")

    files = sorted([path.join(_path_hmi, hmi_dir, file) for file in listdir(path.join(_path_hmi, hmi_dir)) if "field" in file])
    files = filter_files(files)

    obs_time = np.array([fits.getheader(file, 1)["T_REC"] for file in files])

    nrows, ncols = np.shape(fits.getdata(files[0], 1))

    # HERE SHOULD BE RESIZE TO REAL HMI/Hinode FRACTION (Hinode = ??'' / pix; HMI = 0.504264'' / pix)
    output_shape = (2 * nrows, 2 * ncols)

    hmi_b = np.zeros((3, nrows, ncols))
    HMI = np.zeros((len(files), *output_shape, 4))

    for ifile, file in enumerate(files):
        hmi_b[0] = fits.getdata(file, 1)
        hmi_b[1] = fits.getdata(file.replace("field", "inclination"), 1)
        hmi_b[2] = fits.getdata(file.replace("field", "azimuth"), 1)

        disambig = np.array(fits.getdata(file.replace("field", "disambig"), 1), dtype=int)
        disambig = np.array([[f"{num:03b}"[1] for num in row] for row in disambig], dtype=float)
        hmi_b[2] += 180. * disambig

        hmi_b = spherical_to_cartesian(r=hmi_b[0], theta=np.deg2rad(hmi_b[1]), phi=np.deg2rad(hmi_b[2]))

        HMI[ifile, :, :, 1:] = np.transpose(np.array([resize(b_part, output_shape, anti_aliasing=True) for b_part in hmi_b]) / 1000.,
                                            axes=(1, 2, 0))

        HMI[ifile, :, :, 0] = rescale_intensity(resize(fits.getdata(file.replace("field", "1.continuum")
                                                                    .replace("hmi.b", "hmi.ic"), 1),
                                                       output_shape,
                                                       anti_aliasing=True),
                                                thresh=0.9)

    check_dir(final_name)
    with open(final_name, "wb") as f:
        np.savez_compressed(f, obs_time=obs_time, HMI=np.array(HMI, dtype=_wp))


def remove_outliers(image: np.ndarray, kernel_size: int = 7, n_std: float = 3.) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size))
    kernel /= np.sum(kernel)
    ddof = 1 if kernel_size > 1 else 0
    sliding_mean = sliding_window(image=image, kernel=kernel, func=lambda i, k: np.mean(i))
    sliding_std = sliding_window(image=image, kernel=kernel, func=lambda i, k: np.std(i, ddof=ddof))
    mask = np.abs(image - sliding_mean) > n_std * sliding_std

    interp_image = interpolate_mask(image, mask)
    interp_image[~np.isfinite(interp_image)] = image[~np.isfinite(interp_image)]

    return interp_image


def combine_data_to_patches(patch_size: int | None = None, interpolate_outliers: bool = True, suf: str= "") -> None:
    print(f"Combining aligned HMI and SP data and splitting them into the final patches...")
    if patch_size is None: patch_size = conf_grid_setup["patch_size"]
    print(f"Patch size = {patch_size} px")
    print("Outliers will be interpolated.") if interpolate_outliers else print("Outliers will not be interpolated.")

    stored_files = sorted(listdir(_path_sp_hmi))

    hmi = len(stored_files) * [0]
    sp_hmi_like = len(stored_files) * [0]
    sp = len(stored_files) * [0]

    psf = hmi_psf(kernel_size=7, method="Baso",
                  gauss_lorentz_trade_off=0.3, gauss_sigma=2.5,
                  lorentz_width=3.4, lorentz_power=3.0)

    kernel_size, n_std = 7, 3.

    for i, file in tqdm(enumerate(stored_files)):
        data = load_npz(path.join(_path_sp_hmi, file))
        correction = convolve2d(np.ones_like(data[_observations_name][0, :, :, 0]), psf, mode="same")

        data_part = data[_observations_name]  # HMI
        if interpolate_outliers:
            data_part = np.array([remove_outliers(image=data_part[0, :, :, i], kernel_size=kernel_size, n_std=n_std)
                                  for i in range(np.shape(data_part)[-1])])
            data_part = np.expand_dims(np.transpose(data_part, (1, 2, 0)), axis=0)
        hmi[i] = split_data_to_patches(data_part, patch_size=patch_size)

        data_part = data[_label_name]  # SP
        if interpolate_outliers:
            data_part = np.array([remove_outliers(image=data_part[0, :, :, i], kernel_size=kernel_size, n_std=n_std)
                                  for i in range(np.shape(data_part)[-1])])
            data_part = np.expand_dims(np.transpose(data_part, (1, 2, 0)), axis=0)
        sp[i] = split_data_to_patches(data_part, patch_size=patch_size)

        # blur SP
        # convert to spherical
        sp_B = cartesian_to_spherical(y=data_part[0, :, :, 1], x=data_part[0, :, :, 2], z=data_part[0, :, :, 3])
        ind_zeros = sp_B[0] == 0.  # this should not happen (filling factor was sometimes 0, but should not be now)
        # blur continuum and the B amplitude
        data_part[0, :, :, 0] = convolve2d(data_part[0, :, :, 0], psf, mode="same") / correction
        sp_B[0] = convolve2d(sp_B[0], psf, mode="same") / correction
        sp_B[0][ind_zeros] = 0.
        # convert back to cartesian
        data_part[0, :, :, 1:] = np.transpose(spherical_to_cartesian(r=sp_B[0],
                                                                     theta=sp_B[1],
                                                                     phi=sp_B[2]), axes=(1, 2, 0))
        data_part[0, :, :, 0] = rescale_intensity(data_part[0, :, :, 0], thresh=0.9)
        sp_hmi_like[i] = split_data_to_patches(data_part, patch_size=patch_size)

    hmi, sp = np.array(stack(hmi, axis=0)), np.array(stack(sp, axis=0))
    sp_hmi_like = np.array(stack(sp_hmi_like, axis=0))

    # G -> kG to have values close to 1
    hmi[..., 1:] /= 1000.
    sp[..., 1:] /= 1000.
    sp_hmi_like[..., 1:] /= 1000.

    save_data(f"SP_HMI_aligned{suf}.npz", observations=hmi, labels=sp, order="k", subfolder=_path_data,
              other_info={f"{_observations_name}_simulated": sp_hmi_like})


if __name__ == "__main__":
    for ar_number in [11267, 11268, 11269, 11270, 11272, 11276, 11278, 11280, 11281, 11285, 11288, 11294]:
        try:
            prepare_hmi_data(ar_number=ar_number)
        except Exception:
            continue

"""
    skip_jsoq_query = True
    SP_filenames = sorted([sp_file for sp_file in listdir(_path_sp) if "L2.1" not in sp_file])

    if not skip_jsoq_query:
        for SP_filename in tqdm(SP_filenames):
            try:
                jsoc_query_from_sp_name(SP_filename, quantity="I")
                jsoc_query_from_sp_name(SP_filename, quantity="B")
            except Exception:  # sometimes got drms.exceptions.DrmsExportError:  [status=4]
                print(traceback.format_exc())
                continue

    # YOU SHOULD USE coordinates="cartesian"
    for SP_filename in tqdm(SP_filenames):
        try:
            hmi_iptr, sp_iptr = align_SP_file(SP_filename=SP_filename, coordinates="cartesian",
                                              index_continuum=0, align_rules={0: 0, 1: 0, 2: 0, 3: 0},
                                              control_plots=False, quiet=False)
        except Exception:
            # alignment of the SP and HMI data is not perfect (coordinates in header are often not precise enough)
            # possible missing HMI data due to the drms.exceptions.DrmsExportError:  [status=4]
            print(traceback.format_exc())
            continue

        plot_alignment(hmi_iptr, sp_iptr, suf=f"_{SP_filename.replace('.fits', '')}")

    # you should check the control plots and remove the cubes that are poorly matched or with other artifacts...
    combine_data_to_patches(interpolate_outliers=True)
"""
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
