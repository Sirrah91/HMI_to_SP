from modules.utilities_contours import (contour_area, find_contours, do_contours_intersects, contours_distance,
                                        extract_expanded_contour, sum_within_and_on_contour, contour_length)
from modules.utilities import to_list, stack, check_dir
from modules.utilities_data import load_npz

import argparse
from types import SimpleNamespace
import os
from os import path
from pprint import pprint
import socket
from glob import glob
from astropy.io import fits
import numpy as np
from collections import defaultdict


def gimme_quantity(filename: str, quantity: str = "Ic") -> np.ndarray:
    with fits.open(filename) as hdu:
        images = hdu[quantity].data
    return images


def collect_all(filenames: list[str], quantity: str = "Ic") -> np.ndarray:
    print(f"Collecting {quantity} data...")
    return stack([gimme_quantity(filename, quantity) for filename in filenames], axis=0)


def detect_and_track_regions(images: np.ndarray,
                             threshold_intensity: float = 0.7,
                             threshold_min: float | None = None,
                             threshold_max: float | None = None,
                             min_persistence: int = 3) -> dict:
    """
    Detect and track regions across images using contours.

    Parameters:
    - images: List or array of numpy arrays, where each array represents an image.
    - threshold_intensity: Threshold for creating binary mask.
    - threshold_min: Minimum area for a valid region.
    - threshold_max: Maximum area for a valid region.
    - min_persistence: Minimum number of consecutive images a region should appear to be considered persistent.
    - gap_threshold: Maximum number of frames a region can disappear and still be considered the same region.

    Returns:
    - tracked_regions: Dictionary with region IDs and their characteristics.
    """

    print("Region detection...")

    tracked_regions = defaultdict(lambda: {"start": None, "end": None, "images": {}, "contour": None})
    region_id_counter = 1

    for idx, img in enumerate(images):
        # Ensure image is in correct format
        img = np.asarray(img, dtype=np.float32)

        if np.any(~np.isfinite(img)):
            continue

        # Find contours
        contours = find_contours(image=img, level=threshold_intensity)

        # Filter contours based on area thresholds
        filtered_contours = []
        for _contour in contours:
            area = contour_area(contour=_contour)
            if (threshold_min is None or area > threshold_min) and (threshold_max is None or area < threshold_max):
                filtered_contours.append(_contour)

        current_regions = {}
        for _contour in filtered_contours:
            # Track contours in current frame
            current_regions[region_id_counter] = np.reshape(_contour, newshape=(-1, 2))
            region_id_counter += 1

        # Compare current regions with previously tracked regions
        new_tracked_regions = defaultdict(lambda: {"start": None, "end": None, "images": {}, "contour": None})

        for prev_region_id, prev_data in tracked_regions.items():
            # Check overlap with new contours
            matched = False
            for current_region_id, current_contour in current_regions.items():
                if do_contours_intersects(prev_data["contour"], current_contour):
                    new_tracked_regions[prev_region_id] = {
                        "start": prev_data["start"],
                        "end": idx,
                        "images": {**prev_data["images"], idx: current_contour},
                        "contour": current_contour
                    }
                    matched = True
                    break

            if not matched:
                # Region has disappeared but will still be kept in the tracked_regions
                new_tracked_regions[prev_region_id] = {
                    "start": prev_data["start"],
                    "end": idx,  # Update end frame to current frame
                    "images": prev_data["images"],
                    "contour": prev_data["contour"]
                }

        # Add new regions
        for current_region_id, current_contour in current_regions.items():
            if not any(do_contours_intersects(current_contour, prev_data["contour"])
                       for prev_data in new_tracked_regions.values()):
                # If the current region does not overlap with any previous region
                new_tracked_regions[region_id_counter] = {
                    "start": idx,
                    "end": idx,
                    "images": {idx: current_contour},
                    "contour": current_contour
                }
                region_id_counter += 1

        tracked_regions = new_tracked_regions

    # Filter regions based on persistence
    persistent_regions = {
        _region_id: data for _region_id, data in tracked_regions.items()
        if len(data["images"]) >= min_persistence
    }

    return persistent_regions


def merge_close_regions(persistent_regions: dict, closeness_threshold: int) -> dict:
    """
    Merge regions in `persistent_regions` if their contours are close to each other.

    Parameters:
    - persistent_regions: Dictionary containing region data
    - closeness_threshold: Maximum allowed distance between two regions for them to be considered the same.

    Returns:
    - merged_regions: Dictionary of merged region data, where contours are stored as lists of arrays.
    """

    print("Region merging...")

    merged_regions = {}
    region_id_mapping = {}  # Maps old region IDs to new merged region IDs
    new_region_id = 0

    for _region_id, region_data in persistent_regions.items():
        if _region_id in region_id_mapping:
            continue  # Skip already merged regions

        current_region = region_data
        current_contours = current_region["images"]  # Dictionary of contours for the region
        # Start with the current region
        merged_region = {"start": current_region["start"], "end": current_region["end"], "images": defaultdict(list)}

        # Append current contours
        for frame_idx, _contour in current_contours.items():
            merged_region["images"][frame_idx].append(_contour)  # Append the initial contour(s)

        # Check all other regions to see if they are close to the current region
        for other_region_id, other_region_data in persistent_regions.items():
            if other_region_id == _region_id or other_region_id in region_id_mapping:
                continue  # Skip self and already merged regions

            other_contours = other_region_data["images"]
            # Compute minimum distance between any contour in the current region and any contour in the other region
            close = False
            for frame_idx, _contour in current_contours.items():
                for other_contour in other_contours.values():
                    dist = min_distance_between_contours(_contour, other_contour)
                    if dist < closeness_threshold:
                        close = True
                        break
                if close:
                    break

            # If the regions are close, merge them
            if close:
                # Append other region"s contours to the current region"s contours
                for frame_idx, other_contour in other_contours.items():
                    merged_region["images"][frame_idx].append(other_contour)

                # Update the end frame (the latest appearance)
                merged_region["end"] = max(merged_region["end"], other_region_data["end"])

                region_id_mapping[other_region_id] = new_region_id  # Mark the other region as merged

        # Add merged region to the new dictionary
        merged_regions[new_region_id] = merged_region
        new_region_id += 1

    return merged_regions


def min_distance_between_contours(contour1: np.ndarray, contour2: np.ndarray) -> float:
    """
    Calculate the minimum distance between two contours.

    Parameters:
    - contour1: The first contour (numpy array of shape (N, 2))
    - contour2: The second contour (numpy array of shape (M, 2))

    Returns:
    - The minimum distance between any point in contour1 and any point in contour2.
    """
    # Ensure contours are 2-dimensional
    contour1 = np.array(contour1)
    contour2 = np.array(contour2)

    # Check if contours are valid (not empty and 2D)
    if contour1.ndim != 2 or contour1.shape[1] != 2:
        raise ValueError(f"Contour1 is not a 2D array with shape (N, 2). Got shape: {contour1.shape}")
    if contour2.ndim != 2 or contour2.shape[1] != 2:
        raise ValueError(f"Contour2 is not a 2D array with shape (N, 2). Got shape: {contour2.shape}")

    # Calculate pairwise distances between contour points
    return contours_distance(contour1, contour2)


def combine_mask(masks: list[np.ndarray | None]) -> np.ndarray | None:
    # filter None first
    masks = np.array([m for m in masks if m is not None])

    # if masks was full of None, return None
    if np.size(masks) == 0:
        return None

    # create nan-1 mask
    masks = np.where(np.logical_and(np.isfinite(masks), masks > 0.), 1., np.nan)

    # sum the masks (union of the regions)
    masks = np.nansum(masks, axis=0)

    # create nan-1 mask
    return np.where(masks > 0., 1., 0.)


def filter_AR_contours(contours: list[np.ndarray | None]) -> list[np.ndarray] | None:
    # filter None
    contours = [c for c in contours if c is not None]

    return None if not contours else contours


def gimme_name(params) -> str:
    return (f"{params.contour_quantity_calc}-{params.contour_threshold}_"
            f"{params.contour_quantity_eval}-{params.expansion_threshold}")


def find_and_store_regions(filenames: list[str] | str, params) -> dict:
    images_contour = collect_all(to_list(filenames), quantity=params.contour_quantity_calc)

    persistent_regions = detect_and_track_regions(images_contour,
                                                  threshold_intensity=params.contour_threshold,
                                                  threshold_min=params.min_area,
                                                  threshold_max=params.max_area,
                                                  min_persistence=params.min_persistence)

    merged_regions = merge_close_regions(persistent_regions,
                                         closeness_threshold=params.closeness_threshold)

    data_and_metadata = {"persistent_regions": persistent_regions, "merged_regions": merged_regions,
                         "params": vars(params), "filenames": _names}

    return data_and_metadata


if __name__ == "__main__":
    hostname = socket.gethostname()
    print(f"Running on: {hostname}\n")

    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--region_number", type=str, default="*")

    parser.add_argument("--contour_quantity_calc", type=str, default="Ic")
    parser.add_argument("--contour_threshold", type=float, default=0.7)
    parser.add_argument("--contour_quantity_eval", type=str, default="Br")
    parser.add_argument("--expansion_threshold", type=float, default=500.)

    parser.add_argument("--min_area", type=float, default=0.)
    parser.add_argument("--max_area", type=str, default="inf")
    parser.add_argument("--min_persistence", type=int, default=0)
    parser.add_argument("--closeness_threshold", type=int, default=0)
    parser.add_argument("--closing_px", type=int, default=1)
    parser.add_argument("--min_occurrence", type=int, default=0)
    parser.add_argument("--max_occurrence", type=int, default=-1)

    parser.add_argument("--append", action="store_true")
    parser.add_argument("--outdir", type=str, default="/nfsscratch/david/NN/results/AR_stat")

    args, _ = parser.parse_known_args()

    ar_prefix = "AR-"
    if not args.region_number.startswith(ar_prefix):
        args.region_number = f"{ar_prefix}{args.region_number}"

    if args.max_area == "inf":
        args.max_area = np.inf
    else:
        args.max_area = float(args.max_area)

    print("Parameters:")
    pprint(vars(args))
    print()

    check_dir(args.outdir)

    fits_dir = "/nfsscratch/david/NN/results"
    names = sorted(glob(path.join(fits_dir, f"*{args.region_number}*.fits")))

    bare_names = [path.split(name)[-1] for name in names]
    region_names = np.unique([bare_name.split("_")[0] for bare_name in bare_names])

    for region_name in region_names:
        print(f"Active region: {region_name}")
        final_name = path.join(args.outdir, path.split(f"{region_name}_{gimme_name(params=args)}.npz")[1])
        check_dir(final_name)

        if args.append and path.isfile(final_name):
            data_ar = load_npz(final_name)
            regions_dict = dict(data_ar)
            _names, regions, _args = list(data_ar["filenames"]), data_ar["merged_regions"][()], SimpleNamespace(**data_ar["params"][()])
            _args.min_occurrence = args.min_occurrence
            _args.max_occurrence = args.max_occurrence
            args = _args
            data_ar.close()
        else:
            _names = sorted(glob(path.join(fits_dir, f"*{region_name}*.fits")))

            # region-by-region is memory expensive but precise (comparing to file-by-file)
            regions_dict = find_and_store_regions(filenames=_names, params=args)
            regions = regions_dict["merged_regions"]

        images_eval = collect_all(to_list(_names), quantity=args.contour_quantity_eval)

        max_occurrence = max([len(regions[i]["images"]) for i in regions.keys()])
        if args.max_occurrence < 0:
            args.max_occurrence = max_occurrence
        else:
            args.max_occurrence = min(max_occurrence, args.max_occurrence)

        n_regions = len(regions.keys())
        print(f"No. detected regions: {n_regions}")
        print()  # empty line

        occurrence_list = range(args.min_occurrence, args.max_occurrence)

        # compute statistics on the regions
        for _image_index in occurrence_list:  # N-th occurrence
            # print(50 * "-")
            print(f"N-th occurrence: {_image_index + 1}/{args.max_occurrence}")

            # initialize memory
            _sum_in_ic = np.full(n_regions, np.nan)
            _sum_on_ic = np.full(n_regions, np.nan)
            _sum_in_expanded = np.full(n_regions, np.nan)
            # to check correctness, should be roughly args.expansion_threshold * length
            _sum_on_expanded = np.full(n_regions, np.nan)

            _area_ic = np.full(n_regions, np.nan)
            _length_ic = np.full(n_regions, np.nan)
            _area_expanded = np.full(n_regions, np.nan)
            _length_expanded = np.full(n_regions, np.nan)

            for i_region, region_id in enumerate(regions.keys()):
                # print(f"Region progress: {region_id + 1}/{n_regions}")

                region = regions[region_id]["images"]
                image_indices = list(region.keys())
                if _image_index >= len(image_indices):
                    continue
                image_index = image_indices[_image_index]
                image = images_eval[image_index]
                contour = region[image_index]

                _sums_in, _sums_on = zip(*[sum_within_and_on_contour(image=image, vertices=c, margin=10)
                                           for c in contour])
                _sum_in_ic[i_region] = np.sum(_sums_in)
                _sum_on_ic[i_region] = np.sum(_sums_on)
                _area_ic[i_region] = np.sum([contour_area(c) for c in contour])
                _length_ic[i_region] = np.sum([contour_length(c) for c in contour])

                contour_expanded = [extract_expanded_contour(image, c, expansion_threshold=args.expansion_threshold,
                                                             iterations=args.closing_px, return_mask=False)
                                    for c in contour]

                contour_expanded = filter_AR_contours(contour_expanded)

                if contour_expanded is None:
                    _sum_in_expanded[i_region] = np.nan
                    _sum_on_expanded[i_region] = np.nan
                    _area_expanded[i_region] = np.nan
                    _length_expanded[i_region] = np.nan
                else:
                    _sums_in, _sums_on = zip(*[sum_within_and_on_contour(image=image, vertices=c, margin=10)
                                               for c in contour_expanded])
                    _sum_in_expanded[i_region] = np.sum(_sums_in)
                    _sum_on_expanded[i_region] = np.sum(_sums_on)
                    _area_expanded[i_region] = np.sum([contour_area(c) for c in contour_expanded])
                    _length_expanded[i_region] = np.sum([contour_length(c) for c in contour_expanded])

            label = f"occurrence_{_image_index}"
            occurrence_dict = {"sum_in_ic": _sum_in_ic,
                               "sum_on_ic": _sum_on_ic,
                               "sum_in_expanded": _sum_in_expanded,
                               "sum_on_expanded": _sum_on_expanded,
                               "area_ic": _area_ic,
                               "length_ic": _length_ic,
                               "area_expanded": _area_expanded,
                               "length_expanded": _length_expanded}

            regions_dict[label] = occurrence_dict

            # save part
            with open(f"{final_name}.tmp", "wb") as f:
                np.savez_compressed(f, **regions_dict, flag=False)

        # save all to a final file (so that you can see if the job finished)
        with open(final_name, "wb") as f:
            np.savez_compressed(f, **regions_dict, flag=True)
        # remove the temporary file
        os.remove(f"{final_name}.tmp")

    print()
    print("All done")
