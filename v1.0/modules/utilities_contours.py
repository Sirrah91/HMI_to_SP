from modules.utilities import stack, flatten_list_v2, unflatten_list

import numpy as np
import warnings
from skimage import measure
from copy import deepcopy
from scipy.spatial import distance
from skimage.draw import polygon, polygon_perimeter
from skimage.morphology import dilation, erosion, square
from skimage.transform import resize
from shapely.geometry import Polygon, Point, LineString
from astropy.io import fits
from typing import Literal
from os import path


def contour_to_shape(contour: np.ndarray):
    if len(contour) == 1:
        return Point(contour)
    if len(contour) < 3:
        return LineString(contour)
    return Polygon(contour).convex_hull


def contour_center(contour: np.ndarray) -> tuple[float, ...]:
    return contour_to_shape(contour).centroid.coords[0]


def contour_area(contour: np.ndarray) -> float:
    return contour_to_shape(contour).area


def contour_length(contour: np.ndarray) -> float:
    return contour_to_shape(contour).length


def find_contours(image: np.ndarray, level: float = 0.5) -> list[np.ndarray] | tuple[np.ndarray, ...]:
    return measure.find_contours(image, level=level)


def contours_distance(contour1: np.ndarray, contour2: np.ndarray) -> bool:
    # return contour_to_shape(contour1).distance(contour_to_shape(contour2))
    return distance.cdist(contour1, contour2, "euclidean").min()


def compute_contour_areas(image: np.ndarray, level: float) -> np.ndarray:
    return np.array([contour_area(contour) for contour in find_contours(image, level)])


def do_contours_intersects(contour1: np.ndarray, contour2: np.ndarray) -> bool:
    return contour_to_shape(contour1).intersects(contour_to_shape(contour2))


def dilate_mask(mask: np.ndarray, dilate_pixels: int = 0) -> np.ndarray:
    return dilation(mask, square(2 * dilate_pixels + 1))


def erode_mask(mask: np.ndarray, erode_pixels: int = 0) -> np.ndarray:
    return erosion(mask, square(2 * erode_pixels + 1))


def create_contour_mask(contour: np.ndarray | list[np.ndarray], mask_shape: tuple[int, ...],
                        subpixel_factor: int = 5, fill_threshold: float = 0.5,
                        fill_positive: float | bool = 1., fill_negative: float | bool = 0.,
                        border_only: bool = False,
                        expansion_pixels: int = 0) -> np.ndarray:
    # Step 1: Increase resolution by subpixel_factor
    upscale_shape = (mask_shape[0] * subpixel_factor, mask_shape[1] * subpixel_factor)
    upscale_mask = np.zeros(upscale_shape, dtype=np.float32)

    # Step 2: Draw the contour on the upscaled mask
    if isinstance(contour, list) and isinstance(contour[0], np.ndarray) and np.ndim(contour[0]) == 2:
        for _contour in contour:
            rr, cc = polygon(_contour[:, 0] * subpixel_factor, _contour[:, 1] * subpixel_factor, upscale_shape)
            upscale_mask[rr, cc] = 1  # Fill with 1 in high-resolution mask
    else:
        rr, cc = polygon(contour[:, 0] * subpixel_factor, contour[:, 1] * subpixel_factor, upscale_shape)
        upscale_mask[rr, cc] = 1  # Fill with 1 in high-resolution mask

    # Step 3: Downsample the mask back to original resolution
    downscale_mask = resize(upscale_mask, mask_shape, order=1, anti_aliasing=True)

    # Step 4: Apply the threshold (e.g., pixels filled by more than half)
    downscale_mask = downscale_mask > fill_threshold

    downscale_mask = expand_mask(mask=downscale_mask, expansion_pixels=expansion_pixels, border_only=border_only)

    return np.where(downscale_mask, fill_positive, fill_negative)


def combined_contour_mask(contours: list[np.ndarray], mask_shape: tuple[int, ...],
                          fill_positive: float | bool = 1., fill_negative: float | bool = 0.,
                          **kwargs) -> np.ndarray:
    mask = np.any([create_contour_mask(contour=contour, mask_shape=mask_shape,
                                       fill_positive=True, fill_negative=False,
                                       **kwargs) for contour in contours])
    return np.where(mask, fill_positive, fill_negative)


def is_new_contour(contour_tested: Polygon | LineString | Point, contour_base: Polygon | LineString | Point) -> bool:
    return not contour_base.contains(contour_tested.centroid)


def sum_within_and_on_contour(image: np.ndarray, vertices: np.ndarray,
                              scale_factor: int = 5, margin: int | None = None) -> tuple[float, float]:
    if margin is not None:
        margin = max(margin, 0)

        image = deepcopy(image)
        vertices = deepcopy(vertices)

        # Compute bounding box for the contour
        all_y = vertices[:, 0]
        all_x = vertices[:, 1]
        min_y, max_y = all_y.min(), all_y.max()
        min_x, max_x = all_x.min(), all_x.max()

        # Add margin
        min_y = np.floor(max(min_y - margin, 0))
        max_y = np.ceil(min(max_y + margin, image.shape[0] - 1))
        min_x = np.floor(max(min_x - margin, 0))
        max_x = np.ceil(min(max_x + margin, image.shape[1] - 1))

        # Crop the image and vertices
        image = deepcopy(image[int(min_y):int(max_y) + 1, int(min_x):int(max_x) + 1])
        vertices[:, 1] = vertices[:, 1] - int(min_x)
        vertices[:, 0] = vertices[:, 0] - int(min_y)

    # Scale the image to a higher resolution for subpixel accuracy
    new_shape = (image.shape[0] * scale_factor, image.shape[1] * scale_factor)
    high_res_image = resize(image, new_shape, mode="reflect", anti_aliasing=True)

    # Scale the vertices to match the new image resolution
    scaled_vertices = [(y * scale_factor, x * scale_factor) for (y, x) in vertices]

    # Extract row and column positions from the scaled vertices
    r, c = zip(*scaled_vertices)

    # Create a mask in the high resolution space
    mask_inside = np.zeros(high_res_image.shape, dtype=bool)
    mask_contour = np.zeros(high_res_image.shape, dtype=bool)

    # Fill the polygon for the inside mask
    rr, cc = polygon(r, c, high_res_image.shape)
    mask_inside[rr, cc] = True

    # Get the perimeter of the polygon for the contour mask
    rr_contour, cc_contour = polygon_perimeter(r, c, high_res_image.shape, clip=True)
    mask_contour[rr_contour, cc_contour] = True

    # Sum the elements of the high-resolution image that are within and on the closed curve
    sum_inside = np.sum(high_res_image[mask_inside])
    sum_on_contour = np.sum(high_res_image[mask_contour])

    # Since we scaled the image by scale_factor, divide the total sum by the appropriate factor
    sum_inside /= (scale_factor ** 2)  # For area-based sum
    sum_on_contour /= scale_factor      # For perimeter-based sum

    return sum_inside, sum_on_contour


def value_in_contour(image: np.ndarray,
                     contours: np.ndarray | list[np.ndarray],
                     expansion_pixels: int = 0,
                     flux: bool = False,
                     border_only: bool = False,
                     mean_or_sum: Literal["mean", "sum"] = "mean",
                     quiet: bool = True) -> np.ndarray:
    if isinstance(contours, np.ndarray) and np.ndim(contours) == 2:  # a single contour
        contours = [contours]

    # Create a mask for the expanded contour region
    mask = combined_contour_mask(contours=contours, mask_shape=np.shape(image),
                                 fill_negative=False, fill_positive=True,
                                 expansion_pixels=expansion_pixels, border_only=border_only)

    # Extract values within the expanded contour
    sampled_values = image[mask]
    if flux:
        sampled_values *= np.sum(mask)

    # Calculate the final value (mean or sum) based on user input
    if mean_or_sum == "mean":
        final_value = np.nanmean(sampled_values)
        if not quiet:
            print(f"Average value within the expanded contour: {final_value:.3f}")
    else:
        final_value = np.nansum(sampled_values)
        if not quiet:
            print(f"Total value within the expanded contour: {final_value:.3f}")

    return final_value


def expand_mask(mask: np.ndarray,
                expansion_pixels: int = 0,
                border_only: bool = False) -> np.ndarray | tuple[np.ndarray, ...]:

    # Expand the mask using dilation if expansion is required
    if expansion_pixels > 0:
        mask = dilate_mask(mask, dilate_pixels=expansion_pixels)
    elif expansion_pixels < 0:
        mask = erode_mask(mask, erode_pixels=expansion_pixels)

    if border_only:
        # Extract border values by subtracting the dilated mask from itself
        if np.result_type(mask) == "bool":
            mask = mask.astype(float)
            to_bool = True
        else:
            to_bool = False
        mask -= erode_mask(mask, erode_pixels=1)

        if to_bool:
            mask = mask.astype(bool)

    return mask


def extract_expanded_contour(image: np.ndarray,
                             contour: np.ndarray,
                             expansion_threshold: float = 500.,
                             iterations: int = 1,
                             fill_positive: float | bool = 1.,
                             fill_negative: float | bool = 0.,
                             return_mask: bool = False) -> tuple[np.ndarray, ...] | np.ndarray | None:
    # Step 1: Threshold the image based on the given threshold value
    thresh_image = np.where(np.abs(image) > expansion_threshold, 1., 0.)

    # Step 2: Perform erosion to make the contours smaller and break small connections
    if iterations > 0:
        eroded_image = erode_mask(thresh_image, erode_pixels=iterations)
    else:
        eroded_image = thresh_image

    # Step 3: Find contours from the morphologically processed image
    contours = find_contours(eroded_image, level=0.5)

    # Step 4: Select the largest contour that overlaps with the input contour
    largest_contour = None
    largest_contour_mask = None
    max_area = 0

    # Squeeze the input contour to make it (N, 2) shape
    input_contour = np.reshape(contour, newshape=(-1, 2))
    input_contour_poly = contour_to_shape(input_contour)

    # heuristic filtering
    # - contours must be closer than dilation distance
    contours = [cnt for cnt in contours if contours_distance(input_contour, cnt) <= 2. * max(iterations, 0)]
    closest_contour = [contours_distance(input_contour, cnt) for cnt in contours]
    closest_contour = np.argmin(closest_contour) if closest_contour else 0
    for icontour, cnt in enumerate(contours):
        # Squeeze each contour from (N, 1, 2) to (N, 2)
        cnt = np.reshape(cnt, newshape=(-1, 2))

        # Step 5: Find mask for the contour and perform dilation to restore the contour to a slightly expanded state
        mask = create_contour_mask(contour=cnt, mask_shape=np.shape(eroded_image), expansion_pixels=max(iterations, 0))

        # Step 6: Find contour of the dilated mask
        cnt = find_contours(image=mask, level=0.5)
        if cnt:
            cnt = cnt[0]  # only a single contour here
            cnt_poly = contour_to_shape(cnt)
        else:
            continue

        # Step 6: Check if the current contour overlaps (area > 0 == initial) with the input contour and update largest_contour
        if cnt_poly.intersects(input_contour_poly):
            area = cnt_poly.intersection(input_contour_poly).area
        else:
            area = 0.

        if area > max_area or (max_area == 0. and icontour == closest_contour):  # save the largest or the closest
            max_area = area
            largest_contour = cnt
            largest_contour_mask = mask

    if largest_contour is None:
        return None

    # Extract values within the expanded contour
    largest_contour_mask = np.where(largest_contour_mask > 0., fill_positive, fill_negative)

    if return_mask:
        return largest_contour, largest_contour_mask
    else:
        return largest_contour


def select_new_contours(contours: np.ndarray,
                        max_area_px: float | None = None, min_area_px: float | None = None) -> np.ndarray | list:
    def _filter_data(original_list: list, mask: np.ndarray) -> list:
        # Flatten the original list
        flat_list = flatten_list_v2(original_list)

        # Apply the mask
        filtered_flat_list = [item if mask[i] else None for i, item in enumerate(flat_list)]

        # Reshape the filtered flat list back to the original structure
        reshaped_list = unflatten_list(filtered_flat_list, original_list)

        return reshaped_list

    def _process_filtered_list(filtered_list: list) -> list:
        def _list_of_None(lst):
            """Check if a list contains only None values."""
            return all(item is None for item in lst)

        def _remove_None(lst):
            """Remove None values from a list."""
            return [item for item in lst if item is not None]

        """Process the filtered list to replace lists of None with None and remove None from other lists."""
        return [None if sublist is None or _list_of_None(sublist) else _remove_None(sublist) for sublist in filtered_list]

    if max_area_px is None:
        max_area_px = np.inf
    if min_area_px is None:
        min_area_px = 0.

    contours = list(contours)
    # flatten the contours
    contours_flatten = flatten_list_v2(contours)

    # prepare polygons
    contours_flatten = [Polygon(contour) if contour is not None and len(contour) > 2 else None for contour in contours_flatten]

    # filter contours by size
    contours_flatten = [None if contour is None or not (min_area_px <= contour.area <= max_area_px) else contour for contour in contours_flatten]

    # reverse the contours (they are sorted in time, start from latest time and look back)
    contours_flatten = contours_flatten[::-1]
    new_contours_mask = np.ones(len(contours_flatten), dtype=bool)
    if contours_flatten[-1] is None:
        new_contours_mask[-1] = False
    for i in range(len(contours_flatten) - 1):
        contour_tested = contours_flatten[i]
        if contour_tested is None:
            new_contours_mask[i] = False
            continue

        for contour_base in contours_flatten[i+1:]:  # test all older contours
            if contour_base is None:
                continue
            if not is_new_contour(contour_tested, contour_base):
                new_contours_mask[i] = False
                break
    new_contours_mask = new_contours_mask[::-1]
    return _process_filtered_list(_filter_data(contours, new_contours_mask))


def average_value_in_contour(image: np.ndarray,
                             reference_image: np.ndarray,
                             level: float,
                             contour_index: int | Literal["all"] = 0,
                             contour_border_only: bool = True,
                             control_plot: bool = False,
                             quiet: bool = True) -> tuple[np.ndarray, ...] | None:
    """
    Calculate the average value of an image based on the contour of a reference image.

    Parameters:
        image (np.ndarray): The image to analyze.
        reference_image (np.ndarray): The image to derive the contour from.
        level (float): The level at which to find the contour.
        contour_index (int | Literal["all"]): Index of a contour to use or "all" to use all.
        contour_border_only (bool): Whether to use only the contour border or the entire region.
        control_plot (bool): Whether to plot the images and the contour or not.

    Returns:
        np.ndarray: The average values of the image based on the contours. If `contour_index` is an integer,
                    returns a single element array. If `contour_index` is "all", returns an array of average
                    values for each contour.
    Example usage:
        import numpy as np
        data = np.load("/nfsscratch/david/NN/data/SP_HMI_like/20100704_225505.npz")
        image = data["SP"][0, :, :, -1]  # The image to analyze
        reference_image = data["SP"][0, :, :, 0]  # The image to derive the contour from
        avg_value = average_value_in_contour(image, reference_image, level=0.5, contour_border_only=True)
    """

    # Find contours in the reference image
    contours = find_contours_v2(image=np.abs(reference_image), level=level, contour_index=contour_index)
    if contours is None:
        return None

    average_value = np.zeros(len(contours))
    total_value = np.zeros_like(average_value)

    for index, contour in enumerate(contours):
        # Choose a contour to use
        contour = np.array(np.round(contour, decimals=0), dtype=int)

        if contour_border_only:
            _text = "along"
            # Sample the pixel values at these coordinates
            sampled_values = image[contour[:, 0], contour[:, 1]]
        else:
            _text = "within"
            # Create an empty mask
            mask = np.zeros_like(image, dtype=bool)

            # Fill the mask with the contour polygon
            rr, cc = polygon(contour[:, 0], contour[:, 1], np.shape(mask))
            mask[rr, cc] = True

            # Extract the pixel values within the mask
            sampled_values = image[mask]

        if contour_index == "all":
            _index = index
        else:
            _index = contour_index

        # Calculate the average value
        average_value[index] = np.mean(sampled_values)
        total_value[index] = np.sum(sampled_values)

        if not quiet:
            print(f"Average value {_text} the contour with index {_index}: {average_value[index]:.3f}")
            print(f"Total value {_text} the contour with index {_index}: {total_value[index]:.3f}")

    if control_plot:
        import matplotlib
        # Matplotlib backend change
        backend = "TkAgg"  # must be done like this to confuse PyInstaller
        matplotlib.use(backend)  # Set the backend dynamically
        from matplotlib import pyplot as plt  # Import pyplot after setting the backend

        # Plot for visualization (optional)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image, cmap="gray", origin="lower")
        for contour in contours:
            ax[0].plot(contour[:, 1], contour[:, 0], linewidth=2)
        ax[0].set_title("Image to Analyze with Contour")
        ax[0].set_xlabel("Pixel")
        ax[0].set_ylabel("Pixel")

        ax[1].imshow(reference_image, cmap="gray", origin="lower")
        for contour in contours:
            ax[1].plot(contour[:, 1], contour[:, 0], linewidth=2)
        ax[1].set_title("Image with Contour")
        ax[1].set_xlabel("Pixel")
        ax[1].set_ylabel("Pixel")

        plt.show()

    return average_value, total_value


def find_contours_v2(image: np.ndarray, level: float,
                     contour_index: int | Literal["all"] = 0) -> list[np.ndarray] | None:
    # Find contours in the reference image
    contours = measure.find_contours(image, level=level)

    if not contours:
        warnings.warn("No contours found in the reference image at the specified level.", Warning)
        return None

    # Sort contours by their length (number of points)
    contours = sorted(contours, key=lambda x: len(x), reverse=True)

    if isinstance(contour_index, int):
        contours = [contours[contour_index]]

    return contours


def filter_contours(image: np.ndarray, contours: np.ndarray,
                    vmin: float | None = None, vmax: float | None = None, take_abs: bool = False,
                    contour_border_only: bool = True, mean_or_sum: Literal["mean", "sum"] = "mean") -> list | None:
    if vmin is None:
        vmin = -np.inf
    if vmax is None:
        vmax = np.inf

    def _value_within_limit(contour: np.ndarray) -> bool:
        value = value_in_contour(image=image, contours=contour,
                                 border_only=contour_border_only, mean_or_sum=mean_or_sum)
        if take_abs:
            value = np.abs(value)

        return vmin <= value <= vmax

    if contours is None:
        return None
    _contours = [contour for contour in contours if _value_within_limit(contour)]
    if not _contours:
        return None
    return _contours


def collect_contours(filenames: list[str], level: float, image_quantity: str = "Ic",
                     subfolder: str = "/nfsscratch/david/NN/results/") -> np.ndarray:

    contours = np.array([])

    for filename in filenames:
        hdu = fits.open(path.join(subfolder, filename))
        image = np.array(hdu[image_quantity].data, dtype=np.float32)
        hdu.close()

        _contours = np.array([find_contours_v2(image=im, level=level, contour_index="all") for im in image], dtype=object)
        contours = stack((contours, _contours))

    return contours
