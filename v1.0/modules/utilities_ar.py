from glob import glob
from os import path
import numpy as np
from matplotlib import pyplot as plt
from modules.utilities import best_blk


def collect_nth_occurrence(occurrence: int,
                           base_contour: str,
                           eval_contour: str,
                           ar_dir: str = "/nfsscratch/david/NN/results/AR_stat") -> dict:
    """
    occurrence = 0
    base_contour = "Ic-0.7"
    eval_contour = "Br-500.0"
    ar_dir = "/nfsscratch/david/NN/results/AR_stat"
    """

    regex_name = path.join(ar_dir, f"*_{base_contour}_{eval_contour}*.npz")
    filenames = sorted(glob(regex_name))

    if not filenames:
        print(f"No files {regex_name}")

    sum_in_ic, sum_on_ic, area_ic, length_ic = np.array([]), np.array([]), np.array([]), np.array([])
    sum_in_expanded, sum_on_expanded, area_expanded, length_expanded = np.array([]), np.array([]), np.array([]), np.array([])

    for filename in filenames:
        data = np.load(filename, allow_pickle=True)

        if f"occurrence_{occurrence}" not in data.files:
            continue

        data = data[f"occurrence_{occurrence}"][()]

        sum_in_ic = np.concatenate((sum_in_ic, data["sum_in_ic"]))
        sum_on_ic = np.concatenate((sum_on_ic, data["sum_on_ic"]))
        area_ic = np.concatenate((area_ic, data["area_ic"]))
        length_ic = np.concatenate((length_ic, data["length_ic"]))

        sum_in_expanded = np.concatenate((sum_in_expanded, data["sum_in_expanded"]))
        sum_on_expanded = np.concatenate((sum_on_expanded, data["sum_on_expanded"]))
        area_expanded = np.concatenate((area_expanded, data["area_expanded"]))
        length_expanded = np.concatenate((length_expanded, data["length_expanded"]))

    return {"sum_in_ic": sum_in_ic, "sum_on_ic": sum_on_ic,
            "area_ic": area_ic, "length_ic": length_ic,
            "sum_in_expanded": sum_in_expanded, "sum_on_expanded": sum_on_expanded,
            "area_expanded": area_expanded, "length_expanded": length_expanded,
            "mean_in_ic": sum_in_ic / area_ic, "mean_on_ic": sum_on_ic / length_ic,
            "mean_in_expanded": sum_in_expanded / area_expanded, "mean_on_expanded": sum_on_expanded / length_expanded,
            "flux_in_ic": sum_in_ic * area_ic, "flux_on_ic": sum_on_ic * length_ic,
            "flux_in_expanded": sum_in_expanded * area_expanded, "flux_on_expanded": sum_on_expanded * length_expanded}


base_contour = "Ic-0.75"
eval_contour = "Br-500.0"
cut_part = 30
for occurrence in range(20):
    results = collect_nth_occurrence(occurrence=occurrence, base_contour=base_contour, eval_contour=eval_contour)

    fig, ax = plt.subplots(*best_blk(len(results)))
    plt.suptitle(f"{base_contour}_{eval_contour}_{occurrence}-th")
    ax = np.ravel(ax)

    for iax, (key, value) in enumerate(results.items()):
        lower_limit = np.nanpercentile(value, cut_part)  # 1st percentile
        upper_limit = np.nanpercentile(value, 100 - cut_part)  # 99th percentile
        if iax not in [8, 9, 10, 11]:
            if iax in [2, 3, 6, 7]:
                value = value[value <= upper_limit]
            else:
                value = value[(value >= lower_limit) & (value <= upper_limit)]
        ax[iax].hist(value, bins=100)
        ax[iax].set_title(key)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.tight_layout()
