import matplotlib as mpl
mpl.use("Agg")  # should be before importing pyplot
import matplotlib.pyplot as plt

from modules.decorators import safe_call

from modules.NN_losses_metrics_activations import my_quantile

from modules.utilities_data import load_npz, compute_metrics, used_indices, add_unit_to_names, filter_empty_data
from modules.utilities import (check_dir, kernel_density_estimation_2d, safe_arange, my_polyfit, denoise_array, rreplace,
                               stack, best_blk, error_in_bins)

from modules._constants import _path_figures, _path_accuracy_tests, _rnd_seed, _sep_out, _sep_in
from modules._constants import _label_true_name, _label_pred_name, _config_name, _observations_name

from modules.NN_config import quantity_names, quantity_names_short_latex

# defaults only
from modules.NN_config import conf_output_setup

from typing import Literal
from os import path, listdir
import shutil
import numpy as np
import pandas as pd
from astropy.io import fits
from copy import deepcopy
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow.keras.callbacks import History

TEXT_SIZE = 12
SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 28

LW = 1

# good to keep jpg (unless you want it for a paper) because you can easily switch among images
# jpg with higher dpi is much sharper than png with comparable size
default_fig_format = "pdf"

cbar_kwargs = {"position": "right",
               "size": "5%",
               "pad": 0.1}
savefig_kwargs = {"bbox_inches": "tight",
                  "pad_inches": 0.05,
                  "dpi": 100}
pil_kwargs = {}


def change_params(offset: float, reset: bool = False):
    if reset:
        offset = -offset

    # Check if LaTeX is available
    if shutil.which("latex"):
        plt.rc("text", usetex=True)
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
        # plt.rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"]})
    else:
        plt.rc("text", usetex=False)  # Fallback to default Matplotlib rendering

    plt.rc("font", size=TEXT_SIZE + offset)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE + offset)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE + offset)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE + offset)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE + offset)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE + offset)  # fontsize of legend
    plt.rc("figure", titlesize=BIGGER_SIZE + offset)  # fontsize of the figure title


change_params(offset=0.)  # default params

outdir = _path_figures
check_dir(outdir)

outdir_HMI_to_SOT = path.join(outdir, "HMI_to_SOT")
check_dir(outdir_HMI_to_SOT)


@safe_call
def plot_quantity_maps(y_pred: np.ndarray, y_true: np.ndarray | None = None, x_true: np.ndarray | None = None,
                       used_quantities: np.ndarray | None = None,
                       merge_rows: int = 1, merge_cols: int = 1, num_plots: int = 1,
                       title_option: Literal["SP", "HMI"] = "SP",
                       coaligned: bool = False,
                       rnd_seed: int | None = _rnd_seed, offset: float = 0.,
                       subfolder: str = "", suptitle: str | None = None,
                       fig_format: str = default_fig_format,
                       suf: str = "", quiet: bool = False) -> None:
    if not quiet:
        print("Quantity maps")
    change_params(offset)

    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    if isinstance(y_pred, tuple | list):
        # in case some of the observations are empty (skip it)
        y_pred, used_quantities = filter_empty_data(y_pred, used_quantities)
    if isinstance(y_true, tuple | list):
        # in case some of the observations are empty (skip it)
        y_true, _ = filter_empty_data(y_true, used_quantities)
    if isinstance(x_true, tuple | list):
        # in case some of the observations are empty (skip it)
        x_true, _ = filter_empty_data(x_true, used_quantities)

    rng = np.random.default_rng(seed=rnd_seed)

    used_names = add_unit_to_names(quantity_names_short_latex)
    used_names = used_names[used_indices(used_quantities)]

    if title_option == "SP":
        used_columns = np.array(["SDO/HMI-like", "Predictions", "Hinode/SOT", r"$\text{Predictions} - \text{Hinode/SOT}$"])
    else:
        if coaligned:
            used_columns = np.array(["SDO/HMI co-aligned", "Predictions", "SDO/HMI dconS", r"$\text{Predictions} - \text{SDO/HMI dconS}$"])
        else:
            used_columns = np.array(["SDO/HMI", "Predictions", "SDO/HMI dconS", r"$\text{Predictions} - \text{SDO/HMI dconS}$"])
    used_columns = used_columns[[x_true is not None, True, y_true is not None, y_true is not None]]

    num_quantities = len(used_names)
    num_columns = len(used_columns)

    cmap = "gray"

    def join_data(data: np.ndarray | None, indices: np.ndarray) -> np.ndarray | None:
        if data is None:
            return None

        _, ny, nx, nz = np.shape(data)

        x = np.zeros((merge_rows * ny, merge_cols * nx, num_quantities))

        for ir in range(merge_rows):
            for ic in range(merge_cols):
                data_index = np.ravel_multi_index((ir, ic), (merge_rows, merge_cols))
                data_index = indices[data_index]
                x[ir * ny: (1 + ir) * ny, ic * nx: (1 + ic) * nx] = data[data_index]

        return x

    num_plots = np.min((num_plots, len(y_pred) // (merge_rows * merge_cols)))
    data_indices = rng.choice(len(y_pred), (num_plots, merge_rows * merge_cols), replace=False)

    n_digits = len(str(num_plots - 1))  # -1 to count from 0

    for ifig in range(num_plots):
        y_pred_plot = join_data(y_pred, data_indices[ifig])
        y_true_plot = join_data(y_true, data_indices[ifig])
        x_true_plot = join_data(x_true, data_indices[ifig])

        if x_true_plot is None:
            if y_true_plot is None:
                col0, col1, col2, col3 = y_pred_plot, None, None, None
                vmins = np.nanmin(col0, axis=(0, 1))
                vmaxs = np.nanmax(col0, axis=(0, 1))
            else:
                col0, col1, col2, col3 = y_pred_plot, y_true_plot, y_pred_plot - y_true_plot, None
                vmins = np.nanmin((np.nanmin(col0, axis=(0, 1)), np.nanmin(col1, axis=(0, 1))), axis=0)
                vmaxs = np.nanmax((np.nanmax(col0, axis=(0, 1)), np.nanmax(col1, axis=(0, 1))), axis=0)
        else:
            if y_true_plot is None:
                col0, col1, col2, col3 = x_true_plot, y_pred_plot, None, None
                vmins = np.nanmin((np.nanmin(col0, axis=(0, 1)), np.nanmin(col1, axis=(0, 1))), axis=0)
                vmaxs = np.nanmax((np.nanmax(col0, axis=(0, 1)), np.nanmax(col1, axis=(0, 1))), axis=0)
            else:
                col0, col1, col2, col3 = x_true_plot, y_pred_plot, y_true_plot, y_pred_plot - y_true_plot
                vmins = np.nanmin((np.nanmin(col0, axis=(0, 1)), np.nanmin(col1, axis=(0, 1)), np.nanmin(col2, axis=(0, 1))), axis=0)
                vmaxs = np.nanmax((np.nanmax(col0, axis=(0, 1)), np.nanmax(col1, axis=(0, 1)), np.nanmax(col2, axis=(0, 1))), axis=0)
        cols = {"col0": col0, "col1": col1, "col2": col2, "col3": col3}

        fig, ax = plt.subplots(num_quantities, num_columns, figsize=(6. * num_columns, 4. * num_quantities))
        ax = np.reshape(ax, (num_quantities, num_columns))  # force dimensions for the for cycle

        for irow in range(num_quantities):
            for icol in range(num_columns):
                if y_true_plot is not None and icol == num_columns - 1:  # diff panel with different vmin, vmax
                    sp = ax[irow, icol].imshow(cols[f"col{icol}"][:, :, irow], aspect="equal", origin="lower", cmap=cmap)
                else:
                    sp = ax[irow, icol].imshow(cols[f"col{icol}"][:, :, irow], aspect="equal", origin="lower", cmap=cmap,
                                               vmin=vmins[irow], vmax=vmaxs[irow])
                ax[irow, icol].set_xticks([])
                ax[irow, icol].set_yticks([])
                ax[irow, icol].set_yticklabels([])
                ax[irow, icol].set_xticklabels([])

                if irow == 0:
                    ax[irow, icol].set_title(used_columns[icol])

                # add common colorbars
                if icol < num_columns - int(y_true_plot is not None):
                    divider = make_axes_locatable(ax[irow, icol])
                    cax = divider.append_axes(**cbar_kwargs)
                    cbar = plt.colorbar(sp, cax=cax)
                    cbar.ax.set_ylabel(used_names[irow])

                # add colorbar to prediction - SOT column
                if y_true_plot is not None and icol == num_columns - 1:
                    divider = make_axes_locatable(ax[irow, icol])
                    cax = divider.append_axes(**cbar_kwargs)
                    cbar = plt.colorbar(sp, cax=cax)
                    cbar.ax.set_ylabel(used_names[irow])

        """
        # Bt best
        for i in range(4):
            ax[0, i].add_patch(patches.Rectangle((150, 101), 15, 15, linewidth=2, edgecolor="y", facecolor="none", linestyle="-"))

        # Bp worst
        ax[0, 2].add_patch(patches.Rectangle((144, 20), 15, 15, linewidth=2, edgecolor="r", facecolor="none", linestyle="-"))
        ax[0, 2].add_patch(patches.Rectangle((144, 20), 15, 15, linewidth=2, edgecolor="r", facecolor="none", linestyle="-"))
        ax[0, 2].add_patch(patches.Rectangle((17, 50), 15, 15, linewidth=2, edgecolor="r", facecolor="none", linestyle="-"))
        ax[0, 2].add_patch(patches.Rectangle((35, 144), 15, 15, linewidth=2, edgecolor="r", facecolor="none", linestyle="-"))
        ax[0, 2].add_patch(patches.Rectangle((115, 150), 20, 15, linewidth=2, edgecolor="r", facecolor="none", linestyle="-"))
        ax[0, 2].add_patch(patches.Rectangle((170, 160), 20, 40, angle=55, linewidth=2, edgecolor="b", facecolor="none", linestyle="-"))
        
        # Bt worst
        ax[0, 2].add_patch(patches.Rectangle((196, 0), 10, 72, linewidth=2, edgecolor="r", facecolor="none", linestyle="-"))
        
        # Br best
        for i in range(3):
            ax[0, i].add_patch(patches.Rectangle((175, 89), 35, 50, linewidth=2, edgecolor="y", facecolor="none", linestyle="-"))
            ax[0, i].add_patch(patches.Rectangle((175, 89), 35, 50, linewidth=2, edgecolor="y", facecolor="none", linestyle="-"))

        # Br worst
        for i in range(3):
            ax[0, i].add_patch(patches.Rectangle((146, 40), 20, 30, linewidth=2, edgecolor="r", facecolor="none", linestyle="-"))
            ax[0, i].add_patch(patches.Rectangle((170, 5), 33, 18, linewidth=2, edgecolor="b", facecolor="none", linestyle="-"))

        """

        if suptitle is not None:
            plt.suptitle(suptitle)

        plt.draw()
        fig.tight_layout()

        if num_plots == 1:
            fig_name = f"quantity_map{suf}.{fig_format}"
        else:
            fig_name = f"quantity_map{suf}_{ifig:0{n_digits}d}.{fig_format}"

        fig_full_name = path.join(outdir_HMI_to_SOT, subfolder, fig_name)
        check_dir(fig_full_name, is_file=True)
        fig.savefig(fig_full_name, format=fig_format, **savefig_kwargs, **pil_kwargs)
        plt.close(fig)

    change_params(offset, reset=True)


@safe_call
def plot_alignment(image1: np.ndarray, image2: np.ndarray, image3: np.ndarray | None = None,
                   offset: float = 0.,
                   subfolder: str = "alignment", suptitle: str | None = None,
                   fig_format: str = default_fig_format, suf: str = "", quiet: bool = False) -> None:
    if not quiet:
        print("Alignment maps")

    change_params(offset)

    cmap = "gray"

    nrows = 2 + (image3 is not None)

    if np.ndim(image1) == 4:  # prepared for NN (Ndata, ny, nx, nq)
        image1, image2 = image1[0], image2[0]
        image1, image2 = np.transpose(image1, (2, 0, 1)), np.transpose(image2, (2, 0, 1))
        if image3 is not None:
            image3 = image3[0]
            image3 = np.transpose(image3, (2, 0, 1))

    cbarlabels = add_unit_to_names(quantity_names_short_latex, latex_output=True)
    xylabel = "Pixel"
    xticks = np.arange(0, np.shape(image1)[-1], 100)
    yticks = np.arange(0, np.shape(image1)[-2], 100)

    fig, ax = plt.subplots(nrows + 1, len(image1), figsize=(15.21, 5.17 * nrows), sharex=True, sharey=True,
                           gridspec_kw={"width_ratios": [1]*len(image1),
                                        "height_ratios": [0.2] + [1]*nrows})
    for i in range(len(image1)):
        if image3 is not None:
            clim = [np.min([image1[i], image2[i], image3[i]]), np.max([image1[i], image2[i], image3[i]])]
        else:
            clim = [np.min([image1[i], image2[i]]), np.max([image1[i], image2[i]])]
        y_max, x_max = np.shape(image1[i])

        ax[1, i].imshow(image1[i], origin="lower", extent=[0, x_max, 0, y_max], vmin=clim[0], vmax=clim[1],
                        aspect="auto", cmap=cmap)

        im = ax[2, i].imshow(image2[i], origin="lower", extent=[0, x_max, 0, y_max], vmin=clim[0], vmax=clim[1],
                             aspect="auto", cmap=cmap)

        if image3 is not None:
            im = ax[3, i].imshow(image3[i], origin="lower", extent=[0, x_max, 0, y_max], vmin=clim[0], vmax=clim[1],
                                 aspect="auto", cmap=cmap)

        if i == 0:
            ax[1, i].set_ylabel(xylabel)
            ax[1, i].set_yticks(yticks)
            ax[2, i].set_ylabel(xylabel)
            ax[2, i].set_yticks(yticks)

            if image3 is not None:
                ax[3, i].set_ylabel(xylabel)
                ax[3, i].set_yticks(yticks)

        if image3 is None:
            ax[2, i].set_xlabel(xylabel)
            ax[2, i].set_xticks(xticks)
        else:
            ax[3, i].set_xlabel(xylabel)
            ax[3, i].set_xticks(xticks)

        ax[0, i].set_axis_off()
        # to make subplots the same size
        divider = make_axes_locatable(ax[0, i])
        cax = divider.append_axes(position="top", size="100%", pad=0.)
        cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
        cbar.ax.xaxis.set_label_position("top")
        cbar.ax.set_xlabel(cbarlabels[i], labelpad=10)

    if suptitle is not None:
        plt.suptitle(suptitle)

    fig.tight_layout()
    plt.draw()

    fig_name = f"alignment{suf}.{fig_format}"
    fig_full_name = path.join(outdir_HMI_to_SOT, subfolder, fig_name)
    check_dir(fig_full_name, is_file=True)
    fig.savefig(fig_full_name, format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


@safe_call
def plot_scatter_plots(y_true: np.ndarray, y_pred: np.ndarray,
                       used_quantities: np.ndarray | None = None,
                       offset: float = 0., subfolder: str = "", suptitle: str | None = None,
                       fig_format: str = default_fig_format,
                       suf: str = "", quiet: bool = False) -> None:
    if not quiet:
        print("Scatter plots")
    change_params(offset)

    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    if isinstance(y_true, tuple | list):
        # in case some of the observations are empty (skip it)
        y_true, used_quantities = filter_empty_data(y_true, used_quantities)
        y_pred, used_quantities = filter_empty_data(y_pred, used_quantities)

    used_names = quantity_names[used_indices(used_quantities)]
    used_units = add_unit_to_names(quantity_names, return_units=True, latex_output=True)[1][used_indices(used_quantities)]

    used_units_bare = add_unit_to_names(quantity_names, return_units=True)[1][used_indices(used_quantities)]
    used_units_bare = np.array([rreplace(unit.replace("(", "", 1), ")", "", 1) for unit in used_units_bare])
    if used_quantities[0]:
        used_units_bare[0] = ""  # no quiet sun norm.
    rmse_format = np.array(["5.3f", "4.1f", "4.1f", "4.1f"])[used_indices(used_quantities)]

    num_quantities = len(used_names)

    LW_scatter = 2.5

    # limit = 0.25
    s = 30  # scaling parameter (marker size)

    # xticks = safe_arange(0., 100., 25., endpoint=True)
    # yticks = safe_arange(0., 100., 25., endpoint=True)

    # define the lines
    if isinstance(y_true, tuple | list):  # quantity first
        m = np.min(([np.min(y) for y in y_pred], [np.min(y) for y in y_true]), axis=0)
        M = np.max(([np.max(y) for y in y_pred], [np.max(y) for y in y_true]), axis=0)
    else:
        m, M = np.min((y_pred, y_true), axis=(0, 1, 2, 3)), np.max((y_pred, y_true), axis=(0, 1, 2, 3))

    x_line = np.array([np.min(m) - 0.1 * np.abs(np.min(m)), np.max(M) * 1.1])
    y_line = x_line
    y1p_line, y1m_line = y_line * 1.1, y_line / 1.1
    y2p_line, y2m_line = y_line * 1.2, y_line / 1.2
    l0, l10, l20, eb = "y-", "m-", "c-", "r"
    lab_line0, lab_line10, lab_line20 = r"0\% error", r"10\% error", r"20\% error"

    left, right = m - 0.1 * np.abs(m), M * 1.1
    bottom, top = left, right

    if isinstance(y_true, tuple | list):  # quantity first
        RMSE, R2, SAM = zip(*[compute_metrics(y_true[i], y_pred[i], return_r2=True, return_sam=True, all_to_one=False)
                              for i in range(num_quantities)])
        RMSE, R2, SAM = np.ravel(RMSE), np.ravel(R2), np.ravel(SAM)
    else:
        RMSE, R2, SAM = compute_metrics(y_true, y_pred, return_r2=True, return_sam=True, all_to_one=False)

    fig, ax = plt.subplots(1, num_quantities, figsize=(4.5 * num_quantities, 6))
    ax = np.ravel(ax)  # to force iterable for the for cycle

    if isinstance(y_true, tuple | list):  # quantity first
        y_true = deepcopy([np.ravel(y) for y in y_true])
        y_pred = deepcopy([np.ravel(y) for y in y_pred])
    else:
        y_true, y_pred = np.reshape(y_true, (-1, num_quantities)), np.reshape(y_pred, (-1, num_quantities))
        y_true, y_pred = np.transpose(y_true), np.transpose(y_pred)

    for i, axis in enumerate(ax):
        # lines
        lns1 = axis.plot(x_line, y_line, l0, label=lab_line0, linewidth=LW_scatter, zorder=3)
        lns2 = axis.plot(x_line, y1p_line, l10, label=lab_line10, linewidth=LW_scatter, zorder=4)
        axis.plot(x_line, y1m_line, l10, linewidth=LW_scatter, zorder=5)
        lns3 = axis.plot(x_line, y2p_line, l20, label=lab_line20, linewidth=LW_scatter, zorder=6)
        axis.plot(x_line, y2m_line, l20, linewidth=LW_scatter, zorder=7)
        # data
        axis.scatter(y_true[i], y_pred[i], c="black", s=s, zorder=2)
        # axis.errorbar(y_true[i], y_pred[i], yerr=error_pred[i], xerr=error_true[i], fmt=eb, ls="", elinewidth=0.5, zorder=1)

        axis.set_xlabel(f"Actual {used_units[i]}")
        axis.set_ylabel(f"Predicted {used_units[i]}")
        axis.tick_params(axis="both")
        axis.axis("square")
        axis.set_title(used_names[i].capitalize())
        # axis.set_xticks(xticks)
        # axis.set_yticks(yticks)
        axis.set_ylim(bottom=bottom[i], top=top[i])
        axis.set_xlim(left=left[i], right=right[i])

        axis.text(0.8, 0.15,
                   r"\["  # every line is a separate raw string...
                   r"\begin{split}"  # ...but they are all concatenated by the interpreter :-)
                   r"\mathsf{RMSE} &= " + f"{RMSE[i]:{rmse_format[i]}}" + f"\\text{{ {used_units_bare[i]}}}" + r"\\"
                   r"\mathsf{R}^2 &= " + f"{R2[i]:4.2f}" + r"\\"
                   r"\mathsf{SAM} &= " + f"{SAM[i]:4.1f}" + r"\text{ deg}"
                   r"\end{split}"
                   r"\]",
                   horizontalalignment="center", verticalalignment="center", transform=axis.transAxes)

        lns = lns1 + lns2 + lns3
        labs = [l.get_label() for l in lns]
        axis.legend(lns, labs, loc="upper left", frameon=False)

    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.draw()
    fig.tight_layout()

    fig_name = f"scatter_plot{suf}.{fig_format}"
    fig_full_name = path.join(outdir_HMI_to_SOT, subfolder, fig_name)
    check_dir(fig_full_name, is_file=True)
    fig.savefig(fig_full_name, format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


@safe_call
def plot_hist2d_plots(y_true: np.ndarray, y_pred: np.ndarray,
                      used_quantities: np.ndarray | None = None,
                      norm: Literal["asinh", "function", "functionlog", "linear", "log", "logit", "symlog"] |
                            mpl.colors.Normalize | None = None,
                      offset: float = 0., subfolder: str = "", suptitle: str | None = None,
                      fig_format: str = default_fig_format,
                      suf: str = "", quiet: bool = False) -> None:
    if not quiet:
        print("2D histogram plots")
    change_params(offset)

    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    if isinstance(y_true, tuple | list):
        # in case some of the observations are empty (skip it)
        y_true, used_quantities = filter_empty_data(y_true, used_quantities)
        y_pred, used_quantities = filter_empty_data(y_pred, used_quantities)

    used_names = quantity_names[used_indices(used_quantities)]
    used_units = add_unit_to_names(quantity_names, return_units=True, latex_output=True)[1][used_indices(used_quantities)]

    used_units_bare = add_unit_to_names(quantity_names, return_units=True)[1][used_indices(used_quantities)]
    used_units_bare = np.array([rreplace(unit.replace("(", "", 1), ")", "", 1) for unit in used_units_bare])
    if used_quantities[0]:
        used_units_bare[0] = ""  # no quiet sun norm.
    rmse_format = np.array(["5.3f", "4.1f", "4.1f", "4.1f"])[used_indices(used_quantities)]

    num_quantities = len(used_names)

    LW_scatter = 2.5

    # define the lines
    if isinstance(y_true, tuple | list):  # quantity first
        m = np.min(([np.min(y) for y in y_pred], [np.min(y) for y in y_true]), axis=0)
        M = np.max(([np.max(y) for y in y_pred], [np.max(y) for y in y_true]), axis=0)
    else:
        m, M = np.min((y_pred, y_true), axis=(0, 1, 2, 3)), np.max((y_pred, y_true), axis=(0, 1, 2, 3))

    x_line = np.array([np.min(m) - 0.1 * np.abs(np.min(m)), np.max(M) * 1.1])
    y_line = x_line
    y1p_line, y1m_line = y_line * 1.1, y_line / 1.1
    y2p_line, y2m_line = y_line * 1.2, y_line / 1.2
    l0, l10, l20 = "r:", "r--", "r-"
    lab_line0, lab_line10, lab_line20 = r"0\% error", r"10\% error", r"20\% error"

    left, right = m - 0.1 * np.abs(m), M * 1.1
    bottom, top = left, right

    if isinstance(y_true, tuple | list):  # quantity first
        RMSE, R2, SAM = zip(*[compute_metrics(y_true[i], y_pred[i], return_r2=True, return_sam=True, all_to_one=False)
                              for i in range(num_quantities)])
        RMSE, R2, SAM = np.ravel(RMSE), np.ravel(R2), np.ravel(SAM)
    else:
        RMSE, R2, SAM = compute_metrics(y_true, y_pred, return_r2=True, return_sam=True, all_to_one=False)

    if isinstance(y_true, tuple | list):  # quantity first
        y_true = deepcopy([np.ravel(y) for y in y_true])
        y_pred = deepcopy([np.ravel(y) for y in y_pred])
    else:
        y_true, y_pred = np.reshape(y_true, (-1, num_quantities)), np.reshape(y_pred, (-1, num_quantities))
        y_true, y_pred = np.transpose(y_true), np.transpose(y_pred)

    fig, ax = plt.subplots(1, num_quantities, figsize=(4.5 * num_quantities, 6))
    ax = np.ravel(ax)  # to force iterable for the for cycle

    for i, axis in enumerate(ax):
        # lines
        lns1 = axis.plot(x_line, y_line, l0, label=lab_line0, linewidth=LW_scatter, zorder=3)
        lns2 = axis.plot(x_line, y1p_line, l10, label=lab_line10, linewidth=LW_scatter, zorder=4)
        axis.plot(x_line, y1m_line, l10, linewidth=LW_scatter, zorder=5)
        lns3 = axis.plot(x_line, y2p_line, l20, label=lab_line20, linewidth=LW_scatter, zorder=6)
        axis.plot(x_line, y2m_line, l20, linewidth=LW_scatter, zorder=7)
        # data
        axis.hist2d(y_true[i], y_pred[i], bins=100, zorder=2, norm=norm)

        axis.set_xlabel(f"Actual {used_units[i]}")
        axis.set_ylabel(f"Predicted {used_units[i]}")
        axis.tick_params(axis="both")
        axis.axis("square")
        axis.set_title(used_names[i].capitalize())
        # axis.set_xticks(xticks)
        # axis.set_yticks(yticks)
        axis.set_ylim(bottom=bottom[i], top=top[i])
        axis.set_xlim(left=left[i], right=right[i])

        axis.text(0.8, 0.15,
                   r"\["  # every line is a separate raw string...
                   r"\begin{split}"  # ...but they are all concatenated by the interpreter :-)
                   r"\mathsf{RMSE} &= " + f"{RMSE[i]:{rmse_format[i]}}" + f"\\text{{ {used_units_bare[i]}}}" + r"\\"
                   r"\mathsf{R}^2 &= " + f"{R2[i]:4.2f}" + r"\\"
                   r"\mathsf{SAM} &= " + f"{SAM[i]:4.1f}" + r"\text{ deg}"
                   r"\end{split}"
                   r"\]",
                   horizontalalignment="center", verticalalignment="center", transform=axis.transAxes)

        lns = lns1 + lns2 + lns3
        labs = [l.get_label() for l in lns]
        axis.legend(lns, labs, loc="upper left", frameon=False)

    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.draw()
    fig.tight_layout()

    fig_name = f"hist2d_plot{suf}.{fig_format}"
    fig_full_name = path.join(outdir_HMI_to_SOT, subfolder, fig_name)
    check_dir(fig_full_name, is_file=True)
    fig.savefig(fig_full_name, format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


@safe_call
def plot_error_evaluation(y_true: np.ndarray, y_pred: np.ndarray,
                          used_quantities: np.ndarray | None = None,
                          offset: float = 0., fig_format: str = default_fig_format,
                          subfolder: str = "", suf: str = "", quiet: bool = False) -> None:
    if not quiet:
        print("Plot error in quantiles")
    change_params(offset)

    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    if isinstance(y_true, tuple | list):
        # in case some of the observations are empty (skip it)
        y_true, used_quantities = filter_empty_data(y_true, used_quantities)
        y_pred, used_quantities = filter_empty_data(y_pred, used_quantities)

    add_all = False

    axis_B_to_I_ratio = 1000.

    used_names = np.array([name.capitalize() for name in quantity_names])
    # used_names = add_unit_to_names(used_names)
    indices_used = used_indices(used_quantities)
    used_names = used_names[indices_used]

    inds_I = 0 if indices_used[0] else None
    inds_B = np.where(indices_used)[0][1:] if np.any(indices_used[1:]) else None

    percentile = safe_arange(0., 100., 0.5, endpoint=True)

    # define the lines
    x_line = safe_arange(-150., 150., endpoint=True)
    y_line_25 = np.ones(np.shape(x_line)) * 25.
    y_line_50 = np.ones(np.shape(x_line)) * 50.
    y_line_100 = np.ones(np.shape(x_line)) * 100.
    l25, l50, l100 = "k--", "k--", "k--"

    one_sigma = 68.2
    sigma_c, sigma_ls = "k", "--"

    shift = 3.  # Control ranges of axes (from 0 - shift to 100 + shift)

    xticks = safe_arange(0., 100., 10., endpoint=True)

    left, right = -shift, 100. + shift
    bottom, top = -shift, 250 + shift

    if isinstance(y_true, tuple | list):  # quantity first
        quantile = np.array([my_quantile(percentile=percentile, all_to_one=False)(y_true[i], y_pred[i]).numpy()
                             for i in range(len(y_true))])
        quantile = np.transpose([np.ravel(q) for q in quantile])
    else:
        quantile = my_quantile(percentile=percentile, all_to_one=False)(y_true, y_pred).numpy()

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    if add_all:
        used_names = np.insert(used_names, 0, "All data")
        quantile_all = my_quantile(percentile=percentile, all_to_one=True)(y_true, y_pred).numpy()

        ax1.plot(percentile, quantile_all, linewidth=3, zorder=100)

    # constant error lines
    ax1.plot(x_line, y_line_25, l25, zorder=101)
    ax1.plot(x_line, y_line_50, l50, zorder=102)
    ax1.plot(x_line, y_line_100, l100, zorder=103)

    ax1.axvline(one_sigma, color=sigma_c, ls=sigma_ls, zorder=103)

    ax1.set_xlabel("Percentile")
    ax1.set_ylabel("Absolute error (G)")

    ax1.set_xticks(xticks)
    ax1.set_xlim(left=left, right=right)

    if inds_I is not None:
        ax2 = ax1.twinx()  # instantiate a second axis that shares the same x-axis
        lns_i = ax2.plot(percentile, quantile[:, inds_I], linewidth=2)
        ax2.set_ylabel("Absolute error (quiet-Sun normalised)")  # we already handled the x-label with ax1
    else:
        lns_i = []

    if inds_B is not None:
        if inds_I is not None:  # skip the first color if it is used by the intensity
            ax1._get_lines.get_next_color()

        lns_b = ax1.plot(percentile, quantile[:, inds_B], linewidth=2)
        _, ax1_max = ax1.get_ylim()
        ax1.set_ylim(bottom=bottom, top=np.min((top, ax1_max)))
    else:
        lns_b = []

    if inds_I is not None and inds_B is not None:
        # Adjust the limits of ax1 and ax2 to be in the specified ratio
        ax1_min, ax1_max = ax1.get_ylim()
        ax2_min, ax2_max = ax1_min / axis_B_to_I_ratio, ax1_max / axis_B_to_I_ratio
        ax2.set_ylim(ax2_min, ax2_max)

    lns = lns_i + lns_b
    labels = used_names
    ax1.legend(lns, labels, loc="upper left", ncol=1)

    plt.draw()
    fig.tight_layout()

    fig_name = f"quantile_error_plot{suf}.{fig_format}"
    fig_full_name = path.join(outdir_HMI_to_SOT, subfolder, fig_name)
    check_dir(fig_full_name, is_file=True)
    fig.savefig(fig_full_name, format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


@safe_call
def plot_error_in_bins(y_true: np.ndarray, y_pred: np.ndarray,
                       used_quantities: np.ndarray | None = None,
                       offset: float = 0., fig_format: str = default_fig_format,
                       subfolder: str = "", suf: str = "", quiet: bool = False) -> None:
    if not quiet:
        print("Plot error in bins")
    change_params(offset)

    color1, color2 = "b", "r"

    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    bins = [[0., *np.arange(0.3, 1.4, 0.1), 5.],
            [*np.arange(-10000., -2000., 500.), *np.arange(-2000., 2001., 100.), *np.arange(2000., 10000., 500.)],
            [*np.arange(-10000., -2000., 500.), *np.arange(-2000., 2001., 100.), *np.arange(2000., 10000., 500.)],
            [*np.arange(-10000., -2000., 500.), *np.arange(-2000., 2001., 100.), *np.arange(2000., 10000., 500.)]]
    bins = [_bin for i, _bin in enumerate(bins) if used_quantities[i]]

    if isinstance(y_true, tuple | list):
        # in case some of the observations are empty (skip it)
        y_true, used_quantities = filter_empty_data(y_true, used_quantities)
        y_pred, used_quantities = filter_empty_data(y_pred, used_quantities)

    get_data = lambda data, i: data[i] if isinstance(data, tuple | list) else data[..., i]

    errors = []
    counts = []

    for i in range(len(bins)):
        error, count, _ = error_in_bins(array_master=get_data(y_true, i), array_slave=get_data(y_pred, i), bins=bins[i])
        errors.append(error)
        counts.append(count)

    used_names = quantity_names[used_indices(used_quantities)]
    used_units = add_unit_to_names(quantity_names, return_units=True, latex_output=True)[1][used_indices(used_quantities)]

    fig, ax1 = plt.subplots(1, np.sum(used_quantities), figsize=(4.5 * np.sum(used_quantities), 6))
    ax1 = np.ravel(ax1)

    for i in range(len(bins)):
        mask = np.isfinite(errors[i])
        _bins = np.array(bins[i][:-1])[mask]  # errors are only within the bin, but "bins" defines edges (-1)
        ax1[i].plot(_bins, errors[i][mask], color=color1, label="RMSE")
        ax1[i].set_ylim(bottom=0.)
        ax1[i].set_xlabel(f"Predicted {used_names[i]} {used_units[i]}")
        ax1[i].set_ylabel(f"RMSE {used_units[i]}", color=color1)
        ax1[i].tick_params(axis="y", labelcolor=color1)

        ax2 = ax1[i].twinx()
        ax2.plot(_bins, counts[i][mask], color=color2, label="Counts")
        ax2.set_ylabel("Counts", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_yscale("log")
        ax2.minorticks_off()

        if used_quantities[0] and i == 0:
            ax1[i].axhline(0.025, color="k", linestyle="--")
            ax1[i].axhline(0.05, color="k", linestyle="--")
        else:
            ax1[i].axhline(50., color="k", linestyle="--")
            ax1[i].axhline(100., color="k", linestyle="--")

    plt.draw()
    fig.tight_layout()

    fig_name = f"bin_error_plot{suf}.{fig_format}"
    fig_full_name = path.join(outdir_HMI_to_SOT, subfolder, fig_name)
    check_dir(fig_full_name, is_file=True)
    fig.savefig(fig_full_name, format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


@safe_call
def plot_error_hist(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    ae_limits: list[float],
                    n_bins: int = 51,
                    offset: float = 0.,
                    subfolder: str = "",
                    fig_format: str = default_fig_format,
                    suf: str = "") -> None:
    change_params(offset)

    _LW = 2.5

    error = np.ravel(y_true - y_pred)
    y_true = np.ravel(y_true)
    bins = np.linspace(np.min(y_true), np.max(y_true), num=n_bins)
    bin_centres = (bins[:-1] + bins[1:]) / 2.

    if np.min(y_true) < 0.:
        unit = "G"
    else:
        unit = ""

    colors = plt.cm.viridis(np.linspace(0., 1., len(ae_limits)))
    alphas = np.linspace(1., 0.5, len(ae_limits))

    combined_counts = np.zeros(len(bin_centres))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax2 = ax.twinx()

    for ith, ae_limit in enumerate(ae_limits):
        counts = np.zeros(len(bin_centres))
        density = np.zeros(len(bin_centres))

        for i in range(len(bin_centres)):
            mask = np.logical_and(y_true >= bins[i], y_true < bins[i + 1])
            counts[i] = np.sum(np.logical_and(mask, error > ae_limit))
            density[i] = counts[i] / np.clip(np.sum(mask), 0.1, None)  # 0/0 --> 0

        if unit:
            label = f"$\mathsf{{RMSE}} > {ae_limit}$ {unit}"
        else:
            label = f"$\mathsf{{RMSE}} > {ae_limit}$"

        ax.bar(bin_centres, counts, width=np.diff(bins), color=colors[ith], alpha=alphas[ith], label=label)
        ax2.plot(bin_centres, density, linewidth=_LW)
        combined_counts += counts

    # Crop the range
    non_empty_bins = np.where(combined_counts > 0)[0]
    if len(non_empty_bins) > 0:
        x_min, x_max = bins[non_empty_bins[0]], bins[non_empty_bins[-1] + 1]
        ax.set_xlim(x_min, x_max)

    if unit:
        ax.set_xlabel(f"Actual value ({unit})")
    else:
        ax.set_xlabel(f"Actual value")

    ax.set_ylabel("Counts")
    ax2.set_ylabel("Count density")
    ax2.tick_params(axis="y")
    ax2.minorticks_off()
    ax.legend()

    plt.draw()
    fig.tight_layout()

    fig_name = f"error_hist{suf}.{fig_format}"
    fig_full_name = path.join(outdir_HMI_to_SOT, subfolder, fig_name)
    check_dir(fig_full_name, is_file=True)
    fig.savefig(fig_full_name, format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


@safe_call
def quantity_control_plots(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           quantity: Literal["ic", "bp", "bt", "br"],
                           ae_limits: list[float] | None = None,
                           offset: float = 0.,
                           subfolder: str = "",
                           fig_format: str = default_fig_format,
                           quiet: bool = False,
                           suf: str = "") -> None:
    change_params(offset)

    if not quiet:
        print("Quantity control plots")

    if quantity.lower() == "ic":
        rmse_format = "5.3f"
        used_units_bare = ""
        used_units = "(quiet-Sun norm.)"
        quantity_name = "intensity"
        bins = np.array([0., *np.arange(0.3, 1.4, 0.1), 5.])
        lims = [0.05, 0.1]
    else:
        rmse_format = "4.1f"
        used_units_bare = "G"
        used_units = "(G)"
        bins = np.array([*np.arange(-10000., -2000., 500.), *np.arange(-2000., 2001., 100.), *np.arange(2000., 10000., 500.)])
        lims = [50, 100]
        if quantity.lower() == "bp":
            quantity_name = "zonal magnetic field"
        elif quantity.lower() == "bt":
            quantity_name = "azimuthal magnetic field"
        else:
            quantity_name = "radial magnetic field"

    if ae_limits is None:
        ae_limits = lims

    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    bin_centres = (bins[:-1] + bins[1:]) / 2.
    alphas = np.linspace(1., 0.8, len(ae_limits))

    error, count, _ = error_in_bins(array_master=y_true, array_slave=y_pred, bins=bins)
    unsigned_error = np.abs(np.ravel(y_true - y_pred))
    mask = np.isfinite(error)
    _bins = np.array(bin_centres)[mask]
    error, count = error[mask], count[mask]

    color1, color2 = "b", "r"

    LW_scatter = 2.5
    m, M = np.min((y_pred, y_true)), np.max((y_pred, y_true))

    x_line = np.array([np.min(m) - 0.1 * np.abs(np.min(m)), np.max(M) * 1.1])
    y_line = x_line
    y1p_line, y1m_line = y_line * 1.1, y_line / 1.1
    y2p_line, y2m_line = y_line * 1.2, y_line / 1.2
    l0, l10, l20 = "r:", "r--", "r-"
    lab_line0, lab_line10, lab_line20 = r"0\% error", r"10\% error", r"20\% error"

    left, right = m - 0.1 * np.abs(m), M * 1.1
    bottom, top = left, right

    RMSE, R2, SAM = compute_metrics(y_true, y_pred, return_r2=True, return_sam=True, all_to_one=True)

    fig, ax = plt.subplots(1, 3, figsize=(20, 6))

    # lines
    lns1 = ax[0].plot(x_line, y_line, l0, label=lab_line0, linewidth=LW_scatter, zorder=3)
    lns2 = ax[0].plot(x_line, y1p_line, l10, label=lab_line10, linewidth=LW_scatter, zorder=4)
    ax[0].plot(x_line, y1m_line, l10, linewidth=LW_scatter, zorder=5)
    lns3 = ax[0].plot(x_line, y2p_line, l20, label=lab_line20, linewidth=LW_scatter, zorder=6)
    ax[0].plot(x_line, y2m_line, l20, linewidth=LW_scatter, zorder=7)
    # data
    ax[0].hist2d(y_true, y_pred, bins=100, zorder=2, norm="log")

    ax[0].set_xlabel(f"Actual {used_units}")
    ax[0].set_ylabel(f"Predicted {used_units}")
    ax[0].tick_params(axis="both")
    ax[0].axis("square")
    ax[0].set_ylim(bottom=bottom, top=top)
    ax[0].set_xlim(left=left, right=right)

    ax[0].text(0.8, 0.15,
               r"\["  # every line is a separate raw string...
               r"\begin{split}"  # ...but they are all concatenated by the interpreter :-)
               r"\mathsf{RMSE} &= " + f"{RMSE[0]:{rmse_format}}" + f"\\text{{ {used_units_bare}}}" + r"\\"
               r"\mathsf{R}^2 &= " + f"{R2[0]:4.2f}" + r"\\"
               r"\mathsf{SAM} &= " + f"{SAM[0]:4.1f}" + r"\text{ deg}"
               r"\end{split}"
               r"\]",
               horizontalalignment="center", verticalalignment="center", transform=ax[0].transAxes)

    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax[0].legend(lns, labs, loc="upper left")

    ax[1].plot(_bins, error, color=color1, label="RMSE")
    ax[1].set_ylim(bottom=0.)
    ax[1].set_xlim(left=left, right=right)
    ax[1].set_xlabel(f"Actual {used_units}")
    ax[1].set_ylabel(f"RMSE {used_units}", color=color1)
    ax[1].tick_params(axis="y", labelcolor=color1)

    ax2 = ax[1].twinx()
    ax2.plot(_bins, count, color=color2, label="Counts")
    ax2.set_ylabel("Pixel counts", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_yscale("log")
    ax2.minorticks_off()

    for ae_limit in ae_limits:
        ax[1].axhline(ae_limit, color="k", linestyle="--")

    ax2 = ax[2].twinx()
    lns = []

    colors = ["b", "r", "g", "c", "m", "y"]
    tabcolors = ["tab:blue", "tab:red", "tab:green", "tab:cyan", "tab:purple", "tab:olive"]

    for ith, ae_limit in enumerate(ae_limits):
        counts = np.zeros(len(bin_centres))
        density = np.zeros(len(bin_centres))

        for i in range(len(bin_centres)):
            mask = np.logical_and(y_true >= bins[i], y_true < bins[i + 1])
            counts[i] = np.sum(np.logical_and(mask, unsigned_error > ae_limit))
            density[i] = counts[i] / np.clip(np.sum(mask), 0.1, None)  # 0/0 --> 0

        if used_units_bare:
            label = f"$\mathsf{{RMSE}} > {ae_limit}$ {used_units_bare}"
        else:
            label = f"$\mathsf{{RMSE}} > {ae_limit}$"

        # ax[2].bar(bin_centres, counts, width=np.diff(bins), alpha=alphas[ith], label=label, fill=True, hatch="xx", edgecolor="k")
        ln = ax[2].step(bin_centres, counts, where="mid", linewidth=LW_scatter, color=colors[ith], label=label)
        lns += ln
        ax2.plot(bin_centres, density, linewidth=LW_scatter + 1, linestyle="--", color=tabcolors[ith])

    ax[2].set_xlim(left=left, right=right)
    ax[2].set_ylim(bottom=0)

    ax[2].set_xlabel(f"Actual {used_units}")
    ax[2].set_ylabel("Pixel counts")

    labs = [l.get_label() for l in lns]
    ax[2].legend(handles=lns, labels=labs,
                 bbox_to_anchor=(0., 1.02, 1., 0.2), loc="lower left", mode="expand", borderaxespad=0., ncol=2)

    ax2.set_ylabel("Pixel density")
    ax2.tick_params(axis="y")
    ax2.minorticks_off()

    # plt.suptitle(quantity_name.title())
    ax[1].set_title(quantity_name.title())

    plt.draw()
    fig.tight_layout()

    fig_name = f"{quantity}_control_plot{suf}.{fig_format}"
    fig_full_name = path.join(outdir_HMI_to_SOT, subfolder, fig_name)
    check_dir(fig_full_name, is_file=True)
    fig.savefig(fig_full_name, format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


@safe_call
def plot_error_density_plots(y_true: np.ndarray, y_pred: np.ndarray,
                             used_quantities: np.ndarray | None = None,
                             max_samples_to_use: int | None = None,
                             offset: float = 0., subfolder: str = "", suptitle: str | None = None,
                             fig_format: str = default_fig_format,
                             suf: str = "", quiet: bool = False) -> None:
    if not quiet:
        print("Plot error and point densities")
    change_params(offset)

    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]
    if max_samples_to_use is None: max_samples_to_use = len(y_true)

    if isinstance(y_true, tuple | list):
        # in case some of the observations are empty (skip it)
        y_true, used_quantities = filter_empty_data(y_true, used_quantities)
        y_pred, used_quantities = filter_empty_data(y_pred, used_quantities)

    used_titles = quantity_names[used_indices(used_quantities)]
    used_names, used_units = add_unit_to_names(quantity_names_short_latex, return_units=True, latex_output=True)
    used_names = used_names[used_indices(used_quantities)]
    used_units = used_units[used_indices(used_quantities)]

    num_quantities = len(used_names)

    # there is too much data to do the density plot, select random subset
    rng = np.random.default_rng(seed=42)
    nz, ny, nx, _ = np.shape(y_true)
    n_samples = nz * ny * nx
    max_samples_to_use = np.min((max_samples_to_use, n_samples))
    indices = rng.choice(n_samples, size=max_samples_to_use, replace=False)
    i = np.unravel_index(indices, (nz, ny, nx))
    y_true, y_pred = y_true[i], y_pred[i]

    nbins = 50
    cmap = "viridis_r"  # cmap of points
    fs = SMALL_SIZE + 6

    # define the line styles
    ls_hor = "r--"

    fig, ax = plt.subplots(ncols=num_quantities, nrows=1, figsize=(4.5 * num_quantities, 6))
    ax = np.ravel(ax)  # to force iterable for the for cycle

    for i in range(num_quantities):
        xi, yi, zi = kernel_density_estimation_2d(np.ravel(y_true[..., i]), np.ravel(y_pred[..., i]), nbins=nbins)
        ax[i].pcolormesh(xi, yi, zi, shading="gouraud", cmap=cmap)
        ax[i].contour(xi, yi, zi)

        ax[i].axhline(y=0, linestyle=ls_hor[1:], color=ls_hor[0])

        ax[i].set_title(used_titles[i].capitalize(), fontsize=fs)
        ax[i].set_xlabel(used_names[i], fontsize=fs)
        ax[i].set_ylabel(f"Error {used_units[i]}", fontsize=fs)

        ax[i].tick_params(axis="both", labelsize=fs)

    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.draw()
    fig.tight_layout()

    fig_name = f"density_error_plot{suf}.{fig_format}"
    fig_full_name = path.join(outdir_HMI_to_SOT, subfolder, fig_name)
    check_dir(fig_full_name, is_file=True)
    fig.savefig(path.join(outdir_HMI_to_SOT, subfolder, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


@safe_call
def plot_model_history(history: History, dt_string: str | None = None, blur_sigma: float | None = None,
                       offset: float = 0., fig_format: str = default_fig_format,
                       subfolder: str = "", quiet: bool = False) -> None:
    if not quiet:
        print("Model history")
    change_params(offset)

    left, right = 0., history.params["epochs"]
    bottom = 0.

    color1, color2 = "tab:red", "tab:blue"

    history = history.history

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    if blur_sigma is None:
        plot1 = history["loss"]
    else:
        plot1 = denoise_array(history["loss"], sigma=blur_sigma)

    lns1 = ax1.plot(plot1, color=color1, linestyle="-", label="Loss - training")

    metrics_names = [name for name in history if "val_" not in name and "loss" not in name]
    if not metrics_names:
        metric_name = ""
    else:
        if "rmse" in metrics_names:
            metric_name = "rmse"
        elif "mse" in metrics_names:
            metric_name = "mse"
        elif "r2" in metrics_names:
            metric_name = "r2"
        elif "sam" in metrics_names:
            metric_name = "sam"
        else:
            metric_name = metrics_names[0]

    if metric_name:
        metrics = history[metric_name]
        if metric_name == "mse":  # MSE to RMSE
            metrics = np.sqrt(metrics)
            labely = "RMSE"

        elif metric_name in ["mse", "mae", "rmse"]:
            labely = metric_name.upper()

        else:
            labely = metric_name.capitalize()

        if blur_sigma is None:
            plot3 = metrics
        else:
            plot3 = denoise_array(metrics, sigma=blur_sigma)

        labely = str(np.char.replace(labely, "_", " "))

        ax2 = ax1.twinx()  # instantiate a second axis that shares the same x-axis
        lns3 = ax2.plot(plot3, color=color2, linestyle="-", label=f"{labely} - training")

    if "val_loss" in history.keys():
        if blur_sigma is None:
            plot2 = history["val_loss"]
        else:
            plot2 = denoise_array(history["val_loss"], sigma=blur_sigma)

        lns2 = ax1.plot(plot2, color=color1, linestyle=":", label="Loss - validation")

        if f"val_{metric_name}" in history.keys():
            metrics = history[f"val_{metric_name}"]

            if metric_name == "mse":  # MSE to RMSE
                metrics = np.sqrt(metrics)

            if blur_sigma is None:
                plot4 = metrics
            else:
                plot4 = denoise_array(metrics, sigma=blur_sigma)

            lns4 = ax2.plot(plot4, color=color2, linestyle=":", label=f"{labely} - validation")

    if "val_loss" in history.keys():
        if f"val_{metric_name}" in history.keys():
            lns = lns1 + lns2 + lns3 + lns4
        else:
            lns = lns1 + lns2
    else:
        if f"val_{metric_name}" in history.keys():
            lns = lns1 + lns3
        else:
            lns = lns1

    ax1.set_xlabel("Epoch")
    ax1.tick_params(axis="x")
    ax1.set_ylabel("Loss", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(bottom=bottom)
    ax1.set_xlim(left=left, right=right)
    ax1.grid(False)

    if metric_name:
        ax2.set_ylabel(labely, color=color2)  # we already handled the x-label with ax1
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(bottom=bottom)
        ax2.grid(False)

    labs = [l.get_label() for l in lns]
    if metric_name:
        if plot3[0] > plot3[-1]:  # the metric decreases if quality increases
            loc = "upper right"
        else:
            loc = "center right"
    else:
        loc = "upper right"
    ax1.legend(lns, labs, loc=loc)

    fig.tight_layout()  # Otherwise the right y-label is slightly clipped
    # plt.title("Model history")

    plt.draw()

    if dt_string is None:
        fig_name = f"model_history.{fig_format}"
    else:
        fig_name = f"model_history_{dt_string}.{fig_format}"

    fig_full_name = path.join(outdir, subfolder, fig_name)
    check_dir(fig_full_name, is_file=True)
    fig.savefig(fig_full_name, format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


@safe_call
def plot_corr_matrix(labels: np.ndarray, corr_matrix: pd.DataFrame,
                     offset: float = 0., subfolder: str = "", fig_format: str = default_fig_format,
                     suf: str = "", quiet: bool = False) -> None:
    if not quiet:
        print("Correlation matrix")
    change_params(offset)

    # polynom to adjust font size for various corr_matrix sizes
    x, y = [0., 16., 40.], [0.5, 1., 2.]
    fs_multiply = np.polyval(my_polyfit(x, y, 2), len(labels))

    fs_text = SMALL_SIZE * fs_multiply
    fs_small = SMALL_SIZE * fs_multiply
    fs_med = MEDIUM_SIZE * fs_multiply

    xticks, yticks = safe_arange(len(labels)), safe_arange(len(labels))

    fig, ax = plt.subplots(1, 1, figsize=np.shape(corr_matrix))
    im = ax.matshow(corr_matrix, vmin=-1, vmax=1, cmap="seismic")

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    if plt.rcParams["text.usetex"]:
        labels = np.char.replace(labels, "_", r"\_")
    ax.set_xticklabels(labels, rotation=90, fontsize=fs_small)
    ax.set_yticklabels(labels, fontsize=fs_small)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(**cbar_kwargs)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=fs_small)
    cbar.ax.set_ylabel("Correlation coefficient", fontsize=fs_med)

    corr_matrix = np.round(corr_matrix.to_numpy(), 2)

    color = np.full(np.shape(corr_matrix), "w")
    color[np.logical_or(np.abs(corr_matrix) < 0.25, ~np.isfinite(corr_matrix))] = "k"

    for ix in range(len(labels)):
        for iy in range(len(labels)):
            ax.text(iy, ix, f"{corr_matrix[ix, iy]:.2f}",
                    ha="center", va="center", color=color[ix, iy], fontsize=fs_text)

    plt.draw()
    fig.tight_layout()

    fig_name = f"correlation_matrix{suf}.{fig_format}"
    fig_full_name = path.join(outdir, subfolder, fig_name)
    check_dir(fig_full_name, is_file=True)
    fig.savefig(fig_full_name, format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


@safe_call
def plot_ar_maps(ar_numbers: tuple[int] | list[int] | int, offset: float = 0., quiet: bool = False) -> None:
    change_params(offset)

    fits_dir = "/nfsscratch/david/NN/results"

    """
    ar_numbers = np.unique([int(name.split(_sep_out)[0].split(_sep_in)[1]) for name in listdir(fits_dir)])

    ar_numbers = [11067, 11082, 11095, 11096, 11098, 11103, 11116, 11122, 11125, 11132, 11134, 11137, 11139, 11142,
                  11143, 11144, 11145, 11146, 11152, 11154, 11155, 11156, 11159, 11167, 11173, 11179, 11182, 11192,
                  11194, 11206, 11207, 11209, 11211, 11221, 11223, 11229, 11231, 11237, 11239, 11241, 11246, 11247,
                  11249, 11256, 11258, 11262, 11266, 11267, 11268, 11269, 11270, 11272, 11273, 11276, 11278, 11280,
                  11281, 11285, 11288, 11291, 11293, 11294]
    """

    # int to tuple
    if isinstance(ar_numbers, int):
        ar_numbers = ar_numbers,

    for ar_number in ar_numbers:
        fits_names = sorted([path.join(fits_dir, name) for name in listdir(fits_dir) if str(ar_number) in name])

        shift = 0

        for fits_name in fits_names:
            with fits.open(fits_name) as data_sp:
                sp_like = stack((data_sp["Ic"].data, data_sp["Bp"].data, data_sp["Bt"].data, data_sp["Br"].data), axis=3)
                obs_times = np.ravel(list(data_sp["obs_time"].data))
                data_hmi = load_npz(data_sp["obs_time"].header["HMI_FILE"])

            hmi = data_hmi["HMI"]
            hmi[..., 1:] *= 1000.  # kG -> G
            hmi[..., 2] *= -1.  # +N

            for itime in range(len(obs_times)):
                plot_quantity_maps(y_pred=sp_like[[itime]], x_true=hmi[[itime]], suptitle=obs_times[itime],
                                   subfolder=path.join("active_regions", f"AR-{ar_number}"),
                                   offset=offset,
                                   fig_format="jpg",
                                   suf=f"_{itime + shift:04d}",
                                   quiet=quiet)

            shift += len(obs_times)

    change_params(offset, reset=True)


@safe_call
def result_plots(x_true: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                 used_quantities: np.ndarray | None = None,
                 norm: Literal["asinh", "function", "functionlog", "linear", "log", "logit", "symlog"] |
                       mpl.colors.Normalize | None = mpl.colors.LogNorm(),
                 merge_rows: int | None = None, merge_cols: int | None = None, num_quantity_plots: int | None = None,
                 subfolder: str = "",
                 fig_format: str = default_fig_format,
                 suf: str = "", quiet: bool = True) -> None:
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    if "val" in suf or "train" in suf:
        if merge_rows is None: merge_rows = 2
        if merge_cols is None: merge_cols = 3
        if num_quantity_plots is None: num_quantity_plots = 1
    else:
        if merge_rows is None: merge_rows = 2
        if merge_cols is None: merge_cols = 3
        if num_quantity_plots is None: num_quantity_plots = 5

    plot_quantity_maps(y_pred, y_true, x_true, merge_rows=merge_rows, merge_cols=merge_cols,
                       num_plots=num_quantity_plots, used_quantities=used_quantities, subfolder=subfolder,
                       fig_format=fig_format, suf=suf, quiet=quiet)
    """
    plot_scatter_plots(y_true, y_pred, used_quantities=used_quantities, subfolder=subfolder, fig_format=fig_format,
                       suf=suf, quiet=quiet)
    """
    plot_hist2d_plots(y_true, y_pred, norm=norm, used_quantities=used_quantities,
                      subfolder=subfolder, fig_format=fig_format, suf=suf, quiet=quiet)
    plot_error_evaluation(y_true, y_pred, used_quantities=used_quantities, subfolder=subfolder, fig_format=fig_format,
                          suf=suf, quiet=quiet)
    plot_error_in_bins(y_true, y_pred, used_quantities=used_quantities, subfolder=subfolder, fig_format=fig_format,
                       suf=suf, quiet=quiet)


@safe_call
def plot_acc_test_results(filename: str, subfolder: str = "", fig_format: str = default_fig_format) -> None:
    full_path = path.join(_path_accuracy_tests, subfolder, filename)
    data = load_npz(full_path)

    x_true, y_true, y_pred = data[_observations_name], data[_label_true_name], data[_label_pred_name]

    config = data[_config_name][()]
    used_quantities = config["output_setup"]["used_quantities"]

    suf = "_accuracy_test"

    result_plots(x_true, y_true, y_pred, used_quantities=used_quantities, fig_format=fig_format, suf=suf)
