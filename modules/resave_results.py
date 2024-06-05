from os import path
from glob import glob
import numpy as np
from astropy.io import fits
from modules.NN_config import conf_model_setup
from modules.NN_evaluate import evaluate  # , unit_to_gauss
from modules.utilities_data import load_npz
from modules.utilities import check_dir
from modules.collect_data import ar_info
from sklearn.model_selection import KFold
# from modules.control_plots import plot_quantity_maps, outdir_HMI_to_SOT

model_names = ["weights_1111_20240513105130.h5"]

n_splits = 5

data_dir = "/nfsscratch/david/NN/data/datasets"
outdir = "/nfsscratch/david/NN/results"
check_dir(outdir)

for AR_number in [11067, 11273, 11291, 11293]:
    info = ar_info(AR_number)
    info["location"] = info["location"].replace(" ", "")
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
            results[indices[i_indices][1]] = evaluate(model_names=model_names,
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
            if key == "ar_number":
                key = "AR_num"

            hdu_table.header[key] = str(value)
            primary_hdu.header[key] = str(value)

        hdul = fits.HDUList([primary_hdu, hdu_table])
        hdul.writeto(filename, overwrite=True)

"""
x_true = unit_to_gauss(x_true, used_quantities=[True, True, True, True])

for t, x, y in zip(time, x_true, y_pred):
    x = np.reshape(x, (1, *np.shape(x)))
    y = np.reshape(y, (1, *np.shape(y)))

    plot_quantity_maps(y_pred=y, x_true=x, suptitle=t, subfolder=coords, suf=f"_{im_index:03d}")
    im_index += 1
"""

"""
import os
import cv2

image_folder = os.path.join(outdir_HMI_to_SOT, coords)
video_name = f"{coords}.mp4"

images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(video_name, fourcc, 4., (width, height))

for image in images:
    image_path = os.path.join(image_folder, image)
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error reading image: {image_path}")
            continue
    except Exception as e:
        print(f"Error reading image: {image_path}, {e}")
        continue

    # Write the imate to the video
    video.write(frame)

video.release()
"""
