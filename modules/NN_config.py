# This file contains global parameters defining the neural network
import numpy as np
from modules.NN_HP import gimme_hyperparameters
from modules.NN_config_parse import config_check, used_to_bin
from modules._constants import _sep_out


conf_output_setup = {
    # must follow order in CD_config.py
    "used_quantities": np.array([True,  # intensity
                                 True,  # Bp (Bp is positive when pointing west)
                                 True,  # Bt (Bt is positive when pointing north)
                                 True]),  # Br
}

conf_grid_setup = {
    "patch_size": 72
}

# filter only patches with active regions
conf_filtering_setup = {"I": 0.5,  # Ic level
                        "Bp": 0.5,  # kG
                        "Bt": 0.5,  # kG
                        "Br": 0.5}  # kG

conf_data_split_setup = {
    "val_portion": 0.1,  # Set the fraction of data for validation
    "test_portion": 0.1  # Set the fraction of data for tests
}

conf_model_setup = {
    # loss_type and metrics are in NN_HP.py

    # important for HP tuning and early stopping
    "monitoring": {"objective": "val_loss",  # if not loss, must be included in custom_objects and metrics
                   "direction": "min"  # minimise or maximise the objective?
                   },

    "trim_mean_cut": 0.2,  # parameter of trim_mean in evaluation

    "model_subdir": "HMI_to_SOT"  # subdirectory where to save models
}

#
# DO NOT CHANGE THE PART BELOW (unless you know the consequences)
#

# used quantities
conf_output_setup["bin_code"] = used_to_bin(used_quantities=conf_output_setup["used_quantities"])

# hyperparameters
p = gimme_hyperparameters(for_tuning=False)()
conf_model_setup["params"] = p

# model name
conf_model_setup["model_name"] = f"weights{_sep_out}{conf_output_setup['bin_code']}"

# Names of possible labels (must follow order in CD_config.py)

quantity_names = np.array(["intensity",
                           "zonal magnetic field",
                           "meridional magnetic field",
                           "radial magnetic field"])
quantity_names_short = np.array(["Ic", "Bp", "Bt", "Br"])
quantity_names_short_latex = np.array([r"$I_c$", r"$B_p$", r"$B_t$", r"$B_r$"])

config_check(output_setup=conf_output_setup, data_split_setup=conf_data_split_setup, model_setup=conf_model_setup)
