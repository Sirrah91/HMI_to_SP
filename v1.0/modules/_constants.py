# THIS FILE CONTAINS VARIABLES COMMON TO ALL FUNCTIONS (PATHS, CONSTANTS, ETC)
import numpy as np
from os import path

# Base directory of the project (do not use a relative path if you create files from the "modules" folder)
_project_dir = "/nfshome/david/NN"
_data_dir = "/nfsscratch/david/NN/data"
_backup_dir = "/nfsscratch/david/NN/backup"
_result_dir = "/nfsscratch/david/NN/results"

# subdirs in _project_dir (useful for backup)
_subdirs = {"modules": "modules",
            "models": "models",
            "HP_tuning": "tuning_HP",
            "TB_log": "TB_log",

            "datasets": "datasets",
            "HMI": "SDO_HMI",
            "SP": "Hinode_SP",
            "SP_HMI_aligned": "SP_HMI_aligned",
            "SP_HMI_like": "SP_HMI_like",
            "SP_blur": "SP_blurred",

            "figures": "figures",
            "accuracy_tests": "accuracy_tests",

            "backup": "backup"
            }

# other directories
_path_modules = path.join(_project_dir, _subdirs["modules"])  # path to models
_path_model = path.join(_project_dir, _subdirs["models"])  # path to models
_path_hp_tuning = path.join(_project_dir, _subdirs["HP_tuning"])  # path to HP tuning results
_path_tb_logs = path.join(_project_dir, _subdirs["TB_log"])  # path to logs of TensorBoard

_path_data = path.join(_data_dir, _subdirs["datasets"])  # path to datasets
_path_hmi = path.join(_data_dir, _subdirs["HMI"])  # path to HMI fits
_path_sp = path.join(_data_dir, _subdirs["SP"])  # path to SP fits
_path_sp_hmi = path.join(_data_dir, _subdirs["SP_HMI_aligned"])  # path to aligned npz files
_path_sp_hmilike = path.join(_data_dir, _subdirs["SP_HMI_like"])  # path to degraded SP npz files

_path_figures = path.join(_project_dir, _subdirs["figures"])  # path to figures
_path_accuracy_tests = path.join(_project_dir, _subdirs["accuracy_tests"])  # path to accuracy tests

_path_backup = path.join(_backup_dir, _subdirs["backup"])  # path to back-up directory

# config file with base models
_model_config_file = "/nfshome/david/NN/base_models.conf"

# names of the files in *.npz
# If you change any of these, you need to re-save your data, e.g., using change_files_in_npz in modules.utilities.py
_observations_name = "HMI"
_observations_key_name = "quantities"
_lat_name = "lat"
_lon_name = "lon"
_metadata_name = "metadata"
_metadata_key_name = "metadata_key"
_label_name = "SP"

# additional names of the files in *.npz used in accuracy tests
# If you change any of these, you need to re-save your data, e.g., using change_files_in_npz in modules.utilities.py
_label_true_name = "SP_true"
_label_pred_name = "SP_predicted"
_config_name = "config"

_wp = np.float32  # working precision
_num_eps = 1e-7  # num_eps of float32
_b_unit = "kG"  # unit that is used by the network

_model_suffix = "weights.h5"  # suffix of saved models; do not change

_rnd_seed = 42  # to control reproducibility; can be int or None (None for "automatic" random seed)

# verbosity of the code
_show_result_plot = True  # True for showing and saving results plots, False for not
_show_control_plot = True  # True for showing and saving control plots, False for not
_verbose = 2  # Set value for verbose: 0 = no print, 1 = full print, 2 = simple print
_quiet = _verbose == 0

# separators
_sep_in = "-"  # separates units inside one logical structure
_sep_out = "_"  # separates logical structures
