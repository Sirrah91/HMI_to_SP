"""
-----------------------------------------------------
                        I (cont. norm.)   Bp (G)     Bt (G)     Br (G)
Mean train RMSE score:       0.023     18.688     17.876     23.861
Mean validation RMSE score:  0.023     18.439     17.948     23.336
Mean test RMSE score:        0.023     18.918     17.971     23.729
-----------------------------------------------------
                        I (cont. norm.)   Bp (G)     Bt (G)     Br (G)
Mean train RMSE score:       0.022     18.183     17.323     23.111
Mean validation RMSE score:  0.023     17.973     17.457     22.620
Mean test RMSE score:        0.022     18.468     17.423     23.052
-----------------------------------------------------
"""

import h5py
import ast
import os
from modules.utilities_data import adjust_params_from_weights
from modules.NN_config import conf_model_setup

for file in os.listdir("/nfshome/david/NN/models/HMI_to_SOT/"):
    filename = "/nfshome/david/NN/models/HMI_to_SOT/" + file
    f = h5py.File(filename, "r+")
    if "params" in f:
        params = ast.literal_eval(f["params"][()].decode())
    else:
        params = conf_model_setup["params"]

    params["loss_type"] = "mse"
    params["metrics"] = ["mse"]
    params = adjust_params_from_weights(filename=filename, params=params)

    if "params" not in f:
        f.create_dataset("params", data=str(params))
    else:
        data = f["params"]
        data[...] = str(params)
    f.close()

