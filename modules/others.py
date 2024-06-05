from os import environ, listdir
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from modules.NN_data import load_prepared_data, split_data, clean_data
from modules.NN_evaluate import evaluate_test_data
from modules.NN_config import conf_output_setup, conf_model_setup, conf_filtering_setup, conf_data_split_setup

filename_train_data = "SP_HMI_aligned.npz"
print(f"Data file: {filename_train_data}")

# Load the data
x_train, y_train = load_prepared_data(filename_data=filename_train_data,
                                      used_quantities=conf_output_setup["used_quantities"],
                                      use_simulated_hmi=True)

# Split the data
x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_data=x_train, y_data=y_train,
                                                            val_portion=conf_data_split_setup["val_portion"],
                                                            test_portion=conf_data_split_setup["test_portion"])

# Filter out the quiet Sun patches
x_train, y_train = clean_data(x_train, y_train,
                              filtering_setup=conf_filtering_setup,
                              used_quantities=conf_output_setup["used_quantities"])
x_val, y_val = clean_data(x_val, y_val,
                          filtering_setup=conf_filtering_setup,
                          used_quantities=conf_output_setup["used_quantities"])

model_names = ["weights_1111_20240513105441.h5"]

y_pred, _ = evaluate_test_data(model_names=model_names,
                               x_test=x_test, y_test=y_test,
                               x_val=x_val, y_val=y_val,
                               x_train=x_train, y_train=y_train,
                               proportiontocut=conf_model_setup["trim_mean_cut"],
                               subfolder_model=conf_model_setup["model_subdir"])
