from modules.NN_models import MyHyperModel, build_model, batch_size_estimate
from modules.NN_losses_metrics_activations import calc_train_weights
from modules.control_plots import plot_model_history, plot_corr_matrix
from modules.utilities_data import (print_header, print_info, load_txt, gimme_used_from_name, convert_unit,
                                    load_keras_model, weight_bins, load_npz)
from modules.utilities import check_dir, is_empty, sort_df_with_keys, find_nearest, robust_load_weights

from modules.NN_HP import gimme_hyperparameters

from modules._constants import (_path_model, _path_hp_tuning, _model_suffix, _sep_out, _sep_in, _quiet, _verbose,
                                _show_control_plot, _path_tb_logs)

# defaults only
from modules.NN_config import conf_model_setup, conf_output_setup, quantity_names_short

import numpy as np
import pandas as pd
import h5py
from datetime import datetime
from tensorflow.keras.callbacks import TerminateOnNaN, ReduceLROnPlateau, TensorBoard
from modules.NN_callbacks import ReturnBestEarlyStopping, CustomModelCheckpoint
from pprint import pprint
import json
import sympy
import warnings
from os import path, walk, listdir
from itertools import chain

from keras_tuner.tuners import BayesianOptimization, RandomSearch
import keras_tuner as kt
from tensorflow.keras.models import Model


def train(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
          params: dict[str, str | int | float | bool | list[int]] | None = None,
          monitoring: dict | None = None,
          model_subdir: str | None = None,
          model_to_retrain: str | None = None,  # this parameters enables to retrain a model
          model_name: str | None = None,
          metrics: list[str] | None = None
          ) -> str:
    # loading needed values
    if params is None: params = conf_model_setup["params"]
    if monitoring is None: monitoring = conf_model_setup["monitoring"]
    if model_subdir is None: model_subdir = conf_model_setup["model_subdir"]
    if model_name is None: model_name = conf_model_setup["model_name"]
    if metrics is None: metrics = conf_model_setup["params"]["metrics"]

    if not _quiet:
        # empty line after loading the data
        print("\nSetting up the neural network")

    if model_to_retrain is None:
        model_name, model_filename, dt_string = gimme_model_name(model_subdir, model_name)
        used_quantities = gimme_used_from_name(model_name)

        if params["use_weights"]:
            bins, weights = gimme_bins_and_weights(y_true=y_train, mu=params["weight_scale"])
        else:
            bins, weights = None, None

        # Define model architecture
        model = build_model(input_shape=np.shape(x_train)[1:], params=params, metrics=metrics,
                            used_quantities=used_quantities, weights=weights, bins=bins)
        model_checkpoint_limit = None
    else:
        model_name = model_to_retrain.replace(_sep_out, f"{_sep_in}retrained{_sep_out}", 1)  # first occurrence of _sep_out
        model_filename = path.join(_path_model, model_subdir, model_name)
        dt_string = model_name.split(_sep_out)[-1].split(f".{_model_suffix}")[0]

        used_quantities = gimme_used_from_name(model_to_retrain)

        if params["use_weights"]:
            bins, weights = gimme_bins_and_weights(y_true=y_train, mu=params["weight_scale"])
        else:
            bins, weights = None, None

        # read params automatically from the file
        model = load_keras_model(filename=model_to_retrain, input_shape=np.shape(x_train)[1:], subfolder=model_subdir,
                                 params=params, weights=weights, bins=bins)
        model_checkpoint_limit = None

    if "accuracy_test" in model_subdir:
        show_control_plot, verbose = False, 0
    else:
        show_control_plot, verbose = _show_control_plot, _verbose

    if not _quiet:
        print(f"Used quantities: {quantity_names_short[used_quantities]}\n")

    model = fit_model(model, x_train, y_train, x_val, y_val, params=params, monitoring=monitoring,
                      model_filename=model_filename, model_checkpoint_limit=model_checkpoint_limit,
                      dt_string=dt_string, verbose=verbose)

    save_model(model, model_filename=model_filename, params=params)

    if show_control_plot:
        # model.predict removes model history. The plot must be before it.
        plot_model_history(model.history, dt_string=dt_string, quiet=_quiet)

    if not _quiet:
        # batch_size to aim for 1 GB
        batch_size = batch_size_estimate(model=model, gb_memory_target=1.)

        print_header()
        print_info(convert_unit(y_train, initial_unit="kG", final_unit="G", used_quantities=used_quantities),
                   convert_unit(model.predict(x_train, verbose=0, batch_size=batch_size),
                                initial_unit="kG", final_unit="G",
                                used_quantities=used_quantities),
                   which="train")
        if not is_empty(y_val):
            print_info(convert_unit(y_val, initial_unit="kG", final_unit="G", used_quantities=used_quantities),
                       convert_unit(model.predict(x_val, verbose=0, batch_size=batch_size),
                                    initial_unit="kG", final_unit="G",
                                    used_quantities=used_quantities),
                       which="validation")

    return model_name


def fit_model(model: Model,
              x_train: np.ndarray, y_train: np.ndarray,
              x_val: np.ndarray, y_val: np.ndarray,
              params: dict[str, str | int | float | bool | list[int]],
              monitoring: dict[str, str],
              model_filename: str | None = None,
              model_checkpoint_limit: float | None = None,
              dt_string: str | None = None,
              verbose: int = 2) -> Model:
    # visualise the model
    # visualizer(model, filename="architecture", format="png", view=True)

    # Train model
    if verbose > 0:
        print("Hyperparameters:")
        pprint(params)
        print()
        model.summary()
        print("\nTraining...")

    if not is_empty(y_val):
        validation_data = (x_val, y_val)
        val_divisors = sympy.divisors(len(y_val), proper=False)
        validation_batch_size = int(find_nearest(val_divisors, params["batch_size"]))
    else:
        validation_data, monitoring["objective"] = None, monitoring["objective"].replace("val_", "")
        validation_batch_size = None

    # parameters in early stopping and reduce LR
    stopping_patience = np.min((int(params["num_epochs"] * 0.1), 200))
    lr_factor = 0.5
    layer_names = [layer.name for layer in model.layers]
    callbacks = collect_callbacks(monitoring=monitoring,
                                  do_tensorboard=False,
                                  model_filename=model_filename,
                                  layer_names=layer_names,
                                  params=params,
                                  model_checkpoint_limit=model_checkpoint_limit,
                                  stopping_patience=stopping_patience,
                                  reducelr_patience=stopping_patience // 2,
                                  lr_factor=lr_factor,
                                  dt_string=dt_string,
                                  verbose=verbose)

    model.fit(x_train, y_train, validation_data=validation_data, epochs=params["num_epochs"],
              batch_size=params["batch_size"],
              validation_batch_size=validation_batch_size,
              shuffle=True, callbacks=callbacks, verbose=verbose)

    return model


def save_model(model: Model, model_filename: str, params: dict | None = None) -> None:
    check_dir(model_filename)

    if path.isfile(model_filename):
        # Model weights were saved by ModelCheckpoint; restore the best one here
        model = robust_load_weights(model, filename=model_filename)
    else:
        # Model weights were set by EarlyStopping; save the weights here
        model.save_weights(model_filename)

    if params is not None:
        with h5py.File(model_filename, "a") as f:
            # Check if "params" already exists
            if "params" not in f.attrs:
                # load with ast.literal_eval(f.attrs["params"])
                f.attrs["params"] = str(params)

    layer_names = [layer.name for layer in model.layers]
    with h5py.File(model_filename, "a") as f:
        # Check if "layer_names" already exists
        if "layer_names" not in f.attrs:
            f.attrs["layer_names"] = layer_names

    if not _quiet:
        print("Model was saved to disk")


def collect_callbacks(monitoring: dict[str, str],
                      do_tensorboard: bool = False,
                      model_filename: str | None = None,
                      layer_names: list[str] | None = None,
                      params: dict[str, str | int | float | bool | list[int]] | None = None,
                      model_checkpoint_limit: float | None = None,
                      stopping_patience: int = 0,
                      reducelr_patience: int = 0,
                      lr_factor: float = 1.,
                      dt_string: str | None = None,
                      verbose: int = 2) -> list:
    callbacks = [TerminateOnNaN()]

    if do_tensorboard:
        if dt_string is None: dt_string = datetime.utcnow().strftime("%Y%m%d%H%M%S")

        tensorboard = TensorBoard(path.join(_path_tb_logs, dt_string), write_graph=True)
        callbacks.append(tensorboard)

    if model_filename is not None:  # good backup if something happens during training
        checkpoint = CustomModelCheckpoint(filepath=model_filename,
                                           layer_names=layer_names,
                                           model_params=params,
                                           monitor=monitoring["objective"], mode=monitoring["direction"],
                                           save_best_only=True, save_weights_only=True,
                                           initial_value_threshold=model_checkpoint_limit,
                                           verbose=verbose)
        callbacks.append(checkpoint)

    if stopping_patience > 0:
        # Set early stopping monitor so the model will stop training if it does not improve anymore
        # This callback returns the best weights even when the stopping is not activated
        early_stopping_monitor = ReturnBestEarlyStopping(monitor=monitoring["objective"], mode=monitoring["direction"],
                                                         patience=stopping_patience, restore_best_weights=True,
                                                         verbose=verbose)
        callbacks.append(early_stopping_monitor)

    if 0. < lr_factor < 1. and reducelr_patience > 0:
        # set reduction learning rate callback
        reduce_lr = ReduceLROnPlateau(monitor=monitoring["objective"], mode=monitoring["direction"], factor=lr_factor,
                                      patience=reducelr_patience, min_lr=0.000001, min_delta=0.00002, verbose=verbose)
        callbacks.append(reduce_lr)

    return callbacks


def gimme_model_name(model_subdir: str | None, model_name: str | None, append_name: str = "") -> tuple[str, ...]:
    if model_subdir is None: model_subdir = conf_model_setup["model_subdir"]
    if model_name is None: model_name = conf_model_setup["model_name"]

    model_name = model_name.replace(".", "")  # there cannot be . in model_name

    dt_string = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    final_suffix = "" if _model_suffix == "SavedModel" else f".{_model_suffix}"
    model_name_full = f"{model_name}{_sep_out}{dt_string}{append_name}{final_suffix}"

    filename = path.join(_path_model, model_subdir, model_name_full)

    return model_name_full, filename, dt_string


def gimme_bins_and_weights(y_true: np.ndarray, mu: float = 0.15) -> tuple:
    step = 0.05  # 5% intensity of 50 G
    _minimum = np.min(y_true, axis=tuple(range(np.ndim(y_true) - 1)))
    _maximum = np.max(y_true, axis=tuple(range(np.ndim(y_true) - 1)))
    bins = weight_bins(minimum=_minimum, maximum=_maximum, step=step)
    weights = calc_train_weights(y_true=y_true, bins=bins, mu=mu)

    return bins, weights


def hp_tuner(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
             monitoring: dict | None = None,
             model_subdir: str | None = None,
             model_name: str | None = None,
             metrics: list[str] | None = None,
             used_quantities: np.ndarray | None = None
             ) -> list[str]:
    # loading needed values
    if monitoring is None: monitoring = conf_model_setup["monitoring"]
    if model_subdir is None: model_subdir = conf_model_setup["model_subdir"]
    if model_name is None: model_name = conf_model_setup["model_name"]
    if metrics is None: metrics = conf_model_setup["params"]["metrics"]
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    verbose = 2

    params = gimme_hyperparameters(for_tuning=True)()
    method = params["tuning_method"]

    if np.sum(used_quantities) == 1 and "CNN" in params["model_type"] and "CNN_sep" in params["model_type"]:
        # technically, it is possible to use it but both model types are equivalent in this case, and you don't need
        # to iterate over both...
        raise ValueError(f"Cannot use separated model because there is only a single-type quantity in the model.")

    dt_string = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    project_name = dt_string
    directory = path.join(_path_hp_tuning, method, model_subdir)

    # parameters in early stopping and reduce LR
    stopping_patience = int(params["num_epochs"])
    lr_factor = 1.
    callbacks = collect_callbacks(monitoring=monitoring,
                                  do_tensorboard=False,  # does not work for unknown reasons (maybe it does not accept a constant parameter in tuning)
                                  stopping_patience=stopping_patience,
                                  reducelr_patience=stopping_patience // 2,
                                  lr_factor=lr_factor,
                                  dt_string=dt_string,
                                  verbose=0)

    N = 5  # save N best models (if N < max_trials, N = max_trials)

    if params["use_weights"]:
        bins, weights = gimme_bins_and_weights(y_true=y_train, mu=params["weight_scale"])
    else:
        bins, weights = None, None

    # input_shape_CNN is computed from this one
    hypermodel = MyHyperModel(input_shape=np.shape(x_train)[1:],
                              params=params,
                              metrics=metrics,
                              used_quantities=used_quantities,
                              weights=weights,
                              bins=bins)

    if method == "Bayes":
        max_trials = 100

        tuner = BayesianOptimization(hypermodel,
                                     objective=kt.Objective(monitoring["objective"], direction=monitoring["direction"]),
                                     max_trials=max_trials,
                                     beta=5.,
                                     num_initial_points=np.round(max_trials * 2. / 3.),
                                     directory=directory,
                                     project_name=project_name)
    elif method == "Random":
        max_trials = 50

        tuner = RandomSearch(hypermodel,
                             objective=kt.Objective(monitoring["objective"], direction=monitoring["direction"]),
                             max_trials=max_trials,
                             directory=directory,
                             project_name=project_name)
    else:
        raise ValueError('Unknown method. Must be "Bayes" or "Random"')

    # tuner.search_space_summary()

    val_divisors = sympy.divisors(len(y_val), proper=False)
    tuner.search(x_train, y_train, validation_data=(x_val, y_val), epochs=params["num_epochs"], shuffle=True,
                 validation_batch_size=int(find_nearest(val_divisors, 32)),
                 callbacks=callbacks, verbose=verbose)

    # write results to a file
    filename = path.join(directory, f"{project_name}.csv")
    check_dir(filename)

    trials = tuner.oracle.get_best_trials(num_trials=max_trials)
    metrics_all = list(trials[0].metrics.get_config()["metrics"].keys())

    # put metric and val_metric next to each other
    metrics_all = [metric for metric in metrics_all if "val_" in metric]
    metrics_all = list(chain.from_iterable((metric.replace("val_", ""), metric) for metric in metrics_all))

    df = pd.DataFrame(trial.hyperparameters.get_config()["values"] |
                      {metric: trial.metrics.get_last_value(metric) for metric in metrics_all}
                      for trial in trials)

    # sort keys before saving
    filters = [f"num_filters_{i}" for i in range(np.max(df["num_layers"]))] if "num_layers" in df.keys() else []
    order = [*metrics_all, "model_type", "num_layers", "num_residuals", *filters, "num_filters",
             "kernel_size", "kernel_padding",
             "input_activation", "output_activation",
             "dropout_input_hidden", "dropout_hidden_hidden", "dropout_residual_residual", "dropout_hidden_output",
             "L1_trade_off", "L2_trade_off", "max_norm", "optimizer", "learning_rate",
             "batch_size", "apply_batch_norm", "batch_norm_before_activation", "alpha", "c"]

    df = sort_df_with_keys(df, order)
    df.to_csv(filename, sep="\t", index=False, na_rep="N/A")

    # Retrieve the best model
    num_models = np.min((N, max_trials))

    with warnings.catch_warnings():  # This does not help to suppress the warnings
        warnings.simplefilter("ignore")
        best_model = tuner.get_best_models(num_models=num_models)  # raises warnings

    list_of_params = [trial.hyperparameters.get_config()["values"] for trial in trials][:num_models]

    model_names = [gimme_model_name(model_subdir=model_subdir,
                                    model_name=model_name,
                                    append_name=f"{_sep_out}Tuner{_sep_in}{i}")[1]
                   for i in range(len(best_model))]

    for i, model in enumerate(best_model):
        # Save top models to project dir with a timestamp
        save_model(model, model_names[i], list_of_params[i])

    if params["plot_corr_mat"]:
        do_corr_plots(df=df, method=method)

    return model_names


def prepare_hp_for_corr_plot(hp_dirname: str, method: str, subfolder: str = "",
                             save_csv: bool = True, do_plots: bool = True) -> pd.DataFrame:
    file = path.join(_path_hp_tuning, method, subfolder, f"{hp_dirname}.csv")

    if path.isfile(file):
        data = load_txt(file, sep="\t")

        # remove other keys and put metric and val_metric next to each other
        metrics_all = list(data.keys())
        metrics_all = [metric for metric in metrics_all if "val_" in metric]
        metrics_all = list(chain.from_iterable((metric.replace("val_", ""), metric) for metric in metrics_all))

    else:
        metrics_all = []

    if "loss" not in metrics_all:  # if loss is not there, the data is probably unstructured; read it from trials
        # to collect data from json
        folder = path.join(_path_hp_tuning, method, subfolder, hp_dirname)
        print(folder)

        def json_name(x: int) -> str:
            return path.join(folder, f"{files[x]}", "trial.json")

        # to collect data from json
        def collect_info_from_json(max_trials: int) -> tuple[pd.DataFrame, list[str]]:
            hps = max_trials * [0]

            with open(json_name(0), "r") as f:
                stored_json = json.load(f)

            metrics_all = list(stored_json["metrics"]["metrics"].keys())

            # put metric and val_metric next to each other
            metrics_all = [metric for metric in metrics_all if "val_" in metric]
            metrics_all = list(chain.from_iterable((metric.replace("val_", ""), metric) for metric in metrics_all))

            loss_metrics = np.zeros((max_trials, len(metrics_all)))

            for i in range(max_trials):
                with open(json_name(i), "r") as f:
                    stored_json = json.load(f)
                hps[i] = stored_json["hyperparameters"]["values"]
                for j, keys in enumerate(metrics_all):
                    loss_metrics[i, j] = stored_json["metrics"]["metrics"][keys]["observations"][0]["value"][0]

            data = pd.DataFrame(hps)

            for i, keys in enumerate(metrics_all):
                data[keys] = loss_metrics[:, i]

            return data, metrics_all

        files = np.sort(next(walk(folder))[1])

        max_trials = len(files)
        try:
            data, metrics_all = collect_info_from_json(max_trials=max_trials)
        except (FileNotFoundError, KeyError):  # the last one is not computed yet
            data, metrics_all = collect_info_from_json(max_trials=max_trials - 1)

    filters = [f"num_filters_{i}" for i in range(np.max(data["num_layers"]))] if "num_layers" in data.keys() else []
    order = [*metrics_all, "model_type", "num_layers", "num_residuals", *filters, "num_filters",
             "kernel_size", "kernel_padding",
             "input_activation", "output_activation",
             "dropout_input_hidden", "dropout_hidden_hidden", "dropout_residual_residual", "dropout_hidden_output",
             "L1_trade_off", "L2_trade_off", "max_norm", "optimizer", "learning_rate",
             "batch_size", "apply_batch_norm", "batch_norm_before_activation", "alpha", "c"]

    data = sort_df_with_keys(data, order)
    data = data.sort_values(by=["val_loss"], ascending=True, ignore_index=True)

    if save_csv:
        data.to_csv(file, sep="\t", index=False, na_rep="N/A")
    if do_plots:
        do_corr_plots(df=data, method=method)

    return data


def combine_hp_tuning(method: str, subfolder: str = "") -> pd.DataFrame:
    base_dir = path.join(_path_hp_tuning, method, subfolder)
    file = path.join(base_dir, "hp_merged.csv")

    folders = [path.join(base_dir, folder) for folder in listdir(base_dir) if path.isdir(path.join(base_dir, folder))]

    df = pd.concat([prepare_hp_for_corr_plot(folder, method, subfolder, save_csv=False, do_plots=False) for folder in folders],
                   join="outer", ignore_index=True)

    df = df.sort_values(by=["val_loss"], ascending=True, ignore_index=True)
    df.to_csv(file, sep="\t", index=False, na_rep="N/A")

    return df


def do_corr_plots(df: pd.DataFrame, method: str = "Unknown") -> None:
    for model_type in np.unique(df["model_type"]):
        where_models = np.array(df["model_type"] == model_type)

        # construct the correlation matrix only if more than one realisation were computed
        if np.sum(where_models) > 1:
            data_part = df[where_models]

            # remove constant columns and single non-NaN columns
            data_part = data_part.loc[:, data_part.apply(pd.Series.nunique) > 1]

            data_corr = data_part.corr(numeric_only=True)
            data_keys = np.array(data_corr.keys(), dtype=str)

            plot_corr_matrix(data_keys, data_corr, suf=f"{_sep_out}Tuner{_sep_out}{method}{_sep_out}{model_type}")
