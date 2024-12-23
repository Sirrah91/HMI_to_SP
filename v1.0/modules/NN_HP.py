from typing import Callable


def gimme_hyperparameters(for_tuning: bool = False) -> Callable:
    # kern_size, dropout_hidden_hidden, and input_activation are uniform for all layers;
    # modify similarly to num_nodes if needed

    if for_tuning:
        return tuning
    return usage


def usage() -> dict[str, str | int | float | bool | list[int]]:
    p = {
        "model_type": "CNN",  # Convolutional (CNN); separate parameters (CNN_sep); standard (CNN_classic)
        "num_layers": 2,  # Number of CNN layers in standard
        "num_residuals": 4,  # Number of residual layers
        "num_nodes": 64,  # Number of units/filters in the hidden layers (number for residual, list for standard)
        "kern_size": 3,  # Width of the kernel
        "kern_pad": "valid",  # Kernel padding (same or valid; valid stands for reflection padding)
        "input_activation": "relu",  # Activation function of the input and hidden layers
        "output_activation": "softplus_linear",  # Activation function of the output layer

        "dropout_input_hidden": 0.0,  # Dropout rate
        "dropout_residual_residual": 0.0,  # Dropout rate (residual only)
        "dropout_hidden_hidden": 0.0,  # Dropout rate (standard only)
        "dropout_hidden_output": 0.0,  # Dropout rate
        "L1_trade_off": 0.0,  # L1 trade-off parameter
        "L2_trade_off": 0.0,  # L2 trade-off parameter
        "max_norm": 100.0,  # max L2 norm of the weights for each layer
        "optimizer": "Adam",  # see return_optimizer and MyHyperModel.build in NN_models.py for options
        "learning_rate": 0.0001,  # Learning rate
        "batch_size": 32,  # Bath size
        "apply_bs_norm": True,
        "bs_norm_before_activation": True,

        "loss_type": "mse",  # "mse", "Cauchy", or "SSIM"
        "use_weights": True,  # put more stress on underpopulated bins
        "weight_scale": 0.15,  # scaling parameter for underpopulated data -- log(weight_scale * total/counts)
        "metrics": ["rmse"],  # must be in custom_objects in NN_losses_metrics_activations.py
        "alpha": 1.0,  # Trade-off between continuum (I) and magnetic field (B) misfits (alpha x I + B)
        "c": 0.1,  # parameter in Cauchy loss function (low c -> high sensitivity to outliers)
        "num_epochs": 2000,  # Number of epochs
    }

    return p


def tuning() -> dict:
    p = {
        # kern_size, dropout_hidden_hidden, and input_activation are uniform for all layers;
        # modify similarly to num_nodes if needed
        "model_type": ["CNN"],  # Convolutional (CNN); separate parameters (CNN_sep); standard (CNN_classic)
        "num_residuals": [0, 3],  # Number of residual layers
        "num_nodes": [64, 128],  # Number of units/filters in the hidden layers
        "kern_size": [3, 5],  # Width of the kernel
        "kern_pad": ["valid"],  # Kernel padding (same or valid; valid stands for reflection padding)
        "input_activation": ["relu"],  #, "tanh", "sigmoid", "elu"],  # Activation function of the input and hidden layers
        "output_activation": ["softplus_linear"],  # Activation function of the output layer

        "dropout_input_hidden": [0.0, 0.3],  # Dropout rate
        "dropout_residual_residual": [0.0, 0.0],  # Dropout rate (not used in CNN and CNN_sep)
        "dropout_hidden_output": [0.0, 0.0],  # Dropout rate (not used in CNN and CNN_sep)
        "L1_trade_off": [0.0, 1.0],  # L1 trade-off parameter
        "L2_trade_off": [0.0, 1.0],  # L2 trade-off parameter
        "max_norm": [100.0],  # max L2 norm of the weights for each layer
        "optimizer": ["Adam"],  # see return_optimizer and MyHyperModel.build in NN_models.py for options
        "learning_rate": [0.0001, 0.01],  # Learning rate
        "batch_size": [32, 80],  # Bath size
        "apply_bs_norm": [True],
        "bs_norm_before_activation": [True, False],

        # IF YOU USE VAL_LOSS AS A MONITORING QUANTITY, YOU SHOULD NOT USE ALPHA  AMD C IN HP TUNING
        "loss_type": "mse",  # "mse", "Cauchy", or "SSIM"
        "use_weights": True,  # put more stress on underpopulated bins
        "weight_scale": 0.15,  # scaling parameter for underpopulated data -- log(weight_scale * total/counts)
        "metrics": ["rmse"],  # must be in custom_objects in NN_losses_metrics_activations.py
        "alpha": [1.0],  # Trade-off between continuum (I) and magnetic field (B) misfits (alpha x I + B)
        "c": [0.1],  # parameter in Cauchy loss function (low c -> high sensitivity to outliers)
        "num_epochs": 500,  # Number of epochs

        "tuning_method": "Random",  # "Bayes", "Random"
        "plot_corr_mat": True,
    }

    return p
