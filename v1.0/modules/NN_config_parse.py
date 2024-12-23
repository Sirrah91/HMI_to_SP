import numpy as np


def config_check(output_setup: dict, data_split_setup: dict, model_setup: dict) -> None:
    if not np.any(output_setup["used_quantities"]):
        raise ValueError('There is no quantity in "used_quantities".')

    if output_setup["bin_code"] != used_to_bin(output_setup["used_quantities"]):
        raise ValueError('"used_quantities" and "bin_code" are different.')

    if data_split_setup["val_portion"] + data_split_setup["test_portion"] >= 1.:
        raise ValueError('Too high sum of "val_portion" and "test_portion.')

    if not (0. <= model_setup["trim_mean_cut"] < 0.5):
        raise ValueError(f"Trimming parameter must be a non-negative number lower than 0.5 "
                         f"but is {model_setup['trim_mean_cut']}.")

    """  # this is not a problem anymore
    # separation of intensity and magnetic field
    if ((not output_setup["used_quantities"][0] or not np.any(output_setup["used_quantities"][1:])) and
            "_sep" in model_setup["params"]["model_type"]):
        raise ValueError(f"Cannot use separated model because there is only a single-type quantity in the model.")
    """


def used_to_bin(used_quantities: np.ndarray) -> str:
    return "".join(str(int(x)) for x in used_quantities)


def bin_to_used(bin_code: str) -> np.ndarray:
    error_msg = f'Invalid bin code input "{bin_code}".'

    bin_code_num = np.array(list(bin_code), dtype=int)
    if not np.all(np.logical_or(bin_code_num == 1, bin_code_num == 0)):
        raise ValueError(f'{error_msg} Bin code must be made of "1" and "0" only.')

    used_quantities = np.array(np.array(list(bin_code), dtype=int), dtype=bool)

    if not np.any(used_quantities):
        raise ValueError(f'{error_msg} Bin code must contain at least one "1".')

    return used_quantities
