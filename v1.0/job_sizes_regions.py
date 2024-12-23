import numpy as np
import argparse
from glob import glob
from os import path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--region_number", type=str, default="*")
    args, _ = parser.parse_known_args()

    ar_prefix = "AR-"
    if not args.region_number.startswith(ar_prefix):
        args.region_number = f"{ar_prefix}{args.region_number}"

    fits_dir = "/nfsscratch/david/NN/results"
    names = sorted(glob(path.join(fits_dir, f"*{args.region_number}*.fits")))

    bare_names = [path.split(name)[-1] for name in names]
    _, counts = np.unique([bare_name.split("_")[0] for bare_name in bare_names], return_counts=True)

    # print usage for qsub
    print(f"{np.clip(np.ceil(np.max(counts) * 0.8), a_min=1., a_max=None):.0f}")
