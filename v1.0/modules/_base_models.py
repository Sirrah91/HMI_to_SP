# list of base models for evaluation
import os


def load_base_models(model_config_file: str = "/nfshome/david/NN/base_models.conf"):
    model_names = []

    if os.path.isfile(model_config_file):
        with open(model_config_file, "r") as file:
            model_names = file.readlines()

        # Filter out empty lines and lines that start with #
        model_names = [line.strip() for line in model_names if line.strip() and not line.strip().startswith(("#",))]

    if not model_names:
        model_names = ["/nfshome/david/NN/models/HMI_to_SOT/HMI-to-SOT_1000_20241128161222.weights.h5",
                       "/nfshome/david/NN/models/HMI_to_SOT/HMI-to-SOT_1000_20250317091749.weights.h5",
                       "/nfshome/david/NN/models/HMI_to_SOT/HMI-to-SOT_0100_20241128161222.weights.h5",
                       "/nfshome/david/NN/models/HMI_to_SOT/HMI-to-SOT_0100_20250317091749.weights.h5",
                       "/nfshome/david/NN/models/HMI_to_SOT/HMI-to-SOT_0010_20241128161222.weights.h5",
                       "/nfshome/david/NN/models/HMI_to_SOT/HMI-to-SOT_0010_20250317091749.weights.h5",
                       "/nfshome/david/NN/models/HMI_to_SOT/HMI-to-SOT_0001_20241128161222.weights.h5",
                       "/nfshome/david/NN/models/HMI_to_SOT/HMI-to-SOT_0001_20250317091749.weights.h5"]

    return model_names
