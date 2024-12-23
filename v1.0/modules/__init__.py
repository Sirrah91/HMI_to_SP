import os

# Filter out TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Disable GPUs to avoid CUDA-related issues
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Extend PATH to include LaTeX binaries
os.environ["PATH"] += os.pathsep + "/usr/bin:/usr/share/texlive/bin"
