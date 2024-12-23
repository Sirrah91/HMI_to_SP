import os

# Filter out TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Disable GPUs to avoid CUDA-related issues
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Extend PATH to include LaTeX binaries
os.environ["PATH"] += os.pathsep + "/usr/bin:/usr/share/texlive/bin"

"""
1) celkovy tok v oblasti okolo nove vznikele pory
    - kolik je potreba na vytvoreni?
    - v nejake cele oblasti okolo ni, ne jen uvnitr kontury
2) podminky na casove trvani nove pory
    - minimalne existuje min_time cca 3 snimky
    - uvolnit oblast, pokud se tam nic nevyskytuje alespon max_time cca 10 snimku
"""
