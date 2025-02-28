#!/bin/bash

# Activate the Conda environment
# eval "$(/software/anaconda3/bin/conda shell.bash hook)"
env=/clusterhome/david/.conda/envs/NN/bin/python3.10

# Navigate to the directory containing your Python scripts
cd /nfshome/david/NN || exit

# Directory for the compiled binaries
WD=$(cd ./python_compiled/ && pwd)

# Create a directory for the compiled binaries if it doesn't exist
mkdir -p "${WD}"

# Install PyInstaller if not already installed
$env -m pip install pyinstaller

# List Python scripts and compile them into standalone binaries
for script in job_sizes.py main_openpbs.py eval_data.py run_align_data.py; do
# for script in job_sizes.py main_openpbs.py eval_data.py run_align_data.py job_sizes_regions.py ar_region_stat.py; do
    # Extract the filename without the extension
    filename=$(basename -- "$script")
    filename_no_extension="${filename%.*}"

    # Clear the compiled code from the directory
    rm -rf "${WD:?}/${filename_no_extension:?}" "${WD:?}/${filename_no_extension:?}_bin"

    # Compile the Python script into a standalone binary executable
    $env -m PyInstaller --clean \
        --noupx \
        --add-data "/usr/share/texmf:texmf" \
        --add-data "/usr/share/texlive/texmf-dist:texmf-dist" \
        --add-data "/usr/share/texlive/texmf-dist/fonts:fonts" \
        --distpath="${WD}" "$script"

    # Move the .spec file to the compiled_binaries folder
    mv -- "./${filename_no_extension}.spec" "${WD:?}/${filename_no_extension}/${filename_no_extension}.spec"

    # Create a symlink to the binary in the parent directory
    ln -sf "${WD}/${filename_no_extension}/${filename_no_extension}" "${WD}/${filename_no_extension}_bin"

    # Cleanup temporary build files
    rm -rf ./build
done
