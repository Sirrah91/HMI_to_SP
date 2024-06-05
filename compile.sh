#!/bin/bash

# Activate the Conda environment
# eval "$(/software/anaconda3/bin/conda shell.bash hook)"
env=/home/david/.conda/envs/NN/bin/python3.10

# Navigate to the directory containing your Python scripts
cd /nfshome/david/NN || exit

# Create a directory for the compiled binaries if it doesn't exist
mkdir -p python_compiled

# Clear the directory
rm -rf ./python_compiled/*

# Install PyInstaller if not already installed
$env -m pip install pyinstaller

# List Python scripts and compile them into standalone binaries
for script in job_sizes.py main_torque.py; do
    # Extract the filename without the extension
    filename=$(basename -- "$script")
    filename_no_extension="${filename%.*}"

    # Compile the Python script into a standalone binary executable
    $env -m PyInstaller --onefile --distpath=./python_compiled "$script"

    # Move the compiled binary to the appropriate directory with the same name as the Python script
    # mv ./python_compiled/dist/"$filename_no_extension" ./python_compiled/"$filename_no_extension"

    # Move the .spec file to the compiled_binaries folder
    mv ./"$filename_no_extension.spec" ./python_compiled/"$filename_no_extension.spec"

    # Cleanup temporary build files
    rm -rf ./build
    # rm -rf ./python_compiled/build
    # rm -rf ./python_compiled/dist
    # rm -rf ./python_compiled/__pycache__
done
