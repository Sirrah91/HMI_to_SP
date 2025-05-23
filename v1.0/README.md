# New Model Training or Existing Model Retraining

1. **Specify parameters** in `./modules/NN_config.py`, particularly the `conf_output_setup["used_quantities"]` setting. If necessary, adjust the hyperparameters in `./modules/NN_HP.py` to suit your requirements.
2. In `./main.py`, set the path to the training data in the `data_filename` variable. The path can be either absolute or relative to the `_path_data` value in `./modules/_constants.py`.
3. If the data are not in patches, use the `load_data` function from `./modules/NN_data.py` in `./main.py` instead of the `load_prepared_data` function from the same module. Alternatively, load the data as a 4D numpy array with the shape `(num_observations, num_lat, num_lon, num_quantities)`. Ensure `num_lat` and `num_lon` are larger than 50 to minimise the effects of convolution boundary conditions.
4. For **model retraining**, specify the model name in the `model_to_retrain` variable in `./main.py`. This can be an absolute path or relative to the `_path_model` value in `./modules/_constants.py`.
5. Run `./main.py` to initiate the training or retraining process.

# Model Evaluation
1. Data can be collected from FITS files using the `prepare_hmi_data` function in `./modules/utilities_data.py`. This function performs the following steps:
   - a. Magnetic field components are converted to the local reference frame.
   - b. Continuum observations are corrected for limb darkening and normalised to quiet-Sun level.
   - c. All observations are interpolated to the **Hinode/SOT-SP resolution**, with latitude and longitude steps of `dlat = 0.31998` and `dlon = 0.29714` arcsec per pixel.
   - d. Rotates the data such that north is up and west is right.
   - e. The output includes:
      - A 4D NumPy array with shape `(num_observations, num_lat, num_lon, num_quantities)`.
      - An updated FITS header containing information about remapped coordinates, orientation, and resolution.
2. For data evaluation, we recommend using the `process_patches` function in `./modules/NN_evaluate.py`. This function is scalable and allows evaluation of full-disk observations with low memory usage.
   
## Minimum Version
   ```python
   from modules.utilities_data import prepare_hmi_data
   from modules.NN_evaluate import process_patches
   from modules._base_models import load_base_models
   
   # Absolute paths to FITS files
   hmi_fits = {
       "fits_ic": "continuum_intensity_fits_or_None",
       "fits_b": "B_field_strength_fits_or_None",
       "fits_inc": "B_field_inclination_fits_or_None",
       "fits_azi": "B_field_azimuth_fits_or_None",
       "fits_disamb": "B_field_disambig_fits_or_None"
   }

   # Prepare the data from FITS files
   data, header = prepare_hmi_data(**hmi_fits)

   # Select the used models
   model_names = load_base_models("base_models.conf")

   # Make predictions on the data
   predictions_4d = process_patches(model_names=model_names, image_4d=data)
   ```

# Using `eval_data.py` for Full Workflow

The `eval_data.py` script provides an easy way to evaluate the trained model on HMI FITS data. It simplifies the process of preparing data, running the model, and saving the results. Below is an example of how to use the script:

   ```bash
   python eval_data.py --data_dir "${path_to_hmi_fits}" --output_dir "${path_to_output_folder}" --used_quantities iptr --remove_limb_dark --disambiguate --disambiguate_method random --interpolate_outliers --used_B_units G --max_valid_size 256
   ```
## Options:

- `--data_dir`: Path to the folder containing raw HMI FITS files (continuum intensity and magnetic field components).
- `--output_dir`: (Optional) Path to the folder where output FITS files will be saved. If not provided, results will be saved in the input data folder.
- `--used_quantities`: (Optional) Specifies which quantities to use from the input FITS files. The default is `"iptr"`, which includes intensity and all components of the magnetic field.
- `--remove_limb_dark`: (Optional) Flag to remove limb darkening from the intensity data.
- `--disambiguate`: (Optional) Flag to perform azimuthal disambiguation using the disambiguation FITS file in `--data_dir`.
- `--disambiguate_method`: (Optional) Method to use if disambiguation is done. The default is `"random"`.  Other allowed methods are `"radial_acute"` (closest to radial from disk centre) and `"potential_acute"` (closest to potential field).
  > *Note: The models were trained using `"random"` disambiguation. Since this affects the noise statistics, using the same method is highly recommended to ensure optimal performance of the deconvolution model.*
- `--interpolate_outliers`: (Optional) Flag to compute a sliding window z-score and apply linear interpolation to data that are not finite, have magnetic field values above 10 kG, or have local z-scores exceeding 3 sigma.
  > *Note: This option can be slow due to the interpolation process.*
- `--used_B_units`: (Optional) Specifies the units for the magnetic field. The default is `"G"` (Gauss). Other allowed units are `"kG"` (kilogauss), `"T"` (Tesla), and `"mT"` (millitesla).
- `--max_valid_size`: (Optional) Maximum allowed size for the patches to avoid memory issues. The default is `256`, meaning 256x256 pixels. Larger patch sizes may lead to memory overload, so it’s recommended to adjust this parameter based on available system memory.

## Workflow:

1. **Data Preparation**: Uses the `prepare_hmi_data` function to process the raw HMI FITS files. This function resamples the data to Hinode resolution, converts the magnetic field to the local reference frame, and optionally performs other tasks such as removing limb darkening and azimuthal disambiguation.
2. **Model Evaluation**: The `process_patches` function evaluates the prepared data in manageable patches to avoid memory issues, and performs the deconvolution using the trained model.
3. **Results Saving**: The output FITS files are saved to the specified directory, containing the deconvolved intensity and magnetic field data with model predictions.

This option provides a streamlined, command-line interface for evaluating the model on new HMI data. For users who prefer more manual control over the workflow, the individual functions can also be called directly as in the **Minimum Version** above.

# Pretrained Models

For more information about the pretrained models, including details about training, tests, and performance, please refer to the paper by **David Korda, Jan Jurčák, Michal Švanda, and Nazaret Bello González, 2025, A&A, XXX, XXX** (https://doi.org/10.1051/0004-6361/202553918).

The pretrained models are named using a 4-bit binary code, where each bit indicates which quantity is included in the model:

- The **first bit** represents the continuum intensity.
- The **second bit** represents the zonal magnetic field.
- The **third bit** represents the azimuthal magnetic field.
- The **fourth bit** represents the radial magnetic field.

For example, a model with a name containing the binary code `1010` would include the continuum intensity and azimuthal magnetic field, but not the zonal or radial fields.

# Compilation

To compile the code into executable binaries that can be run on a cluster, a script `./compile.sh` is provided. This script compiles all necessary source files and prepares the executables. To run the compilation, follow these steps:

1. Navigate to the project directory.
2. Run the following command in your terminal:

   ```bash
   ./compile.sh
   ```
3. The script will handle the compilation process and output the binaries in the appropriate directory. Once the compilation is complete, the executables will be ready to run on your cluster.

> *Note: Ensure that the necessary compilers and dependencies are installed before running the script. If you encounter any issues, check the installation requirements or consult the system administrator for support.*
