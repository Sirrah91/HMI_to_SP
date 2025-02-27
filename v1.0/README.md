# New Model Training or Existing Model Retraining

1. **Specify parameters** in `./modules/NN_config.py`, particularly the `conf_output_setup["used_quantities"]` setting. If necessary, adjust the hyperparameters in `./modules/NN_HP.py` to suit your requirements.
2. In `./main.py`, set the path to the training data in the `data_filename` variable. The path can be either absolute or relative to the `_path_data` value in `./modules/_constants.py`.
3. If the data are not in patches, use the `load_data` function from `./modules/NN_data.py` in `./main.py` instead of the `load_prepared_data` function from the same module. Alternatively, load the data as a 4D numpy array with the shape `(num_observations, num_lat, num_lon, num_quantities)`. Ensure `num_lat` and `num_lon` are larger than 50 to minimise the effects of convolution boundary conditions.
4. For **model retraining**, specify the model name in the `model_to_retrain` variable in `./main.py`. This can be an absolute path or relative to the `_path_model` value in `./modules/_constants.py`.
5. Run `./main.py` to initiate the training or retraining process.

# Model Evaluation

1. **Load your data** as a 4D numpy array. The expected shape is `(num_observations, num_lat, num_lon, num_quantities)`.
2. Data can be collected from FITS files using the `prepare_hmi_data` function in `./modules/utilities_data.py`. We note that `prepare_hmi_data` rotates the data such that north is up and west is right.
3. We recommend using the `process_patches` function in `./modules/NN_evaluate.py` for evaluation. This function is scalable and allows evaluation of full-disk observations with low memory usage.
4. **Minimum version**:
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
   data = prepare_hmi_data(**hmi_fits)

   # Load the used models
   model_names = load_base_models("base_models.conf")

   # Make predictions on the data
   predictions_4d = process_patches(model_names=model_names, image_4d=data)
5. The data are provided at the **Hinode/SOT-SP resolution**, with latitude and longitude steps of `dlat = 0.319978` and `dlon = 0.29714` arcsec per pixel.

# Pretrained Models

For more information about the pretrained models, including details about training, tests, and performance, please refer to the paper by **David Korda, Jan Jurčák, Michal Švanda, and Nazaret Bello González, 2025, A&A, XXX, XXX** (https://doi.org/XXX).

The pretrained models are named using a 4-bit binary code, where each bit indicates which quantity is included in the model:

- The **first bit** represents the continuum intensity.
- The **second bit** represents the zonal magnetic field.
- The **third bit** represents the azimuthal magnetic field.
- The **fourth bit** represents the radial magnetic field.

For example, a model with a name containing the binary code `1010` would include the continuum intensity and azimuthal magnetic field, but not the zonal or radial fields.
