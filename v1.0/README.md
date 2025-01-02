# New Model Training or Existing Model Retraining

1. **Specify parameters** in `./modules/NN_config.py`, particularly the `conf_output_setup["used_quantities"]` setting.
2. In `./main.py`, set the path to the training data in the `data_filename` variable. The path can be either absolute or relative to the `_path_data` value in `./modules/_constants.py`.
3. If the data are not in patches, use the `load_data` function in `./main.py` (located in `./modules/NN_data.py`) instead of the `load_prepared_data` function (also located in `./modules/NN_data.py`).
4. For **model retraining**, specify the model name in the `model_to_retrain` variable in `./main.py`. This can be an absolute path or relative to the `_path_model` value in `./modules/_constants.py`.
5. Run `./main.py` to initiate the training or retraining process.

# Model Evaluation

1. **Load your data** as a 4D numpy array. The expected shape is `(num_observations, lat, lon, num_quantities)`.
2. Data can be collected from FITS files using the `prepare_hmi_data` function in `./modules/utilities_data.py`. We note that `prepare_hmi_data` rotates the data such that north is up and west is left.
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
   # WARNING: This step may result in a large variable "data" and could cause memory error issues
   # depending on the size of your input data.
   data = prepare_hmi_data(**hmi_fits)

   # Load the used models
   model_names = load_base_models("base_models.conf")

   # Make predictions on the data
   predictions_4d = process_patches(model_names=model_names, image_4d=data)
5. The data are provided at the **Hinode/SOT-SP resolution**, with latitude and longitude steps of `dlat = 0.319978` and `dlon = 0.29714` arcsec per pixel.
