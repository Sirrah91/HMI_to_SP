# New Model Training or Existing Model Retraining

1. **Specify parameters** in `./modules/NN_config.py`, especially the `"used_quantities"` setting.
2. In `./main.py`, specify the path to the training data in the `"data_filename"` variable. The path can be either absolute or relative to the `_path_data` value in `./modules/_constants.py`.
3. For **model retraining**, specify the model name in the `"model_to_retrain"` variable in `./main.py`. This can be an absolute path or relative to the `_path_model` value in `./modules/_constants.py`.
4. Run `./main.py` to start the training or retraining process.

# Model Evaluation

1. **Load your data** as a 4D numpy array. The expected shape of the data is `(num_observations, lat, lon, num_quantities)`.
2. The data can be collected from FITS files using the `prepare_hmi_data` function in `./modules/utilities_data.py`.
3. We recommend using the `process_patches` function in `./modules/NN_evaluate.py` for evaluation.
4. **Minimum version**:
   ```python
   from modules.utilities_data import prepare_hmi_data
   from modules.NN_evaluate import process_patches
   
   data = prepare_hmi_data(hmi_fits)
   predictions_4d = process_patches(model_names=[list, of, used, models], image_4d=data)
