# New model training or existing model retraining

1. Specify parameters In ./modules/NN_config.py, especially used_quantities.
2. In ./main.py, specify the path to the training data in "data_filename". It can be absolute path or relative to "_path_data" in ./modules/_constants.py.
3. For model retraining, specify the model name in "model_to_retrain" in ./main.py. It can be absolute path or relative to "_path_model" in ./modules/_constants.py.
4. Run ./main.py

# Model evaluation

1.
