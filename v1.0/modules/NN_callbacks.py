from modules._constants import _wp  # just to suppress tensorflow warnings (to run __init__.py)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import h5py


class ReturnBestEarlyStopping(EarlyStopping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            if self.verbose > 0:
                print(f"\nEpoch {self.stopped_epoch + 1}: early stopping")
        elif self.restore_best_weights:
            if self.verbose > 0:
                print(f"Restoring model weights from the end of the best epoch: {self.best_epoch + 1}.")
            self.model.set_weights(self.best_weights)


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, model_params=None, layer_names=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_params = model_params
        self.layer_names = layer_names

    def _save_model(self, epoch, batch, logs=None):
        # Call the original save model method
        super()._save_model(epoch, batch, logs)

        # After saving the model, add params and layer names as attributes if not already present
        if self.model_params is not None or self.layer_names is not None:
            with h5py.File(self.filepath, "a") as f:
                # Check if "params" already exists
                if self.model_params is not None and "params" not in f.attrs:
                    # load with ast.literal_eval(f.attrs["params"])
                    f.attrs["params"] = str(self.model_params)
                # Check if "layer_names" already exists
                if self.layer_names is not None and "layer_names" not in f.attrs:
                    f.attrs["layer_names"] = self.layer_names
