from modules._constants import _wp, _num_eps

from tensorflow.keras.layers import Layer, InputSpec
import tensorflow.keras.backend as K
from tensorflow.python.keras.utils import conv_utils
import tensorflow as tf
import keras
from packaging import version

K.set_epsilon(_num_eps)
K.set_floatx(str(_wp).split(".")[-1].split("'")[0])

if version.parse(keras.__version__) < version.parse("2.2.1"):
    current_version = False
    normdata = conv_utils.normalize_data_format
else:
    # normdata = K.normalize_data_format
    def normdata(value):
        if value is None:
            value = K.image_data_format()
        data_format = value.lower()
        if data_format not in {"channels_first", "channels_last"}:
            raise ValueError(f'The "data_format" argument must be one of "channels_first", "channels_last". '
                             f'Received: "{value}"')
        return data_format


def spatial_reflection_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    """Pads the 2nd and 3rd dimensions of a 4D tensor.
    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 tuples, padding pattern.
        data_format: One of `channels_last` or `channels_first`.
    # Returns
        A padded 4D tensor.
    # Raises
        ValueError: if `data_format` is neither
            `channels_last` or `channels_first`.
    """
    assert len(padding) == 2
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(f"Unknown data_format {data_format}")

    if data_format == "channels_first":
        pattern = [[0, 0],
                   [0, 0],
                   list(padding[0]),
                   list(padding[1])]
    else:
        pattern = [[0, 0],
                   list(padding[0]), list(padding[1]),
                   [0, 0]]
    return tf.pad(x, pattern, "REFLECT")


class ReflectionPadding2D(Layer):
    """Reflection-padding layer for 2D input (e.g. picture).
    This layer can add rows and columns or zeros
    at the top, bottom, left and right side of an image tensor.
    # Arguments
        padding: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
            - If int: the same symmetric padding
                is applied to width and height.
            - If tuple of 2 ints:
                interpreted as two different
                symmetric padding values for height and width:
                `(symmetric_height_pad, symmetric_width_pad)`.
            - If tuple of 2 tuples of 2 ints:
                interpreted as
                `((top_pad, bottom_pad), (left_pad, right_pad))`
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, rows, cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, rows, cols)`
    # Output shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, padded_rows, padded_cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, padded_rows, padded_cols)`
    """

    def __init__(self,
                 padding=(1, 1),
                 data_format=None,
                 **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.data_format = normdata(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, "__len__"):
            if len(padding) != 2:
                raise ValueError(f"`padding` should have two elements. "
                                 f"Found: {padding}")
            height_padding = conv_utils.normalize_tuple(padding[0], 2, "1st entry of padding")
            width_padding = conv_utils.normalize_tuple(padding[1], 2, "2nd entry of padding")
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError(f"`padding` should be either an int, "
                             f"a tuple of 2 ints "
                             f"(symmetric_height_pad, symmetric_width_pad), "
                             f"or a tuple of 2 tuples of 2 ints "
                             f"((top_pad, bottom_pad), (left_pad, right_pad)). "
                             f"Found: {padding}")
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            if input_shape[2] is not None:
                rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[3] is not None:
                cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return (input_shape[0],
                    input_shape[1],
                    rows,
                    cols)
        elif self.data_format == "channels_last":
            if input_shape[1] is not None:
                rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[2] is not None:
                cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return (input_shape[0],
                    rows,
                    cols,
                    input_shape[3])

    def call(self, inputs):
        return spatial_reflection_2d_padding(inputs,
                                             padding=self.padding,
                                             data_format=self.data_format)

    def get_config(self):
        config = {"padding": self.padding,
                  "data_format": self.data_format}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
