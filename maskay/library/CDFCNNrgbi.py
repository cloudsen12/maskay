import pathlib
import gdown

import numpy as np

from maskay.tensorflow import Module
from maskay.utils import get_models_path


class CDFCNNrgbi(Module):
    def __init__(self):
        super().__init__()
        self.model = model_setup()

    def forward(self, x):
        return self.model(x)

    def inProcessing(self, tensor: np.ndarray):
        # If all the pixels are zero skip the run and outProcessing.
        if np.sum(tensor) == 0:
            shp = tensor.shape
            tensor = np.zeros(
                (shp[0], 1, shp[1], shp[2])
            ) # 1 is the number of the output classes
            return [tensor]
        return tensor / 10000

    def outProcessing(self, tensor: np.ndarray):
        return (tensor * 10000).astype(np.int16)


## Auxiliary functions
def model_setup():
    # Check if packages are installed
    is_external_package_installed = []

    try:
        import tensorflow as tf
    except ImportError:
        is_external_package_installed.append("tensorflow")

    if is_external_package_installed != []:
        nopkgs = ', '.join(is_external_package_installed)
        raise ImportError(
            f"Please install the following packages: {nopkgs}."
        )

    def conv_blocks(
        ip_,
        nfilters,
        axis_batch_norm,
        reg,
        name,
        batch_norm,
        remove_bias_if_batch_norm=False,
        dilation_rate=(1, 1),
    ):
        use_bias = not (remove_bias_if_batch_norm and batch_norm)

        conv = tf.keras.layers.SeparableConv2D(
            nfilters,
            (3, 3),
            padding="same",
            name=name + "_conv_1",
            depthwise_regularizer=reg,
            pointwise_regularizer=reg,
            dilation_rate=dilation_rate,
            use_bias=use_bias,
        )(ip_)

        if batch_norm:
            conv = tf.keras.layers.BatchNormalization(axis=axis_batch_norm, name=name + "_bn_1")(conv)

        conv = tf.keras.layers.Activation("relu", name=name + "_act_1")(conv)

        conv = tf.keras.layers.SeparableConv2D(
            nfilters,
            (3, 3),
            padding="same",
            name=name + "_conv_2",
            use_bias=use_bias,
            dilation_rate=dilation_rate,
            depthwise_regularizer=reg,
            pointwise_regularizer=reg,
        )(conv)

        if batch_norm:
            conv = tf.keras.layers.BatchNormalization(axis=axis_batch_norm, name=name + "_bn_2")(conv)

        return tf.keras.layers.Activation("relu", name=name + "_act_2")(conv)


    def build_unet_model_fun(x_init, weight_decay=0.05, batch_norm=True, final_activation="sigmoid"):

        axis_batch_norm = 3

        reg = tf.keras.regularizers.l2(weight_decay)

        conv1 = conv_blocks(x_init, 32, axis_batch_norm, reg, name="input", batch_norm=batch_norm)

        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pooling_1")(conv1)

        conv2 = conv_blocks(pool1, 64, axis_batch_norm, reg, name="pool1", batch_norm=batch_norm)

        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pooling_2")(conv2)

        conv3 = conv_blocks(pool2, 128, axis_batch_norm, reg, name="pool2", batch_norm=batch_norm)

        up8 = tf.keras.layers.concatenate(
            [
                tf.keras.layers.Conv2DTranspose(
                    64, (2, 2), strides=(2, 2), padding="same", name="upconv1", kernel_regularizer=reg
                )(conv3),
                conv2,
            ],
            axis=axis_batch_norm,
            name="concatenate_up_1",
        )

        conv8 = conv_blocks(up8, 64, axis_batch_norm, reg, name="up1", batch_norm=batch_norm)

        up9 = tf.keras.layers.concatenate(
            [
                tf.keras.layers.Conv2DTranspose(
                    32, (2, 2), strides=(2, 2), padding="same", name="upconv2", kernel_regularizer=reg
                )(conv8),
                conv1,
            ],
            axis=axis_batch_norm,
            name="concatenate_up_2",
        )

        conv9 = conv_blocks(up9, 32, axis_batch_norm, reg, name="up2", batch_norm=batch_norm)

        conv10 = tf.keras.layers.Conv2D(
            1, (1, 1), kernel_regularizer=reg, name="linear_model", activation=final_activation
        )(conv9)

        return conv10

    def load_model(shape=(None, None), bands_input=4, weight_decay=0.0, final_activation="sigmoid"):
        ip = tf.keras.layers.Input(shape + (bands_input,), name="ip_cloud")
        c11 = tf.keras.layers.Conv2D(bands_input, (1, 1), name="normalization_cloud", trainable=False)
        x_init = c11(ip)
        conv2d10 = build_unet_model_fun(
            x_init, weight_decay=weight_decay, final_activation=final_activation, batch_norm=True
        )
        return tf.keras.models.Model(inputs=[ip], outputs=[conv2d10], name="UNet-Clouds")
        
    filename = (pathlib.Path(get_models_path()) / "CD-FCNN-RGBI.hdf5")
    
    # Download the model if it doesn't exist
    if not filename.is_file():
        # download file using gdown
        url = "https://drive.google.com/uc?id=1TpRE1pPtnBVW7XIJvUOMwCP2nwMmEnij"
        gdown.download(url, filename.as_posix())
    
    # Load the model
    model = load_model(shape=(None, None), bands_input=4)
    model.load_weights(filename.as_posix())
    
    return model

