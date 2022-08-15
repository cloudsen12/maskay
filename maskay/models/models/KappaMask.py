### KappaZeta model ###
# Adapted from:
#  - https://github.com/spaceml-org/ml4floods
#  - https://github.com/kappazeta/km_predict

import itertools
import numpy as np
import tensorflow as tf
from typing import Callable, Tuple, Optional

S2L1C = {
    "features": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"],
    "max_v": np.array([0.21, 0.326, 0.264, 0.298, 0.245, 0.258, 0.266, 0.364, 0.259, 0.165, 0.055, 0.203, 0.208])
}

S2L2A = {
    "features": ["AOT", "B01", "B02", "B03", "B04", "B05", "B06", "B08", "B8A", "B09", "B11", "B12", "WVP"],
    "max_v": np.array([0.003906309604644775, 0.2833447754383087, 0.27905699610710144, 0.2668497860431671, 0.2558480203151703, 0.250980406999588, 0.24597543478012085, 0.24182498455047607, 0.24206912517547607, 0.2535133957862854, 0.2232242375612259, 0.23524834215641022, 0.07557793706655502]) 
}

CLASSES_KAPPAZETA = ["UNDEFINED", "CLEAR", "CLOUD_SHADOW", "SEMI_TRANSPARENT_CLOUD", "CLOUD", "MISSING"]


def KappaModelUNetL1C():
    return KappaModelUNet("L1C")

def KappaModelUNetL2A():
    return KappaModelUNet("L2A")

class KappaModelUNet:
    def __init__(self, version:str="L1C", norm_factor = 65_635):
        if version == "L1C":
            self.features = S2L1C["features"]
            self.max_v = S2L1C["max_v"]
        if version == "L2A":
            self.features = S2L2A["features"]
            self.max_v = S2L2A["max_v"]
        self.classes = CLASSES_KAPPAZETA
        self.norm_factor = norm_factor # 2**16

    def construct(self, width=None, height=None, pretrained_weights:str=None):
        """
        Construct the model.
        """
        # For symmetrical neighbourhood, width and height must be odd numbers.
        input_shape = (width, height, len(self.features))

        with tf.name_scope("Model"):
            inputs = tf.keras.layers.Input(input_shape, name='input')

            conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
            conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv1)

            conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
            conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
            pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv2)

            conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
            conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
            pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv3)

            conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
            conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
            drop4 = tf.keras.layers.Dropout(0.5)(conv4)
            pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(drop4)

            conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
            conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
            drop5 = tf.keras.layers.Dropout(0.5)(conv5)

            up6 = tf.keras.layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                tf.keras.layers.UpSampling2D(size=(2, 2))(drop5))
            merge6 = tf.keras.layers.concatenate([drop4, up6], axis=3)
            conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
            conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

            up7 = tf.keras.layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                tf.keras.layers.UpSampling2D(size=(2, 2))(conv6))
            merge7 = tf.keras.layers.concatenate([conv3, up7], axis=3)
            conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
            conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

            up8 = tf.keras.layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                tf.keras.layers.UpSampling2D(size=(2, 2))(conv7))
            merge8 = tf.keras.layers.concatenate([conv2, up8], axis=3)
            conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
            conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

            up9 = tf.keras.layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                tf.keras.layers.UpSampling2D(size=(2, 2))(conv8))
            merge9 = tf.keras.layers.concatenate([conv1, up9], axis=3)

            conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
            conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv9 = tf.keras.layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv10 = tf.keras.layers.Conv2D(len(self.classes), (1, 1), activation='softmax')(conv9)

            self.model = tf.keras.Model(inputs, conv10)

            if pretrained_weights is not None:
                self.model.load_weights(pretrained_weights)

            return self.model

    def predict(self, input:np.ndarray, tile_size:int=1280) -> np.ndarray:
        """

        Args:
            input: (C, H, W) tensor expected to have len(self.features) channels
            tile_size: tile size to predict

        Returns:
            (H, W) with the most likely class among the self.classes classes
        """
        assert input.ndim == 3, f"Expected 3 dims in format CHW found {input.shape}"
        assert input.shape[0] == len(self.features), f"Input tensor shape {input.shape} unexpected\n" \
                                                     f"Expected format CHW with {len(self.features)} channels corresponding to S2 bands: {self.features}"

        data_to_pred = input / self.norm_factor / self.max_v[:, None, None]

        pad_r = find_padding(data_to_pred.shape[1])
        pad_c = find_padding(data_to_pred.shape[2])
        data_to_pred = np.pad(
            data_to_pred, ((0, 0), (pad_r[0], pad_r[1]), (pad_c[0], pad_c[1])), "reflect"
        )
        data_to_pred_tf = np.transpose(data_to_pred, (1, 2, 0))

        if (data_to_pred_tf.shape[0] > tile_size) or (data_to_pred_tf.shape[1] > tile_size):
            classes_prob = predbytiles(lambda patch: self.model.predict(patch[None])[0],
                                       data_to_pred_tf)
        else:
            classes_prob = self.model.predict(data_to_pred_tf[None])[0]

        slice_rows = slice(pad_r[0], None if pad_r[1] <= 0 else -pad_r[1])
        slice_cols = slice(pad_c[0], None if pad_c[1] <= 0 else -pad_c[1])
        classes_prob = classes_prob[(slice_rows, slice_cols)]
        return np.argmax(classes_prob, axis=2).astype(np.uint8)


def find_padding(v:int, divisor=16) -> Tuple[int, int]:
    v_divisible = max(divisor, int(divisor * np.ceil(v / divisor)))
    total_pad = v_divisible - v
    pad_1 = total_pad // 2
    pad_2 = total_pad - pad_1
    return pad_1, pad_2


def predbytiles(pred_function: Callable, input_batch: np.ndarray,
                tile_size=1280, pad_size=32) -> np.ndarray:
    """
    Apply a pred_function by tiling the input_batch array.
    The purpose is to run `pred_function(input_batch)` avoiding memory errors.
    It tiles and stiches the patches with padding using the strategy described in: https://arxiv.org/abs/1805.12219

    Args:
        pred_function: pred_function to call
        input_batch: torch.Tensor in (H, W, C) format (tensorflow format!)
        tile_size: Size of the tiles.
        pad_size: each tile is padded before calling the pred_function.
        device: Device to save the predictions

    Returns:
        tensor (H, W, C) format (H and W as input_batch)

    """
    pred_continuous_tf = None
    assert input_batch.ndim == 3, f"Expected batch of images found {input_batch.shape}"

    for i, j in itertools.product(range(0, input_batch.shape[0], tile_size),
                                  range(0, input_batch.shape[1], tile_size)):

        slice_current = (slice(i, min(i + tile_size, input_batch.shape[0])),
                         slice(j, min(j + tile_size, input_batch.shape[1])))
        slice_pad = (slice(max(i - pad_size, 0), min(i + tile_size + pad_size, input_batch.shape[0])),
                     slice(max(j - pad_size, 0), min(j + tile_size + pad_size, input_batch.shape[1])))

        slice_save_i = slice(slice_current[0].start - slice_pad[0].start,
                             None if (slice_current[0].stop - slice_pad[0].stop) == 0 else slice_current[0].stop -
                                                                                           slice_pad[0].stop)
        slice_save_j = slice(slice_current[1].start - slice_pad[1].start,
                             None if (slice_current[1].stop - slice_pad[1].stop) == 0 else slice_current[1].stop -
                                                                                           slice_pad[1].stop)

        slice_save = (slice_save_i, slice_save_j)

        slice_current = slice_current + (slice(None),)
        slice_pad = slice_pad + (slice(None),)
        slice_save = slice_save + (slice(None),)

        vals_to_predict = input_batch[slice_pad]
        cnn_out = pred_function(vals_to_predict)

        assert cnn_out.ndim == 3, f"Expected 3-band prediction (after softmax) found {cnn_out.shape}"

        if pred_continuous_tf is None:
            pred_continuous_tf = np.zeros((input_batch.shape[0], input_batch.shape[1],
                                           cnn_out.shape[-1]),
                                          dtype=np.float32)

        pred_continuous_tf[slice_current] = cnn_out[slice_save]

    return pred_continuous_tf