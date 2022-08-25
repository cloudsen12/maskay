import pathlib

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from maskay.module import ModuleTorch
from maskay.utils import get_models_path, softmax


class UnetMobV2(ModuleTorch):
    def __init__(self):
        super().__init__()
        self.model = model_setup()

    def forward(self, x):
        return self.model(x)

    def inProcessing(self, tensor: np.ndarray):
        return tensor / 10000

    def outProcessing(self, tensor: np.ndarray):
        return (softmax(tensor, axis=1) * 10000).astype(np.int16)


class UnetMobV2Class(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=13,
            classes=4,
        )

    def forward(self, x):
        return self.model(x)


def model_setup():
    filename = pathlib.Path(get_models_path()) / "unetmobv2.ckpt"
    # Download the model if it doesn't exist
    if not filename.is_file():
        # download file using gdown
        url = "https://drive.google.com/uc?id=1o9LeVsXCeD2jmS-G8s7ZISfciaP9v-DU"
        gdown.download(url, filename.as_posix())
    # Load the model
    model = UnetMobV2Class().load_from_checkpoint(filename.as_posix())
    model.eval()
    return model
