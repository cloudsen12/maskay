import re
import time
from sys import getsizeof
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio

# remove one dim
import tensorflow as tf
import xarray

import maskay
import maskay.download
import maskay.preprocessing

# Download a s2 image
productid = "S2A_MSIL1C_20190212T142031_N0207_R010_T19FDF_20190212T191443"
# s2idpath = maskay.download.s2.SAFE(productid, "/content/", quiet=False)


# List files
s2idpath = (
    "/home/csaybar/S2A_MSIL1C_20190212T142031_N0207_R010_T19FDF_20190212T191443.SAFE"
)
S2files = maskay.MaskayDict(
    path=s2idpath,
    pattern="\.jp2$",
    full_names=True,
    recursive=True,
    sensor="Sentinel-2",
)

# Order the bands according to wavelength
# Create A TensorSat object
Dataset = maskay.TensorSat(**S2files.dict)


class UnetMov2(MaskayModule):
    def __init__(
        self,
        cropsize: int = 512,
        overlap: int = 32,
        device: int = "cpu",
        batchsize: int = 1,
        quiet: int = False,
    ):
        super().__init__(cropsize, overlap, device, batchsize, quiet)
        self.model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=13,
            classes=4,
        )

    def setup(self):
        filename = pathlib.Path(get_models_path()) / "unetmobv2.ckpt"

        # Download the model if it doesn't exist
        if not filename.is_file():
            if not quiet:
                print(f"Downloading: UNetMobV2 -> {filename}")

            # download file using gdown
            url = "https://drive.google.com/uc?id=1o9LeVsXCeD2jmS-G8s7ZISfciaP9v-DU"
            gdown.download(url, filename.as_posix(), quiet=quiet)

        # Load the model
        model = self.model.load_from_checkpoint(filename.as_posix())
        model.eval()


UnetMov2().setup()
# Predictions!!
params = dict(cropsize=512, overlap=32, device="cpu", batch_size=1, quiet=False)

# Predict a UNetMobV2 # CPU: 65 seconds, GPU = 12 seconds
tensor = s2_arr
model = "UNetMobV2"
cropsize = 256
overlap = 32
device = "cpu"
batch_size = 1
quiet = False

# Available models
# maskay.DLModels()

cloudprob2 = None

# Run models
cloudprob1 = maskay.predict(tensor=s2_arr, model="KappaMaskL1C", **params)
cloudprob2 = maskay.predict(tensor=s2_arr, model="UnetMobV2", **params)
cloudprob3 = maskay.predict(tensor=s2_arr, model="CDFCNNrgbiswir", **params)

s2_arr = dynamicworld(s2_arr)
landcover1 = maskay.predict(tensor=s2_arr, model="DynamicWorld", **params)

kappamask = np.argmax(cloudprob1, axis=0)
cloudsen12 = np.argmax(cloudprob2, axis=0)
ispvalencia = np.argmax(cloudprob3, axis=0)
dynamicworld = np.argmax(landcover1, axis=0)

plt.imshow(kappamask)
plt.imshow(cloudsen12)
plt.imshow(ispvalencia)
plt.imshow(dynamicworld)
plt.show()


a = time.time()
with rio.open(
    "/home/csaybar/S2A_MSIL1C_20190212T142031_N0207_R010_T19FDF_20190212T191443.SAFE/GRANULE/L1C_T19FDF_A019026_20190212T143214/IMG_DATA/T19FDF_20190212T142031_B04.jp2"
) as src:
    tensor = src.read()
getsizeof(tensor) / 1000 / 1024
time.time() - a


a = time.time()
ddd = xarray.open_dataset(
    "/home/csaybar/S2A_MSIL1C_20190212T142031_N0207_R010_T19FDF_20190212T191443.SAFE/GRANULE/L1C_T19FDF_A019026_20190212T143214/IMG_DATA/T19FDF_20190212T142031_B04.jp2"
)

getsizeof(ddd.band_data.values) / 1000 / 1024
time.time() - a

tensor = dynamicworld(tensor)
# IPbatch = np.moveaxis(tensor[[1, 2, 3, 4, 5, 6, 7, 11, 12]], 0, -1)


landcover1 = maskay.predict(tensor=tensor, model="DynamicWorld", cropsize=1024)
lulc_probx = tf.nn.softmax(landcover1, axis=0)
lulc_prob_int16 = (np.array(lulc_probx) * 10000).astype(np.uint16)

lulc_prob_int16
plt.imshow(lulc_prob_int16[0])

with rio.open(
    "/home/csaybar/test/dynamic/AERONET_LANDUSE_Banizoumbou_L1C_20190509T101039_20190509T102007_T31PDR.tif"
) as src:
    xx = src.read()

plt.imshow(xx[0])
plt.show()
