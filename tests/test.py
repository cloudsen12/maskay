import maskay
from maskay.library.unetmobv2 import UnetMobV2
import os

# Donwload S2 an create a TenSorSat object
productid = "S2A_MSIL1C_20190212T142031_N0207_R010_T19FDF_20190212T191443"
#s2idpath = maskay.download.s2.SAFE(productid, "/content/", quiet=False)

path_save_products = "folder_test"
s2idpath = os.path.join(path_save_products, f"{productid}.SAFE")

if not os.path.exists(s2idpath):
        s2idpath = maskay.download.s2.SAFE(productid, path_save_products, quiet=False)

S2files = maskay.utils.MaskayDict(
    path=s2idpath,
    pattern="\.jp2$",
    full_names=True,
    recursive=True,
    sensor="Sentinel-2"
)
tensor = maskay.TensorSat(**S2files.to_dict(), cache=True, align=True)

# Make a prediction
model = UnetMobV2()
#model = maskay.library.KappaModelUNetL1C()
#model = maskay.library.DynamicWorld()
# model = maskay.library.CDFCNNrgbi()
predictor = maskay.Predictor(
    cropsize = 512,
    overlap = 32,
    device = "cpu",
    batchsize = 1,
    quiet = False,
    order = "BCHW"
)
result = predictor.predict(model, tensor)
result.shape
predictor.result.rio.to_raster(os.path.join(path_save_products, "outensor2.tif"))
bands = S2files[[7, 3, 2, 1]]
 
import matplotlib.pyplot as plt
import numpy as np

xxx = result.to_numpy()[0:3]
rtoplot = np.moveaxis(xxx, 0, -1)
plt.imshow(rtoplot/10000)
plt.show()
 

