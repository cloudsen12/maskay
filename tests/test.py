import maskay

# Donwload S2 an create a TenSorSat object
productid = "S2A_MSIL1C_20190212T142031_N0207_R010_T19FDF_20190212T191443"
s2idpath = maskay.download.s2.SAFE(productid, "/content/", quiet=False)
s2idpath = "/home/csaybar/S2A_MSIL1C_20190212T142031_N0207_R010_T19FDF_20190212T191443.SAFE"
S2files = maskay.utils.MaskayDict(
    path=s2idpath,
    pattern="\.jp2$",
    full_names=True,
    recursive=True,
    sensor="Sentinel-2"
)
tensor = maskay.TensorSat(**S2files.to_dict(), cache=True, align=False)

# Make a prediction
model = maskay.library.UnetMobV2()
predictor = maskay.Predictor(cropsize = 512, overlap = 32, device = "cpu", batchsize = 1, quiet = False)
predictor.predict(model, tensor)
predictor.rio.to_raster("/content/demo.tif")
