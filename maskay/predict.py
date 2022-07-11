import torch
import os
import shutil
import re
import rasterio as rio
import numpy as np
import tempfile
import pathlib
from .utils import list_files, obtain_img_metadata, define_iteration, rasterio_windows_gen, img_to_numpy, fix_batch_size, getbestmodel
import requests
from .models.models import CLOUDMODELS
from .models.CloudModel import *

# Band spatial resolution
__S2BAND10 = ["_B02.jp2", "_B03.jp2", "_B04.jp2", "_B08.jp2"]
__S2BAND20 = ["_B05.jp2", "_B06.jp2", "_B07.jp2", "_B8A.jp2", "_B11.jp2", "_B12.jp2"]
__S2BAND60 = ["_B01.jp2", "_B09.jp2", "_B10.jp2"]


def predict_tensor(tensor: torch.Tensor, device: str = "cuda", model: str = "adan") -> torch.Tensor:
    """Predict cloud cover from a SEN2 tensor.

    Args:
        tensor (torch.Tensor): A SEN2 with shape (B, C, H, W).
        device (str, optional): The device to use. Defaults to "cuda".
        model (str, optional): Define the key of the model to use. Defaults to "adan".

    Raises:
        ValueError: If the model is not supported.

    Returns:

        torch.Tensor: A tensor with shape (B, 4, H, W) with the cloud cover probabilities.
    """
    # Load the Segmentation Model
    SegModel = retrieve_model(model=model)
    if device == "cuda":
        SegModel = SegModel.cuda()
        tensor = tensor.cuda()
        
    # make a prediction with the model
    bresults = SegModel(tensor).softmax(dim=1).squeeze(0).detach().cpu().numpy()
    return (bresults*10000).astype(np.uint16)


def predict_SAFE(folder:str, output: str, device: str = "cuda", model: str = "adan", batch_size:int = 576, res: int = 34.3125) -> bool:
    """Predict cloud cover from a SAFE folder.
        
    Args:
        folder (str): The path to the SAFE folder.
        device (str, optional): The device to use. Defaults to "cuda".
        model (str, optional): Define the key of the model to use. Defaults to "adan".
        output (str): The output directory.
        
    Raises:
        ValueError: If the model is not supported.

    Returns:
        bool: True if a GeoTIFF was created.
    """
    
    # Create temporary folder
    tmp_folder = pathlib.Path(tempfile.mkdtemp(suffix="__cloudmask"))
        
    # Load the Segmentation Model
    SegModel = retrieve_model(model=model)
    if device == "cuda":
        SegModel = SegModel.cuda()
    
        
    # Identify L1C images
    S2files = list_files(folder, pattern="\.jp2$", full_names=True, recursive=True)
    
    # Select only the S2 L1C bands
    ## 10m
    rgx_expr10 = "|".join(__S2BAND10)
    allS2bands10 = [f for f in S2files if re.search(rgx_expr10, f)]
    
    ## 20m
    rgx_expr20 = "|".join(__S2BAND20)
    allS2bands20 = [f for f in S2files if re.search(rgx_expr20, f)]

    ## 60m
    rgx_expr60 = "|".join(__S2BAND60)
    allS2bands60 = [f for f in S2files if re.search(rgx_expr60, f)]    
    
    ## Obtain the dimensions of the images
    xinit, yinit, dim10, dim20, dim60, s2crs = obtain_img_metadata(
        allS2bands10=allS2bands10,
        allS2bands20=allS2bands20,
        allS2bands60=allS2bands60
    )

    # Obtain the spatial resolution
    SPATIAL_RESOLUTION = res
    dim = int((dim10[0]*10)//SPATIAL_RESOLUTION)
    
    if dim % SegModel.maskayparams.mindimreq != 0:        
        possible_dim = SegModel.maskayparams.mindimreq*np.arange(10, 500)
        possible_res = dim10[0]/possible_dim*10
        possible_res = [str(x) for x in possible_res if str(x)[::-1].find('.') < 6]
        raise ValueError(
            "The dimension must be multiple of %s to avoid problems with the models. %s: [%s]" % (
                SegModel.maskayparams.mindimreq,
                "Tentative resolution according to the model",
                ", ".join(possible_res)
            ))

   ## Fix batch_size if it does not match the model d
    batch_size = fix_batch_size(
        batch_size=batch_size,
        dim=dim,
        mindimreq=SegModel.maskayparams.mindimreq
    )
        
    # DL dimensions required
    if batch_size % SegModel.maskayparams.mindimreq != 0:
        raise ValueError(
            "The batch_size must be multiple of {} to avoid problems with the models".format(
                SegModel.maskayparams.mindimreq
            )
        )
        
    # Read data
    container_10m = img_to_numpy(bands=allS2bands10, dim=dim)
    container_20m = img_to_numpy(bands=allS2bands20, dim=dim)
    container_60m = img_to_numpy(bands=allS2bands60, dim=dim)
        
    s2_arr = np.array([
            container_60m[0], container_10m[0], container_10m[1],
            container_10m[2], container_20m[0], container_20m[1],
            container_20m[2], container_10m[3], container_20m[3],
            container_60m[1], container_60m[2], container_20m[4],
            container_20m[5]
    ])
            
    # Define the iteration
    iterchunks = define_iteration(dimension=s2_arr.shape[1:], chunk_size=batch_size)
            
    # Define the windows (chunk)    
    for index in range(len(iterchunks)):
        # Create windows
        w = rasterio_windows_gen(iterchunks[index], batch_size)
        offset_y = w.row_off
        offset_x = w.col_off        

        #  Obtain s2 by batch
        s2_arrb = s2_arr[:, w.row_off:(w.row_off + w.height), w.col_off:(w.col_off + w.width)]
                
        # from numpy to torch
        s2_arrb = s2_arrb.astype(np.float32)/10000
        s2_torch = torch.tensor(s2_arrb)[None,:]
        
        if device == "cuda":
            s2_torch = s2_torch.cuda()
        
        # make a prediction with the model
        bresults = SegModel(s2_torch).softmax(dim=1).squeeze(0).detach().cpu().numpy()
        cloudprob = (bresults*10000).astype(np.uint16)
                
        # Save the results
        transform = rio.transform.from_origin(
            xinit + offset_x*SPATIAL_RESOLUTION,
            yinit - offset_y*SPATIAL_RESOLUTION,
            SPATIAL_RESOLUTION,
            SPATIAL_RESOLUTION
        )
        
        options = {
            "driver": "GTiff",
            "height": s2_arrb.shape[1],
            "width": s2_arrb.shape[1],
            "count": 4,
            "dtype": str(cloudprob.dtype),
            "crs": s2crs,
            "transform": transform
        }
        
        output_batch = tmp_folder/("cloudmask%03d.tif" % index)
        with rio.open(output_batch, "w", **options) as dst:
            for index2, band in enumerate(cloudprob, 1):
                dst.write(band, index2)
    
    # merge all the tiles if it is necessary
    if index > 0:
        os.system("gdal_merge.py -o %s %s/*.tif" % (output, tmp_folder))
    else:
        # file copy
        shutil.copy(output_batch, output)
    
    # remove tmp/ folder and files
    shutil.rmtree(tmp_folder)
    
    return True


def retrieve_model(model: str = "adan"):
    """Retrieve a pretrained model.
    Args:
        model (str, optional): Define the key of the model to use. Defaults to "adan".
    Raises:
        ValueError: If the model is not supported.
    Returns:
        AdanModel: A pretrained model.
    """
    _models_available = list(CLOUDMODELS.CloudModels.keys())
    
    # Search if the model is available
    model = model.lower()
    available_models = [_model for _model in _models_available if re.search(model, _model)]
    
    if len(available_models) == 0:
        raise ValueError(f"Model {model} is not supported. Available models: {_models_available}")
    
    bestmodel = getbestmodel(available_models)

    # Get model information
    __model__ = CLOUDMODELS.CloudModels[bestmodel]

    # if pretrained model does not exist, download it!
    if not pathlib.Path(__model__.filename).is_file():
        r = requests.get(__model__.url, allow_redirects=True)
        with open(__model__.filename, "wb") as f:
            f.write(r.content)
    
    # Load the model
    model = eval(__model__.classname)().load_from_checkpoint(__model__.filename)
    model.set_maskay_params(__model__)
    model.eval()
    
    return model

if __name__ == "__main__":
    tensor = torch.randn(1, 13, 512, 512)
    cloudprob = predict(tensor)

    folder = "/home/csaybar/S2B_MSIL1C_20190316T141049_N0207_R110_T19FDF_20190316T203712.SAFE/"
    output = "/home/csaybar/cloud_mask_10.tif"
    import time
    a = time.time()
    predict_SAFE(folder, output, device="cpu", model="adan", batch_size=10000, res=34.3125)
    time.time() - a
