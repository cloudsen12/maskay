import re
import torch
import pathlib
import gdown
import itertools
import numpy as np
from .utils import MaskayArray
from .models.dataclass import DLMODELS
from tqdm import tqdm
from .utils import color
from .models.models import *
from typing import Union, Dict, List
from .models.models import *


def predict(
    tensor: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
    cropsize: int = 512,
    overlap: int = 128,
    batch_size: int = 1,
    model: str = "UnetMobV2",
    device: str = "cpu",
    quiet: bool = False):
    """Make predictions using a pre-trained model in torch or tensorflow.

    Args:
        tensor (Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]): A SEN2/L8 
            array with shape (B, C, H, W). The bands can be defined as a list or a 
            dictionary.
        cropsize (int, optional):   The size of the crops to use. Defaults to 512.
        overlap (int, optional):   The overlap between the crops. Defaults to 32.
        batch_size (int, optional):  The batch size to use in model prediction.
            Useful for large prediction running on a GPU. Defaults to 32.
        model (str, optional): Define the key of the model to use. Defaults to "UnetMobV2".
        device (str, optional): The device to use. Defaults to "cpu".
        quiet (bool, optional): If True, do not print any messages. Defaults to False.

    Returns:
        np.darray: A tensor with shape (C, H, W) with the cloud cover probabilities.
    
    Examples:
        >>> import maskay         
        >>> # Download a S2 image
        >>> s2idfull = "S2A_MSIL1C_20190212T142031_N0207_R010_T19FDF_20190212T191443"
        >>> s2idpath = maskay.download.s2.SAFE(s2idfull, "/home/user/")
        >>> # Read the image as tensor
        >>> tensor = maskay.S2tensor(s2idpath)
        >>> # Make predictions
        >>> tensorprob = maskay.predict(tensor)
    """
    
    # Perform pre-processing
    DLModel, params = retrieve_model(model=model, quiet=quiet)
    
    # Create the model
    if params.framework == "torch":
        from .models.TorchRun import TorchRun
        
        # Send the model to the GPU device
        if device == "cuda":
            DLModel = DLModel.cuda()
        
        # Run the model under evaluation mode
        DLModel.eval()
    elif params.framework == "tensorflow":
        from .models.TensorFlowRun import TensorFlowRun
    else:
        raise ValueError("DL Framework not supported")

        
    # Chop the tensor into smaller image patches
    IPs = __MagickCrop(tensor, cropsize=cropsize, overlap=overlap, quiet=quiet)
            
    # Set parameter for run the model
    nbatch, xdim, ydim  = IPs.shape[0], IPs.shape[2], IPs.shape[3]
    obands = params.output_bands
    bands = params.bands
    order = params.order
    
    # Create an empty tensor with the output bands (B, C, H, W) order
    tensorprob = MaskayArray(
        array=np.zeros((nbatch, obands, xdim, ydim)),
        coordinates=IPs.coordinates,
        cropsize=IPs.cropsize,
        overlap=IPs.overlap
    )
    
    # Make predictions on the image patches
    if params.framework == "torch":
        tensorprob = TorchRun(DLModel, IPs, tensorprob, bands, batch_size, order, device, quiet)
    elif params.framework == "tensorflow":        
        tensorprob = TensorFlowRun(DLModel, IPs, tensorprob, bands, batch_size, order, device, quiet)
        
    
    # Gather the image patches and merge them into a single tensor
    outensor = np.zeros((obands, tensor.shape[1], tensor.shape[2]))
    
    outclassprob = __MagickGather(
        tensorprob=tensorprob,
        outensor=outensor,
        quiet=quiet
    )
    
    return outclassprob


# Sub main functions -------------------------------------------------------------
def __MagickCrop(tensor, cropsize: int = 256, overlap: int = 0, quiet: bool = False):
    tshp = tensor.shape
    
    # Check tensor dimensions
    if len(tshp) != 3:
        raise ValueError("The tensor must have shape (C, H, W)")
    if tshp[0] > tshp[1] or tshp[0] > tshp[2]:
        raise ValueError("The tensor must have shape (C, H, W)")
    
    # if cropsize > tshp
    if (tshp[1] < cropsize) and (tshp[2] < cropsize):
        return tensor

    # Define relative coordinates.
    xmn, xmx, ymn, ymx = (0, tshp[1], 0, tshp[2])
    
    if overlap > cropsize:
        raise ValueError("The overlap must be smaller than the cropsize")
    
    xrange = np.arange(xmn, xmx, (cropsize - overlap))
    yrange = np.arange(ymn, ymx, (cropsize - overlap))
    
    # If there is negative values in the range, change them by zero.
    xrange[xrange < 0] = 0
    yrange[yrange < 0] = 0
    
    # Remove the last element if it is outside the tensor
    xrange = xrange[xrange - (tshp[1] - cropsize) <= 0]
    yrange = yrange[yrange - (tshp[1] - cropsize) <= 0]
        
    # If the last element is not (tshp[1] - cropsize) add it!
    if xrange[-1] != (tshp[1] - cropsize):
        xrange = np.append(xrange, tshp[1] - cropsize)
    if yrange[-1] != (tshp[2] - cropsize):
        yrange = np.append(yrange, tshp[2] - cropsize)
    
    # Create all the relative coordinates
    mrs = list(itertools.product(xrange, yrange))
    
    # Create an empty tensor with batch dimension
    newarr = np.zeros((len(mrs), tshp[0], cropsize, cropsize))
    
    if not quiet:
        print(
            color.BLUE + color.BOLD + 
            " Splitting the tensor into smaller image patches ..." + color.END
        )
        
    for i, mr in enumerate(tqdm(mrs, disable=quiet)):
        newarr[i] = tensor[None, :, mr[0]:(mr[0] + cropsize), mr[1]:(mr[1] + cropsize)]
    
    if not quiet:
        print("")
                
    return MaskayArray(newarr, mrs, overlap, cropsize)


def __MagickGather(
    tensorprob: torch.Tensor,
    outensor: torch.Tensor,
    quiet: bool = False
):
    # Get CropMagick properties
    coordinates = tensorprob.coordinates  # Coordinates <xmin, ymax> of each tensor
    overlap = tensorprob.overlap  # Overlap between tensors
    cropsize = tensorprob.cropsize  # Cropsize of each tensor
    xmin, xmax = (0, outensor.shape[1])  # X borders of the output tensor
    ymin, ymax = (0, outensor.shape[2])  # Y borders of the output tensor

    if not quiet:
        print(
            color.BLUE + color.BOLD + 
            " Gathering the image patches into a single tensor ..." + color.END
        )
            
    for index, ip in enumerate(tqdm(tensorprob, disable=quiet)):
        # |*-----| -> '*' is the initial coordinate ("coord")
        # |------| --> overlap is the amount of pixels to overlap between tensors
        # |------| --> We define safe/danger zones based on the overlap to perform the gather.
        coord = coordinates[index]
        
        #  Define X dimension pixels
        if coord[0] == xmin:
            Xmin = coord[0]
            XIPmin = 0
        else:
            Xmin = coord[0] + overlap//2
            XIPmin = overlap//2
            
        if (coord[0] + cropsize) == xmax:
            Xmax = coord[0] + cropsize
            XIPmax = cropsize
        else: 
            Xmax = coord[0] + cropsize - overlap//2
            XIPmax = cropsize - overlap//2
        
        
        #  Define Y dimension pixels
        if coord[1] == ymin:
            Ymin = coord[1]
            YIPmin = 0
        else:
            Ymin = coord[1] + overlap//2
            YIPmin = overlap//2
            
        if (coord[1] + cropsize) == ymax:
            Ymax = coord[1] + cropsize
            YIPmax = cropsize
        else:
            Ymax = coord[1] + cropsize - overlap//2
            YIPmax = cropsize - overlap//2
            coord
        # Put the IP tensor in the output tensor
        outensor[:, Xmin: Xmax, Ymin: Ymax] = ip[:, XIPmin: XIPmax, YIPmin: YIPmax]

    if not quiet:
        print("")
                        
    return outensor


# Auxiliary functions ---------------------------------------------------------
def retrieve_model(model: str = "UnetMobV2", quiet: bool = False):
    """Retrieve a pretrained model.
    Args:
        model (str, optional): Define the key of the model to use. Defaults to "UnetMobV2".
    Raises:
        ValueError: If the model is not supported.
    Returns:
        AdanModel: A pretrained model.
    """    
    
    # Search if the model is available
    _models_available = list(DLMODELS.DLModels.keys())
    model = model.lower()
    available_models = [_model for _model in _models_available if re.search(model, _model)]
    
    if len(available_models) == 0:
        raise ValueError(f"Model {model} is not supported. Available models: {_models_available}")
    
    bestmodel = getbestmodel(available_models)
    

    # Get model information
    __model__ = DLMODELS.DLModels[bestmodel]
    
    # if pretrained model does not exist, download it!
    if not pathlib.Path(__model__.filename).is_file():
        if not quiet:
            print(f"Downloading: {bestmodel} -> {__model__.filename}")

        # download file using request        
        gdown.download(__model__.url, __model__.filename, quiet=quiet)
                        
    # Load the model
    model = eval(__model__.name)()
    
    return model, __model__

def getbestmodel(_models_available: list):
    """Retrieve the best model.

    Args:
        _models_available (list): List of available models.

    Returns:
        str: Key of the most recent model.
    """
    
    best_version = 0
    # get the model with the highest version    
    for model in _models_available:
        try:
            version = int(re.search(r'_[0-9].*', model).group(0)[1:])
        except:
            version = 1
        if version > best_version:
            best_version = version
            best_model = model
        if version > best_version:
            best_version = version
            best_model = model
    return best_model