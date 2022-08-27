import zipfile
import pathlib
import gdown

import numpy as np

from maskay.tensorflow import Module
from maskay.utils import get_models_path, softmax

class DynamicWorld(Module):
    def __init__(self):
        super().__init__()
        self.model = model_setup()

    def forward(self, x):
        return self.model(x)
    
    def inProcessing(self, tensor: np.ndarray) -> np.ndarray:
        # If all the pixels are zero skip the run and outProcessing.
        if np.sum(tensor) == 0:
            shp = tensor.shape
            tensor = np.zeros(
                (shp[0], 9, shp[1], shp[2])
            ) # 9 is the number of the output classes
            return [tensor]
        
        NORM_PERCENTILES = np.array(
            [
                [1.7417268007636313, 2.023298706048351],
                [1.7261204997060209, 2.038905204308012],
                [1.6798346251414997, 2.179592821212937],
                [1.7734969472909623, 2.2890068333026603],
                [2.289154079164943, 2.6171674549378166],
                [2.382939712192371, 2.773418590375327],
                [2.3828939530384052, 2.7578332604178284],
                [2.1952484264967844, 2.789092484314204],
                [1.554812948247501, 2.4140534947492487]            
            ]
        )
        tensor = np.log(tensor * 0.005 + 1)
        tensor = (tensor - NORM_PERCENTILES[:, 0]) / NORM_PERCENTILES[:, 1]
        tensor = np.exp(tensor * 5 - 1)
        tensor = tensor / (tensor + 1)
        return tensor.astype("float32")

    def outProcessing(self, tensor: np.ndarray) -> np.ndarray:
        return (softmax(tensor) * 10000).astype(np.int16)
    
# random int
def model_setup():
    # Check if packages are installed
    is_external_package_installed = []

    try:
        import tensorflow as tf
    except ImportError:
        is_external_package_installed.append("tensorflow")

    filename = (pathlib.Path(get_models_path()) / "dynamicworld.zip")

    # Download the model if it doesn't exist
    if not filename.is_file():
        # download file using gdown
        url = "https://drive.google.com/uc?id=1DcvM2AvkJB0qBzymp-MBgNtYcO47ILBF"
        gdown.download(url, filename.as_posix(), quiet=False)
    
    # if a zip file
    if filename.suffix == ".zip" and filename.is_file():
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(filename.parent)
    FolderName = str(filename.parent / "DynamicWorld")
    
    return tf.saved_model.load(FolderName)
