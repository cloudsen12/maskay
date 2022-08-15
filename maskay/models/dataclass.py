from typing import Dict, Union
from pydantic import BaseModel
from ..utils import get_models_path
import pathlib

class DLModel(BaseModel):
    """
    Python dataclass for a Cloud Model
    """
    name: str    
    output_bands: int
    model_params: dict = None
    preprocessing: str = "nopreprocessing"    
    version: str
    url: str
    bands: Union[list, str] = "ALL"
    order: str
    filename: str
    trainsize: int
    framework: str
    info: str = "First version"

class DLModelsCollection(BaseModel):
    """
    Python dataclass for a Cloud Models Collection
    """
    DLModels: Dict[str, DLModel]


DLMODELS = DLModelsCollection(
    DLModels=dict(
        unetmobv2=DLModel(
            name="UnetMobV2",
            output_bands=4,
            preprocessing="nopreprocessing",
            model_params={
                "model_name": "U-Net",
                "encoder_name": "mobilenet_v2",
                "loss": "torch.nn.functional.cross_entropy",
                "augmentation": "dehidral_group_D4",
                "batch_size": 32,
                "lr": 0.001,
                "optimizer": "torch.optim.Adam",
                "schedule": "torch.optim.lr_scheduler.ReduceLROnPlateau",
            },            
            version="0.0.1",
            url="https://drive.google.com/uc?id=1o9LeVsXCeD2jmS-G8s7ZISfciaP9v-DU",
            bands="ALL",
            order="BCHW",
            filename=(pathlib.Path(get_models_path()) / "unetmobv2.ckpt").as_posix(),
            framework="torch",
            trainsize=512
        ),
        dynamicworld=DLModel(
            name="DynamicWorld",
            output_bands=9,
            preprocessing="dynamicworld",
            model_params={
                "model_name": "F-CNN",
                "encoder_name": "simple",
                "loss": "torch.nn.functional.cross_entropy",
                "augmentation": "dehidral_group_D4",
                "batch_size": 32,
                "lr": 0.001,
                "optimizer": "torch.optim.Adam",
                "schedule": "torch.optim.lr_scheduler.ReduceLROnPlateau",
            },            
            version="0.0.1",
            url="https://drive.google.com/uc?id=1DcvM2AvkJB0qBzymp-MBgNtYcO47ILBF",
            bands=[2, 3, 4, 5, 6, 7, 8, 12, 13],
            order="BHWC",
            filename=(pathlib.Path(get_models_path()) / "dynamicworld.zip").as_posix(),
            framework="tensorflow",
            trainsize=512
        ),
    )
)