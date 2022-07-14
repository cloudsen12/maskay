from typing import Dict
from pydantic import BaseModel
from ..utils import get_models_path
import pathlib


class CloudModel(BaseModel):
    """
    Python dataclass for a Cloud Model
    """

    name: str
    model_params: dict = None
    classname: str
    version: str
    url: str
    bands: list
    filename: str
    mindimreq: int
    info: str = "First version"


class CloudModelsCollection(BaseModel):
    """
    Python dataclass for a Cloud Models Collection
    """

    CloudModels: Dict[str, CloudModel]


CLOUDMODELS = CloudModelsCollection(
    CloudModels=dict(
        adan_001=CloudModel(
            name="adam_v0.0.1",
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
            classname="CloudModelAdan",
            version="0.0.2",
            url="https://zenodo.org/record/6807536/files/lilith.ckpt",
            bands=[
                "B01",
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B8A",
                "B09",
                "B10",
                "B11",
                "B12",
            ],
            filename=(pathlib.Path(get_models_path()) / "adan.ckpt").as_posix(),
            mindimreq=32,  # minimum dimension required for the model
        )
    )
)
