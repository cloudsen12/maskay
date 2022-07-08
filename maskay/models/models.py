from .CloudModel import CloudModel
from typing import Dict
from pydantic import BaseModel


class CloudModelsCollection(BaseModel):
    """
    Python dataclass for a Cloud Models Collection
    """
    CloudModels: Dict[str, CloudModel]


CLOUDMODELS = CloudModelsCollection(
    CloudModels=dict(
        Adan001=CloudModel(
            model='Adan',
            version='0.0.1',
            url='https://...',            
            doi='https://...',
            sen2level='L1C',
            filename='adan_0.0.1.ckpt'
        ),
    )
)