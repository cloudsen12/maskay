from pydantic import BaseModel

class CloudModel(BaseModel):
    """
    Python dataclass for a Cloud Model
    """
    model: str
    version: str
    url: str
    doi: str
    sen2level: str
    filename: str