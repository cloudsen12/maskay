import tensorflow as tf
import zipfile
import pathlib
from ...utils import get_models_path

def DynamicWorld():
    model = (pathlib.Path(get_models_path()) / "dynamicworld.zip")
    
    # if a zip file
    if model.suffix == ".zip" and model.is_file():
        with zipfile.ZipFile(model, "r") as zip_ref:
            zip_ref.extractall(model.parent)
    
    # delete zip file if exists
    #if model.is_file():
    #    model.unlink()
    
    FolderName = str(model.parent/ "DynamicWorld")
    return tf.saved_model.load(FolderName)
