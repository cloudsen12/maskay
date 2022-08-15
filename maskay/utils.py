import numpy as np
import requests
import pathlib
import re

class color:
    """Color class for printing colored text."""
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class MaskayArray(np.ndarray):
    def __new__(cls, array, coordinates, overlap, cropsize):
        obj = np.asarray(array).view(cls)
        obj.coordinates = coordinates
        obj.overlap = overlap
        obj.cropsize = cropsize
        return obj

def get_models_path():
    """ Get the path to the cloud detection models."""
        
    # Path to save the models
    cred_path = pathlib.Path(
        '~/.config/maskaymodels/',
    ).expanduser()
    
    # create the folder if it does not exist
    if not cred_path.is_dir():
        cred_path.mkdir(parents=True, exist_ok=True)
  
    return cred_path.as_posix()


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                
def list_files(path, pattern=None, full_names=False, recursive=False):
    """list.file (like in R) function"""
    files = list(list_file_gen(path, pattern, full_names, recursive))
    files_str = [str(file) for file in files]
    files_str.sort()
    return files_str


def list_file_gen(path, pattern=None, full_names=False, recursive=False):
    """List files like in R - generator"""
    path = pathlib.Path(path)
    for file in path.iterdir():
        if file.is_file():
            if pattern is None:
                if full_names:
                    yield file
                else:
                    yield file.name
            elif pattern is not None:
                regex_cond = re.compile(pattern=pattern)
                if regex_cond.search(str(file)):
                    if full_names:
                        yield file
                    else:
                        yield file.name
        elif recursive:
            yield from list_files(file, pattern, full_names, recursive)
