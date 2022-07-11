import rasterio as rio
import pathlib
import re
import itertools
import cv2
import numpy as np

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


def list_files(path, pattern=None, full_names=False, recursive=False):
    """list.file (like in R) function"""
    files = list(list_file_gen(path, pattern, full_names, recursive))
    files_str = [str(file) for file in files]
    files_str.sort()
    return files_str


def list_file_gen(path, pattern=None, full_names=False, recursive=False):
    """List files R style function"""
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
            

def obtain_img_metadata(allS2bands10: list, allS2bands20: list, allS2bands60: list):
    """ Obtain the dimensions of the images

    Args:
        allS2bands10 (list): List of the 10 meter images.
        allS2bands20 (list): List of the 20 meter images.
        allS2bands60 (list): List of the 60 meter images.

    Returns:
        dict: Dictionary with the dimensions and xinit and yinit of the S2 scene.
    """
    # 10 meters
    with rio.open(allS2bands10[0]) as src:
        ncols10, nrows10 = src.meta['width'], src.meta['height']
    # 20 meters
    with rio.open(allS2bands20[0]) as src:
        ncols20, nrows20 = src.meta['width'], src.meta['height']
    # 60 meters
    with rio.open(allS2bands60[0]) as src:
        ncols60, nrows60 = src.meta['width'], src.meta['height']
    
    xinit = src.meta['transform'].c
    yinit = src.meta['transform'].f
    crs = src.meta["crs"]
    return [xinit, yinit, (ncols10, nrows10), (ncols20, nrows20), (ncols60, nrows60), crs]


def fix_lastchunk(iterchunks, s2dim, chunk_size=512):
    """Fix the last chunk of the overlay.

    Args:
        iterchunks (list): List of the chunks. Created by itertools.product.
        s2dim (_type_): Dimension of the S2 images.
        chunk_size (int, optional): Size of the chunks. Defaults to 512.

    Returns:
        list: List of the chunks.
    """
    
    itercontainer = list()
    for index_i, index_j in iterchunks:
        # Check if the chunk is out of bounds      
        checki = s2dim[0] - index_i
        checkj = s2dim[1] - index_j
        
        # If the chunk is out of bounds, then we need to fix the last chunk
        if checki < chunk_size:
            index_i = s2dim[0] - chunk_size
        
        if checkj < chunk_size:
            index_j = s2dim[1] - chunk_size
        
        # Add the chunk to the list
        itercontainer.append((index_i, index_j))
        
    return itercontainer


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
        version = int(re.search(r'_[0-9].*', model).group(0)[1:])
        if version > best_version:
            best_version = version
            best_model = model
        if version > best_version:
            best_version = version
            best_model = model
    return best_model


def define_iteration(dimension: tuple, chunk_size: int):
    """ Define the iteration strategy to walk through the image.

    Args:
        dimension (tuple): Dimension of the S2 image.
        chunk_size (int): Size of the chunks.

    Raises:
        ValueError: The dimension (x, y) of the S2 image must be equal.

    Returns:
        _type_: List of the chunks.
    """
    if dimension[0] != dimension[1]:
        raise ValueError('The dimension of the images must be equal.')
    
    if chunk_size > dimension[0]:
        return [(0, 0)]
    
    dim = dimension[0]
    iterchunks = list(itertools.product(range(0, dim, chunk_size), range(0, dim, chunk_size)))
    iterchunks_fixed = fix_lastchunk(
        iterchunks=iterchunks,
        s2dim=dimension,
        chunk_size=chunk_size
    )
    return iterchunks_fixed


def rasterio_windows_gen(iterchunks: list, chunk_size: int=512):
    """Generate rasterio windows.

    Args:
        iterchunks (list): A single chunk.
        chunk_size (int, optional): Size of the chunks. Defaults to 512.

    Returns:
        rasterio.windows.Window: A rasterio window.
    """
    # Set the x and y coordinates of the windows
    icol, irow = iterchunks
    
    # Set the rasterio window
    rio_windows = rio.windows.Window(
        col_off=icol,
        row_off=irow,
        width=chunk_size,
        height=chunk_size
    )
    
    return rio_windows


def img_to_numpy(bands: list, dim: int = 2880):
    """Convert the image to numpy array.

    Args:
        bands (list): List of the bands.
        res (int, optional): Resolution of the image. Defaults to 40.

    Returns:
        list: List of the numpy arrays.
    """
    container_bands = list()
    for band in bands:
        with rio.open(band) as src:
            band_arr = src.read(1)            
            if src.meta["width"] != dim:
                band_arr = cv2.resize(band_arr, (dim, dim))
        container_bands.append(band_arr)
    return container_bands


def fix_batch_size(batch_size: int, dim: tuple, mindimreq: int=32):
    """Fix the batch size.

    Args:
        batch_size (int): Batch size.
        dim (tuple): Dimension of the S2 image.
        mindimreq (int, optional): Minimum dimension required. Defaults to 32.

    Returns:
        int: Batch size.
    """
    batch_size = int(batch_size/mindimreq)*mindimreq
    maxsize = dim
    efactor = np.ceil(maxsize/(mindimreq))
    maxbatchsize = int(mindimreq*efactor)
    if maxbatchsize < batch_size:
        return maxbatchsize
    return batch_size
