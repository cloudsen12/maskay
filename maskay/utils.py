import pathlib
import re
from collections import OrderedDict
from typing import Dict, List, Union

import numpy as np
import requests


# Beautiful print --------------------------------------------------------------
class color:
    """Color class for printing colored text."""

    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


# Setup auxiliary functions ---------------------------------------------------
def get_models_path():
    """ Get the path to the cloud detection models."""

    # Path to save the models
    cred_path = pathlib.Path(
        "~/.config/maskaymodels/",
    ).expanduser()

    # create the folder if it does not exist
    if not cred_path.is_dir():
        cred_path.mkdir(parents=True, exist_ok=True)

    return cred_path.as_posix()


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


# Processing auxiliary functions ----------------------------------------------
def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()

    return p


# Maskay dict object ----------------------------------------------------------
def ListFiles(path, pattern=None, full_names=False, recursive=False):
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
            yield from ListFiles(file, pattern, full_names, recursive)


def OrderBands(files: List[str], sensor: str = "Sentinel-2"):
    if sensor == "Sentinel-2":
        bandS2 = {
            "Aerosol": "_B01.jp2",
            "Blue": "_B02.jp2",
            "Green": "_B03.jp2",
            "Red": "_B04.jp2",
            "RedEdge1": "_B05.jp2",
            "RedEdge2": "_B06.jp2",
            "RedEdge3": "_B07.jp2",
            "NIR": "_B08.jp2",
            "NIR2": "_B8A.jp2",
            "WaterVapor": "_B09.jp2",
            "Cirrus": "_B10.jp2",
            "SWIR1": "_B11.jp2",
            "SWIR2": "_B12.jp2",
        }

        S2r = {}
        for key, value in bandS2.items():
            for file in files:
                if re.search(value, file):
                    S2r[key] = file
                    continue
        return MaskayObject(S2r)
    else:
        raise ValueError("The sensor is not supported yet!")


def dictreverser(dict, ref):
    refnames = np.array(list(ref.keys()))
    dictnames = np.array(list(dict.keys()))
    condition = ~np.isin(dictnames, refnames)
    if np.any(condition):
        raise ValueError(
            "The band names '%s' are invalid" % ", ".join(dictnames[condition])
        )
    dictreverser = {}
    for dictname in dictnames:
        dictreverser[str(ref[dictname])] = dictname
    return dictreverser


class MaskayObject:
    def __init__(self, dict: Dict[str, str]) -> None:
        self.ref = OrderedDict(
            {
                "Aerosol": 0,
                "Blue": 1,
                "Green": 2,
                "Red": 3,
                "RedEdge1": 4,
                "RedEdge2": 5,
                "RedEdge3": 6,
                "NIR": 7,
                "NIR2": 8,
                "WaterVapor": 9,
                "Cirrus": 10,
                "SWIR1": 11,
                "SWIR2": 12,
                "TIR1": 13,
                "TIR2": 14,
                "HV": 15,
                "VH": 16,
                "HH": 17,
                "VV": 18,
            }
        )
        self.ref_rev = OrderedDict({v: k for k, v in self.ref.items()})
        self.dict = OrderedDict(dict)

    def to_dict(self):
        return self.dict

    def to_list(self):
        return list(self.dict.values())

    def __getitem__(self, index):
        # If is a range object, convert it to a list
        if isinstance(index, range):
            index = list(index)
        elif isinstance(index, int):
            band = self.ref_rev[index]
            return MaskayObject({band: self.dict[band]})
        elif isinstance(index, str):
            band = self.dict[index]
            return MaskayObject({index: band})
        elif isinstance(index, slice):
            # Obtain the bands between the start and stop slice
            container_list = list()
            for key in self.ref_rev.keys():
                if key >= index.start and key <= index.stop:
                    container_list.append(key)

            # Using the indexes of the bands, create a new dictionary
            container_dict = {}
            for element in container_list:
                band = self.ref_rev[element]
                container_dict[band] = self.dict[band]
            return MaskayObject(container_dict)

        if isinstance(index, list):
            container_dict = OrderedDict()
            for element in index:
                if isinstance(element, int):
                    band = self.ref_rev[element]
                    container_dict[band] = self.dict[band]
                elif isinstance(element, str):
                    container_dict[element] = self.dict[element]
            return MaskayObject(container_dict)
        else:
            raise ValueError("Subset index is not valid")

    def __repr__(self):
        MAXSTRING = 30  # Maximum string length to show
        ESPACE = 15  # Establish a constant space between the elements

        msg = ""
        for key, value in self.dict.items():
            # Add three dots if the string is longer than MAXSTRING
            if len(value) > MAXSTRING:
                value_string = value[0:30] + " ... " + value[-30:-1] + value[-1]
            else:
                value_string = value

            # Constant space between the elements
            white_space = " " * (ESPACE - len("%s [%02d]" % (key, 0)))
            name = color.BLUE + color.BOLD + key + color.END
            order = color.BLUE + color.BOLD + "%02d" % self.ref[key] + color.END

            # Print it!
            msg += "%s %s[%s]: %s\n" % (name, white_space, order, value_string)
        return msg


def MaskayDict(
    path: Union[str, pathlib.Path],
    pattern: str = None,
    full_names: bool = False,
    recursive: bool = False,
    sensor: str = "Sentinel-2",
) -> MaskayObject:
    """Create a MaskayDict object from a path.

    Args:
        path (Union[str, pathlib.Path]): Path to the folder containing the files.
        pattern (str, optional): Pattern to match the files. Defaults to None.
        full_names (bool, optional): If True, return the full path of the files. Defaults to False.
        recursive (bool, optional): If True, recursively list the files. Defaults to False.
        sensor (str, optional): Sensor name. Defaults to "Sentinel-2".

    Returns:
        MaskayObject: MaskayDict
    """
    # List all the files according to a pattern
    files: List[str] = ListFiles(
        path=path, pattern=pattern, full_names=full_names, recursive=recursive
    )

    return OrderBands(files, sensor=sensor)
