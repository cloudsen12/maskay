import os
import pathlib
import re
from xml.dom import minidom

import ee
import requests


def SEN2downloadGEE(
    s2id: str,
    output: str,
    runchecks: bool = True,
    quiet: bool = True,
    method: str = "requests",
):
    """Download a Sentinel-2 image using Google Earth Engine IDs.

    Args:
        s2id (str): The GEE Sentinel-2 ID.
        output (str): The output directory.
        runchecks (bool, optional): If True, run checks on the Earth Engine and gsutil. Defaults to True.
        quiet (bool, optional): If True, do not print info messages. Defaults to True.
        method (str, optional): The method to use to download the image. Defaults to "requests".

    Returns:
        str: The path to the downloaded image.

    Examples:
        >>> import ee
        >>> ee.Initialize()
        >>> s2id = "20190212T142031_20190212T143214_T19FDF"
        >>> s2idpath = SEN2downloadGEE(s2id, "/home/user/")
        >>> print(s2idpath)
        >>> "/home/user/S2A_MSIL1C_20190221T142031_N0213_R092_T19FDF_20190212T142031.SAFE/"
    """

    # check if ee is initialized
    if runchecks:
        __eecheck()

    # Identify S2 image using GEE S2 ID
    s2 = ee.ImageCollection("COPERNICUS/S2")
    s2 = s2.filter(ee.Filter.eq("system:index", s2id)).first()

    # Get tile information.
    productid = s2.get("PRODUCT_ID").getInfo()

    # Download S2 image.
    SEN2download(productid, output, quiet)

    return True


def SEN2download(
    productid: str,
    output: str,
    runchecks: bool = True,
    quiet: bool = True,
    method: str = "requests",
):
    """Download a Sentinel-2 image using a product ID.

    Args:
        productid (str): The full id of the Sentinel-2 product.
        output (str): The output directory to save the image.
        runchecks (bool, optional): If True, run checks on the gsutil. Only used if method is "gsutil". Defaults to True.
        quiet (bool, optional): If True, do not print info messages. Defaults to True.
        method (str, optional): The method to use to download the image. Defaults to "requests".

    Returns:
        str: The path to the downloaded image.

    Examples:
        >>> import ee
        >>> ee.Initialize()
        >>> s2idfull = "S2A_MSIL1C_20190221T142031_N0213_R092_T19FDF_20190212T142031"
        >>> s2idpath = SEN2download(s2idfull, "/home/user/")
        >>> print(s2idpath)
        >>> "/home/user/S2A_MSIL1C_20190221T142031_N0213_R092_T19FDF_20190212T142031.SAFE/"
    """
    # check if gsutil is installed
    if runchecks:
        __gsutilcheck()

    # Get tile grid info.
    rgx_expr = "(.*)_(.*)_(.*)_(.*)_(.*)"
    tile_info = re.search(rgx_expr, productid, re.IGNORECASE).group(4)[1:]
    p01 = tile_info[0:2]
    p02 = tile_info[2]
    p03 = tile_info[3:5]

    # 4. Download the S2 L1C image folder.
    if method == "gsutil":
        # 3. Create the path to download the image.
        base_uri = "gs://gcp-public-data-sentinel-2/tiles"
        BASEPATH = "%s/%s/%s/%s/%s.SAFE/" % (base_uri, p01, p02, p03, productid)
        if quiet:
            os.system("gsutil -q cp -r %s %s" % (BASEPATH, output))
        else:
            print("Running: gsutil -m cp -r %s %s" % (BASEPATH, output))
            os.system("gsutil cp -r %s %s" % (BASEPATH, output))

    if method == "requests":
        base_uri = "https://storage.googleapis.com/gcp-public-data-sentinel-2/tiles"
        folder_img = "%s/%s/%s/%s/%s.SAFE/" % (base_uri, p01, p02, p03, productid)

        # Create the output folder if it does not exist.
        PATH = pathlib.Path(output)
        if not PATH.exists():
            PATH.mkdir(parents=True, exist_ok=True)

        # download MTD_MSIL1C.xml
        mtd_msil1c_xml = "%sMTD_MSIL1C.xml" % folder_img
        r = requests.get(mtd_msil1c_xml, allow_redirects=True)

        # write a xml file
        BASEPATH = pathlib.Path((PATH / productid).as_posix() + ".SAFE")
        BASEPATH.mkdir(exist_ok=True)
        S2xmlFile = BASEPATH / "MTD_MSIL1C.xml"
        with open(S2xmlFile, "wb") as f:
            f.write(r.content)

        # read a XML file
        file = minidom.parse(S2xmlFile.as_posix())

        # download images
        models = file.getElementsByTagName("IMAGE_FILE")
        for model in models:
            if not quiet:
                print("Downloading: %s" % model.firstChild.data)

            # Get S2 image from google cloud storage
            S2todownload = (folder_img + model.firstChild.data) + ".jp2"

            # where to save the S2 image
            S2tosave = BASEPATH / model.firstChild.data

            # Create the folder if it does not exist.
            S2tosave.parent.mkdir(parents=True, exist_ok=True)
            S2tosave = S2tosave.as_posix() + ".jp2"

            # Download S2 images
            r = requests.get(S2todownload, allow_redirects=True)
            with open(S2tosave, "wb") as f:
                f.write(r.content)
    return BASEPATH


def __eecheck():
    """Simple check to see if the Earth Engine is working.
    Raises:
        Exception: If the Earth Engine is not working.
    Returns:
        bool: True if gsutil is installed.
    """
    try:
        ee.Image(0)
    except:
        raise Exception("ee is not initialized")

    return True


def __gsutilcheck():
    """Check if the gsutil is installed.

    Raises:
        Exception: If the gsutil is not installed.
        Exception: If the gsutil is not working.

    Returns:
        bool: True if gsutil is installed.
    """

    # check if gsutil is installed silent
    if not os.system("gsutil --version > /dev/null 2>&1") == 0:
        raise Exception("gsutil is not installed")

    # check if gsutil is working
    # if not os.system("gsutil ls > /dev/null 2>&1") == 0:
    #    raise Exception("gsutil is not working")
    return True
