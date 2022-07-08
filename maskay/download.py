import os
import re

import ee


def SEN2downloadGEE(s2id: str, output: str, runchecks: bool = True, quiet: bool = True):
    """Download a Sentinel-2 image using Google Earth Engine IDs.

    Args:
        s2id (str): The GEE Sentinel-2 ID.
        output (str): The output directory.
        runchecks (bool, optional): If True, run checks on the Earth Engine and gsutil. Defaults to True.
        quiet (bool, optional): If True, do not print info messages. Defaults to True.
    Returns:
        bool: True if the image was downloaded.

    Examples:
        >>> import ee
        >>> ee.Initialize()
        >>> s2id = "20190212T142031_20190212T143214_T19FDF"
        >>> SEN2downloadGEE(s2id, "./")
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
    productid: str, output: str, runchecks: bool = True, quiet: bool = True
):
    """Download a Sentinel-2 image using a product ID.

    Args:
        s2id (str): The full id of the Sentinel-2 product.
        output (str): The output directory.
        runchecks (bool, optional): If True, run checks on the gsutil. Defaults to True.
        quiet (bool, optional): If True, do not print info messages. Defaults to True.
    Returns:
        bool: True if the image was downloaded.

    Examples:
        >>> import ee
        >>> ee.Initialize()
        >>> s2idfull = "S2A_MSIL1C_20190221T142031_N0213_R092_T19FDF_20190212T142031"
        >>> SEN2download(s2idfull, "./")
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

    # 3. Create the path to download the image.
    base_uri = "gs://gcp-public-data-sentinel-2/tiles"
    file_img = "%s/%s/%s/%s/%s.SAFE/" % (base_uri, p01, p02, p03, productid)

    # 4. Download the S2 L1C image folder.
    # create a downloading bar
    if quiet:
        os.system("gsutil -q cp -r %s %s" % (file_img, output))
    else:
        os.system("gsutil cp -r %s %s" % (file_img, output))

    return True


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
    if not os.system("gsutil ls > /dev/null 2>&1") == 0:
        raise Exception("gsutil is not working")

    return True
