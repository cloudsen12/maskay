from .safe import SAFE


def SAFEGEE(
    s2id: str,
    output: str,
    runchecks: bool = True,
    quiet: bool = False,
    method: str = "requests",
):
    """Download a Sentinel-2 image using Google Earth Engine IDs.

    Args:
        s2id (str): The GEE Sentinel-2 ID.
        output (str): The output directory.
        runchecks (bool, optional): If True, run checks on the Earth Engine and gsutil. Defaults to True.
        quiet (bool, optional): If True, do not print info messages. Defaults to False.
        method (str, optional): The method to use to download the image. Defaults to "requests". Available options are "requests" and "gsutil".

    Raises:
        Exception: If the Earth Engine is not working.

    Returns:
        str: The path to the downloaded image.

    Examples:
        >>> import ee
        >>> import maskay
        >>> ee.Initialize()
        >>> s2id = "20190212T142031_20190212T143214_T19FDF"
        >>> s2idpath = maskay.download.s2.SAFEGEE(s2id, "/home/user/")
    """

    try:
        import ee
    except ImportError:
        print("Please install the following packages: earthengine-api.")

    # check if ee is initialized
    if runchecks:
        __eecheck()

    # Identify S2 image using GEE S2 ID
    s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    s2 = s2.filter(ee.Filter.eq("system:index", s2id)).first()

    # Get tile information.
    productid = s2.get("PRODUCT_ID").getInfo()

    # Download S2 image.
    BASEPATH = SAFE(productid, output, runchecks, quiet, method)

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
