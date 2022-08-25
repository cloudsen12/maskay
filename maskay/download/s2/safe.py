import os
import pathlib
import re
from xml.dom import minidom

import requests

from ...utils import color


def SAFE(
    productid: str,
    output: str,
    runchecks: bool = True,
    quiet: bool = False,
    method: str = "requests",
):
    """Download a Sentinel-2 image using a product ID.

    Args:
        productid (str): The full id of the Sentinel-2 product.
        output (str): The output directory to save the image.
        runchecks (bool, optional): If True, run checks on the gsutil. Only used if method is "gsutil". Defaults to True.
        quiet (bool, optional): If True, do not print info messages. Defaults to False.
        method (str, optional): The method to use to download the image. Defaults to "requests". Available options are "requests" and "gsutil".

    Returns:
        str: The path to the downloaded image.

    Raises:
        Exception: If the productid is not valid.
        Exception: If gsutil is not installed.

    Examples:
        >>> import maskay
        >>> s2idfull = "S2A_MSIL1C_20190212T142031_N0207_R010_T19FDF_20190212T191443"
        >>> s2idpath = maskay.download.s2.SAFE(s2idfull, "/home/user/")
    """

    # Get tile grid info.
    rgx_expr = "(.*)_(.*)_(.*)_(.*)_(.*)"
    tile_info = re.search(rgx_expr, productid, re.IGNORECASE).group(4)[1:]
    p01 = tile_info[0:2]
    p02 = tile_info[2]
    p03 = tile_info[3:5]

    # Download the S2 L1C image folder.
    if method == "gsutil":
        # check if gsutil is installed
        if runchecks:
            __gsutilcheck()

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

        # if models is empty return an exception
        if len(models) == 0:
            raise Exception(
                color.RED
                + "No images found. Please check the"
                + color.BOLD
                + " productid"
                + color.END
                + "."
            )

        # Download all S2L1C bands
        for model in models:

            # Get S2 image band from google cloud storage
            S2todownload = (folder_img + model.firstChild.data) + ".jp2"
            band = re.search(
                string=pathlib.Path(model.firstChild.data).stem,
                pattern="(.*)_(.*)_(.*)",
                flags=re.IGNORECASE,
            ).group(3)

            # NO download the TCI image if it is exists.
            if band == "TCI":
                continue

            if not quiet:
                print(
                    color.BLUE
                    + color.BOLD
                    + "Downloading [%s]: " % band
                    + color.END
                    + color.UNDERLINE
                    + S2todownload
                    + color.END
                )

            # where to save the S2 image
            S2tosave = BASEPATH / model.firstChild.data

            # Create the folder if it does not exist.
            S2tosave.parent.mkdir(parents=True, exist_ok=True)
            S2tosave = S2tosave.as_posix() + ".jp2"

            # Download S2 images
            r = requests.get(S2todownload, allow_redirects=True)
            with open(S2tosave, "wb") as f:
                f.write(r.content)
    return BASEPATH.as_posix()


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
