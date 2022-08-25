import numpy as np
import rasterio.features

from pyproj import Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

S2BANDS_L2A = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
    "SCL",
    "AOT",
    "WVP",
]


def PC(coords, iniDate, endDate, buffer, bands=None):
    # Check if packages are installed
    is_external_package_installed = []

    try:
        import planetary_computer as pc
    except ImportError:
        is_external_package_installed.append("planetary_computer")

    try:
        import stackstac
    except ImportError:
        is_external_package_installed.append("stackstac")

    try:
        import pystac_client
    except ImportError:
        is_external_package_installed.append("pystac_client")

    if is_external_package_installed != []:
        nopkgs = ', '.join(is_external_package_installed)
        raise ImportError(
            f"Please install the following packages: {nopkgs}."
        )

    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(coords[0], coords[1], coords[0], coords[1]),
    )
    epsg = int(utm_crs_list[0].code)

    aoi = towerFootprint(coords[0], coords[1], buffer)
    bbox = rasterio.features.bounds(towerFootprint(coords[0], coords[1], buffer, False))

    CATALOG = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )

    SEARCH = CATALOG.search(
        intersects=aoi, datetime=f"{iniDate}/{endDate}", collections=["sentinel-2-l2a"]
    )

    items = list(SEARCH.get_items())

    if bands is None:
        bands = S2BANDS_L2A

    S2 = signAndStack(items, bbox, epsg, bands)

    return S2


# Auxiliary functions ------------------------------------------------------
def signAndStack(items, bbox, epsg, bands):
    signed_items = []

    for item in items:
        item.clear_links()
        signed_items.append(pc.sign(item).to_dict())

    S2 = stackstac.stack(
        signed_items,
        assets=bands,
        resolution=10,
        bounds=bbox,
        epsg=epsg,
    ).where(
        lambda x: x > 0, other=np.nan
    )  # NO DATA IS ZERO -> THEN TRANSFORM ZEROS TO NO DATA

    return S2


def towerFootprint(x, y, distance, latlng=True, resolution=10):
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(x, y, x, y),
    )
    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:" + utm_crs_list[0].code, always_xy=True
    )
    inverse_transformer = Transformer.from_crs(
        "EPSG:" + utm_crs_list[0].code, "EPSG:4326", always_xy=True
    )
    newCoords = transformer.transform(x, y)

    newCoords = [round(i / resolution) * resolution for i in newCoords]

    E = newCoords[0] + distance
    W = newCoords[0] - distance
    N = newCoords[1] + distance
    S = newCoords[1] - distance

    polygon = [
        [W, S],
        [E, S],
        [E, N],
        [W, N],
        [W, S],
    ]

    if latlng:
        polygon = [list(inverse_transformer.transform(x[0], x[1])) for x in polygon]

    footprint = {
        "type": "Polygon",
        "coordinates": [polygon],
    }

    return footprint
