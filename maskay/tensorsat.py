from collections import OrderedDict
from typing import Union

import numpy as np
import rioxarray
import xarray as xr
from affine import Affine
from rasterio.enums import Resampling

from .utils import color


class TensorSat:
    def __init__(
        self,
        cache=False,
        align=False,
        Aerosol=None,
        Blue=None,
        Green=None,
        Red=None,
        RedEdge1=None,
        RedEdge2=None,
        RedEdge3=None,
        NIR=None,
        NIR2=None,
        WaterVapor=None,
        Cirrus=None,
        SWIR1=None,
        SWIR2=None,
        TIR1=None,
        TIR2=None,
        HV=None,
        VH=None,
        HH=None,
        VV=None,
    ):
        self.cache: bool = cache
        self.align: bool = align
        self.Aerosol: Union[str, np.ndarray, xr.DataArray] = Aerosol
        self.Blue: Union[str, np.ndarray, xr.DataArray] = Blue
        self.Green: Union[str, np.ndarray, xr.DataArray] = Green
        self.Red: Union[str, np.ndarray, xr.DataArray] = Red
        self.RedEdge1: Union[str, np.ndarray, xr.DataArray] = RedEdge1
        self.RedEdge2: Union[str, np.ndarray, xr.DataArray] = RedEdge2
        self.RedEdge3: Union[str, np.ndarray, xr.DataArray] = RedEdge3
        self.NIR: Union[str, np.ndarray, xr.DataArray] = NIR
        self.NIR2: Union[str, np.ndarray, xr.DataArray] = NIR2
        self.WaterVapor: Union[str, np.ndarray, xr.DataArray] = WaterVapor
        self.Cirrus: Union[str, np.ndarray, xr.DataArray] = Cirrus
        self.SWIR1: Union[str, np.ndarray, xr.DataArray] = SWIR1
        self.SWIR2: Union[str, np.ndarray, xr.DataArray] = SWIR2
        self.TIR1: Union[str, np.ndarray, xr.DataArray] = TIR1
        self.TIR2: Union[str, np.ndarray, xr.DataArray] = TIR2
        self.HV: Union[str, np.ndarray, xr.DataArray] = HV
        self.VH: Union[str, np.ndarray, xr.DataArray] = VH
        self.HH: Union[str, np.ndarray, xr.DataArray] = HH
        self.VV: Union[str, np.ndarray, xr.DataArray] = VV
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
        self.dictarray = None
        self.bounds = None

    def to_xarray(self, object: Union[str, np.ndarray] = None) -> xr.DataArray:
        """From string and numpy array to xr.DataArray.

        Args:
            object (Union[str, np.ndarray], optional): Object to convert.
                Defaults to None.

        Raises:
            ValueError: If the object is not a string or a numpy array.

        Returns:
            xr.DataArray: A xr.DataArray object.
        """
        if isinstance(object, str):
            with rioxarray.open_rasterio(object, "r") as src:
                raster = np.squeeze(src, axis=0)
                if self.cache:
                    raster = raster.compute()
                return raster

        elif isinstance(object, np.ndarray):
            coords = [list(range(d)) for d in object.shape]
            dims = ["y", "x"]
            return (
                xr.DataArray(object, coords=coords, dims=dims)
                .rio.write_nodata(-999)
                .rio.write_transform(Affine(1.0, 0.0, -0.5, 0.0, 1.0, -0.5))
            )
        elif isinstance(object, xr.DataArray):
            return object
        elif object is None:
            return None
        else:
            raise ValueError("object must be a string, or an array (numpy, xarray).")

    @property
    def Aerosol(self):
        return self._Aerosol

    @Aerosol.setter
    def Aerosol(self, value):
        self._Aerosol = self.to_xarray(value)

    @property
    def Blue(self):
        return self._Blue

    @Blue.setter
    def Blue(self, value):
        self._Blue = self.to_xarray(value)

    @property
    def Green(self):
        return self._Green

    @Green.setter
    def Green(self, value):
        self._Green = self.to_xarray(value)

    @property
    def Red(self):
        return self._Red

    @Red.setter
    def Red(self, value):
        self._Red = self.to_xarray(value)

    @property
    def RedEdge1(self):
        return self._RedEdge1

    @RedEdge1.setter
    def RedEdge1(self, value):
        self._RedEdge1 = self.to_xarray(value)

    @property
    def RedEdge2(self):
        return self._RedEdge2

    @RedEdge2.setter
    def RedEdge2(self, value):
        self._RedEdge2 = self.to_xarray(value)

    @property
    def RedEdge3(self):
        return self._RedEdge3

    @RedEdge3.setter
    def RedEdge3(self, value):
        self._RedEdge3 = self.to_xarray(value)

    @property
    def NIR(self):
        return self._NIR

    @NIR.setter
    def NIR(self, value):
        self._NIR = self.to_xarray(value)

    @property
    def NIR2(self):
        return self._NIR2

    @NIR2.setter
    def NIR2(self, value):
        self._NIR2 = self.to_xarray(value)

    @property
    def WaterVapor(self):
        return self._WaterVapor

    @WaterVapor.setter
    def WaterVapor(self, value):
        self._WaterVapor = self.to_xarray(value)

    @property
    def Cirrus(self):
        return self._Cirrus

    @Cirrus.setter
    def Cirrus(self, value):
        self._Cirrus = self.to_xarray(value)

    @property
    def SWIR1(self):
        return self._SWIR1

    @SWIR1.setter
    def SWIR1(self, value):
        self._SWIR1 = self.to_xarray(value)

    @property
    def SWIR2(self):
        return self._SWIR2

    @SWIR2.setter
    def SWIR2(self, value):
        self._SWIR2 = self.to_xarray(value)

    @property
    def TIR1(self):
        return self._TIR1

    @TIR1.setter
    def TIR1(self, value):
        self._TIR1 = self.to_xarray(value)

    @property
    def TIR2(self):
        return self._TIR2

    @TIR2.setter
    def TIR2(self, value):
        self._TIR2 = self.to_xarray(value)

    @property
    def HV(self):
        return self._HV

    @HV.setter
    def HV(self, value):
        self._HV = self.to_xarray(value)

    @property
    def VH(self):
        return self._VH

    @VH.setter
    def VH(self, value):
        self._VH = self.to_xarray(value)

    @property
    def HH(self):
        return self._HH

    @HH.setter
    def HH(self, value):
        self._HH = self.to_xarray(value)

    @property
    def VV(self):
        return self._VV

    @VV.setter
    def VV(self, value):
        self._VV = self.to_xarray(value)

    @property
    def dictarray(self):
        return self._dictarray

    @dictarray.setter
    def dictarray(self, value):
        dictarray = OrderedDict()
        for _, value in self.ref_rev.items():
            obj = eval("self.%s" % value)
            if isinstance(obj, xr.DataArray):
                dictarray[value] = obj

        if self.align:
            resdata = {k: v.rio.resolution()[0] for k, v in dictarray.items()}
            reftensor = dictarray[min(resdata, key=resdata.get)]
            reftensor_res = reftensor.rio.resolution()[0]
            for key, value in dictarray.items():
                if reftensor_res != value.rio.resolution()[0]:
                    dictarray[key] = value.rio.reproject(
                        dst_crs=reftensor.rio.crs,
                        shape=reftensor.shape,
                        transform=reftensor.rio.transform(),
                        resampling=Resampling.nearest,
                    )
                else:
                    dictarray[key] = value
        self._dictarray = dictarray

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        bounds = set([x.rio.bounds() for x in self.dictarray.values()])
        if len(bounds) > 1:
            raise ValueError("All bands must have the same bounds.")
        self._bounds = bounds.pop()

    def to_dict(self):
        return self.dictarray

    def to_list(self):
        return list(self.dictarray.values())

    def resolution(self):
        dict_res = OrderedDict()
        for key, value in self.dictarray.items():
            dict_res[key] = value.rio.resolution()[0]
        return dict_res

    def rasterbase(self):
        res = self.resolution()
        # Obtain the raster resolution for each band.
        raster_res = [value for key, value in res.items()]

        # Select the raster with the lowest resolution.
        rasterbase_res_index = np.array(raster_res).argmin()
        rasterbase_res_band = list(res.keys())[rasterbase_res_index]
        return self.dictarray[rasterbase_res_band]

    def rasterbase_name(self):
        res = self.resolution()
        # Obtain the raster resolution for each band.
        raster_res = [value for key, value in res.items()]

        # Select the raster with the lowest resolution.
        rasterbase_res_index = np.array(raster_res).argmin()
        return list(res.keys())[rasterbase_res_index]

    def message(self, object: xr.DataArray, band: str, order: int):
        if isinstance(object, xr.DataArray):
            equal_space = 13
            white_space = " " * (equal_space - len("%s [%02d]" % (band, 0)))
            resraster = object.rio.resolution()[0]
            band = color.BLUE + color.BOLD + band + color.END
            order = color.BLUE + color.BOLD + ("%02d" % order) + color.END
            resolution = color.BLUE + color.BOLD + ("%0.1f" % resraster) + color.END

            shp = (
                band,
                white_space,
                order,
                object.shape[0],
                object.shape[1],
                resolution,
            )
            return "%s %s[%s]: <xarray.DataArray (y: %s, x: %s, res: %s)> \n" % shp
        return ""

    def __repr__(self) -> str:
        msg = ""
        for key, value in self.ref_rev.items():
            obj = eval("self.%s" % value)
            if isinstance(obj, xr.DataArray):
                msg += self.message(self.dictarray[value], value, key)
        return msg
