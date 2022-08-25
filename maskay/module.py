import itertools
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import xarray as xr
from rasterio.enums import Resampling
from tqdm import tqdm

from maskay.utils import color

from .tensorsat import TensorSat


class MaskayModule:
    def __init__(self, cropsize, overlap, device, batchsize, quiet):
        self.cropsize = cropsize
        self.overlap = overlap
        self.device = device
        self.batchsize = batchsize
        self.quiet = quiet

    @property
    def cropsize(self) -> int:
        return self._cropsize

    @cropsize.setter
    def cropsize(self, value: int):
        if not isinstance(value, int):
            raise TypeError("cropsize must be an integer")
        self._cropsize = value

    @property
    def overlap(self) -> int:
        return self._overlap

    @overlap.setter
    def overlap(self, value: int):
        if not isinstance(value, int):
            raise TypeError("overlap must be an integer")
        self._overlap = value

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, value: str):
        if not isinstance(value, str):
            raise TypeError("device must be a string")
        if not value in ["cpu", "cuda"]:
            raise ValueError("device must be either cpu or cuda")
        self._device = value

    @property
    def batchsize(self) -> int:
        return self._batchsize

    @batchsize.setter
    def batchsize(self, value: int):
        if not isinstance(value, int):
            raise TypeError("batchsize must be an integer")
        self._batchsize = value

    @property
    def quiet(self) -> bool:
        return self._quiet

    @quiet.setter
    def quiet(self, value: int):
        if not isinstance(value, int):
            raise TypeError("quiet must be an integer")
        self._quiet = value

    def get_resolution(self, tensor: TensorSat):
        return tensor.resolution()

    def get_cropsize(self, tensor: TensorSat):
        # Base parameters
        cropsize_base = self.cropsize

        # Resolution of the base raster (lowest resolution)
        res_base = tensor.rasterbase().rio.resolution()[0]

        # Resolution of each raster
        res_arrays = self.get_resolution(tensor)

        # Estimate zero-coordinate from each image patch
        dict_cropsize = OrderedDict()
        for key, _ in tensor.dictarray.items():
            factor = res_base / res_arrays[key]
            dict_cropsize[key] = int(np.ceil(cropsize_base * factor))
        return dict_cropsize

    def get_overlap(self, tensor: TensorSat):
        # Base parameters
        overlap_base = self.overlap
        res_base = tensor.rasterbase().rio.resolution()[0]

        # Resolution of each raster
        res_arrays = self.get_resolution(tensor)

        # Estimate zero-coordinate from each image patch
        dict_cropsize = OrderedDict()
        for key, _ in tensor.dictarray.items():
            factor = res_base / res_arrays[key]
            dict_cropsize[key] = int(np.ceil(overlap_base * factor))
        return dict_cropsize

    def get_ips(self, zerocoord: OrderedDict):
        return len(list(zerocoord.values())[0])

    def _MagickCrop(self, tensor: TensorSat):
        # Get the cropsize and overlap for each raster
        tensor_cropsize = self.get_cropsize(tensor)
        tensor_overlap = self.get_overlap(tensor)

        # Estimate zero-coordinate from each image patch
        dict_mrs = OrderedDict()
        for key, value in tensor.dictarray.items():
            cropsize_array = tensor_cropsize[key]
            overlap_array = tensor_overlap[key]
            mrs = self._Crop(
                tensor=value, cropsize=cropsize_array, overlap=overlap_array
            )
            dict_mrs[key] = mrs
        return dict_mrs

    def _Crop(self, tensor: xr.DataArray, cropsize: int, overlap: int):
        # Select the raster with the lowest resolution
        tshp = tensor.shape

        # if the image is too small, return (0, 0)
        if (tshp[0] < cropsize) and (tshp[1] < cropsize):
            return [(0, 0)]

        # Define relative coordinates.
        xmn, xmx, ymn, ymx = (0, tshp[0], 0, tshp[1])

        if overlap > cropsize:
            raise ValueError("The overlap must be smaller than the cropsize")

        xrange = np.arange(xmn, xmx, (cropsize - overlap))
        yrange = np.arange(ymn, ymx, (cropsize - overlap))

        # If there is negative values in the range, change them by zero.
        xrange[xrange < 0] = 0
        yrange[yrange < 0] = 0

        # Remove the last element if it is outside the tensor
        xrange = xrange[xrange - (tshp[0] - cropsize) <= 0]
        yrange = yrange[yrange - (tshp[1] - cropsize) <= 0]

        # If the last element is not (tshp[1] - cropsize) add it!
        if xrange[-1] != (tshp[0] - cropsize):
            xrange = np.append(xrange, tshp[0] - cropsize)
        if yrange[-1] != (tshp[1] - cropsize):
            yrange = np.append(yrange, tshp[1] - cropsize)

        # Create all the relative coordinates
        mrs = list(itertools.product(xrange, yrange))

        return mrs

    def _align(self, reftensor: xr.DataArray, tensor: xr.DataArray):
        # Select a raster base based on the lowest resolution.
        return tensor.rio.reproject(
            dst_crs=reftensor.rio.crs,
            shape=reftensor.shape,
            transform=reftensor.rio.transform(),
            resampling=Resampling.nearest,
        )

    def _MagickGather(self, outensor: np.ndarray, batch_step: List):
        """Gather the image patches and merge them into a single tensor.

        Args:
            tensorprob (np.ndarray): A tensor with shape (B, C, H, W).
            outensor (np.ndarray): A tensor with shape (C, H, W).
            quiet (bool, optional): If True, do not print any messages. Defaults to False.

        Returns:
            np.ndarray: A np.ndarray with shape (C, H, W) with the image patches.
        """

        # Get CropMagick properties
        xmin, xmax = (0, outensor.shape[-1])  # X borders of the output tensor
        ymin, ymax = (0, outensor.shape[-2])  # Y borders of the output tensor

        container = list()
        for coord in batch_step:
            #  Define X dimension pixels
            if coord[0] == xmin:
                Xmin = coord[0]
                XIPmin = 0
            else:
                Xmin = coord[0] + self.overlap // 2
                XIPmin = self.overlap // 2

            if (coord[0] + self.cropsize) == xmax:
                Xmax = coord[0] + self.cropsize
                XIPmax = self.cropsize
            else:
                Xmax = coord[0] + self.cropsize - self.overlap // 2
                XIPmax = self.cropsize - self.overlap // 2

            #  Define Y dimension pixels
            if coord[1] == ymin:
                Ymin = coord[1]
                YIPmin = 0
            else:
                Ymin = coord[1] + self.overlap // 2
                YIPmin = self.overlap // 2

            if (coord[1] + self.cropsize) == ymax:
                Ymax = coord[1] + self.cropsize
                YIPmax = self.cropsize
            else:
                Ymax = coord[1] + self.cropsize - self.overlap // 2
                YIPmax = self.cropsize - self.overlap // 2

            # Put the IP tensor in the output tensor
            container.append(
                {
                    "outensor": [(Xmin, Ymin), (Xmax, Ymax)],
                    "ip": [(XIPmin, YIPmin), (XIPmax, YIPmax)],
                }
            )
        return container


class ModuleTorch(MaskayModule):
    def __init__(
        self,
        cropsize: int = 512,
        overlap: int = 32,
        device: str = "cpu",
        batchsize: int = 1,
        quiet: int = False,
    ):
        super().__init__(cropsize, overlap, device, batchsize, quiet)

    def inProcessing(self, tensor: np.ndarray):
        pass

    def forward(self, x):
        pass

    def outProcessing(self, tensor: np.ndarray):
        pass

    def _run(self, tensor: np.ndarray):
        tensor = torch.Tensor(tensor)

        # Run the model
        with torch.no_grad():
            if self.device == "cuda":
                tensor = tensor.cuda()
            tensor = self.forward(tensor).detach().cpu().numpy()
            torch.cuda.empty_cache()
        return tensor

    def _predict(self, tensor: TensorSat):
        # Obtain the zero coordinate to create an IP
        zero_coord = self._MagickCrop(tensor)

        # Number of image patches (IPs)
        IPslen = self.get_ips(zero_coord)

        # Get the cropsize for each raster
        tensor_cropsize = self.get_cropsize(tensor)

        # Raster ref (lowest resolution)
        rbase = tensor.rasterbase()
        rbase_name = tensor.rasterbase_name()

        # Create outensor
        outensor = None

        batch_iter = range(0, IPslen, self.batchsize)
        for index in tqdm(batch_iter, disable=self.quiet):
            batched_IP = list()
            zerocoords = list()
            for index2 in range(index * self.batchsize, (index + 1) * self.batchsize):
                # Container to create a full IP with all bands with the same resolution
                IP = list()

                # Reference raster IP
                bmrx, bmry = zero_coord[rbase_name][index2]
                rbase_ip = rbase[
                    bmrx : (bmrx + self.cropsize), bmry : (bmry + self.cropsize)
                ]
                base_cropsize = tensor_cropsize[rbase_name]

                for key, _ in zero_coord.items():
                    # Select the zero coordinate
                    mrx, mry = zero_coord[key][index2]

                    # Obtain the specific cropsize for each raster
                    cropsize = tensor_cropsize[key]

                    # Crop the full raster using specific coordinates
                    tensorIP = tensor.dictarray[key][
                        mrx : (mrx + cropsize), mry : (mry + cropsize)
                    ]

                    # Resample the raster to the reference raster
                    if base_cropsize != cropsize:
                        tensorIP = self._align(rbase_ip, tensorIP)

                    # Append the IP to the container
                    IP.append(tensorIP.to_numpy())

                # Stack the IP
                IP = np.stack(IP, axis=0)

                # Append the IP to the batch
                zerocoords.append((mrx, mry))
                batched_IP.append(IP)

            # Stack the batch
            batched_IP = np.stack(batched_IP, axis=0)

            # Run the preprocessing
            batched_IP = self.inProcessing(batched_IP)

            # Run the model
            batched_IP = self._run(batched_IP)

            # Run the postprocessing
            batched_IP = self.outProcessing(batched_IP)

            # If is the first iteration, create the output tensor
            if outensor is None:
                classes = batched_IP.shape[1]
                dtype = batched_IP.dtype
                outensor = np.zeros(
                    shape=(classes, rbase.shape[0], rbase.shape[1]), dtype=dtype
                )

            # Copy the IP values in the outputtensor
            gather_zerocoord = self._MagickGather(outensor, zerocoords)

            for index, zcoords in enumerate(gather_zerocoord):
                (Xmin, Ymin), (Xmax, Ymax) = zcoords["outensor"]
                (XIPmin, YIPmin), (XIPmax, YIPmax) = zcoords["ip"]
                outensor[:, Xmin:Xmax, Ymin:Ymax] = batched_IP[
                    index, :, XIPmin:XIPmax, YIPmin:YIPmax
                ]

        # Create the output tensor rio xarray object
        xcoord, ycoord = rbase.coords["y"].values, rbase.coords["x"].values
        coords = [np.arange(0, classes), xcoord, ycoord]
        dims = ["band", "y", "x"]

        return (
            xr.DataArray(outensor, coords=coords, dims=dims)
            .rio.write_nodata(-999)
            .rio.write_transform(rbase.rio.transform())
        )

    def __repr__(self) -> str:
        message = (
            color.BLUE
            + color.BOLD
            + "Maskay object parameters: "
            + color.END
            + "\n"
            + color.BLUE
            + color.BOLD
            + "   cropsize  : "
            + color.END
            + str(self.cropsize)
            + "\n"
            + color.BLUE
            + color.BOLD
            + "   overlap   : "
            + color.END
            + str(self.overlap)
            + "\n"
            + color.BLUE
            + color.BOLD
            + "   batchsize : "
            + color.END
            + str(self.batchsize)
            + "\n"
            + color.BLUE
            + color.BOLD
            + "   device    : "
            + color.END
            + self.device
            + "\n"
            + color.BLUE
            + color.BOLD
            + "   framework : "
            + color.END
            + "torch"
        )
        return message

    def __str__(self) -> str:
        return self.__repr__()
