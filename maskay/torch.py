from tqdm import tqdm
from maskay.utils import color
import numpy as np
import xarray as xr

try:
    import torch
except ImportError:
    raise ImportError("Please install the following packages: torch.")

from maskay.module import MaskayModule
from maskay.tensorsat import TensorSat


class Module(MaskayModule):
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
                zerocoords.append((bmrx, bmry))                
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
            
            for index3, zcoords in enumerate(gather_zerocoord):
                # Coordinates to copy the IP
                (Xmin, Ymin), (Xmax, Ymax) = zcoords["outensor"]
                (XIPmin, YIPmin), (XIPmax, YIPmax) = zcoords["ip"]
                
                # Copy the IP
                outensor[:, Xmin:Xmax, Ymin:Ymax] = batched_IP[
                    index3, :, XIPmin:XIPmax, YIPmin:YIPmax
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
