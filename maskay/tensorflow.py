import numpy as np

try:
    import tensorflow as tf
except ImportError:
    raise ImportError("Please install the following packages: torch.")

from maskay.module import MaskayModule


class Module(MaskayModule):
    def __init__(
        self,
        cropsize: int = 512,
        overlap: int = 32,
        device: str = "cpu",
        batchsize: int = 1,
        order: str = "BCWH",
        quiet: int = False,
    ):
        super().__init__(cropsize, overlap, device, batchsize, order, quiet)

    def inProcessing(self, tensor: np.ndarray):
        pass

    def forward(self, x):
        pass

    def outProcessing(self, tensor: np.ndarray):
        pass

    def _run(self, tensor: np.ndarray):
        if self.device == "cuda":
            with tf.device("/GPU:0"):
                tensor = np.moveaxis(self.forward(tensor).numpy(), 3, 1)
        elif self.device == "cpu":
            with tf.device("/CPU:0"):
                tensor = np.moveaxis(self.forward(tensor).numpy(), 3, 1)
        else:
            raise ValueError("Unknown device: " + str(self.device))
        return tensor