from maskay.tensorsat import TensorSat


class Predictor:
    def __init__(
        self,
        cropsize: int = 512,
        overlap: int = 32,
        device: str = "cpu",
        batchsize: int = 1,
        order: int = "BCHW",
        quiet: int = False,
    ):
        self.batchsize = batchsize
        self.cropsize = cropsize
        self.overlap = overlap
        self.device = device
        self.order = order
        self.quiet = quiet
        self.result = None

    def predict(self, model, tensor: TensorSat):
        model.batchsize = self.batchsize
        model.cropsize = self.cropsize
        model.overlap = self.overlap
        model.device = self.device
        model.order = self.order
        model.quiet = self.quiet
        self.result = model._predict(tensor=tensor)
        return self.result

    def __repr__(self) -> str:
        # no long line
        msg = (
            "Predictor("
            + "cropsize="
            + str(self.cropsize)
            + ", "
            + "overlap="
            + str(self.overlap)
            + ", "
            + "device="
            + str(self.device)
            + ", "
            + "order="
            + str(self.order)
            + "batchsize="
            + str(self.batchsize)
            + ", "
            + "quiet="
            + str(self.quiet)
            + ")"
        )

        return msg

    def __str__(self) -> str:
        return self.__repr__()
